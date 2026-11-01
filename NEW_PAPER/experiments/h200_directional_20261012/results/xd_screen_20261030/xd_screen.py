#!/usr/bin/env python3
"""Cross-domain layout mismatch screening (author directive 2026-10-30).
Read-only offline analysis. No GPU. No system or parameter changes.
For each candidate A->B domain pair: evaluate the SAME B-domain routing counts
under layouts {identity I, A-profile layout flat_place(A counts), B-dedicated
reference layout flat_place(B counts)}; report per-layer rank-load metrics and
whether the A-profile vs B-reference gap PERSISTS across 4 contiguous B segments.
B-dedicated layout is a DIAGNOSTIC REFERENCE ONLY (oracle-flavored: built from
the full B counts incl all eval segments). It is NOT called optimal and no
count improvement is converted into expected speedup anywhere.
Same logical-expert caliber as proven imbalance_offline.py:
  hr = sum_l max(0, max_rank_load_l - max(total_l/8, max_expert_l)) / sum_l total_l
       (identical formula to P0 controller h_rate; physical slot space 8 ranks x 16 slots)
  r_raw = mean_l(max_rank/mean_rank);  cv = mean_l(std_rank/mean_rank)
Layout construction: deepseek.py rebalance_experts(weight, num_replicas=128,
num_groups=1, num_nodes=1, num_gpus=8, enable_hierarchical=False).
"""
import json, glob, os, importlib.util, hashlib, datetime
import numpy as np
import torch

D = "/data/minghua/sjq/PAPER_EXPERIMENTS"
X = D + "/cross_domain_e2e_20261027"
O = "/data/minghua/sjq/OEPLBdata"
OUT = "/tmp/xd_screen"
ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

spec = importlib.util.spec_from_file_location(
    "ds_eplb",
    "/opt/conda/lib/python3.11/site-packages/sglang/srt/eplb/eplb_algorithms/deepseek.py")
dse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dse)

def flat_place(counts_le):
    w = torch.tensor(np.asarray(counts_le), dtype=torch.float32)
    p2l, _l2p, _c = dse.rebalance_experts(
        weight=w, num_replicas=128, num_groups=1, num_nodes=1,
        num_gpus=8, enable_hierarchical=False)
    return np.asarray(p2l, dtype=np.int64)

def rank_loads(counts, p2l):
    c = np.asarray(counts, dtype=np.float64)
    p = np.asarray(p2l, dtype=np.int64)
    L = c.shape[0]
    pl = p.reshape(L, 8, 16)
    loads = np.zeros((L, 8), dtype=np.float64)
    for r in range(8):
        loads[:, r] = np.take_along_axis(c, pl[:, r, :], axis=1).sum(1)
    return loads

def metrics(counts, p2l):
    c = np.asarray(counts, dtype=np.float64)
    loads = rank_loads(c, p2l)
    total = c.sum(1)
    max_e = c.max(1)
    floor = np.maximum(total / 8.0, max_e)
    headroom = np.maximum(loads.max(1) - floor, 0.0)
    mean_l = loads.mean(1)
    rr = loads.max(1) / np.maximum(mean_l, 1e-12)
    return {
        "hr": float(headroom.sum() / max(float(total.sum()), 1.0)),
        "r_raw": float(rr.mean()),
        "cv": float((loads.std(1) / np.maximum(mean_l, 1e-12)).mean()),
        "r_raw_min": float(rr.min()),
        "r_raw_max": float(rr.max()),
        "per_layer_r_raw": [float(v) for v in rr],
        "count_sum": int(c.sum()),
    }

result = {"meta": {
    "ts": ts,
    "purpose": "author directive 2026-10-30: low-cost cross-domain placement-mismatch screening, offline only",
    "metric_formulas": "hr identical to P0 controller h_rate (8 ranks x 16 slots physical); r_raw mean per-layer max/mean; cv mean per-layer std/mean",
    "layout_construction": "deepseek.py rebalance_experts flat (num_replicas=128, num_groups=1, num_nodes=1, num_gpus=8, non-hierarchical), same as proven imbalance_offline.py",
    "caveats": [
        "B-dedicated reference layout is DIAGNOSTIC REFERENCE ONLY, built from full B counts including all eval segments (oracle-flavored); NOT called optimal",
        "no count-level improvement is converted into expected speedup anywhere",
        "e3rec domains are uniform in512/out1 retokenized caliber; external validity to native-length domains (e.g. book 4438tok) checked separately via book_pub supplementary rows",
    ],
}}

# ---- sources ----
pt = X + "/prep/record_boot2/expert_distribution_recorder_1793078907.6631618.pt"
result["meta"]["pt_sha256"] = hashlib.sha256(open(pt, "rb").read()).hexdigest()
d = torch.load(pt, weights_only=False, map_location="cpu")
a = d["logical_count"].to(torch.int64).numpy()
result["meta"]["pt_shape"] = list(a.shape)

# slice convention (same as imbalance_offline.py: boundary+4 .. next_boundary-4; boundaries empirical 132/260/388; step = 8192 hidden rows, top-8 assignments x 94 layers)
slices = {"sharegpt": (4, 128), "prover": (136, 256), "book": (264, 384), "code": (392, 512)}
counts = {k: a[lo:hi].sum(0).astype(np.float64) for k, (lo, hi) in slices.items()}
def seg_counts(k, nseg=4):
    lo, hi = slices[k]
    edges = [lo + (hi - lo) * i // nseg for i in range(nseg + 1)]
    return [a[edges[i]:edges[i + 1]].sum(0).astype(np.float64) for i in range(nseg)], edges
segs = {}; edges = {}
for k in slices:
    segs[k], edges[k] = seg_counts(k)

# book published traces (pb stage2 exact source population, prefill-only filter)
files = sorted(glob.glob(O + "/datasets/prefill_decode_correlation/traces/book_4438tok_O10/*.npz"))
book_pub = np.zeros((94, 128), dtype=np.float64)
book_segs = [np.zeros((94, 128), dtype=np.float64) for _ in range(4)]
nfw = 0; nprefill = 0
for i, f in enumerate(files):
    z = np.load(f)
    ip = np.asarray(z["is_prefill"], dtype=bool)
    lh = np.asarray(z["layer_hists"], dtype=np.int64)
    s = lh[ip].sum(0).astype(np.float64)
    book_pub += s
    book_segs[i * 4 // len(files)] += s
    nfw += int(len(ip)); nprefill += int(ip.sum())
segs["book_pub"] = book_segs
counts["book_pub"] = book_pub

# DF deployed profile counts (pb/pc ES+DF arms frozen placement source, sha PL_DF c1787ad1...)
dff = D + "/prover_book_switch_20261027/placements/datafore_prover_counts.json"
import time as _t
def read_retry(path, tries=20, gap=3):
    for i in range(tries):
        try:
            with open(path, "rb") as fh: return fh.read()
        except Exception:
            _t.sleep(gap)
    raise RuntimeError("unreadable after retries: " + path)
dfc = None
try:
    dfb = read_retry(dff, tries=6, gap=2)
    dfsha = hashlib.sha256(dfb).hexdigest()
    assert dfsha == "c1787ad19f16ef5d840d5a7c7e364ecf23fb3caafba682e56c0f21991a298f29", "DF profile sha mismatch: " + dfsha
    dfc = np.array(json.loads(dfb.decode())["logical_count"], dtype=np.float64)
    assert dfc.shape == (94, 128)
    result["meta"]["datafore_sha256"] = dfsha
    result["meta"]["datafore_source"] = "pb archive frozen placements copy; sha matches pb code_lock PL_DF/PL_ES lock c1787ad1..."
except Exception as e:
    result["meta"]["datafore_status"] = "MISSING at run time (all known copies unreadable, transient fuse failures): " + repr(e)[:160]
    result["meta"]["datafore_source"] = "attempted: " + dff

result["provenance"] = {
    "e3rec_sharegpt": {"domain": "sharegpt (mixed zh/en chat)", "input_len": "512 tok (uniform retok)", "output_rule": "max_tokens=1; temperature not separately recorded = MISSING", "stage": "all-forward logical counts, identity basis, O=1 prefill-dominated; step=8192 hidden rows", "sample": "2048 reqs; slice steps[4:128] (124 steps); count_sum=" + str(int(counts["sharegpt"].sum()))},
    "e3rec_prover": {"domain": "prover (DeepSeek-Prover-V1 retok)", "input_len": "512 tok (uniform retok)", "output_rule": "max_tokens=1; temperature MISSING", "stage": "same recording", "sample": "2048 reqs; slice steps[136:256] (120 steps); count_sum=" + str(int(counts["prover"].sum()))},
    "e3rec_book": {"domain": "book (BookCorpus retok)", "input_len": "512 tok (uniform retok)", "output_rule": "max_tokens=1; temperature MISSING", "stage": "same recording", "sample": "2048 reqs; slice steps[264:384] (120 steps); count_sum=" + str(int(counts["book"].sum()))},
    "e3rec_code": {"domain": "code (retok L512 5-source splice, same population as pc stage2)", "input_len": "512 tok (uniform retok)", "output_rule": "max_tokens=1; temperature MISSING", "stage": "same recording", "sample": "2048 reqs; slice steps[392:512] (120 steps); count_sum=" + str(int(counts["code"].sum()))},
    "book_pub": {"domain": "book published traces (exact pb stage2 source population)", "input_len": "~4438 tok", "output_rule": "max_tokens=10 (O10); temperature MISSING", "stage": "prefill forwards only (is_prefill filter)", "sample": str(len(files)) + " npz, " + str(nfw) + " forwards (" + str(nprefill) + " prefill), 1000 published rows; count_sum=" + str(int(book_pub.sum()))},
}
if dfc is not None:
    result["provenance"]["datafore_DF_profile"] = {"domain": "prover (DF/ES deployed arm profile, paper 5.3.1 lineage)", "input_len": "prover ~2048 tok lineage; exact capture batch not recorded alongside file = MISSING", "output_rule": "MISSING", "stage": "MISSING (2026-09-14 baseline_comparison capture)", "sample": "logical_count sum=" + str(int(dfc.sum()))}
else:
    result["provenance"]["datafore_DF_profile"] = {"domain": "prover (DF/ES deployed arm profile, paper 5.3.1 lineage)", "input_len": "MISSING (all source copies unreadable at run time)", "output_rule": "MISSING", "stage": "MISSING", "sample": "MISSING"}

# ---- layouts ----
ident = np.tile(np.arange(128), (94, 1)).astype(np.int64)
prof = {k: flat_place(counts[k]) for k in ("sharegpt", "prover", "book", "code")}
prof["book_pub"] = flat_place(book_pub)          # book_pub dedicated reference (pb-caliber)
df_deploy = flat_place(dfc) if dfc is not None else None   # deployed DF/ES arm layout (frozen old layout of pb/pc)

def eval_B(Bname, Bfull, Bsegs, Alayouts):
    """Alayouts: dict label -> p2l. Returns per-layout full metrics + per-segment hr table."""
    out = {}
    for label, lay in Alayouts.items():
        m = metrics(Bfull, lay)
        rr_A = np.array(m["per_layer_r_raw"])
        out[label] = {"full": {k: v for k, v in m.items() if k != "per_layer_r_raw"},
                      "per_layer_r_raw_min_max": [m["r_raw_min"], m["r_raw_max"]],
                      "segments": []}
        for s in Bsegs:
            out[label]["segments"].append(metrics(s, lay)["hr"])
        out[label]["_rr"] = rr_A
    return out

def pair_row(Aname, Bname, Bfull, Bsegs, extra=None):
    lays = {"I_identity": ident, "A_profile_" + Aname: prof[Aname] if Aname in prof else extra}
    if extra is not None and Aname in prof:
        lays["DF_deployed_prover"] = df_deploy
    lays["B_ref_" + Bname] = prof[Bname]
    ev = eval_B(Bname, Bfull, Bsegs, lays)
    rr_ref = ev["B_ref_" + Bname]["_rr"]
    row = {"A": Aname, "B": Bname, "layouts": {}}
    for label, e in ev.items():
        rr = e.pop("_rr")
        lw = int((rr > rr_ref + 1e-12).sum()) if label != "B_ref_" + Bname else 0
        hrA = e["full"]["hr"]
        e["layer_worse_than_Bref"] = lw
        e["seg_hr"] = e.pop("segments")
        row["layouts"][label] = e
    ap = "A_profile_" + Aname
    row["gap_full_Aprof_minus_Bref"] = row["layouts"][ap]["full"]["hr"] - row["layouts"]["B_ref_" + Bname]["full"]["hr"]
    row["gap_full_Aprof_minus_I"] = row["layouts"][ap]["full"]["hr"] - row["layouts"]["I_identity"]["full"]["hr"]
    row["seg_gaps"] = [g - r for g, r in zip(row["layouts"][ap]["seg_hr"], row["layouts"]["B_ref_" + Bname]["seg_hr"])]
    row["persistent_mismatch"] = bool(all(g > 0 for g in row["seg_gaps"]))
    row["min_seg_gap"] = min(row["seg_gaps"])
    return row

pairs = []
domains = ["sharegpt", "prover", "book", "code"]
for A in domains:
    for B in domains:
        if A == B: continue
        pairs.append(pair_row(A, B, counts[B], segs[B]))
result["pairs_e3rec"] = pairs

# supplementary: B = book_pub (pb native caliber), A profiles from e3rec + DF deployed
sup = []
sup_note_df = False
for A in domains:
    r = pair_row(A, "book_pub", book_pub, book_segs)
    sup.append(r)
if df_deploy is not None:
    r = {"A": "DF_deployed(prover2048-lineage)", "B": "book_pub", "layouts": {}}
    ev = eval_B("book_pub", book_pub, book_segs, {"I_identity": ident, "DF_deployed_prover": df_deploy, "B_ref_book_pub": prof["book_pub"]})
    rr_ref = ev["B_ref_book_pub"]["_rr"]
    for label, e in ev.items():
        rr = e.pop("_rr")
        e["layer_worse_than_Bref"] = int((rr > rr_ref + 1e-12).sum())
        e["seg_hr"] = e.pop("segments")
        r["layouts"][label] = e
    r["seg_gaps"] = [g - b for g, b in zip(r["layouts"]["DF_deployed_prover"]["seg_hr"], r["layouts"]["B_ref_book_pub"]["seg_hr"])]
    r["persistent_mismatch"] = bool(all(g > 0 for g in r["seg_gaps"]))
    r["gap_full_Aprof_minus_Bref"] = r["layouts"]["DF_deployed_prover"]["full"]["hr"] - r["layouts"]["B_ref_book_pub"]["full"]["hr"]
    sup.append(r)
else:
    sup_note_df = True
result["pairs_bookpub_supplementary"] = sup

with open(OUT + "/xd_screen.json", "w") as f:
    json.dump(result, f, indent=1)

# ---- markdown ----
L = []
L.append("# Cross-domain placement-mismatch screening (offline, read-only, " + ts + ")")
L.append("")
L.append("Caliber: same logical-expert basis as proven imbalance_offline.py. hr = P0 controller h_rate formula (8 ranks x 16 slots); r_raw = mean per-layer max/mean rank load; layouts = deepseek flat rebalance. B-dedicated reference layout is DIAGNOSTIC REFERENCE ONLY (built from full B counts incl all eval segments), NOT optimal; no count gap is converted to expected speedup. Segments = 4 contiguous slices of B counts (e3rec: equal step chunks; book_pub: file-order chunks).")
L.append("")
L.append("## Provenance (input len / output rule / stage / sample; MISSING marked)")
L.append("| count source | domain | input_len | output_rule | stage | sample |")
L.append("|---|---|---|---|---|---|")
for k, v in result["provenance"].items():
    L.append("| %s | %s | %s | %s | %s | %s |" % (k, v["domain"], v["input_len"], v["output_rule"], v["stage"], v["sample"]))
L.append("")
def main_table(rows, title):
    L.append("## " + title)
    L.append("| A->B | hr_I | hr_Aprof | hr_Bref | gap(A-Bref) | gap(A-I) | r_raw I/A/Bref | layers worse/94 | seg gaps g1..g4 | persistent |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo = r["layouts"]
        ap = [k for k in lo if k.startswith("A_profile") or k == "DF_deployed_prover"][0]
        br = [k for k in lo if k.startswith("B_ref")][0]
        sg = " ".join("%.4f" % g for g in r["seg_gaps"])
        L.append("| %s->%s | %.5f | %.5f | %.5f | %+.5f | %+.5f | %.3f/%.3f/%.3f | %d | %s | %s |" % (
            r["A"], r["B"], lo["I_identity"]["full"]["hr"], lo[ap]["full"]["hr"], lo[br]["full"]["hr"],
            r["gap_full_Aprof_minus_Bref"], r.get("gap_full_Aprof_minus_I", float("nan")),
            lo["I_identity"]["full"]["r_raw"], lo[ap]["full"]["r_raw"], lo[br]["full"]["r_raw"],
            lo[ap]["layer_worse_than_Bref"], sg, "YES" if r["persistent_mismatch"] else "no"))
    L.append("")
main_table(result["pairs_e3rec"], "Primary matrix: 12 A->B pairs on e3rec counts (uniform in512/out1 caliber)")
main_table(result["pairs_bookpub_supplementary"], "Supplementary: B = book_pub (pb native caliber ~4438tok, prefill-only, O10)")
# DF deployed rows on e3rec B domains
if df_deploy is None:
    L.append("## DF deployed layout rows: MISSING this run (datafore_prover_counts.json unreadable at run time on all known copies; transient storage failures). Full-count DF numbers remain available in the archived imbalance_offline.json grid (prover_frozen_DF_repro rows); segment-level DF evaluation deferred until the source is readable again.")
    L.append("")
else:
    L.append("## DF deployed layout (literal frozen old layout of pb/pc arms) on each e3rec B domain")
    L.append("| B | hr_I | hr_DFdeployed | hr_Bref | gap(DF-Bref) | seg gaps | persistent |")
    L.append("|---|---|---|---|---|---|---|")
    for B in domains:
        ev = eval_B(B, counts[B], segs[B], {"I": ident, "DF": df_deploy, "BR": prof[B]})
        g = [x - y for x, y in zip(ev["DF"]["segments"], ev["BR"]["segments"])]
        L.append("| %s | %.5f | %.5f | %.5f | %+.5f | %s | %s |" % (
            B, ev["I"]["full"]["hr"], ev["DF"]["full"]["hr"], ev["BR"]["full"]["hr"],
            ev["DF"]["full"]["hr"] - ev["BR"]["full"]["hr"], " ".join("%.4f" % x for x in g),
            "YES" if all(x > 0 for x in g) else "no"))
        for e in ev.values(): e.pop("_rr", None)
    L.append("")
with open(OUT + "/xd_screen.md", "w") as f:
    f.write("\n".join(L) + "\n")
print("WROTE", OUT + "/xd_screen.json and .md")
print("pairs:", len(pairs), "sup:", len(sup))
for r in pairs:
    print("  %s->%s gap=%+.5f persistent=%s min_seg=%.5f" % (r["A"], r["B"], r["gap_full_Aprof_minus_Bref"], r["persistent_mismatch"], r["min_seg_gap"]))
