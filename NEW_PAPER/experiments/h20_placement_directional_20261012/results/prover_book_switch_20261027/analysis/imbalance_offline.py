#!/usr/bin/env python3
"""Offline imbalance comparison per author directive 2026-10-27.

Compares the SAME recorded code routing counts under layouts:
identity / prover-frozen (DF repro) / e3-fair ILP / MT pinned / code-adapted
greedy (offline-constructed) / book-adapted greedy, using:
  - headroom_rate = sum_l max(0, max_rank_load_l - max(total_l/8, max_expert_l)) / sum_l total_l
    (identical formula to the P0 controller h_rate, PHYSICAL slot space, 8 ranks x 16 slots)
  - r_raw = mean_l (max_rank_load / mean_rank_load)   (cdx0 metric)
  - cv    = mean_l (std_rank_load / mean_rank_load)
P0 live h_rate plateaus are extracted from v2p1 logs for commensurate contrast.
All read-only. No GPU.
"""
import json, os, glob, importlib.util, traceback
import numpy as np
import torch

H = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
X = H + "/results/cross_domain_e2e_20261027"
PC = H + "/results/prover_code_switch_20261027"
PB = H + "/results/prover_book_switch_20261027"
B = "/data/minghua/sjq/OEPLBdata/experiment_logs/baseline_comparison_20260914"
M = "/data/minghua/sjq/OEPLBdata/experiment_logs/moetuner_baseline_20260916"
DST = "/data/minghua/sjq/OEPLBdata/datasets/prefill_decode_correlation"
OUT = PB + "/analysis"
errors = []

spec = importlib.util.spec_from_file_location(
    "ds_eplb",
    "/opt/conda/lib/python3.11/site-packages/sglang/srt/eplb/eplb_algorithms/deepseek.py")
dse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dse)

def flat_place(counts_le):
    w = torch.tensor(np.asarray(counts_le), dtype=torch.float32)
    p2l, _l2p, _cnt = dse.rebalance_experts(
        weight=w, num_replicas=128, num_groups=1, num_nodes=1,
        num_gpus=8, enable_hierarchical=False)
    return np.asarray(p2l)

def metrics(counts, p2l):
    c = np.asarray(counts, dtype=np.float64)
    p = np.asarray(p2l, dtype=np.int64)
    if c.ndim == 1:
        c = c[None, :]
        p = p[None, :]
    L = c.shape[0]
    pl = p.reshape(L, 8, 16)
    loads = np.zeros((L, 8), dtype=np.float64)
    for r in range(8):
        loads[:, r] = np.take_along_axis(c, pl[:, r, :], axis=1).sum(1)
    total = c.sum(1)
    max_e = c.max(1)
    floor = np.maximum(total / 8.0, max_e)
    headroom = np.maximum(loads.max(1) - floor, 0.0)
    mean_l = loads.mean(1)
    return {
        "headroom_rate": float(headroom.sum() / max(float(total.sum()), 1.0)),
        "r_raw": float(np.mean(loads.max(1) / np.maximum(mean_l, 1e-12))),
        "cv": float(np.mean(loads.std(1) / np.maximum(mean_l, 1e-12))),
        "per_layer_r_raw_min_max": [float(np.min(loads.max(1) / np.maximum(mean_l, 1e-12))), float(np.max(loads.max(1) / np.maximum(mean_l, 1e-12)))],
    }

result = {"meta": {
    "purpose": "author directive: attribute stage2 deficit - overhead vs insufficient static gain",
    "headroom_formula": "sum_l max(0, max_rank - max(total/8, max_expert)) / sum_l total; same as P0 h_rate; 8 ranks x 16 slots",
    "code_counts_provenance": "record_boot2 e3_multidomain_switch recording 2026-10-27 05:17-05:28Z conc256 identity-basis logical counts; stream = 4 blocks x 2048 (sharegpt, prover, book, code); step=8192 hidden rows; boundaries empirically at steps 132/260/388; code slice steps[392:512] primary, steps[404:512] sensitivity; pc stage2 code block = SAME retok L512 5-source splice population",
}}

# ---- 1. e3 record slices
try:
    pt = X + "/prep/record_boot2/expert_distribution_recorder_1793078907.6631618.pt"
    d = torch.load(pt, weights_only=False, map_location="cpu")
    a = d["logical_count"].to(torch.int64).numpy()
    tot = a.sum(axis=(1, 2))
    result["meta"]["pt_shape"] = list(a.shape)
    result["meta"]["nonzero_steps"] = [int(np.nonzero(tot)[0][0]), int(np.nonzero(tot)[0][-1])]
    result["meta"]["tail_step_totals_505_520"] = [[i, int(tot[i])] for i in range(505, min(520, len(tot)))]
    code_c = a[392:512].sum(0)
    code_alt = a[404:512].sum(0)
    book_c = a[264:384].sum(0)
    prov_c = a[136:256].sum(0)
    result["meta"]["slice_sums"] = {"code_392_512": int(code_c.sum()), "code_404_512": int(code_alt.sum()), "book_264_384": int(book_c.sum()), "prover_136_256": int(prov_c.sum())}
except Exception as e:
    errors.append(("e3_pt", traceback.format_exc()[-400:]))
    code_c = code_alt = book_c = prov_c = None

# ---- 2. layouts (94-layer)
try:
    prov_df = np.array(json.load(open(B + "/datafore_prover_placement.json"))["logical_count"], dtype=np.int64)
    p2l_df = flat_place(prov_df)
    e3fair = np.array(json.load(open(X + "/placements/placement_e3_fair.json"))["physical_to_logical_map"], dtype=np.int64)
    mt = np.array(json.load(open(M + "/artifacts/placement_pinned_fair.json"))["physical_to_logical_map"], dtype=np.int64)
    p2l_id = np.tile(np.arange(128), (94, 1))
    p2l_code = flat_place(code_c) if code_c is not None else None
    p2l_book = flat_place(book_c) if book_c is not None else None
    layout_sets = {"identity": p2l_id, "prover_frozen_DF_repro": p2l_df, "e3_fair_ILP": e3fair, "MT_pinned_fair": mt}
    if p2l_code is not None:
        layout_sets["code_adapted_greedy_offline"] = p2l_code
    if p2l_book is not None:
        layout_sets["book_adapted_greedy_offline"] = p2l_book
except Exception:
    errors.append(("layouts", traceback.format_exc()[-400:]))
    layout_sets = {}

# ---- 3. grid on identical counts
grid = {}
count_sets = {}
if code_c is not None:
    count_sets["code_e3rec_392_512"] = code_c
    count_sets["code_e3rec_404_512"] = code_alt
    count_sets["book_e3rec_264_384"] = book_c
    count_sets["prover_e3rec_136_256"] = prov_c
try:
    prov_df2 = prov_df
    count_sets["prover_datafore_full"] = prov_df2
except Exception:
    pass
for cn, cc in count_sets.items():
    grid[cn] = {ln: metrics(cc, pl) for ln, pl in layout_sets.items()}
result["grid_94layer"] = grid

# ---- 4. cross-checks
try:
    num = (prov_c.astype(float) * prov_df.astype(float)).sum(1)
    den = np.maximum(np.linalg.norm(prov_c.astype(float), axis=1) * np.linalg.norm(prov_df.astype(float), axis=1), 1e-9)
    cos = num / den
    result["crosscheck_prover_e3rec_vs_datafore_cosine"] = {"mean": float(cos.mean()), "min": float(cos.min()), "max": float(cos.max())}
except Exception:
    errors.append(("crosscheck_cos", traceback.format_exc()[-300:]))

# ---- 5. published book traces (exact counts for pb stage2 source)
try:
    book_pub = np.zeros((94, 128), dtype=np.int64)
    nfiles = 0
    for f in sorted(glob.glob(DST + "/traces/book_4438tok_O10/*.npz")):
        z = np.load(f)
        ip = np.asarray(z["is_prefill"], dtype=bool)
        lh = np.asarray(z["layer_hists"], dtype=np.int64)
        book_pub += lh[ip].sum(0)
        nfiles += 1
    result["meta"]["book_pub_trace_files"] = nfiles
    result["meta"]["book_pub_sum"] = int(book_pub.sum())
    lm = dict(layout_sets)
    lm["book_adapted_greedy_on_pub"] = flat_place(book_pub)
    result["grid_book_published"] = {ln: metrics(book_pub, pl) for ln, pl in lm.items()}
except Exception:
    errors.append(("book_pub", traceback.format_exc()[-400:]))

# ---- 6. cdx0 exact layer 24/71 (archived code-adapted layouts + counts)
cdx = {}
for layer in (24, 71):
    try:
        tag = "%03d" % layer
        lj = json.load(open(H + "/inputs/cdx0_v1/layouts/cdx0_cdx0-evaluation-code-layer-" + tag + ".json"))
        rm = json.load(open(H + "/inputs/cdx0_v1/route_metadata/cdx0-evaluation-code-layer-" + tag + ".json"))
        counts = np.array(rm["route_metadata"]["expert_counts"], dtype=np.float64)
        cols = {}
        for el in lj["layouts"]:
            p2l1 = np.array(el["p2l"], dtype=np.int64)
            m1 = metrics(counts, p2l1)
            m1["archived_r_raw"] = el.get("r_raw")
            m1["archived_padded_max_over_mean"] = el.get("padded_max_over_mean")
            cols[el["layout_id"]] = m1
        rmc = json.load(open(H + "/inputs/cdx0_v1/route_metadata/cdx0-calibration-code-layer-" + tag + ".json"))
        ccounts = np.array(rmc["route_metadata"]["expert_counts"], dtype=np.float64)
        cols_calib = {}
        for el in lj["layouts"]:
            p2l1 = np.array(el["p2l"], dtype=np.int64)
            cols_calib[el["layout_id"]] = metrics(ccounts, p2l1)
        cdx["layer_" + tag] = {"eval_code_counts": cols, "calib_code_counts": cols_calib,
                               "total_assignments_eval": rm["route_metadata"]["total_assignments"]}
    except Exception:
        errors.append(("cdx0_" + str(layer), traceback.format_exc()[-400:]))
result["cdx0_layers"] = cdx

# ---- 7. P0 live h_rate trajectories (pc done; pb if present)
def v2p1_table(path):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    per_w = {}
    for r in rows:
        w = r.get("window_id")
        if w is not None and w not in per_w:
            per_w[w] = r
    return per_w

def summarize_pc(per_w):
    ws = sorted(per_w)
    def avg(lo, hi, key):
        vals = [per_w[w].get(key) for w in ws if lo <= w <= hi and isinstance(per_w[w].get(key), float)]
        return (sum(vals) / len(vals)) if vals else None
    return {
        "n_windows": len(ws),
        "prover_plateau_w2_29_h_rate": avg(2, 29, "h_rate"),
        "boundary_w30_h_rate": per_w.get(30, {}).get("h_rate"),
        "boundary_w30_z": per_w.get(30, {}).get("z"),
        "w31_h_rate": per_w.get(31, {}).get("h_rate"),
        "code_plateau_w32_39_h_rate": avg(32, 39, "h_rate"),
        "code_plateau_w33_39_h_rate": avg(33, 39, "h_rate"),
        "code_plateau_w32_39_z": avg(32, 39, "z"),
        "per_window": [[w, per_w[w].get("h_rate"), per_w[w].get("z"), per_w[w].get("event")] for w in ws],
    }

live = {"pc": {}, "pb": {}}
for b in ("P0_r1", "P0_r2", "P0_r3"):
    t = v2p1_table(PC + "/boots/" + b + "/v2p1_rank0.jsonl")
    if t:
        live["pc"][b] = summarize_pc(t)
for bd in sorted(glob.glob(PB + "/boots/P0_*")):
    t = v2p1_table(bd + "/v2p1_rank0.jsonl")
    if t:
        ws = sorted(t)
        live["pb"][os.path.basename(bd)] = {
            "n_windows": len(ws),
            "per_window": [[w, t[w].get("h_rate"), t[w].get("z"), t[w].get("event")] for w in ws],
        }
result["p0_live"] = live
result["errors"] = errors

with open(OUT + "/imbalance_offline.json", "w") as f:
    json.dump(result, f, indent=1)

# ---- 8. readable summary
lines = []
lines.append("# Offline imbalance comparison (author directive 2026-10-27)")
lines.append("")
lines.append("Metric definitions: headroom_rate = P0 controller h_rate formula (physical space, 8 ranks x 16 slots); r_raw = mean per-layer max/mean rank load; cv = mean per-layer std/mean.")
lines.append("")
lines.append("## 94-layer grid: SAME counts x layouts")
for cn, cols in grid.items():
    lines.append("")
    lines.append("### counts = " + cn)
    lines.append("| layout | headroom_rate | r_raw | cv |")
    lines.append("|---|---|---|---|")
    for ln, m in cols.items():
        lines.append("| %s | %.5f | %.4f | %.4f |" % (ln, m["headroom_rate"], m["r_raw"], m["cv"]))
if "grid_book_published" in result:
    lines.append("")
    lines.append("### counts = book_published_traces (exact, pb stage2 source, prefill forwards)")
    lines.append("| layout | headroom_rate | r_raw | cv |")
    lines.append("|---|---|---|---|")
    for ln, m in result["grid_book_published"].items():
        lines.append("| %s | %.5f | %.4f | %.4f |" % (ln, m["headroom_rate"], m["r_raw"], m["cv"]))
lines.append("")
lines.append("## cdx0 exact layers (eval-code counts, 2026-10-16 captures)")
for lk, lv in cdx.items():
    lines.append("")
    lines.append("### " + lk)
    lines.append("| column | headroom_rate | r_raw (ours) | r_raw (archived) | cv |")
    lines.append("|---|---|---|---|---|")
    for ln, m in lv["eval_code_counts"].items():
        ar = m.get("archived_r_raw")
        lines.append("| %s | %.5f | %.4f | %s | %.4f |" % (ln, m["headroom_rate"], m["r_raw"], ("%.4f" % ar) if isinstance(ar, float) else str(ar), m["cv"]))
lines.append("")
lines.append("## P0 live h_rate (controller own metric on live traffic)")
for camp, boots in live.items():
    for b, s in boots.items():
        if camp == "pc":
            lines.append("- pc %s: prover plateau(w2-29)=%.5f boundary(w30)=%.5f z30=%.1f w31=%.5f code plateau(w33-39)=%.5f z plateau=%.2f" % (
                b,
                s["prover_plateau_w2_29_h_rate"] or -1,
                s["boundary_w30_h_rate"] or -1,
                s["boundary_w30_z"] or -1,
                s["w31_h_rate"] or -1,
                s["code_plateau_w33_39_h_rate"] or -1,
                s["code_plateau_w32_39_z"] or -1))
        else:
            lines.append("- pb %s: %d windows so far (trajectory in json)" % (b, s["n_windows"]))
lines.append("")
lines.append("## Cross-checks")
if "crosscheck_prover_e3rec_vs_datafore_cosine" in result:
    cc = result["crosscheck_prover_e3rec_vs_datafore_cosine"]
    lines.append("- prover slice vs datafore prover counts per-layer cosine: mean=%.4f min=%.4f max=%.4f" % (cc["mean"], cc["min"], cc["max"]))
lines.append("- cdx0 archived r_raw vs recomputed: see table above (identity column anchors pipeline)")
if errors:
    lines.append("")
    lines.append("## ERRORS (sections failed)")
    for tag, tb in errors:
        lines.append("- " + tag + ": " + tb.replace("\n", " ")[-200:])
with open(OUT + "/imbalance_offline.md", "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT + "/imbalance_offline.json", "and .md")
print("errors:", [e[0] for e in errors])
for cn, cols in grid.items():
    print("==", cn)
    for ln, m in cols.items():
        print("   %-28s hr=%.5f r_raw=%.4f cv=%.4f" % (ln, m["headroom_rate"], m["r_raw"], m["cv"]))
