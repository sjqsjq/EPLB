"""Phase-4 supplement analysis — implements ANALYSIS_PLAN_SUPP.md (pre-declared, pre-boots).

Primary Goodput@1.0 metric = goodness_rate (GOOD_count / total_requests); goodput_req_s
reported descriptively. Paired bootstrap over 8 blocks, 10000 draws, seed 20261021,
percentile 95% CI, two-sided p = min(1, 2*min(frac(draw<=0), frac(draw>=0))).
Holm within each family {P-I, P-E_native} per stream per metric (tps, goodness_rate@1.0).
P-F secondary (raw p, flagged). No sample deletion. Non-significant != equivalent.
"""
import json, os, re, glob, random
B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # phase4_core
G = os.path.dirname(B)                                                     # followup root
BOOTS = f"{B}/boots"
REF = json.load(open(f"{G}/phase3_prep/SLO_REFERENCE_FROZEN.json"))["reference"]
STREAMS = ["E3", "E4"]; ARMS = ["I", "E", "P", "F"]; BLOCKS = list(range(1, 9))
KS = [0.8, 1.0, 1.2]; PRIMARY_K = 1.0
NDRAW = 10000; SEED = 20261021

def p95(xs):
    xs = sorted(xs); return xs[int(0.95 * (len(xs) - 1))] if xs else None

def boot_metrics(stream, arm, blk):
    lab = f"s_{stream}_{arm}_b{blk}"; d = f"{BOOTS}/{lab}"
    j = json.load(open(f"{d}/{lab}.json"))
    rows = [json.loads(l) for l in open(f"{d}/{lab}_perreq.jsonl") if l.strip()]
    ref_t, ref_p = REF[stream]["ref_p95_TTFT_s"], REF[stream].get("ref_p95_TPOT_s")
    out = {"label": lab, "tps": j["tps"], "ok": j["ok"], "n": j["num_requests"], "errors": j["errors"],
           "total_time_s": j["total_time_s"], "ttft_p95_ms": j["ttft_p95_ms"],
           "tpot_p95_ms": j.get("tpot_p95_ms")}
    for k in KS:
        good = 0
        for r in rows:
            if r.get("error") or r.get("ttft_s") is None: continue
            if r["ttft_s"] > k * ref_t: continue
            if stream == "E4" and r.get("comp_tokens", 1) > 1:
                tpot = (r["elapsed_s"] - r["ttft_s"]) / (r["comp_tokens"] - 1)
                if tpot > k * ref_p: continue
            good += 1
        out[f"good_rate@{k}"] = good / j["num_requests"]
        out[f"goodput_req_s@{k}"] = good / j["total_time_s"]
    if arm in ("P", "F"):
        log = open(f"{d}/server.log", errors="replace").read()
        out["issued_windows"] = len(set(re.findall(r"\[PB-OEPLB\] window#(\d+): issued", log)))
        v = [json.loads(l) for l in open(f"{d}/v2p1_rank0.jsonl") if l.strip()]
        steps = sum(r.get("steps_actual", 0) or 0 for r in v)
        out["windows"] = len(v)
        out["S>1_share"] = (sum(r.get("steps_actual", 0) or 0 for r in v if (r.get("sp_s") or 0) > 1) / steps) if steps else None
        out["sp_swaps_started"] = sum(r.get("sp_swaps_started", 0) or 0 for r in v)
        out["sp_s_values"] = sorted({r.get("sp_s") for r in v})
        out["recording_ratio"] = (sum(r.get("sp_recorded_steps", 0) or 0 for r in v) /
                                  sum(r.get("true_steps_total", 0) or 0 for r in v)) if steps else None
    # p95 TTFT/TPOT recomputed from perreq for the table
    okr = [r for r in rows if not r.get("error") and r.get("ttft_s") is not None]
    out["ttft_p95_s_perreq"] = round(p95([r["ttft_s"] for r in okr]), 4)
    if stream == "E4":
        tp = [(r["elapsed_s"] - r["ttft_s"]) / (r["comp_tokens"] - 1) for r in okr if r.get("comp_tokens", 1) > 1]
        out["tpot_p95_s_perreq"] = round(p95(tp), 5) if tp else None
    return out

def boot_delta_stats(deltas):
    """Paired bootstrap over blocks: mean, 95% percentile CI, two-sided p."""
    n = len(deltas); mean = sum(deltas) / n
    rng = random.Random(SEED)
    draws = []
    for _ in range(NDRAW):
        s = 0.0
        for _ in range(n): s += deltas[rng.randrange(n)]
        draws.append(s / n)
    draws.sort()
    ci = (draws[int(0.025 * NDRAW)], draws[int(0.975 * NDRAW) - 1])
    le = sum(1 for x in draws if x <= 0) / NDRAW
    ge = sum(1 for x in draws if x >= 0) / NDRAW
    p = min(1.0, 2 * min(le, ge))
    return {"mean": mean, "ci95": list(ci), "p_raw": p, "ndraw": NDRAW, "nblocks": n}

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); m = len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj

data, cells = {}, {}
for st in STREAMS:
    for arm in ARMS:
        cells[(st, arm)] = [boot_metrics(st, arm, b) for b in BLOCKS]
        for m in cells[(st, arm)]: data[m["label"]] = m

results = {"seed": SEED, "ndraw": NDRAW,
           "primary_goodput_metric": "good_rate@1.0 (GOOD/total_requests); goodput_req_s descriptive",
           "families": "Holm per stream per metric over {P-I, P-E}; P-F secondary raw p",
           "cells": {}, "hypotheses": {}, "per_boot": data}
for st in STREAMS:
    for arm in ARMS:
        ms = cells[(st, arm)]
        agg = {}
        for key in ["tps", "good_rate@0.8", "good_rate@1.0", "good_rate@1.2", "goodput_req_s@1.0",
                    "ttft_p95_s_perreq", "tpot_p95_s_perreq"]:
            vals = [m[key] for m in ms if m.get(key) is not None]
            if not vals: continue
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
            agg[key] = {"mean": round(mean, 6), "sd": round(sd, 6), "n": len(vals)}
        if arm in ("P", "F"):
            agg["issued_windows"] = [m["issued_windows"] for m in ms]
            agg["S>1_share"] = [m["S>1_share"] for m in ms]
            agg["recording_ratio"] = [round(m["recording_ratio"], 6) for m in ms]
            agg["sp_s_values"] = sorted({s for m in ms for s in m["sp_s_values"]})
        results["cells"][f"{st}_{arm}"] = agg
    # hypotheses
    hyp = {}
    for metric in ["tps", "good_rate@1.0"]:
        P = [m[metric] for m in cells[(st, "P")]]
        fam = {}
        for X in ["I", "E", "F"]:
            d = [P[b] - cells[(st, X)][b][metric] for b in range(8)]
            fam[f"P-{X}"] = boot_delta_stats(d)
        ps = [fam["P-I"]["p_raw"], fam["P-E"]["p_raw"]]
        adj = holm(ps)
        fam["P-I"]["p_holm"] = adj[0]; fam["P-E"]["p_holm"] = adj[1]
        fam["P-F"]["secondary"] = True
        hyp[metric] = fam
    results["hypotheses"][st] = hyp

os.makedirs(f"{B}/analysis", exist_ok=True)
json.dump(results, open(f"{B}/analysis/SUPP_RESULTS.json", "w"), indent=1)

# ---- markdown report ----
L = ["# Phase-4 supplement results (64 boots, v2 code, locked matrix d3ff2b09)", "",
     f"Bootstrap: paired over 8 blocks, {NDRAW} draws, seed {SEED}, percentile 95% CI,",
     "two-sided p = min(1, 2*min(frac<=0, frac>=0)). Holm per stream per metric over {P-I, P-E}.",
     "P-F secondary (raw p). Primary Goodput@1.0 = goodness_rate (GOOD/total). 不显著≠等价。", ""]
for st in STREAMS:
    L += [f"## {st}", "", "| arm | tps mean±sd | good_rate@1.0 | @0.8 | @1.2 | goodput_req/s@1.0 | p95 TTFT(s) | p95 TPOT(s) |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        c = results["cells"][f"{st}_{arm}"]
        f6 = lambda k, n=4: (f"{c[k]['mean']:.{n}f}±{c[k]['sd']:.{n}f}" if k in c else "-")
        tp = c.get("tpot_p95_s_perreq"); tps_ = f"{tp['mean']:.5f}" if tp else "-"
        L.append(f"| {arm} | {f6('tps',2)} | {f6('good_rate@1.0')} | {f6('good_rate@0.8')} | "
                 f"{f6('good_rate@1.2')} | {f6('goodput_req_s@1.0',2)} | {f6('ttft_p95_s_perreq',3)} | {tps_} |")
    if "P" in [a for a in ARMS]:
        cp = results["cells"][f"{st}_P"]; cf = results["cells"][f"{st}_F"]
        L += ["", f"P arm migration activity: issued_windows per boot {cp['issued_windows']} (total {sum(cp['issued_windows'])}); "
              f"S>1 share {cp['S>1_share']}; recording_ratio {cp['recording_ratio']}; sp_s values {cp['sp_s_values']}",
              f"F arm: issued_windows per boot {cf['issued_windows']} (total {sum(cf['issued_windows'])})", ""]
    L += ["### Hypotheses (primary: Holm over {P-I, P-E} per metric)", "",
          "| metric | delta | mean | 95% CI | p_raw | p_Holm |", "|---|---|---|---|---|---|"]
    for metric in ["tps", "good_rate@1.0"]:
        for pair in ["P-I", "P-E", "P-F"]:
            h = results["hypotheses"][st][metric][pair]
            ph = f"{h['p_holm']:.4f}" if "p_holm" in h else f"(secondary) {h['p_raw']:.4f}"
            L.append(f"| {metric} | {pair} | {h['mean']:.4f} | [{h['ci95'][0]:.4f}, {h['ci95'][1]:.4f}] | {h['p_raw']:.4f} | {ph} |")
    L.append("")
open(f"{B}/analysis/RESULTS_SUPP.md", "w").write("\n".join(L))
print("WROTE SUPP_RESULTS.json + RESULTS_SUPP.md")
for st in STREAMS:
    for metric in ["tps", "good_rate@1.0"]:
        for pair in ["P-I", "P-E"]:
            h = results["hypotheses"][st][metric][pair]
            print(f"{st} {metric} {pair}: mean={h['mean']:.4f} CI=[{h['ci95'][0]:.4f},{h['ci95'][1]:.4f}] p_holm={h['p_holm']:.4f}")
        h = results["hypotheses"][st][metric]["P-F"]
        print(f"{st} {metric} P-F (secondary): mean={h['mean']:.4f} CI=[{h['ci95'][0]:.4f},{h['ci95'][1]:.4f}] p_raw={h['p_raw']:.4f}")
