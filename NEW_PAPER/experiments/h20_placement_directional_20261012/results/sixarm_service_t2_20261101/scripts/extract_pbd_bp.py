#!/usr/bin/env python3
"""extract_pbd_bp.py - PBD_BP (book->prover corrected stream) r4/r5 extraction
+ side-by-side vs original prover->book r1-r3 per-arm means. Tolerates
missing boots (queue in progress): extracts what exists, marks gaps.
Outputs: analysis/EXPORT_t2_pbd_bp.tsv + analysis/SUMMARY_pbd_bp.md"""
import json, os, glob, statistics

T2 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101"
ARMS = ["I", "ES", "ED", "MT", "DF", "P0"]
rows = []

def mech_ok(art):
    for pat in ("mech_t2.json", "mech.json", "mech*.json"):
        hits = sorted(glob.glob(os.path.join(art, pat)))
        for h in hits:
            try:
                return json.load(open(h)).get("all_ok")
            except Exception:
                pass
    return None

def collect(label, stream, arm, rnd):
    art = os.path.join(T2, "boots", label)
    agg = os.path.join(art, label + ".json")
    if not os.path.exists(agg):
        return None
    d = json.load(open(agg))
    rc = None
    ep = os.path.join(art, "boot_meta.json.end")
    if os.path.exists(ep):
        rc = json.load(open(ep)).get("bench_rc")
    tps_tok = d.get("tps")
    req_s = round(d.get("ok") / d.get("total_time_s"), 3) if d.get("total_time_s") else None
    r = {"label": label, "stream": stream, "arm": arm, "round": rnd,
         "ok": d.get("ok"), "errors": d.get("errors"), "tps": tps_tok, "req_s": req_s,
         "total_time_s": d.get("total_time_s"), "lat_mean_ms": d.get("lat_mean_ms"),
         "lat_p50_ms": d.get("lat_p50_ms"), "lat_p95_ms": d.get("lat_p95_ms"),
         "lat_p99_ms": d.get("lat_p99_ms"), "ttft_mean_ms": d.get("ttft_mean_ms"),
         "ttft_p50_ms": d.get("ttft_p50_ms"), "ttft_p95_ms": d.get("ttft_p95_ms"),
         "ttft_p99_ms": d.get("ttft_p99_ms"), "tpot_mean_ms": d.get("tpot_mean_ms"),
         "total_comp_tokens": d.get("total_comp_tokens"),
         "mech_all_ok": mech_ok(art), "bench_rc": rc}
    return r

for a in ARMS:
    for rnd in ("r1", "r2", "r3"):
        r = collect("PBD_" + a + "_" + rnd, "prover_book", a, rnd)
        if r: rows.append(r)
    for rnd in ("r4", "r5"):
        r = collect("PBD_BP_" + a + "_" + rnd, "book_prover", a, rnd)
        if r: rows.append(r)

cols = ["label", "stream", "arm", "round", "ok", "errors", "tps", "req_s", "total_time_s",
        "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "ttft_mean_ms",
        "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "tpot_mean_ms",
        "total_comp_tokens", "mech_all_ok", "bench_rc"]
os.makedirs(os.path.join(T2, "analysis"), exist_ok=True)
with open(os.path.join(T2, "analysis", "EXPORT_t2_pbd_bp.tsv"), "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")

def means(stream):
    out = {}
    for a in ARMS:
        sel = [r for r in rows if r["arm"] == a and r["stream"] == stream]
        if not sel:
            out[a] = None
            continue
        out[a] = {"n": len(sel),
                  "tps_mean": round(statistics.mean(x["tps"] for x in sel), 3),
                  "tps_vals": [x["tps"] for x in sel],
                  "req_s_mean": round(statistics.mean(x["req_s"] for x in sel), 3),
                  "req_s_vals": [x["req_s"] for x in sel],
                  "lat_p99_mean": round(statistics.mean(x["lat_p99_ms"] for x in sel if x["lat_p99_ms"] is not None), 1) if any(x["lat_p99_ms"] is not None for x in sel) else None,
                  "ttft_p95_mean": round(statistics.mean(x["ttft_p95_ms"] for x in sel if x["ttft_p95_ms"] is not None), 1) if any(x["ttft_p95_ms"] is not None for x in sel) else None}
    return out

mo, mn = means("prover_book"), means("book_prover")
lines = []
lines.append("# PBD_BP extraction: corrected book->prover stream (r4/r5) vs original prover->book (r1-r3)")
lines.append("")
lines.append("NOTE: r4/r5 run on pbd_book_prover_double_out32_ie.jsonl (sha256 7b42568a..., author-ordered correction);")
lines.append("r1-r3 ran on the frozen prover->book stream (sha cdd1d864...). Cross-stream comparison is descriptive")
lines.append("(order-effect view), NOT a same-stream A/B. Missing boots marked n<expected.")
lines.append("")
lines.append("Primary caliber = req_s (ok/total_time_s), identical to frozen T2 main-table caliber (agg tps tokens/s = req_s x 32 under O32 ignore_eos).")
lines.append("")
lines.append("| arm | n(r4r5) | req_s_mean(r4r5) | req_s_vals | n(r1r3) | req_s_mean(r1r3) | delta% | lat_p99_mean(r4r5) | ttft_p95_mean(r4r5) |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for a in ARMS:
    x, y = mn[a], mo[a]
    if x is None and y is None:
        lines.append("| " + a + " | 0 | - | - | 0 | - | - | - | - |")
        continue
    xs = ("%s" % x["n"], "%s" % x["req_s_mean"], str(x["req_s_vals"])) if x else ("0", "-", "-")
    ys = ("%s" % y["n"], "%s" % y["req_s_mean"]) if y else ("0", "-")
    d = ("%+.1f%%" % (100.0 * (x["req_s_mean"] - y["req_s_mean"]) / y["req_s_mean"])) if (x and y and y["req_s_mean"]) else "-"
    lines.append("| " + a + " | " + xs[0] + " | " + xs[1] + " | " + xs[2] + " | " + ys[0] + " | " + ys[1] + " | " + d + " | " + str(x["lat_p99_mean"] if x else "-") + " | " + str(x["ttft_p95_mean"] if x else "-") + " |")
with open(os.path.join(T2, "analysis", "SUMMARY_pbd_bp.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("ROWS=" + str(len(rows)) + " bp_rows=" + str(len([r for r in rows if r["stream"] == "book_prover"])) + "/12")
for a in ARMS:
    if mn[a]:
        print(a, "r4r5_req_s", mn[a]["req_s_vals"], "r1r3_mean", mo[a]["req_s_mean"] if mo[a] else None)
