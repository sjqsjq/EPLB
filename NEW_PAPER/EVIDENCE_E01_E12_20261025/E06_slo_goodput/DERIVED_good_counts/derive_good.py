#!/usr/bin/env python3
# DERIVED artifact: per-boot integer GOOD counts for phase-4 (64 boots).
# GOOD_int@k = round(good_rate@k * n) computed EXACTLY from the frozen SUPP_RESULTS.json
# (no re-analysis, no new statistics, no raw perreq re-reading).
# GOOD semantics (frozen, SLO_REFERENCE_FROZEN.json): GOOD = error==null AND TTFT<=k*ref_p95_TTFT
#   AND (E4 only) TPOT<=k*ref_p95_TPOT. We do NOT recompute membership; we only convert the
#   stored rates back to integers and flag any non-exact conversion.
import json, hashlib, os
SRC = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020/phase4_core/analysis/SUPP_RESULTS.json"
OUT = os.path.dirname(os.path.abspath(__file__))
sha = hashlib.sha256(open(SRC,"rb").read()).hexdigest()
d = json.load(open(SRC))
pb = d["per_boot"]
rows = []
flags = []
for label in sorted(pb):
    b = pb[label]
    parts = label.split("_")  # s_E3_P_b1
    stream = parts[1] if len(parts) > 1 else ""
    arm = parts[2] if len(parts) > 2 else ""
    boot = parts[3] if len(parts) > 3 else ""
    n = b.get("n")
    row = {"label": label, "stream": stream, "arm": arm, "boot": boot, "n": n,
           "ok": b.get("ok"), "errors": b.get("errors"), "total_time_s": b.get("total_time_s"),
           "ttft_p95_ms": b.get("ttft_p95_ms"), "tpot_p95_ms": b.get("tpot_p95_ms")}
    for k in ("0.8", "1.0", "1.2"):
        rate = b.get("good_rate@" + k)
        row["good_rate@" + k] = rate
        if rate is not None and n:
            prod = rate * n
            gi = int(round(prod))
            if abs(prod - gi) > 1e-6:
                flags.append((label, k, prod))
            row["GOOD_int@" + k + "_DERIVED"] = gi
        else:
            row["GOOD_int@" + k + "_DERIVED"] = None
    row["goodput_req_s@1.0"] = b.get("goodput_req_s@1.0")
    row["ttft_p95_s_perreq"] = b.get("ttft_p95_s_perreq")
    rows.append(row)
cols = ["label","stream","arm","boot","n","ok","errors","total_time_s","ttft_p95_ms","tpot_p95_ms",
        "good_rate@0.8","GOOD_int@0.8_DERIVED","good_rate@1.0","GOOD_int@1.0_DERIVED",
        "good_rate@1.2","GOOD_int@1.2_DERIVED","goodput_req_s@1.0","ttft_p95_s_perreq"]
with open(os.path.join(OUT,"DERIVED__good_counts_per_boot.tsv"),"w") as f:
    f.write("\t".join(cols)+"\n")
    for r in rows:
        f.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols)+"\n")
with open(os.path.join(OUT,"DERIVED__README.md"),"w") as f:
    f.write("""# DERIVED per-boot integer GOOD counts (phase-4, 64 boots)
Status: **DERIVED arithmetic**, NOT a new measurement and NOT a re-analysis.
Source (frozen): %(src)s
Source sha256: %(sha)s
Method: GOOD_int@k = round(good_rate@k * n) from stored SUPP_RESULTS.json per_boot values only.
Non-exact conversions (|rate*n - round| > 1e-6): %(nflag)d %(flags)s
GOOD definition (frozen in phase3_prep/SLO_REFERENCE_FROZEN.json, frozen 2026-10-20T09:24:53Z):
GOOD = error==null AND TTFT <= k * ref_p95_TTFT AND (stream E4 only) TPOT <= k * ref_p95_TPOT.
Membership was decided by the frozen analyze_supp.py at analysis time; this file only converts rates to integers.
Per-request raw logs for independent recomputation: E05_endtoend/phase4_v2_fixed/boots/*/*_perreq.jsonl (all 64 packaged).
Generated: 2026-10-25 by derive_good.py (in this directory; script packaged for audit).
""" % {"src": SRC, "sha": sha, "nflag": len(flags), "flags": str(flags[:10])})
print("ROWS", len(rows), "FLAGS", len(flags), flags[:5])
print("SUPP_SHA", sha)
