# DERIVED per-boot integer GOOD counts (phase-4, 64 boots)
Status: **DERIVED arithmetic**, NOT a new measurement and NOT a re-analysis.
Source (frozen): /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020/phase4_core/analysis/SUPP_RESULTS.json
Source sha256: 2b3f7a57b5bd7c5303c318f4a3f9177efc9e124f9b951c1466cee94a961572bc
Method: GOOD_int@k = round(good_rate@k * n) from stored SUPP_RESULTS.json per_boot values only.
Non-exact conversions (|rate*n - round| > 1e-6): 0 []
GOOD definition (frozen in phase3_prep/SLO_REFERENCE_FROZEN.json, frozen 2026-10-20T09:24:53Z):
GOOD = error==null AND TTFT <= k * ref_p95_TTFT AND (stream E4 only) TPOT <= k * ref_p95_TPOT.
Membership was decided by the frozen analyze_supp.py at analysis time; this file only converts rates to integers.
Per-request raw logs for independent recomputation: E05_endtoend/phase4_v2_fixed/boots/*/*_perreq.jsonl (all 64 packaged).
Generated: 2026-10-25 by derive_good.py (in this directory; script packaged for audit).
