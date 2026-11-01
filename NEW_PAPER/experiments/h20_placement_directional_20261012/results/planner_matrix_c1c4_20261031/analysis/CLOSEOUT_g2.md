# CLOSEOUT_g2.md - planner_matrix_c1c4_20261031 (closed 2026-10-31T04:53:26Z)

## 0. Discipline (author fix3)
CPU deterministic workload arithmetic ONLY. No timing, no GPU, and NO
verified-real-GPU-gain claims anywhere in this campaign. The executed
planner is the frozen byte copy (sha256 c250026bfa2bdd39a4931c6d18584a254f5133acd2b2355e221d4faac873dd2d)
observed via sys.settrace anchors; every cell asserts control-vs-traced
plan identity op-by-op (52/52 True).

## 1. Protocol constants (T1 restoration, verbatim)
num_ranks=8, num_local=16, identity p2l, max_swaps_per_layer=64 (param
unused in frozen body), max_total_swap_layers=94, acceptance margin
new_ratio < ratio - 0.0005, padded reporting w_e=128*ceil(n_e/128).
Threshold apertures: 1.0962402945 = production PRIMARY (max(1.02 flag,
auto dead-zone)); 1.02 rows = algorithm-diagnostic under that flag.

## 2. Snapshots (all REAL, all kept per author group-2 revision)
| id | source | sum | init max ratio (raw) |
|----|--------|-----|----------------------|
| C1 | counts_calibration.npz (T1-comparable) | 6077073680 | 1.5514 |
| C2 | P_e3.npz crossdomain | 3154479072 | 1.5318 |
| C3 | book profile logical_count | 2921972704 | 2.3106 |
| C4 | datafore prover logical_count | 48563408 | 2.4984 |

## 3. Full 52-cell result table (plan_len / final max ratio raw)
| cell | plan | finmax_raw | finmax_pad | rounds | md_trials | gap_trials | identical |
|------|------|-----------|-----------|--------|-----------|------------|-----------|
| C1_gap_32_prod1096 | 32 | 1.3296 | 1.3295 | 32 | 32 | 0 | True |
| C1_gap_32_diag102 | 32 | 1.3296 | 1.3295 | 32 | 32 | 0 | True |
| C1_gap_128_prod1096 | 128 | 1.1974 | 1.1974 | 128 | 128 | 0 | True |
| C1_gap_128_diag102 | 128 | 1.1974 | 1.1974 | 128 | 128 | 0 | True |
| C1_gap_300_prod1096 | 281 | 1.163 | 1.163 | 282 | 269 | 25 | True |
| C1_gap_300_diag102 | 299 | 1.163 | 1.163 | 300 | 285 | 36 | True |
| C1_maxdelta_32_prod1096 | 32 | 1.3296 | 1.3295 | 32 | 32 | 0 | True |
| C1_maxdelta_32_diag102 | 32 | 1.3296 | 1.3295 | 32 | 32 | 0 | True |
| C1_maxdelta_128_prod1096 | 128 | 1.1974 | 1.1974 | 128 | 128 | 0 | True |
| C1_maxdelta_128_diag102 | 128 | 1.1974 | 1.1974 | 128 | 128 | 0 | True |
| C1_maxdelta_300_prod1096 | 262 | 1.163 | 1.163 | 281 | 413 | 0 | True |
| C1_maxdelta_300_diag102 | 274 | 1.163 | 1.163 | 300 | 758 | 0 | True |
| C2_gap_32_prod1096 | 32 | 1.314 | 1.314 | 32 | 32 | 0 | True |
| C2_gap_32_diag102 | 32 | 1.314 | 1.314 | 32 | 32 | 0 | True |
| C2_gap_128_prod1096 | 128 | 1.1783 | 1.1782 | 128 | 128 | 0 | True |
| C2_gap_128_diag102 | 128 | 1.1783 | 1.1782 | 128 | 128 | 0 | True |
| C2_gap_300_prod1096 | 251 | 1.0962 | 1.0963 | 251 | 238 | 13 | True |
| C2_gap_300_diag102 | 299 | 1.0937 | 1.0937 | 300 | 264 | 53 | True |
| C2_maxdelta_32_prod1096 | 32 | 1.314 | 1.314 | 32 | 32 | 0 | True |
| C2_maxdelta_32_diag102 | 32 | 1.314 | 1.314 | 32 | 32 | 0 | True |
| C2_maxdelta_128_prod1096 | 128 | 1.1783 | 1.1782 | 128 | 128 | 0 | True |
| C2_maxdelta_128_diag102 | 128 | 1.1783 | 1.1782 | 128 | 128 | 0 | True |
| C2_maxdelta_300_prod1096 | 239 | 1.1616 | 1.1615 | 251 | 329 | 0 | True |
| C2_maxdelta_300_diag102 | 266 | 1.1616 | 1.1615 | 300 | 852 | 0 | True |
| C3_gap_32_prod1096 | 32 | 1.8496 | 1.8493 | 32 | 32 | 0 | True |
| C3_gap_32_diag102 | 32 | 1.8496 | 1.8493 | 32 | 32 | 0 | True |
| C3_gap_128_prod1096 | 128 | 1.4342 | 1.4341 | 128 | 127 | 1 | True |
| C3_gap_128_diag102 | 128 | 1.4342 | 1.4341 | 128 | 127 | 1 | True |
| C3_gap_300_prod1096 | 300 | 1.2157 | 1.2156 | 300 | 229 | 71 | True |
| C3_gap_300_diag102 | 300 | 1.2157 | 1.2156 | 300 | 229 | 71 | True |
| C3_maxdelta_32_prod1096 | 32 | 1.8496 | 1.8493 | 32 | 32 | 0 | True |
| C3_maxdelta_32_diag102 | 32 | 1.8496 | 1.8493 | 32 | 32 | 0 | True |
| C3_maxdelta_128_prod1096 | 127 | 1.4345 | 1.4343 | 128 | 145 | 0 | True |
| C3_maxdelta_128_diag102 | 127 | 1.4345 | 1.4343 | 128 | 145 | 0 | True |
| C3_maxdelta_300_prod1096 | 229 | 1.4345 | 1.4343 | 300 | 1312 | 0 | True |
| C3_maxdelta_300_diag102 | 229 | 1.4345 | 1.4343 | 300 | 1683 | 0 | True |
| C4_gap_32_prod1096 | 32 | 1.8637 | 1.8454 | 32 | 32 | 0 | True |
| C4_gap_32_diag102 | 32 | 1.8637 | 1.8454 | 32 | 32 | 0 | True |
| C4_gap_128_prod1096 | 128 | 1.4599 | 1.4528 | 128 | 125 | 3 | True |
| C4_gap_128_diag102 | 128 | 1.4599 | 1.4528 | 128 | 125 | 3 | True |
| C4_gap_300_prod1096 | 300 | 1.2281 | 1.2251 | 300 | 226 | 74 | True |
| C4_gap_300_diag102 | 300 | 1.2281 | 1.2251 | 300 | 226 | 74 | True |
| C4_maxdelta_32_prod1096 | 32 | 1.8637 | 1.8454 | 32 | 32 | 0 | True |
| C4_maxdelta_32_diag102 | 32 | 1.8637 | 1.8454 | 32 | 32 | 0 | True |
| C4_maxdelta_128_prod1096 | 125 | 1.512 | 1.5029 | 128 | 169 | 0 | True |
| C4_maxdelta_128_diag102 | 125 | 1.512 | 1.5029 | 128 | 183 | 0 | True |
| C4_maxdelta_300_prod1096 | 222 | 1.512 | 1.5029 | 300 | 1383 | 0 | True |
| C4_maxdelta_300_diag102 | 222 | 1.512 | 1.5029 | 300 | 1763 | 0 | True |
| C1_gap_300_prod1096_BI | 268 | 1.163 | 1.163 | 269 | 0 | 0 | True |
| C2_gap_300_prod1096_BI | 246 | 1.0957 | 1.0957 | 246 | 0 | 0 | True |
| C3_gap_300_prod1096_BI | 299 | 1.3195 | 1.3194 | 300 | 0 | 0 | True |
| C4_gap_300_prod1096_BI | 300 | 1.2111 | 1.2086 | 300 | 0 | 0 | True |

## 4. Findings
F1 APERTURE EFFECT is budget-gated: at budgets 32/128 every snapshot gives
   identical plans under both apertures (budget-bound); at 300 the
   production aperture yields smaller plans (C1 281 vs 299; C2 251 vs 299)
   and floors the final max ratio at the threshold itself (C2 finmax
   1.0962 prod vs 1.0937 diag) - the dead-zone floor made visible.
F2 DEEP-IMBALANCE SNAPSHOTS saturate the budget: C3 (book, init 2.3106)
   and C4 (prover, init 2.4984) hit plan=300 at both apertures with
   identical finmax (1.2157 / 1.2281) - threshold stops being the binding
   constraint, budget is.
F3 HALF-GAP FEASIBILITY (author highlight): the gap branch (code criterion
   hot0_load > gap) activates only at loose budget: 183 decisions across
   C1-C4; in 100% of them a feasible delta>0 candidate existed, and in
   172/183 the best candidate sat within 0.25*gap of the half-gap target.
F4 MAX-DELTA OVERSHOOT (author highlight): under the STRICT flip criterion
   (delta_max > gap/2 inverts rank order) the count is ZERO in all 52
   cells, including env=maxdelta cells (4617 maxdelta-branch trials total).
   On these real snapshots the code criterion (hot0 > gap) fires while the
   strict criterion never does - the guard is strictly conservative here.
   Reported as measured fact, not spun.
F5 SEARCH EFFICIENCY: env=maxdelta burns pair trials without producing
   proportionally more ops (C3_maxdelta_300: 1312 trials -> 229 ops;
   C1_maxdelta_300 diag: 758 -> 274) whereas gap-rule cells at budgets
   32/128 show ZERO rejected trials (fast branch accepts first pair).
F6 BI CONTROL (P11): best-improvement planner yields 268/246/299/300 ops
   (C1-C4, prod aperture, budget 300) with finmax 1.1630/1.0957/1.3195/
   1.2111 - same skeleton, different pair enumeration; trial-level exhibit
   counters are structurally zero for BI (no gapcalc anchor in its body);
   plan_len is the control endpoint.

## 5. Execution permission chain (per production semantics)
All 52 cells: init max ratio > thr (no early exit), plan_len >= 8
(meets_min_swap_ops=True everywhere), defer gate UNREACHABLE by structure
(score >= 0.00962402945 = 3.208 x 0.003 whenever gate reachable; see
LEDGER DEFER_GATE_READONLY_VERIFICATION: 0 gate hits / 628 logs, 688
min_swap_ops skip hits = the gate that actually fires), bias inert
(bias_correct default False). Moved-bytes ESTIMATE per cell =
plan_len x 37.74 MB in cells/*.json (labeled estimate).

## 6. Limitations
Inventory is a RESTORATION consistent with T1 inputs (not new captures);
excluded classes documented in INVENTORY_RESTORED row. Deterministic
arithmetic: no run-to-run variance exists to report. CPU-only: nothing
here evidences realized GPU speedups.

## 7. Artifact index
analysis/matrix_results.json (52 cells full), analysis/cells/*.json,
analysis/plans/*.json (full op lists), analysis/events/*.jsonl (anchor
event traces), code_lock/rebalancer_frozen_c250026b.py, scripts/
driver_g2.py, LEDGER_g2.tsv (append-only).
