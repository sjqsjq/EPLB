# Deliverable E — Step-6 formal comparison protocol (PRE-REGISTERED, frozen before any formal run)

Date frozen: 2026-10-16. Status: ACTIVE. This protocol was written BEFORE any Phase A/B/C start.
Nothing in it may be changed using results from the runs it governs. Expansion is prohibited
(指令七: no full-factorial expansion; 指令八: fixed start counts/order/comparison family, no
significance-driven additions).

## 1. Code lock (verified by sha256 at every driver invocation; mismatch = abort, no run)

Installed /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/:
- controller.py 2a4dedb22d4bcaba203233c0f9a5ae6c211370f524addb7d166669406690878b
- window_v2.py  810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74
- config.py     180fa41eae6e9394186afceb2ceadf9468ebbacda69591fb9b9897a2e6c2680c
v2 knobs: FREEZE_record_v2_knobs.md (single authority; no post-freeze tuning).

## 2. Conditions (datasets frozen by sha256, step6_dataset_hashes.txt)

| cond | file | n | structure | role |
|------|------|---|-----------|------|
| W0 | calibration.jsonl (7d53aa0b…) | 8037 | 4 domain blocks, boundaries rows 2075/4052/6068; #3 routing-invisible | dev/calibration + known benchmark regression (NOT unseen validation) |
| W1 | w1_stable_code_4000.jsonl (f0391f7c…) | 4000 | code-only, 0 switches | stable holdout (pre-declared rule: first contiguous labeled single-domain block of spliced_code_chinese_O1) |
| W2 | w2_switch_code_chinese_8000.jsonl (9d852a75…) | 8000 | code×4000 → chinese×4000, exactly 1 switch at row 4000 | predefined-real-change holdout |

All: O1 prefill-replay (max_tokens=1), concurrency 256, run_grid_bench.py, single node 8×H20 EP=8.
Labels (domain fields) are used ONLY in offline analysis, never fed to the controller.

## 3. Arms (phase-1 isolation: same α=0.0, D OFF, heuristic_first_improvement planner, same
##    service path; ONLY window control differs — 指令五)

- **V2**: OEPLB_WINDOW_CONTROLLER=2, frozen knobs, W_init=16, live W.
- **F-global**: OEPLB_WINDOW_CONTROLLER=1 (v1 byte-identical legacy path), EA_W=16 fixed.
  W=16 pre-declared from FACT-09 grid (prior campaign); NOT re-derived from Phase A.
- **F-local(c)**: controller=1, EA_W = argmin total_time_s over Phase-A grid {8,16,32,64} on
  condition c (1 start/cell). Tie → smaller W. If argmin=16, F-local(c) merges with F-global
  for that condition (same config; the merged cell's 2 Phase-B starts serve both comparisons).
- Fixed arms get the same calibration opportunity as V2 (指令四/六 fairness); post-hoc best
  fixed across ALL cells is reported only as a labeled reference, never as the comparator.

## 4. Phases, start counts, exact order (TOTAL BUDGET: 12 + ≤18 + 6 = ≤36 starts, ~5.5h)

**Phase A — F-local calibration grids (12 starts, fixed order):**
W0-8, W0-16, W0-32, W0-64, W1-8, W1-16, W1-32, W1-64, W2-8, W2-16, W2-32, W2-64.
Grid starts are calibration ONLY; they are NOT reused as Phase-B comparison starts.

**Phase B — formal comparison (2 starts/cell, round-robin 2 cycles; cell order fixed):**
cells = W0-V2, W0-FG, W0-FL, W1-V2, W1-FG, W1-FL, W2-V2, W2-FG, W2-FL.
Cycle 1 = one start per cell in order; Cycle 2 = same order again. Round-robin interleaving
spreads time-varying machine state equally across arms (pre-declared fairness rule).

**Phase C — mechanism ablations (exploratory-labeled; 6 starts, config-only, frozen code):**
Order: W0-M2off, W0-M3off, W0-M1p, W2-M2off, W2-M3off, W2-M1p.
- M2-off: OEPLB_V2_THETA_H_LOW=1e18, THETA_H_ACT=0 (headroom gating removed → z-only control)
- M3-off: DWELL_SHRINK=0, DWELL_GROW=0, M_SHRINK=1, M_GROW=1, URGENT_EPISODE_CAP=99
  (hysteresis/confirmation/rate-limit removed → per-window jumping)
- M1-partial: BETA=0.0 (ref = last window, no EWMA memory → maximal noise sensitivity)
  HONEST LIMITATION: fresh-sample normalization (z=s_change/(κ/n_layer)) and the UNCERTAIN
  machinery are code-level and NOT ablatable via config under the code freeze; M1-partial
  ablates only the ref-memory component. This limitation is part of the pre-registration.
Ablation baselines reuse Phase-B V2 (Full) and fixed cells from the SAME campaign (指令: reuse).

## 5. Endpoints (指令八)

**Primary**: total_time_s for the same completed token workload (ok==n, errors==0) — fixed
caliber including ALL online overhead (decision costs, heartbeats, legacy allreduce kept in
both arms). Secondary fixed-caliber: tps.
**Comparisons (pre-declared, per condition, never pooled across conditions)**:
V2−F-global and V2−F-local(c), reported as per-start-pair deltas AND both raw starts shown.
n=2/cell: report raw values; language "compatible with zero" when |Δ| is within run-to-run
spread; NO significance testing, NO significance-driven additions.
**Behavior evidence (V2 cells)**: W trajectory + dwell distributions (window/step/time-weighted),
event census, detection latency at known boundaries (W0 rows 2075/4052/6068 → windows via
cumulative-n_fresh mapping; W2 row 4000 same), per-change z/h_w/reason, c_window_ms,
prof cost series, migration/plan counts. Migration wall-clock reported ALONGSIDE, never subtracted.
**Required narrative per condition**: why W changed → whether expected behavior occurred →
whether net service time improved.

## 6. Failure handling (standing constraints)

- Boot failure, errors>0, desync event, or v2-error fallback in a V2/ablation start →
  start RETIRED as failed (mechanism did not run as specified), documented in the phase log
  with its artifacts preserved, NEVER retried under this frozen campaign. Cell then reports
  n=1 (or n=0 with the failure record). No imputation, no replacement.
- Failed starts do not shift the round-robin order; subsequent starts proceed as scheduled.
- No best-start selection anywhere; both starts of a cell are always reported.

## 7. Run hygiene

Each start: kill server → wait GPU clear (<1000MiB) → verify code+dataset hashes → boot
(poll "ready to roll", ≤2400s) → bench (timeout 3600s) → copy artifacts
(results/step6/<phase>/<cell>__<startN>/) → verify ok==n → teardown. Server log, bench log,
bench json, v2 rank0 jsonl (V2/ablation arms) archived per start. Timing excludes
migrations/setup/JIT/warmup only in the sense that the bench client measures steady serving;
all controller overhead is INSIDE the measured window.

## 8. Deliverable F analysis spec (pre-declared tables)

T1 raw service times per cell (all starts); T2 per-condition deltas V2−FG / V2−FL with
compatible-with-zero language; T3 behavior chains per V2 cell; T4 ablation vs Full (service
time + churn counters: W-change count, rejected/blocked counts) with the M1-partial limitation
restated; T5 applicability scope statement (which conditions v2 helped/hurt/neutral, and the
pre-declared note that W0 is a known-regression benchmark not unseen validation).
