# Phase-2 acceptance evidence — CODE VERSION v2 (round-2 fix FIX-3)

Date: 2026-10-21T03:2xZ (queue complete 03:08:18Z). Supersedes for all future
boots: `ACCEPTANCE_EVIDENCE.md` (v1). v1 evidence remains on file as the
record for the phase-2 6/6 jobs and prep boots C1/C2/A1/A2/B1(FAIL).

## Version identity (post-freeze fix ⇒ NEW version, no backfilling)

| file | v1 (round-1 freeze) | v2 (round-2 freeze) |
|---|---|---|
| async_swapper.py | 85a2dac882177667 | **2e4652a0306cb315** |
| controller.py | 4964cfc8e8af397a | 4964cfc8e8af397a (unchanged) |

src == deployed verified by diff; frozen copies: `freeze/code_fixed_v2/`,
`freeze/SHA256SUMS_code_fixed_v2.txt`. v1 freeze untouched.

## Why v2 exists (one line)

v1 still deadlocked on `prep3_B1_P_E3` window#17 (same signature as retired
m_E3_P_b3): root cause = NCCL budget-split of a single large coalesced p2p
group into sequential per-rank kernel plans whose cut points vary by rank
under asymmetric participation (upstream pytorch/pytorch#174288, OPEN,
repro'd ≤ NCCL 2.29; deployment torch 2.9.1 / NCCL 2.27.5). Full chain:
`ROOT_CAUSE_ROUND2.md`. FIX-3 posts per-peer subgroups in ascending peer
order (mirror-symmetric lists ⇒ aligned cuts; ascending order ⇒ deadlock-free
chain argument; same-peer gangs preserved) + DIAG-2 per-peer histogram makes
pairwise mirror symmetry measured on every live window.

## Round-2 verification matrix (all on v2)

| # | evidence | result |
|---|---|---|
| 1 | Offline mechanism proof, `gpu_harness/test_split_deadlock_gpu.py`, 8×GPU: 299-op partitioned skewed plans (window#17 shape), plan identity gloo-asserted per iter | single-group path: **DEADLOCK at iter 0**, exact B1 signature (SeqNum COALESCED timeout, stuck cluster GPUs 100 %, no WARN, opposite cluster completed+waited). Per-peer path: **40/40 iters PASS**, canary_bad=0 |
| 2 | Offline regression j4+j5 suites (17 scenarios: empty plan, same-rank, 8/298/299/300/301-op plans, 5×300 & 3×300 consecutive under SM churn, real-planner 174/300-op plans) importing deployed v2 module | **17/17 PASS**, all canary_bad=0, all plan hashes rank-identical (`gpu_j4_results_v2.json`, `gpu_j5_results_v2.json`) |
| 3 | Live R2_P_E3 (P arm, E3, decisive B1-class trajectory; 8192 reqs) | **PASS**: 14 swap windows, up to 300 ops/504 P2P-ops per rank, zero watchdog, bench complete; TIMING on all ranks all windows |
| 4 | Live R2_F_E3 (F "1 16 16", E3; J1 original-fault path) | **PASS**: 14 swap windows (> old hang's 13), zero watchdog, bench complete |
| 5 | Live R2_P_E4 (P arm, E4 full duplex; J6 path) | **PASS**: 13 windows, zero watchdog, bench complete |
| 6 | Live R2_F_E4 (F arm, E4; J3 path) | **PASS**: 15 windows, zero watchdog, bench complete |
| 7 | DIAG-2 mirror symmetry (`assert_r2_extra.py`) on all 4 live jobs | **PASS**: 448 P2PPEERS lines, **1527 pair-window occurrences**, send(i→j)==recv(j←i) everywhere; discriminating (fails cleanly on v1 B1 log which lacks the marker) |
| 8 | Phase-2 12-check assert (`assert_server_job.py`: watchdog/desync/planhash-identity/swaps_min/large_plan/v2p1/p2l∈{issued−1,issued}/bench/perreq) on all 4 jobs | **pass=true** ×4 |

Round-2 ledger: 4 live server jobs (`LEDGER_r2.tsv`, halt-on-fail, all PASS)
— separate from the spent phase-2 6-job v1 ledger; offline suites recorded in
`gpu_harness.log` / `split_deadlock_logs/`.

## Sync-cost retention (directive: 修复若增加同步其成本必须保留)

Per-peer serialization cost is retained and measured, not hidden. ~300-op
windows, rank0 p2p phase: v1 (B1 window#9, 299 ops) 20.1 ms; v2 (R2_P_E3
windows #9/#17/#25, 299-300 ops) 22.4-26.4 ms — same bandwidth-dominated
regime, no material regression. First-window (channel warmup): v1 1158 ms vs
v2 605 ms. All TIMING lines are preserved verbatim in server logs.

## Directive compliance

- Control law untouched: adaptive thresholds, S/H/W state table, α/β,
  candidate rules, action budgets, plan content/order — the v1→v2 diff is
  confined to the NCCL posting layer inside begin() + one diagnostic log line.
- Migrations NOT disabled; adaptive NOT turned off; large transactions NOT
  trimmed (every op still moves in its window; the split is per-peer posting,
  not plan chunking).
- Fix rounds: 2 of ≤2 used. No further fix rounds remain; any future
  same-class fault ⇒ STOP + blocker report.
- New version frozen; no backfilling of v1 evidence; old pool untouched.

## Validity of pre-v2 artifacts under v2

- C1/C2 (I arm) and A1/A2 (E arm) never execute `async_swapper.begin()`
  (the only changed code path) ⇒ results valid; **frozen SLO reference
  (SLO_REFERENCE_FROZEN.json) stands**.
- B1 (P arm, v1): FAIL-kept, excluded from eligibility pooling (incomplete);
  B-arm eligibility will pool the four completed v2 boots.

## Conclusion

**Round-2 correctness gate: PASSED.** Phase-3 prep resumes from B-arm under
v2 (B1r, B2, B3, B4).
