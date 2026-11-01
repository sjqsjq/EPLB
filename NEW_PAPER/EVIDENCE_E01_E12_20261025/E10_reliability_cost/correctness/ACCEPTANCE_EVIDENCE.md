# Phase-2 acceptance evidence — migration-race fix (final_followup_20261020)

Date: 2026-10-20 (UTC). Fix round usage: **1 of ≤2** (round 1 = FIX-1/FIX-2
sync points + DIAG-1 PLANHASH log; no further code change was needed).

## Frozen code identity
- src /workspace/EPLB/OEPLB/src == deployed
  /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb:
  **0 diffs** across all .py files (verified post-run).
- async_swapper.py sha256[0:16] = 85a2dac882177667 (pre-fix: 93fa4ccdd653443a)
- controller.py    sha256[0:16] = 4964cfc8e8af397a (pre-fix: c6d2f76454679dff)
- Snapshot: freeze/code_fixed/ + freeze/SHA256SUMS_code_fixed.txt
- Pre-fix snapshot: code_lock/ (commit 2f775a7). Diffs: forensics/MINIMAL_DIFF_*.patch
- Control law untouched: adaptive thresholds, S/H/W state table, α/β,
  candidate rules, action budgets, max_total_ops=300 all unchanged.
  Migrations NOT disabled; large transactions NOT trimmed; sync cost RETAINED
  (two full-device synchronizations per swap batch, by design per directive).

## CPU correctness (pre-GPU gate)
test_swap_race_cpu.py (gloo, 8 ranks, real deployed executor + real planner):
11/11 PASS — commit e863632 (cpu_test_results.json, cpu_test_full.log).

## GPU correctness jobs — 6 of ≤6 budget, all PASS
| job | kind | coverage | verdict |
|---|---|---|---|
| j4 | standalone NCCL×8, 12L, real row bytes (13.9MB/op) | empty / same-rank / 8 / **298 / 299 / 300 / 301** hand plans, 5×300 consecutive under side-stream SM churn, real planner plan (174 ops); canary 0 bad rows all scenarios; PLANHASH+comm-order identical | PASS |
| j5 | standalone NCCL×8, **94L full geometry (20.9GB/GPU)** | 3×300 consecutive under churn + **real planner cap-300 plan**; canary 0 | PASS |
| J1_E3_F | real server, E3 stream, F arm "1 16 16" = **original fault path** (retired m_E3_F_b3 hung at window#17, 299 ops) | 14 swap windows incl. **300-op plans at windows #9/#17/#25**; 8192/8192 ok | PASS* |
| J2_E3_P | real server, E3, P arm (adaptive) | 14 swap windows, max 300 ops, 8192/8192 ok | PASS |
| J3_E4_F | real server, E4 fullduplex, F arm | **19 swap windows** (retired m_E4_F_b4 hung with 13 prior), max 300 ops, 2048/2048 | PASS |
| J6_E4_P | real server, E4, P arm | 13 swap windows, max 300 ops, 2048/2048 | PASS |

\* J1's in-run verdict initially read FAIL due to an assert-script bug (wrong
summary filename + wrong p2l invariant); offline re-adjudication of the SAME
on-disk artifacts passes all 12 checks. See HARNESS_BUG_NOTES.md.

Per server job, all held: zero Watchdog/COALESCED/scheduler-exception/RANK
DESYNC lines; PLANHASH identical across 8 ranks on every plan window;
v2p1 h_min==h_max and desync=false on all rows; p2l_version ∈ {issued−1,
issued} (invariant validated on old VALID boots); bench complete with
per-request rows == NREQ.

## Directive coverage checklist
- 真实合法计划于299/300及相邻边界: j4 G4-G7 (298/299/300/301), j5 H4
  (planner-generated 300), server jobs max_n_ops=300 (×4 jobs). ✔
- 空计划: j4 G1. ✔  连续迁移: j4 G8 (5×300), j5 H1-H3 (3×300). ✔
- 原始故障路径: J1 (exact m_E3_F_b3 config; passed hang window #17 with a
  300-op plan), J3 (m_E4_F_b4 stream/arm; exceeded its 13-swap hang point). ✔
- 每rank通信顺序: comm_order_hash determinism (j4/j5) + per-window PLANHASH
  8-rank identity (all server jobs). ✔
- 权重/scale/p2l/stats一致性与commit version: W+S canary rows verified
  elementwise per scenario (0 bad); p2l_version invariant; v2p1 stats
  hash-identical, desync-free. ✔
- 输出/canary正确性: bench ok==NREQ err==0 (all 4 server jobs); canary
  formula-fill verification (j4/j5). ✔
- Assertion-suite discrimination: run against retired m_E3_F_b3 artifacts →
  correctly FAILS (watchdog/COALESCED/scheduler_exc all hit). ✔

## Harness infrastructure bugs (documented, NOT fix-code failures)
HARNESS_BUG_NOTES.md: attempt-1 NameError (dead code), attempt-2 replica
identity-reset (canary false-positive on exactly the previous scenario's
slots; debug_g3.py fresh-state rerun PASSED with identical plan),
attempt-3 churn-thread restart, J1 assert filename+invariant bugs. Deployed
fix-code sha unchanged across all attempts; no fix-round budget consumed.

## Conclusion
The circular-wait path (coalesced P2P group ↔ next-forward DeepEP spin
kernels) is closed by the two sync points on real 8×H20 serving at
production scale, with all correctness dimensions exercised and no control-law
change. **Phase 2 gate PASSED — cleared to proceed to phase 3 prep boots.**
