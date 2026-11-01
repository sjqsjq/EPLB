# Round-2 correctness diagnosis & fix — NCCL coalesced-group budget-split deadlock

Date: 2026-10-20 (UTC). Campaign: final_followup_20261020, phase-3 prep boot
`prep3_B1_P_E3` (P arm @ E3) hung at swap window #17 on the round-1 frozen
code (async_swapper 85a2dac8…, controller 4964cfc8…). This document records
the round-2 (final permitted) diagnosis, root cause, fix, and verification.
Governance: B1 failure artifacts kept; prep queue halted at B1; package
stopped and returned to correctness diagnosis per directive.

## 1. Measured facts from the B1 hang (all from preserved logs)

- Window #17, 299 ops, plan_hash `fbcf602d440b7317` **identical on all 8 ranks**
  (DIAG-1; the hash covers the ordered `(layer, phys_a, phys_b)` triples, and
  phys slots encode ranks ⇒ identical hash ⇒ identical ordered plans).
- Issued windows before #17: #1(268), #2(49), #9(299), #10(259) — all completed
  on all 8 ranks (TIMING lines; batches 20–1160 ms). `p2l_version=4` ✔.
- Ranks {2,3,4,7}: TIMING at 09:40:27 (batch_isend_irecv ≈ 21.6–27.9 ms),
  logged `window#17: issued`, returned to the event loop, then blocked
  host-side in `recv_requests → broadcast_pyobj` (gloo) waiting for the other
  four ranks; GPUs 0 % (pyspy dumps from the sglang watchdog on DP2/4/7).
- Ranks {0,1,5,6}: **all four enqueued** the window-#17 coalesced group —
  watchdog: `WorkNCCL(SeqNum=5, OpType=COALESCED) ran for 600042+ ms`
  (enqueued 09:40:28.5, i.e. ~1.5 s AFTER the fast cluster's group had
  already completed); `last enqueued work: 5`, identical PG state on all
  four; GPUs 100 % (resident spinning kernels); **zero NCCL WARN lines**;
  no retry-path log (FIX-2 retry never fired); last app logs: VERIFY BEFORE
  (ranks 0/5, the plan[0] participants) / PLANHASH (ranks 1/6) — both
  upstream of the post, consistent with being inside `begin()`.
- `issued` is logged after `begin()` returns (controller.py L1372 vs L1367)
  ⇒ the four stuck ranks never returned from `begin()`.

### Timing paradox ⇒ the window-#17 plan is partitioned
The fast cluster's group completed at ~09:40:27.0; the slow cluster posted at
~09:40:28.5. An NCCL send cannot complete before the peer posts the matching
recv. Therefore ranks {2,3,4,7} had **no ops with {0,1,5,6}** in window #17:
the 299-op plan splits into two 4-rank clusters. The fast cluster completed
independently; the slow cluster's group — posted by all four of its ranks,
kernels resident — never completed and never errored.

## 2. Root cause (external, known, OPEN upstream)

`torch.distributed.batch_isend_irecv` wraps **all** ops in a single
`ncclGroupStart/End` (torch 2.9.1 `distributed_c10d.py`, `_coalescing_manager`).
NCCL splits a large p2p group into **multiple sequential kernel plans** via an
internal budget (`scheduleP2pTasksToPlan`, nccl `enqueue.cc` ~L1094), and the
NCCL source itself states: *"p2p work isn't collective. If we to drain p2p
first, the place where we cut the kernel could vary by rank"* (~L1465-1468).
With **asymmetric per-rank participation** (every real swap plan: rank op
counts differ), the budget cut points differ between peers, so matched
send/recv pairs land in **different plan indices** on the two sides. Plans
execute sequentially on the stream ⇒ each rank's current kernel waits for a
peer op that sits in a later kernel the peer never reaches ⇒ silent circular
deadlock: everything posted, kernels resident (GPU 100 %), no NCCL error,
watchdog fires at the 600 s timeout.

- Upstream: [pytorch/pytorch#174288](https://github.com/pytorch/pytorch/issues/174288)
  "Batched isend/irecv with NCCL backend hangs on high load" — **OPEN**;
  reproduced by multiple users on 4×H100 up to NCCL 2.29; root-cause analysis
  in-thread (RohitRathore1, 2026-02-12) cites exactly the budget-split
  mechanism and proposes per-peer group splitting; deployment here:
  torch 2.9.1+cu128 / NCCL 2.27.5.
- Related background: [NVIDIA/nccl#2130](https://github.com/nvidia/nccl/issues/2130)
  (coalesced NCCL p2p = one indivisible batch wait),
  [pytorch/pytorch#175145](https://github.com/pytorch/pytorch/issues/175145).

This explains **every** recorded fact: identical plans yet divergent fate;
all-posted non-completion; the rank split (one cluster's cuts happened to
align, the other's didn't); no WARN; the identical old-campaign signature
(m_E3_P_b3: same window #17, same trajectory, same ranks — same NCCL); why
windows #1/#2/#9/#10 and the J-series passed (cut alignment is
structure-dependent luck); and why round-1 FIX-1/FIX-2 could not close it
(they fix host-side ordering, not NCCL's intra-group plan scheduling).
Round-1's diagnosis was incomplete: the run-ahead mechanism was real but a
second, independent mechanism (this one) remained. Both produce the same
watchdog signature.

## 3. Offline proof (8×GPU, no server; `correctness/gpu_harness/test_split_deadlock_gpu.py`)

Deterministic generator mimics window #17: 299 ops, partitioned into
{0,1,5,6}/{2,3,4,7}, **skewed per-rank participation**, 2 weights/op
(13.9 MB + 0.9 MB), unique slots; plan identity asserted per iteration via
gloo all-gather of the plan hash; formula canary on every received tensor.

| posting path | result |
|---|---|
| `single` (round-1 production path: one `batch_isend_irecv(all ops)`) | **DEADLOCK at iter 0** in the skewed cluster: watchdog `WorkNCCL(SeqNum=1, OpType=COALESCED)` timeout at 180 s; stuck-cluster GPUs 100 %, opposite cluster completed and waited at the gloo barrier; no NCCL WARN — exact B1 signature (`split_deadlock_logs/single2.log`, rc=1) |
| `perpeer` (FIX-3 candidate: per-peer subgroups, ascending peer order, wait each) | **40/40 iterations PASS**, total_canary_bad=0, wall 40 s (`split_deadlock_logs/perpeer.log`, rc=0) |

Note the first (crashed) single-mode attempt was a harness NameError
(print after `del`), fixed before the successful repro — documented here for
transparency; it consumed no fix-round budget.

## 4. FIX-3 (+ DIAG-2), async_swapper.py `begin()` p2p section

- **FIX-3**: replace the single coalesced post with **per-peer subgroups in
  ascending peer order**, host-wait each subgroup, keep FIX-1
  (empty_cache+device sync before posting) and FIX-2 (device sync before
  return). Per-subgroup retry-once replaces the whole-batch retry (a whole
  sequence retry after partial completion would double-post completed peers).
  Safety argument:
  1. both sides of a pair derive **mirror-symmetric** op lists (same
     count/sizes/order) from the identical plan ⇒ identical budget inputs ⇒
     aligned plan cuts within each single-peer subgroup;
  2. ascending-peer + wait-per-subgroup is deadlock-free: any blocked chain
     strictly descends in rank (a rank blocked on peer p implies p is at an
     earlier, i.e. lower-ranked, peer) — impossible in a finite set;
  3. gang safety (send/recv to the same peer mutually unblocking) preserved:
     same-peer ops always stay in one group.
- **DIAG-2**: one log line per rank per window `[PB-OEPLB-P2PPEERS]` with the
  per-peer send/recv histogram ⇒ pairwise mirror symmetry becomes *measured*
  on every live window. Pure logging.
- Untouched, per directive: adaptive thresholds, S/H/W state table, α/β,
  candidate rules, action budgets, plan content and ordering; migrations stay
  enabled; large transactions stay whole (no chunking of plan content — the
  split is at the NCCL posting layer only, every op still moves in its
  window); **the added serialization cost is retained and measured** (TIMING
  line will show it; reported in phase-4 notes, not hidden).
- New version (post-freeze fix ⇒ new version, no backfilling):
  `async_swapper.py` sha256 `2e4652a0306cb315…` (v2; v1 frozen copy remains
  under `freeze/code_fixed/`). `controller.py` unchanged (`4964cfc8e8af397a…`).
  src == deployed verified by diff.

## 5. Verification plan (round-2 ledger, separate from the spent phase-2 6-job ledger)

1. Offline regression: j4 + j5 suites (17 scenarios incl. 298/299/300/301-op
   plans, empty plan, same-rank, consecutive churn, real-planner plans) on
   the v2 module — **DONE, 17/17 PASS**, all canary_bad=0, all plan hashes
   identical across ranks (j4 2026-10-20T17:03:53→17:04:32Z, j5
   →17:05:37Z; `gpu_harness/gpu_j4_results_v2.json`,
   `gpu_j5_results_v2.json`; v1 results archived alongside).
2. Offline mechanism proof above (DONE: repro + fix pass).
3. Live 8-rank server verification, ≤4 jobs (round-2 ledger) — **DONE,
   4/4 PASS** (LEDGER_r2.tsv, 2026-10-21T02:50:29→03:08:18Z): R2_P_E3
   (14 windows, max 300 ops / 504 P2P ops per rank, 8192/8192), R2_F_E3
   (14 windows), R2_P_E4 (13), R2_F_E4 (15); zero watchdogs; phase-2
   12-check assert pass=true ×4; DIAG-2 mirror symmetry PASS on 448 lines /
   1527 pair-window occurrences. Note: R2_P_E3's stochastic trajectory
   differed from B1's (P arm is measurement-driven), so B1's exact
   window-#17 plan was not replayed live; the deterministic evidence for
   that shape is the offline proof (step 2), and the live jobs exercise the
   same code path at equal-or-larger scale (504 P2P ops/rank > B1's load).
4. All passed ⇒ v2 frozen (`freeze/code_fixed_v2/`,
   `SHA256SUMS_code_fixed_v2.txt`), `ACCEPTANCE_EVIDENCE_v2.md` written,
   committed; phase-3 resumes from B1 (rerun under v2; B1 failure kept).
   Validity of already-PASSED prep boots under v2: C1/C2 (I arm) and A1/A2
   (E arm) never execute `async_swapper.begin()` p2p posting (FIX-3's only
   behavior change), so their results and the frozen SLO reference remain
   valid; B-boots (P arm) all run under v2. Documented in the version note.
5. **If the live verification hangs again: fix rounds are exhausted (2/2) ⇒
   STOP all performance packages and report the blocker with the specific
   gap, per directive.**

## 6. Budget/ledger status after this round

- Fix rounds: 2 of ≤2 used (r1 = FIX-1/FIX-2 + DIAG-1; r2 = FIX-3 + DIAG-2).
- Prep boots: 5 of ≤12 used (C1, C2, A1, A2 PASS; B1 FAIL-kept). Remaining 7
  must cover: B1r, B2, B3, B4 (+3 reserve).
- Formal boots: 0 of ≤104 used. Phase-2 correctness ledger: 6/6 (v1, closed).
  Round-2 verification ledger: ≤4 live server jobs + offline suites (this doc).

Core dumps from the offline single-mode deadlock (3×1.4 GB,
`core-pt_nccl_watchdg-*`) are preserved OUTSIDE the git tree at
`/workspace/EPLB/cores_round2_repro/` (binary size; logs are the evidence of
record in `split_deadlock_logs/`).
