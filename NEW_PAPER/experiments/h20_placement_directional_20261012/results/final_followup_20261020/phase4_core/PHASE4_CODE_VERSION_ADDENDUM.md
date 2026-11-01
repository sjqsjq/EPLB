# Phase-4 executed code version — addendum (2026-10-21, pre-boots)

The locked matrix `supplement_matrix_locked.csv`
(sha256 `d3ff2b09a27a683378b356e9a818a2f209c93af708147e8f31607c3b6044d71e`)
was generated BEFORE the round-2 correctness fix; its `code_freeze` column
therefore names v1 (`async_swapper=85a2dac882177667…`). Per governance
(post-freeze fix = new version, no backfilling), the CSV is **not edited** —
the lock and its sha stand — and the executed code for all phase-4 boots is the
round-2 frozen **v2**:

| component | sha256 | note |
|---|---|---|
| async_swapper.py | `2e4652a0306cb315…` (full: 2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f) | v2 = v1 + FIX-3 (per-peer ascending subgroup posting) + DIAG-2 (mirror-symmetry log); control law untouched |
| controller.py | `4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697` | unchanged since v1 |

Why v2 is the correct execution version: v1 contains a proven deadlock
(NCCL coalesced-group budget-split, pytorch/pytorch#174288 class) that killed
old-campaign boots (m_E3_P_b3 and 4 other retired hangs) and prep B1; running
64 formal boots on v1 would violate the phase-2 acceptance gate (now v2,
`../correctness/ACCEPTANCE_EVIDENCE_v2.md`, `../freeze/code_fixed_v2/`).
`run_supp_boot.sh` guards both shas per boot (mismatch ⇒ BOOT_ABORT, rc=2) and
records them in each `boot_meta.json`; `queue_phase4.sh` verifies the CSV sha at
start. Per-boot stream sha also verified against the locked CSV column.

All boots in this package run under v2; no v1/v2 mixing within the new pool.
Formal budget: 64 of ≤104 (phase-5 conditional ≤16 and other packages accounted
separately per directive). Halt-on-fail; failed boot kept, package stopped,
blocker reported (fix rounds 2/2 exhausted — no third round without author).
