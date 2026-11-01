# Author-directed diag #2 — GitHub-pristine comparison (2026-10-21)

Author correction honored: "最原始版" = GitHub origin/main (git@github.com:sjqsjq/EPLB.git)
OEPLB/src, never pushed over, never modified — NOT the code_lock pre-fix copies.

## Pristine package (8 files, verified shas)
- async_swapper.py 93fa4ccdd653443a… — IDENTICAL to code_lock "original" async_swapper
- controller.py    ada04359c2a7ffa2… — differs from code_lock controller c6d2f764 (1467 diff lines;
  code_lock "orig" controller was itself campaign-modified vs GitHub)
- config.py        232ecbd529d0b109… — differs from deployed v2 config 52f7f4d7 (255 diff lines)
- + __init__.py, fast_metadata.py, per_request_tracer.py, rebalancer.py, routing_tracer.py
- Pristine src has NO window_v2/window_v2p1/sp_state/v2p1_signals/probe_admission modules.
- Full deployed dir replaced with exactly these 8 files (n=8 verified); v2.1 env knobs
  (OEPLB_SP_MODE/WINDOW_CONTROLLER/V21_*) inert on pristine code by construction; server_args
  showed pb_oeplb_adaptive_window=False → pristine ran fixed sync_window=16, CLI flags identical
  to all other arms.

## Compatibility
Campaign sglang glue works with pristine package unmodified: __init__ exports
set/get_pb_oeplb_controller; config has from_server_args (getattr-based, tolerates extra fields);
controller has (cfg, model_runner) ctor, record_next_layer, on_forward_pass_end;
on_forward_pass_start is hasattr-guarded → skipped on pristine.

## Results (E3, 8192 req, conc 256, single adhoc boots; ±1–3 tps single-boot noise; errors=0)
| Code version                                   | P arm tps | I arm tps |
|------------------------------------------------|-----------|-----------|
| v2 frozen (current; FIX-1/2/3+DIAG)            | 44.1      | 42.8      |
| code_lock pre-fix hybrid (orig ctrl c6d2f764 + v2 rebalancer/modules) | HUNG rc=124 (window#17, 299 swaps, SeqNum=5 COALESCED 600s watchdog, GPU split 4×100%/4×0%) | 42.1 |
| GitHub pristine (origin/main, ctrl ada04359)   | 38.5      | 42.3      |

Phase-4 formal pool reference (8 boots each): E3 P=43.06±1.25, I=41.92±0.98 — v2 adhoc matches formal.

## Reading
1. Author's degradation suspicion NOT supported on E3: current v2 P (44.1) is the FASTEST of all
   three code versions; GitHub-pristine P is ~13% slower (38.5). The fixes did not cost throughput
   here — v2 adhoc P/I match the 64-boot formal means.
2. GitHub-pristine P completed E3 with heavy migration (window#1: 261 swaps; ≥16 issued windows;
   zero watchdog/COALESCED faults) — pristine controller pacing differs from the pre-fix campaign
   controller, which deadlocked deterministically at window#17/299 swaps with the SAME async_swapper.
   This sharpens root cause: the deadlock class was triggered by the campaign-era controller's
   coalesced-group plan issuance, not by pristine plan shapes on this stream.
3. I baseline stable across all code versions (42.1–42.8) as expected (plain arm; package import-only).
4. assert_prep_boot.py flags planhash_identity/v2p1_no_desync=false on gh boots — v2-specific checks,
   undefined for pristine code; bench_complete/swaps_min/no_watchdog/no_coalesced_err all pass.

## Governance
- Separate author_diag ledger (LEDGER_adhoc.tsv, code=gh_pristine rows); NOT in formal ≤104 / prep ≤12
  pools; descriptive only; no pool mixing with s_* phase-4 data.
- Containment: bench timeout 1200s, watchdog 600s, kill+GPU-drain between boots.
- Full deployed dir backed up to /tmp/dep_pre_gh_backup before swap; trap EXIT restored it;
  RESTORE_v2_full OK, post-run shas verified v2 (async 2e4652a0, controller 4964cfc8, config 52f7f4d7),
  __pycache__ cleared.
- No budget extension, no control-law change, no parameter search; local commit only, never push.
