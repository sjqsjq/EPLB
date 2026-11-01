# PROTOCOL_bsk.md - book_single_knobs_20261027 (EXPLORATORY single-domain knob trial)

## 1. Authorization and research question
- Author directive 2026-10-27 (verbatim): 可以尝试一下，看看会不会在单域的情况下就有好效果。
- Context: pbd campaign DIAGNOSIS_P0_BOOK row established that P0 online adaptation loses its book-phase gain to a late-book swap treadmill (17 of 23 swap windows after mid-book, cumulative swap wall time 24-27pct of serving, tps dips to 2.90 aligned with swap bursts, residual z never converges).
- Question A (pathology): does the treadmill appear in SINGLE-DOMAIN pure book (no domain switch), starting from the prover-oriented initial placement?
- Question B (mitigation): do the implemented-but-disabled config knobs OEPLB_BIAS_GATE (plan-quality gate) and OEPLB_SWAP_BUDGET_FRAC (cumulative swap wall-time cap) reduce churn cost and improve book throughput? Locked pb_oeplb code is NOT modified; knobs are env-read config fields with existing controller.py enforcement (L1014-1038 budget, L1234-1244 bias gate).

## 2. Stream
- bs_book_single.jsonl: 2000 rows, byte-preserved extraction (rows 3345-5344) of the frozen pbd doubled stream book block. sha256: 2e9fe867cf2cb317ea56492d0297e09f8b5303d08a9276dec81058ffda8d156a. Single domain (book), no switch. NREQ=2000 CONC=32 BENCH_TIMEOUT=1800.

## 3. Arms and knob calibration
- P0  = P RG=0 baseline (knobs off)
- P0B = + OEPLB_BIAS_GATE=0.5 (documented: act only when margin at least 2x estimate noise)
- P0S = + OEPLB_SWAP_BUDGET_FRAC=0.08 (cap cumulative swap wall time at 8pct of elapsed serving time)
- P0F = + both knobs
- ED, DF = reference arms (event-driven EPLB champion; static prover-frozen placement)
- Calibration (measured from logs, LEDGER KNOB_CALIBRATION row): spent_frac = cum_swap_s / log span: pb_P0_r1 0.4046 (216 events, b_late 3.97), pb_P0_r2 0.0884 (48 events, book mean 4.72 = best of both campaigns), pbd_P0_r1 0.2654 (192 events), pbd_P0_r2 0.2373 (192 events). Budget 0.08 anchored to the healthy pb_P0_r2 profile.
- launcher launch_arm_bsk.sh sha256 2896f550e39529392a7b528087007dcfa6875f5b61701af8c9c80eb22c4d6372 (cloned from frozen launch_arm_pc.sh cfbdaee7; deltas: knob arms appended, PL to bsk placements).

## 4. Boot runner and guard chain (cloned frozen pattern, deltas)
- run_bsk_boot.sh cloned from run_pbd_boot.sh: paths, stream sha, launcher sha, purpose bsk_single_book, POLICY case for knob arms, LEDGER_bsk.tsv, mech_check_bsk.py. Code locks (async_swapper/controller/sp_state/window_v2 shas) UNCHANGED - knobs do not touch code. Placement locks unchanged. bench_client_v2.py unchanged.
- mech_check_bsk.py: facts-only, completeness rc gate; v2p1 required for arms P/P0B/P0S/P0F; mechanism facts NEVER affect rc (anti-tuning clause preserved).

## 5. Queue behavior
- 12 boots: r1 P0 P0B P0S P0F ED DF | r2 P0B P0S P0F ED DF P0. rc=2 GUARD_ABORT halts queue (report to author, no blind restart). rc=1 logged continue; INFRA issues allow up to 2 zero-data restarts for that arm only, never silently dropped, INFRA_RESTART ledger row. Completion token emitted by the queue script itself.

## 6. Pre-registered analysis (written before results inspection)
- Per boot: tps, lat_mean_ms, ttft_mean_ms; half-split [0:1000) vs [1000:2000); 250-req bucket trajectory; swap stats from server.log (events, cum_swap_s, spent_frac, budget-exhausted line presence, bias-gate skipped-window line count); v2p1 facts (windows, swap windows, z trajectory/plateau, events).
- Pre-registered interpretation rules (DESCRIPTIVE, n=2, no significance claims):
  R1 pathology-in-single-domain: baseline P0 shows second-half bucket dips aligned with swap windows and spent_frac >> 0.08 -> treadmill exists without a domain switch. If baseline instead converges cleanly (few swaps, flat buckets) -> pathology is switch-specific; report as negative result with equal prominence.
  R2 budget knob: P0S/P0F second_half >= P0 second_half and spent_frac <= ~0.08 with budget-exhausted line present -> cap bites as designed.
  R3 bias gate: P0B vs P0 swap-window count reduction attributes phantom-skew chasing; if P0B ~ P0, plan noise is not the driver.
  R4 net effect: P0F book mean vs ED 4.58-equivalent single-domain reference and vs DF static; knob arms must not fall below DF mean by more than early-adaptation transient explains.
- Standing rules: h_rate/h_w are opportunity proxies, NEVER interpreted as remaining speedup headroom; no payback solver claims; descriptive deltas only.

## 7. Freeze rules
- This PROTOCOL, the stream, and all locks freeze at first boot START. Analysis script (descriptive_bsk.py) written from sec.6 BEFORE results inspection. Corrections append-only to LEDGER_bsk.tsv.

## 8. Artifact index
- queue.log, LEDGER_bsk.tsv, B_data/registry.json, scripts/{run_bsk_boot.sh, queue_bsk.sh, mech_check_bsk.py, launch_arm_bsk.sh}, code_lock/placement_shas.env, placements/, boots/<LABEL>/{boot_meta.json, server.log, bench.log, v2p1_rank0.jsonl, sp_run_end.json, MECH_CHECK.json, <LABEL>.json, <LABEL>_perreq.jsonl}, analysis/.
