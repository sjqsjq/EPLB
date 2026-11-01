# PROTOCOL_g3.md - pf_all_phase_select_20261030 (FROZEN 2026-10-31)

Status: frozen BEFORE any GPU boot of this campaign. Amendments only via
appended LEDGER_g3 rows with author authorization; this file is not edited
after freeze - corrections are appended as new sections.

## 1. Question and framing
Does letting the SP record gate admit DECODE batches (in addition to
PREFILL) change planning-window behavior and wall-clock cost on the frozen
pb_oeplb production stack? Two frozen workloads bracket the PD-correlation
spectrum (author revision 2026-10-30): ARC-C aggregate = MAIN scenario
(PD correlation ~0.849), prover = BOUNDARY contrast (~0.443). No blind data
swap; both arms use identical generation rules and identical stream files.
Results are DESCRIPTIVE timing/behavior comparisons at n=4 boots per
task x arm cell: NOT non-inferiority or superiority claims, NOT a
paper-level online-algorithm claim, and no verified-GPU-gain language
beyond the measured wall times themselves (author fix3 discipline).

## 2. Arms and sole delta
- G3_PF : OEPLB_RECORD_ALL_PHASE=0 (prefill-only recording; status quo)
- G3_ALL: OEPLB_RECORD_ALL_PHASE=1 (decode batches also recorded)
Everything else identical (launch_arm_g3.sh, byte-splice clone of frozen
PBS mother launch_arm_pc.sh; clone diffs archived in scripts/):
- BASE + PB server flags verbatim; NO --init-expert-location => identity
  initial layout in both arms (init_by_eplb lines must be 0).
- Threshold aperture: frozen PB passes --pb-oeplb-threshold-ratio 1.02;
  OEPLB_DEAD_ZONE_RATIO unset => auto dead_zone_ratio = 1.0962402945
  (EP power law, EP=8, PROVEN); effective thr = max(...) = 1.0962402945
  = production aperture. Any 1.02 row anywhere is labeled
  algorithm-diagnostic under that flag value (author fix3).
- pb_env_G3 = pb_env_common (frozen V21 knobs verbatim, EA_ALPHA=0.0,
  OEPLB_WINDOW_CONTROLLER=21, EA_W=16, probe off) + FIXED SP:
  OEPLB_SP_MODE=fixed SP_S=1 SP_W=16 SP_REV=0 SP_SEED=20261018; SP_H
  unset => H=W (config default 0). PD-12 (config.py:243) forbids REV=1
  without adaptive mode; PD-13 (config.py:228) requires an SP mode for
  RECORD_ALL_PHASE=1 - fixed satisfies it.
- Record gate (controller.py:460-465, frozen 4964cfc8): record iff
  gate (S=1 => every forward) AND not idle AND (prefill OR
  record_all_phase); decode_record_batches increments at :465 only when
  a non-prefill batch is recorded. cfg.min_record_tokens = 32 (frozen
  config.py dataclass field AND server_args getter default, extracted
  programmatically 2026-10-31) identical in both arms.

## 3. Streams (frozen; code_lock/stream_shas.env written once)
- arcc_1024_out32.jsonl      formal ARC   (delta vs source: max_tokens
  10->32 only; temperature 0 and ignore_eos true already in source)
- prover_1024_out32_ie.jsonl formal prover (delta vs E11 source:
  ignore_eos false->true only)
- arcc_512_smoke.jsonl / prover_512_smoke.jsonl: byte-exact first 512
  lines of the corresponding formal stream (prefix proofs PASS).
Full sha256 values live in code_lock/stream_shas.env + B_data/registry
.json (field-level delta accounting + builder sha). Both arms always
consume the SAME stream file for a given task/segment.

## 4. Boot matrix and order
SMOKE FIRST (author fix1): 2 boots, each boot sequentially covers BOTH
tasks x 512-row short streams at C256, per-task accounting via
segment_meta.jsonl (timestamps + v2p1/server.log line offsets):
  1. SM_PF  (G3_PF)  seg1 arcc_512_smoke  seg2 prover_512_smoke
  2. SM_ALL (G3_ALL) seg1 arcc_512_smoke  seg2 prover_512_smoke
SM_ALL checker runs the cross-boot gate (reads SM_PF g3_facts.json).
Any nonzero rc STOPS the queue; NO single-arm patches; any fix requires
a new frozen design row in LEDGER first.
P-criteria (exact observables, checker check_g3_boot.py):
  P1 ALL: sp_run_end.json decode_record_batches > 0.
  P2 PF : sp_run_end.json decode_record_batches == 0.
  P3    : per segment (each task), ALL v2p1 sums strictly exceed PF sums
          for BOTH sp_valid_rows and sp_recorded_steps.
  P4    : (revised per author fix1) decode records reach the FULL
          planning interval: in each ALL segment, windows with
          sp_recorded_steps>0 occur in BOTH halves of the segment window
          range and window count > 0; EMPTY PLANS ALLOWED (issued swaps /
          swaps_started may be 0 - P4 is record coverage, not swap
          activity; no actual swap is required).
  P5    : sole-delta discipline: (a) static launcher case-block diff
          leaves exactly one differing line (RECORD_ALL_PHASE 0 vs 1);
          (b) runtime ARMFLAGS echoes (8 per boot) differ between arms
          in record_all_phase only; (c) min_record_tokens config default
          = 32 identical (same frozen code shas, guarded pre-launch);
          (d) zero CALIB lines in both server.logs; (e) Init min_tok
          echo equal across arms.
FORMAL (only after SMOKE gate PASS; queue enforces gate file): 16 boots
= 2 tasks x 4 paired blocks x 2 arms, arm order alternating per block,
exact queue order:
  b1: G3_ARC_PF_b1,  G3_ARC_ALL_b1,  G3_PROV_PF_b1,  G3_PROV_ALL_b1
  b2: G3_ARC_ALL_b2, G3_ARC_PF_b2,   G3_PROV_ALL_b2, G3_PROV_PF_b2
  b3: G3_ARC_PF_b3,  G3_ARC_ALL_b3,  G3_PROV_PF_b3,  G3_PROV_ALL_b3
  b4: G3_ARC_ALL_b4, G3_ARC_PF_b4,   G3_PROV_ALL_b4, G3_PROV_PF_b4
C256, full 1024-row streams, O32 real generation, bench_client_v2
(frozen 06668243...), BENCH_TIMEOUT 1800s. rc=2 => STOP (guard abort);
rc=1/3 => log and continue; zero-data INFRA restarts (<=2) are
supervisor-judged at queue end, never automatic mid-queue.

## 5. Endpoints
Primary: total_time_s per boot; ALL 16 rows reported, no selective drops.
Secondary: output tps; per-phase record volumes (decode_record_batches,
sample_opportunities, recorded_steps, sp_fwd_total; per-segment v2p1 sums
of sp_valid_rows / sp_recorded_steps / n_fresh_assignments /
sp_cover_steps); planning-migration ledger (PLANHASH/DIAG/issued counts,
swaps_started_total, committed windows, p2l_version_max, c_window_ms
mean/max, moved-bytes ESTIMATE = issued_ops x 2 x 18.87 MB per FP8
expert copy, labeled ESTIMATE wherever reported).
Mech: ready_to_roll==1, traceback==0, oom==0, ARMFLAGS 8-identical with
expected fields, auto dead_zone 8 hits value 1.096, init_by_eplb==0
(identity layout), CALIB==0, desync==0, lockstep clean.

## 6. Budget and preservation
G3 GPU cap <= 5h hard (global campaign cap 6.5h; G1 consumed ~22min).
Estimate: smoke ~30min + formal 16 x ~7min ~= 2.2h, total ~2.7h.
On overrun risk: abort accounting row in LEDGER, no silent truncation.
LEDGER_g3.tsv append-only. On close: recon facts json + mirror to
/data/minghua/sjq/PAPER_EXPERIMENTS/pf_all_phase_select_20261030 +
closure row with guardrails (n=4 per cell, descriptive only).

## 7. Evidence naming discipline (author fix2)
Offline/expected artifacts are labeled EXPECTED; runtime artifacts
RUNTIME. This campaign has NO runtime p2l export endpoint (verified
absent: 0 endpoint hits); layout statements rest on code-sha guards +
ARMFLAGS/Init echoes + v2p1 p2l_version counters only. No file sha is
described as a runtime fingerprint.
