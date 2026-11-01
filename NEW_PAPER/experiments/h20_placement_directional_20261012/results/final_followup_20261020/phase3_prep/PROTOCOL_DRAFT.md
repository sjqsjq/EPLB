# Phase-3 prep protocol — DRAFT (gated on phase-2 GPU correctness PASS)

Budget: ≤12 prep serving boots (separate from ≤104 formal). Locked allocation
proposed (8 planned + 4 reserve):

## A. E_native official baseline (2 boots)
- Definition: launch_arm.sh arm E "auto 16 256" = official sglang EPLB path
  (--enable-eplb, algorithm auto, R=16 redundant experts, rebalance period
  256, --deepep-mode normal per pre-registered deviation sec.6). No PB flags,
  no pb_env_common, no v21 controller — official execution path control.
- A1: E_native @ E3 stream (8192 req, conc 256); A2: E_native @ E4 (2048, 256).
- Purpose: (i) verify official path runs on fixed deployment; (ii) baseline
  per-request logs for SLO eval; (iii) sanity check E vs old-campaign E
  behavior (no pool mixing: old E numbers stay in old pool).

## B. Sampling eligibility for P (4 boots: E3×2, E4×2)
- Arm P (SP_MODE=adaptive, S=1 init, W=16, seed 20261018, REV=1), fixed code.
- Metrics from v2p1_rank0.jsonl (per-window rows):
  - real_forward_share_S>1 = Σ steps_actual[sp_s>1] / Σ steps_actual
  - recording_ratio = Σ sp_recorded_steps / Σ true_steps_total
- FROZEN A3 reference (computed 2026-10-20 from 8 old A3 boots, pooled:
  rec=6375 true=163200): recording_ratio_A3 = 0.039062 (per-boot range
  0.03902-0.03912; sp_s==1 in 100% of A3 windows -- the quantified
  unexercised-sampling baseline). Frozen BEFORE any P prep boot.
- Decision rule (locked pre-boots): POOLED over all 4 prep boots --
  ELIGIBLE iff pooled S>1 share ≥ 0.10 AND pooled recording_ratio < 0.039062.
  Per-boot values always reported alongside (transparency; no post-hoc
  boot selection). Rationale for pooled: historical old-campaign P boots
  show high between-boot variance in S escalation (only 1/9 old boots
  reached S>1 share ≥10%: m_E3_P_b1 at 0.1875, sp_s=4; E4 P boots have
  rec_ratio 0.0032 << A3 but S==1 always) -- single-boot adjudication
  would be noise-dominated. Reference only; eligibility judged solely on
  NEW prep boots on FIXED code.
- If NOT ELIGIBLE: STOP — no phase-6 sampling supplement; report "sampling
  not exercised under P either" as a finding (未行使≠无用 still applies).

## C. SLO / Goodput protocol (2 calibration boots)
- Independent calibration: arm I (static initial placement, neutral — no
  rebalancing/migration mechanism) on calib_confirm.jsonl, ×2 boots.
  Reference frozen BEFORE any comparison arm is evaluated:
  - ref_p95_TTFT, ref_p95_TPOT (per-request logs: ttft_s; TPOT =
    (t_last_wall_s − first-token time)/comp_tokens from perreq jsonl).
- Goodput definition (frozen): request GOOD iff TTFT ≤ k·ref_p95_TTFT AND
  TPOT ≤ k·ref_p95_TPOT AND no error; Goodput = GOOD_count / total × req/s.
- Sensitivity: k ∈ {0.8, 1.0, 1.2} reported for every arm; primary = 1.0.
- Applies to phase-4 core supplement eval and E_native comparison.

## Governance
- Phase-3 boots run only after phase-2 freeze commit.
- Fixed transaction counts + timeouts per boot (same as GPU correctness jobs).
- Any boot with weight/mapping error or migration-class fault → keep failure,
  stop package, return to correctness diagnosis (fix-round budget: 1 left).
- Savings not convertible; no new model/EP grids; old pool untouched.
