# Phase-3 prep protocol — FROZEN before any prep boot (2026-10-20)

Budget: ≤12 prep boots. Planned: 8 (order C→A→B so the SLO reference is
frozen before any comparison arm runs). Reserve 4 for invalid-boot reruns
only (launch failure / infra fault); savings NOT convertible.
All boots on FIXED code (async_swapper 85a2dac882177667, controller
4964cfc8e8af397a; freeze snapshot in ../freeze/). Fixed transaction counts
+ timeouts per boot: E3=8192 req / E4=2048 req, conc 256, bench timeout
3600s, ready poll 225×4s. Serial queue, halt-on-fail.

## C. SLO calibration (2 boots, run FIRST)
- C1 = arm I @ E3 (label prep3_C1_I_E3); C2 = arm I @ E4 (prep3_C2_I_E4).
- Rationale: arm I = static initial placement, no rebalancing mechanism of
  any kind (no PB-OEPLB, no official EPLB) → most neutral latency anchor;
  independent of every compared arm and of phase-4 blocks.
- Reference (frozen from C boots' per-request logs, before A/B evaluation):
  - ref_p95_TTFT(E3), ref_p95_TTFT(E4), ref_p95_TPOT(E4).
  - TPOT per request = (elapsed_s − ttft_s) / (comp_tokens − 1), defined
    only for comp_tokens > 1. E3 is prefill-heavy with comp_tokens=1
    (verified: old VALID E3 boots avg_output_tokens=1; E4=128) → E3 SLO is
    TTFT-only; E4 SLO is TTFT+TPOT. PRE-DECLARED.
- Goodput (frozen definition): request GOOD iff error==null AND
  TTFT ≤ k·ref_p95_TTFT AND (E4 only) TPOT ≤ k·ref_p95_TPOT.
  Goodput = GOOD_count / wall_time_s [req/s]; goodness_rate = GOOD/total.
- Sensitivity: k ∈ {0.8, 1.0, 1.2}; primary = 1.0. Same k applied to all
  arms on a stream. Reported for every phase-4 arm and for A boots here.

## A. E_native official baseline (2 boots)
- A1 = arm E "auto 16 256" @ E3 (prep3_A1_Enative_E3); A2 = same @ E4.
- Official execution path: --enable-eplb --eplb-algorithm auto
  --ep-num-redundant-experts 16 --eplb-rebalance-num-iterations 256,
  --deepep-mode normal (pre-registered deviation sec.6, sealed precedent
  d38_L512_O1_comparison). No PB env/flags.
- Purpose: official-path control on fixed deployment; per-request logs feed
  SLO eval; sanity vs old-campaign E behavior (old pool untouched).

## B. P sampling eligibility (4 boots)
- B1,B2 = arm P @ E3 (prep3_B1_P_E3, prep3_B2_P_E3);
  B3,B4 = arm P @ E4 (prep3_B3_P_E4, prep3_B4_P_E4).
- Metrics from v2p1_rank0.jsonl: S>1 share = Σ steps_actual[sp_s>1] /
  Σ steps_actual; recording_ratio = Σ sp_recorded_steps / Σ true_steps_total.
- FROZEN A3 reference: recording_ratio_A3 = 0.039062 (pooled over 8 old A3
  boots, rec=6375/true=163200; A3 sp_s==1 in 100% of windows).
- Locked decision rule (pre-boots): ELIGIBLE iff POOLED over all 4 B boots:
  S>1 share ≥ 0.10 AND recording_ratio < 0.039062. Per-boot values always
  reported; no post-hoc boot selection. (Historical context only: 1/9 old P
  boots reached S>1 share ≥10% — m_E3_P_b1 0.1875 with rec_ratio 0.0519
  > A3; E4 P boots rec_ratio 0.0032 but S==1 always. Fixed-code behavior
  may differ; judged solely on NEW boots.)
- If NOT ELIGIBLE: STOP — no phase-6 sampling supplement; report finding.

## Governance
- Prep boots run only on frozen fixed code (phase-2 gate PASSED, commit 26d3bba).
- Any boot with weight/mapping error or migration-class fault → keep failure
  artifacts, STOP package, return to correctness diagnosis (fix rounds: 1 of
  ≤2 used; one remains).
- No sample deletion by mean/p/boot-draw; 不显著≠等价; 未行使≠无用.
- Old-pool data used ONLY for frozen reference constants (A3 ratio) and
  stream/recipe reuse; no mixing of results pools.

---

## ADDENDUM A1 — round-2 interruption and v2 resume (appended 2026-10-21; frozen text above unchanged)

**Interruption record.** prep3_B1_P_E3 (2026-10-20T09:37Z, v1 code) hung at swap
window#17: NCCL coalesced-group budget-split deadlock (pytorch/pytorch#174288
class; identical signature to old m_E3_P_b3). Root cause proven offline
(deterministic 8-GPU repro: single-group posting deadlocks at iter 0, per-peer
posting 40/40 clean). Fixed by FIX-3 (per-peer ascending subgroup posting with
host wait) + DIAG-2 (mirror-symmetry log) in async_swapper v2
sha256 2e4652a0306cb315…; controller unchanged 4964cfc8e8af397a…. Control law,
thresholds, S/H/W tables, α/β, candidate rules, action budget untouched.
Verification: offline j4/j5 v2 17/17 PASS; live R2_{P,F}_{E3,E4} 4/4 PASS
(1527 mirror pair occurrences, 0 watchdogs). Evidence:
`../correctness/ROOT_CAUSE_ROUND2.md`, `../correctness/ACCEPTANCE_EVIDENCE_v2.md`,
`../freeze/code_fixed_v2/` (+ SHA256SUMS_code_fixed_v2.txt).

**Governance update.** Fix rounds now **2 of ≤2 USED**. Any future weight/mapping
error or same-class migration fault ⇒ STOP all performance packages and report
blocker to author (per directive); no third fix round without authorization.

**B1 disposition.** Kept as failure evidence at `boots/prep3_B1_P_E3/` (server.log,
v2p1_rank0.jsonl 17 windows, watchdog core dumps `core-*` in this directory,
gitignored). NOT re-run over, NOT deleted, **excluded from eligibility pooling**
(v1 code + incomplete: hung before bench).

**Pre-v2 artifact validity.** C1/C2 (I arm) and A1/A2 (E arm) never execute
async_swapper.begin() p2p posting ⇒ valid under v2 without re-run.
`SLO_REFERENCE_FROZEN.json` (frozen 2026-10-20T09:24:53Z) stands.

**Resume under v2.** B-arm boots with parameters IDENTICAL to section B above:
prep3_B1r_P_E3 (B1 slot re-execution), prep3_B2_P_E3, prep3_B3_P_E4,
prep3_B4_P_E4 — all on v2 frozen code. Runner: `queue_prep_resume.sh`
(halt-on-fail, ledger appends to LEDGER_prep.tsv).

**Eligibility pooling (rule unchanged).** Pooled over the four v2 B boots
(B1r, B2, B3, B4): S>1 share ≥ 0.10 AND recording_ratio < A3=0.039062, else
STOP sampling supplement (no phase-6). Adjudicator:
`extract_eligibility_resume.py` (same frozen rule; original
`extract_eligibility.py` kept untouched as the pre-interruption artifact;
only its B1→B1r label substitution differs). Output: `SAMPLING_ELIGIBILITY.json`.

**Prep budget accounting.** Used: C1, C2, A1, A2, B1(FAIL) = 5; resume:
B1r, B2, B3, B4 = 4 → **9 of ≤12**, 3 reserve kept for this package only
(unmet-precondition savings NOT converted to other packages).
