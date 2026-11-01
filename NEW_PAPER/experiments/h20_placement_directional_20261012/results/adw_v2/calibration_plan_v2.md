# ADW v2 Step-5 calibration plan (PRE-REGISTERED fitting rules)

Status: pre_registered_before_fitting_data. Date: 2026-10-16.
Anti-tuning rule: every fitting rule below is fixed BEFORE the calibration
run is analyzed. Knob values are frozen on W0 (calibration.jsonl) ONLY and
never re-tuned on holdout conditions or after seeing formal results. If a
rule turns out to be bad, the failure is reported as-is (no silent re-fit).

## 1. Calibration run design (single logging run on W0)

- Workload: calibration.jsonl (8037 requests, O1, 3 known domain switches).
- Arm: OEPLB_WINDOW_CONTROLLER=2, W_init=16 (grid middle), alpha=0.0,
  adaptive flags OFF, default planner - identical to the phase-1 V2 arm
  except knobs below.
- Knob settings FOR THE FITTING RUN ONLY (chosen so W cannot move, making
  the logged series a fixed-W observation, no feedback):
  kappa=1.0 (=> logged z == s_change * n_layer), z_lo=z_hi=z_urgent=1e12
  (no shrink/grow can trigger), theta_h_*=inert, n_min=0, dwell/m/beta at
  spec defaults. W stays 16 for the whole run by construction.
- 1 start is sufficient for signal fitting (this run calibrates SIGNAL
  SCALES, not service-time effects; service-time behavior is validated
  separately in the behavior run, §4).
- Logs: rank0 JSONL (s_change, z(=s_change*n_layer), h_w, n_fresh per
  window) + server log.

## 2. Stationary-window definition (offline; labels NEVER given to controller)

calibration.jsonl preserves the source order of prefill_heavy_universal.jsonl
(4 contiguous 4000-line domain blocks, hash-line split keeps ~half of each
block, order preserved). Offline domain position of calibration line i:
quarter boundaries at cumulative counts 2009/4018/6027 (+-1 rounding of
8037/4). A WINDOW is stationary iff its request span lies entirely inside
one quarter, excluding a guard of +-250 requests around each boundary
(switch-adjacent windows are used later as change evidence, not for kappa).
Window->request-span mapping comes from logged steps_actual + n_fresh
(proportional allocation; documented approximation).

## 3. Fitting rules (applied blind, in this order)

Let Z_s = {z of stationary windows}, H_s = {h_w of stationary windows},
Z_c = {z of windows whose span covers a boundary guard region}, N16 =
{n_fresh of stationary windows}.

- kappa      = median(Z_s)                      [so stationary z has median 1]
- z_lo       = p90(Z_s) / kappa_rescaled -> after kappa applied: z_lo = p90(Z_s/kappa) = p90 of rescaled stationary z
- z_hi       = p99(Z_s rescaled)
- z_urgent   = max( 3*z_hi , p50(Z_c rescaled) )   [urgent must fire on real switches]
- theta_h_low = p30(H_s)   [grow allowed only when headroom below this]
- theta_h_act = p60(H_s)   [shrink actionable only when headroom above this; theta_h_low < theta_h_act by construction]
- n_min      = 0.25 * median(N16) * (16/8)  -> tokens-per-step floor scaled to the SMALLEST grid window: n_min = 0.25 * median(N16) / 2  (a W=8 window normally has ~half the tokens of W=16; flag uncertain below a quarter of that)
- m_shrink=2, m_grow=3, dwell_shrink=2, dwell_grow=2, u_max=4, beta=0.5,
  urgent_episode_cap=3, w_uncertain_target=W_init: SPEC DEFAULTS KEPT.
  No data-driven changes in phase 1 (documented decision).
- W_init     = 16 (the W0-grid selection W* - unchanged; v2 arms and
  F-global share it).

Sanity gates on the fit (report, do not silently fix):
- separation: p50(Z_c rescaled) > z_hi must hold, else report "switch
  signal not separable at fitted kappa" as a limitation;
- stability: IQR(Z_s rescaled) reported; if p90/p50 > 10, report heavy tail.

## 4. Behavior validation run (after freeze, still on W0)

- Same arm with FROZEN knobs; W is live.
- Outputs (deliverable D): W time/forward/token-weighted dwell distribution;
  per-window evidence chain (z, h_w, n_fresh, event, reasons, dwell,
  uncertain/carry); check/plan/migration counts + cumulative cost
  (record/allreduce/planbuild/finalize, migration walls reported ALONGSIDE,
  never subtracted); detection latency around the 3 known switches
  (windows from boundary to first confirmed/urgent shrink); c_window_ms.
- Cost budget check: v2 extra cost vs v1-profile expectations (hash
  all_reduce 2 elems / 16 windows; vectorized decision ~1-2 ms/window).
- Rules frozen after this run: if behavior shows a defect, the defect is
  DOCUMENTED; knob changes require a new pre-registered rule + re-freeze,
  and any holdout run already executed stays reported as-is.

## 5. Fixed-arm calibration (fairness rule, Section 六)

- F-local grids on W1/W2: fixed W in {8,16,32,64}, alpha=0.0, 1 start per
  cell (selection by argmin; ties -> smaller W). Fixed arms get the SAME
  calibration access as v2; v2 gets NOTHING from W1/W2 before formal runs.
- F-global = W16 (already selected from the W0 8037-request grid, FACT-09).
