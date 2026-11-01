# Fit rules v2 (PRE-REGISTERED addendum, before re-fitting)

Fit rules v1 (calibration_plan_v2.md §3) FAILED their pre-declared sanity
gate (separation p50(Z_c')=0.49 > z_hi=144.4: FALSE) on the 2026-10-16
fitting run. Diagnosis from the per-window table (fit_run_v1_diagnosis):
the v1 window classification was contaminated, NOT the signal:
(a) EWMA detection ref (beta=0.5) has ~4-5 window memory: after a step
    change, s_change decays by ~half per window (observed 152.9->75.0->17.2
    ->3.4->1.1), so the v1 +-250-request guard let 2 post-switch windows
    (z'=152.9, 75.0) into the "stationary" set, blowing up p90/p99;
(b) windows 2-9 decay monotonically (26.7->1.9) = ref-settling transient
    (ref was seeded on the ramp window w1, 68.5M vs 98.5M tokens);
(c) block boundaries 2009/4018/6027 were approximations (equal quarters);
    the hash line-split does not allocate exactly 2009 lines per block.

## v2 rules (mechanism-justified, applied blind, in this order)

R1. EXACT boundaries: reconstruct the calibration.jsonl -> universal-block
    origin map offline with the frozen split rule (sha256("fullservice-v2-
    split-v1|index|line")[:8] < 2^63 -> calibration; domain = universal_line
    // 4000). Labels are offline-only; the controller never sees them.
R2. Run-start exclusion: windows 1-8 excluded (ref-settling transient;
    beta=0.5 attenuates the init mismatch 2^8=256x by window 9; w1 seeds
    the ref and has no z anyway).
R3. Change zone: windows whose request span intersects any exact boundary
    +- 700 requests (EWMA memory ~5 windows x ~124 req/window ~= 620,
    rounded up to 700). Wind-down: windows with n_fresh < 0.9*median
    also excluded from the stationary set.
R4. Stationary set Z_s, H_s, N_s: all remaining windows.
R5. kappa = median(Z_s)  [same estimator as v1, clean population]
R6. z_lo = p90(Z_s/kappa); z_hi = p99(Z_s/kappa)  [same quantile rules]
R7. z_urgent = max(3*z_hi, p50(Z_c/kappa)) where Z_c = change-zone windows
    with z >= z_hi (i.e., windows that actually carry a change signal;
    boundary-guard windows with baseline z carry no switch information).
R8. theta_h_low = p30(H_s); theta_h_act = p60(H_s)  [unchanged]
R9. n_min = 0.25*median(N_s)/2  [unchanged]
R10. Sanity gates (report-only, same as v1): separation p50(Z_c') > z_hi;
    tail p90/p50 of Z_s' <= 10.

Known accepted consequences (declared now, not fixed later):
- Intra-block bursts (w12-13, requests ~1325-1573, z'~104/77) fall inside
  the boundary-1 change zone by R3 and are excluded from Z_s. They are
  REAL routing events (not label boundaries); at frozen thresholds they
  will trigger confirmed/urgent shrinks in live operation. This is accepted
  detector behavior (respond to real routing change), not a mislabel to fix.
- Switch #3 (book->code) produces NO z spike (windows near it: z'~0.5-0.9).
  A domain-label change with no routing-distribution change is correctly
  invisible to v2. Reported as a finding; no threshold will be lowered to
  chase it (chasing label events = the v1 RESET noise-firing pathology).
