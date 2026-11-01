# Deliverable D — ADW v2 behavior+cost validation run (step 5 §4)

Date: 2026-10-16. Run: 11:59:42 launch → 12:07:04 BEHAVIOR_DONE (boot rc=0, bench rc=0).
Driver: scripts/run_adw_v2_behavior.sh. Dataset: W0 calibration.jsonl (8037 requests, O1 prefill-replay, concurrency 256).
Arm: OEPLB_WINDOW_CONTROLLER=2, α=0.0, D OFF, heuristic_first_improvement planner, knobs = frozen-candidate
values from fit_results_v2.json (fit blind on W0 fixed-W16 fit run only; NOT tuned on this run).
Logs: v2behavior_rank0.jsonl (65 windows), behavior_server.log, v2behavior.json, behavior_analysis.json.
Code: controller.py 2a4dedb2…, window_v2.py 810c54da…, config.py 180fa41e… (see code_change_note_v2_deploy.md).

## 1. Headline

- ok=8037/8037, errors=0, zero RANK DESYNC, zero v2 error fallbacks, zero uncertain events (n_fresh ≥ n_min all windows at saturation).
- Service time 365.34 s vs 363.5 s (fit run, W locked 16) and 362.37 s (grid fixed W16): +0.51% / +0.82%.
  Single run each — behavioral evidence that live W movement (9 changes, 8↔32 full range) adds no measurable
  overhead; NOT a performance claim. Formal comparison is step 6.
- All three known domain boundaries behaved exactly as pre-declared in fit_rules_v2 (accepted consequences):
  switch #1 and #2 detected with urgent shrinks; switch #3 (book→code) produced NO z spike (max z=1.95 < z_hi=19.86)
  and v2 correctly stayed silent — no threshold chasing.

## 2. Evidence chain — every W change (9 total)

| w# | W | event | z | h_w (assign) | reason (from log) |
|----|---|-------|---|--------------|--------------------|
| 2  | 16→8  | confirmed_shrink | 21.9 | 1,209,397 | alert_k≥2 and h_w≥θ_act (run-start ref-settling transient; pre-declared accepted consequence) |
| 24 | 8→16  | confirmed_grow | 0.2 | 539,751 | stable_s≥3 and h_w<θ_H_low |
| 30 | 16→8  | urgent_shrink | 234.0 | 4,041,515 | z≥z_urgent and h_w≥θ_act (boundary 1) |
| 35 | 8→16  | confirmed_grow | 1.3 | 421,659 | stable_s≥3 and h_w<θ_H_low |
| 38 | 16→32 | confirmed_grow | 0.7 | 888,285 | stable_s≥3 and h_w<θ_H_low |
| 43 | 32→16 | urgent_shrink | 204.1 | 6,617,825 | boundary 2, episode shrink 1/3 |
| 44 | 16→8  | urgent_shrink | 217.7 | 3,349,749 | boundary 2, episode shrink 2/3 (cap 3 not reached) |
| 49 | 8→16  | confirmed_grow | 2.1 | 572,816 | stable_s≥3 and h_w<θ_H_low |
| 52 | 16→32 | confirmed_grow | 0.9 | 866,905 | stable_s≥3 and h_w<θ_H_low |

z at boundaries reproduces the fit run (switch #1: 234.0 here vs z'≈232.5 in fit; switch #2: 204–218 vs 206) —
the signal is reproducible across runs, not a fitting artifact.

M2 negative decisions (change credible, benefit absent → NO churn): shrink_rejected_low_headroom at
w20 (z=34.7, h=721,768), w31 (z=55.4, h=595,966), w45 (z=37.6, h=582,786) — all h < θ_act=1,065,054.
w31 is the post-urgent tail of boundary 1: headroom already recovered by the w30 shrink+migration, so the
residual z elevation was correctly refused. This is the designed "high raw signal ≠ time benefit" behavior.

## 3. Detection latency (boundary → first shrink)

| boundary | row | window | first shrink | latency (windows) | z_max near |
|----------|-----|--------|--------------|--------------------|------------|
| #1 (→sharegpt-like mix) | 2075 | 28 | w30 urgent | 2 | 234.0 |
| #2 | 4052 | 42 | w43 urgent | 1 | 217.7 |
| #3 (book→code) | 6068 | 56 | none | — (routing-invisible; correct silence) | 1.95 |

Mapping: cumulative n_fresh fraction × 8037 rows (n_fresh ∝ executed steps at saturation; same
dataset+protocol as the fit run that established the exact row boundaries 2075/4052/6068).

## 4. Dwell distributions

| W | windows | steps | est. time (s)* |
|---|---------|-------|----------------|
| 8  | 32 | 280 | 96.9 |
| 16 | 15 | 232 | 80.3 |
| 32 | 18 | 544 | 188.2 |

*steps-fraction × 365.34 s; assumes uniform step duration under saturation. W=64 never reached
(grow_blocked_headroom 14×: headroom stayed above θ_H_low whenever stable_s≥3 at W=32 tail — M2 working).
Token-weighted = step-weighted here (n_fresh ∝ steps at fixed saturation batch).

## 5. Event census (65 windows)

init 1, hold 16, hold_alert 4, midband_hold 18, confirmed_shrink 1, urgent_shrink 3,
shrink_rejected_low_headroom 3, confirmed_grow 5, grow_blocked_headroom 14.
No: urgent_shrink_capped, shrink_blocked_dwell, grow_blocked_dwell, uncertain_*, forced_*, w_desync_freeze.

## 6. Costs

- v2 decision cost c_window_ms: median 0.61, mean 0.64, p90 0.86, max 1.25 (excl. first window;
  first window 16.78 ms = lazy init, one-time, inside its own window accounting).
  Cumulative v2 decision cost ≈ 40 ms over the whole 365 s run.
- Legacy 94-layer ratio allreduce (prof_allreduce_cum): 1,318.9 ms — kept identically in BOTH arms
  (F12 cost symmetry); not attributable to v2.
- Heartbeat: hash-consistency allreduce every 16 windows (2 checks this run, w16/w32, silent pass).

## 7. Pathology checks — all pass

- Floor ratchet: NONE. Every shrink episode recovered (w2→w24 recovery took 22 windows — slow but by design:
  grow requires 3 consecutive calm windows AND h<θ_H_low, and the volatile region kept headroom high;
  this is M2, not a stuck state). Run ended at W=32, full 8↔32 range exercised both directions.
- Ping-pong (grow immediately after shrink): 0 occurrences; dwell guards never even needed
  (no *_blocked_dwell events — the m_shrink/m_grow confirmation counters filtered first).
- shrink→noise→shrink feedback (M1 target pathology): absent — ref frozen only on uncertain/freeze events
  (none occurred); ref updates gated by fresh-sample normalization (z = s_change/(κ/n_layer)).
- Desync: all 65 windows desync=false on rank0 log; hash checks at w16/w32 passed on all 8 ranks.
- Uncertain path: never triggered at saturation (n_fresh=98.6M/window at W16 ≫ n_min=12.3M). The UNCERTAIN
  machinery is exercised only by unit tests + will engage under low occupancy; documented limitation.

## 8. Mechanism attribution

- M1 (change vs uncertainty): midband_hold 18× (alerts cleared without action), confirmed shrinks only after
  m_shrink=2 consecutive z≥z_hi with fresh-sample-normalized z; no |Δr|-style raw judgment anywhere.
- M2 (benefit space gates action): all grows required h<θ_H_low (verified true for all 5); all shrinks
  required h≥θ_act (verified true for all 4); 3 credible changes REFUSED for low headroom; grow blocked 14×
  for high headroom. Window length tracks recoverable benefit, not raw routing change.
- M3 (cost/hysteresis/bounded response): single authority (all 9 changes from _window_control_v2; 0 legacy
  mutation lines), grid {8,16,32,64} bounded, dwell=2, urgent episode cap 3 (max observed 2), all ranks
  consistent, heartbeat counted (§6), no fallback-equals-fixed claim made.

## 9. Freeze decision

No defect requiring knob change was observed. Every surprising behavior (w2 transient shrink, boundary-3
silence, slow w2 recovery) was pre-declared as an accepted consequence in fit_rules_v2_PREREGISTERED.md
BEFORE this run. Knobs are FROZEN as-is (see FREEZE_record_v2_knobs.md). Per plan §4, any future knob
change requires a new pre-registered rule + re-freeze, and results already collected stand.

## 10. Constraint compliance

One run, no repeats, no selection (八/standing). Fixed start protocol honored (boot rc=0 once).
No labels fed to controller (boundaries used only in THIS offline analysis). Migration walls reported
alongside, never subtracted. Service time reported with fixed caliber including all online overhead.
No push performed.
