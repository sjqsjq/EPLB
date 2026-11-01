# FREEZE RECORD — ADW v2 controller knobs (step 5 complete)

Frozen at: 2026-10-16, after the pre-registered behavior validation run (deliverable D,
results/adw_v2/calib_step5/behavior_report_D.md) showed no defect requiring knob change.

## Frozen values (single authority for all step-6 arms using OEPLB_WINDOW_CONTROLLER=2)

| knob | value | source rule (fit_rules_v2_PREREGISTERED.md) |
|------|-------|----------------------------------------------|
| grid | {8,16,32,64} | design (M3 bounded response) |
| W_init | 16 | calibration workload default (§四: no hardcoded universal claim; fixed arms equally calibrated) |
| kappa | 2806.9 | R5: median(Z_s) stationary rescale |
| z_lo | 3.737 | R6: p90 of rescaled stationary |
| z_hi | 19.864 | R6: p99 of rescaled stationary |
| z_urgent | 115.619 | R7: max(3·z_hi, p50(Z_c signal windows)) |
| theta_h_low | 988725 | R8: p30 of H_s (assign units) |
| theta_h_act | 1065054 | R8: p60 of H_s |
| n_min | 12320768 | R9: 0.25·median(N_s)/2 |
| m_shrink / m_grow | 2 / 3 | design |
| dwell_shrink / dwell_grow | 2 / 2 | design |
| u_max | 4 | design |
| beta (ref EWMA) | 0.5 | design |
| w_uncertain_target | 16 | design |
| urgent_episode_cap | 3 | design |
| hash check interval | every 16 windows | design |

Fit data: fit_results_v2.json (fit run: W0, W locked 16, 8037/8037 ok, 363.5 s).
Gates passed at fit time: separation p50(Z_c')=115.6 > z_hi=19.9 (5.8× margin); stationary tail p90/p50=3.74 ≤ 10.

## Code under freeze (sha256, deployed to /opt/conda/.../pb_oeplb/, backups in legacy_eav1_locked_20261016/)

- controller.py 2a4dedb22d4bcaba203233c0f9a5ae6c211370f524addb7d166669406690878b
- window_v2.py  810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74
- config.py     180fa41eae6e9394186afceb2ceadf9468ebbacda69591fb9b9897a2e6c2680c

## Freeze rules

1. These knobs are the ONLY v2 configuration for step-6 formal comparison. No post-freeze tuning on
   evaluation results (directive: 不根据评估结果反复改数据/参数直到赢).
2. If a step-6 run exposes a genuine defect, the defect is DOCUMENTED with its run; a knob change requires
   a NEW pre-registered rule + re-freeze memo; already-collected results stand and are reported as-is.
3. Fixed baselines are calibrated under the same rules (F-global W=16 from FACT-09 grid; F-local per-condition
   argmin over {8,16,32,64}×α=0 grid run ON that condition, 1 start/cell) — both reported, never only the easier one.
4. Failure handling: failed starts retired, never retried under the same frozen campaign; no best-start selection.
5. No push or external publication without separate user instruction.
