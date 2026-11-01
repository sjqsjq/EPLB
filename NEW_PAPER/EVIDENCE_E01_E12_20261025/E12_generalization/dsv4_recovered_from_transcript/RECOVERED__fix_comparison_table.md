# DSV4 fix campaign: layer-alignment patch validation (2026-10-22)

Patch: topk.py record hook sets _layer_counter = layer_id - offset (host-side; frozen pb_oeplb byte-identical).
Root cause fixed: 40 records/forward vs 43 slots rotated layer attribution 3 slots/forward (gcd(3,43)=1).

| stream | arm | rounds | tps | wall_s | ttft_p95_ms | tpot_ms | errors | P-vs-I tps |
|---|---|---|---|---|---|---|---|---|
| e2 | I | 2 | 42.6 | 95.7 | 6848.2 | 0.0 | 0 | - |
| e2 | P_broken | 2 | 34.5 | 118.3 | 9562.3 | 0.0 | 0 | -19.1% |
| e2 | P_fix | 2 | 39.8 | 102.6 | 8228.5 | 0.0 | 0 | -6.7% |
| e3 | I | 2 | 44.9 | 182.4 | 6692.8 | 0.0 | 0 | - |
| e3 | P_broken | 2 | 37.2 | 220.6 | 9175.5 | 0.0 | 0 | -17.1% |
| e3 | P_fix | 1 | 38.6 | 212.4 | 10813.1 | 0.0 | 0 | -14.0% |
| e4 | I | 2 | 3215.1 | 81.5 | 5822.3 | 54.5 | 0 | - |
| e4 | P_broken | 2 | 2624.5 | 100.0 | 9661.9 | 63.1 | 0 | -18.4% |
| e4 | P_fix | 1 | 2699.6 | 97.1 | 8732.2 | 61.5 | 0 | -16.0% |

## Controller behavior (P_fix cells)

| cell | windows | swaps_total | z_med | s_change_med |
|---|---|---|---|---|
| bench_fixcheck_p_e2 | 24 | 2 | 0.113 | 0.00082 |
| bench_p_fix_e2_r2 | 39 | 2 | 0.096 | 0.00101 |
| bench_p_fix_e3_r1 | 71 | 13 | 2.893 | 0.02402 |
| bench_p_fix_e4_r1 | 75 | 10 | 1.225 | 0.05863 |

## Broken-arm controller for contrast

| bench_e2_p1 | 28 | 25 | 43.609 | 0.33420 |
| bench_p_e3_r1 | 46 | 45 | 28.755 | 0.21592 |
| bench_p_e4_r1 | 75 | 19 | 11.560 | 0.24851 |

## Function preservation (fixed arm): MARGINAL_tokens_equal all_tokens_same=True worst_pre_divergence_dlogprob=0.29895


Per-round tps: {'e2/I': [42.5, 42.8], 'e2/P_broken': [34.9, 34.1], 'e2/P_fix': [39.2, 40.4], 'e3/I': [44.5, 45.3], 'e3/P_broken': [38.1, 36.3], 'e3/P_fix': [38.6], 'e4/I': [3227.1, 3203.0], 'e4/P_broken': [2696.6, 2552.4], 'e4/P_fix': [2699.6]}
