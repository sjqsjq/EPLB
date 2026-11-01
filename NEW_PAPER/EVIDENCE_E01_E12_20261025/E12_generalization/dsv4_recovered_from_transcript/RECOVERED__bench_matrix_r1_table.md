| stream | arm | ok/err | time_s | tps | lat_p50 | lat_p95 | ttft_p50 | ttft_p95 | ttft_p99 | tpot_mean | tpot_p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2 | I | 4082/0 | 96.0 | 42.5 | 5,914.2 | 6,777.3 | 5,914.1 | 6,777.2 | 6,951.1 | - | - |
| e2 | P | 4082/0 | 116.9 | 34.9 | 6,993.2 | 9,987.8 | 6,993.2 | 9,987.8 | 11,206.1 | - | - |
| e3 | I | 8192/0 | 184.0 | 44.5 | 5,738.1 | 6,823.0 | 5,738.0 | 6,822.8 | 7,352.8 | - | - |
| e3 | P | 8192/0 | 215.3 | 38.1 | 6,578.4 | 8,604.1 | 6,578.3 | 8,604.0 | 9,752.5 | - | - |
| e4 | I | 2048/0 | 81.2 | 3,227.1 | 10,249.1 | 11,189.9 | 3,240.6 | 5,840.5 | 6,545.7 | 54.1 | 74.8 |
| e4 | P | 2048/0 | 97.2 | 2,696.6 | 11,807.0 | 16,353.1 | 4,106.1 | 8,736.1 | 11,010.3 | 62.1 | 100.8 |

## P-vs-I deltas (positive = P better for tps/ok; negative = P slower)
| stream | tps_I | tps_P | d_tps | d_tps% | ttft_p95_I | ttft_p95_P | d_ttft_p95% | tpot_mean_I | tpot_mean_P | d_tpot% |
|---|---|---|---|---|---|---|---|---|---|---|
| e2 | 42.5 | 34.9 | -7.6 | -17.9% | 6,777.2 | 9,987.8 | +47.4% | - | - |  |
| e3 | 44.5 | 38.1 | -6.4 | -14.4% | 6,822.8 | 8,604.0 | +26.1% | - | - |  |
| e4 | 3,227.1 | 2,696.6 | -530.5 | -16.4% | 5,840.5 | 8,736.1 | +49.6% | 54.1 | 62.1 | +14.7% |

## P-arm controller artifacts
- e2 (bench_e2_p1): {"mode": "adaptive", "state_final": "TRACK", "S_final": 1, "W_final": 16, "window_count": 28, "swaps_started_total": 25, "true_steps_total": 452, "recorded_steps": 369, "transitions": 0, "seals": 0}
- e3 (bench_p_e3_r1): {"mode": "adaptive", "state_final": "TRACK", "S_final": 1, "W_final": 16, "window_count": 46, "swaps_started_total": 45, "true_steps_total": 736, "recorded_steps": 702, "transitions": 0, "seals": 0}
- e4 (bench_p_e4_r1): {"mode": "adaptive", "state_final": "TRACK", "S_final": 1, "W_final": 16, "window_count": 75, "swaps_started_total": 19, "true_steps_total": 1200, "recorded_steps": 182, "transitions": 0, "seals": 0}
