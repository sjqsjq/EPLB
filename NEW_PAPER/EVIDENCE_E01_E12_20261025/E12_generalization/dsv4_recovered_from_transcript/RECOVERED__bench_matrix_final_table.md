## DSV4 I-vs-P (median of available rounds; r1+r2)
| stream | arm | rounds | tps | total_time_s | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | tpot_mean_ms | tpot_p95_ms | ok/err |
|---|---|---|---|---|---|---|---|---|---|---|
| e2 | I | 2 | 42.6 | 95.7 | 5,823.3 | 6,848.2 | 7,074.4 | - | - | 4082/0/4082/0 |
| e2 | P | 2 | 34.5 | 118.3 | 7,190.6 | 9,562.3 | 10,407.5 | - | - | 4082/0/4082/0 |
| e3 | I | 2 | 44.9 | 182.4 | 5,660.5 | 6,692.8 | 7,151.1 | - | - | 8192/0/8192/0 |
| e3 | P | 2 | 37.2 | 220.6 | 6,649.2 | 9,175.5 | 10,561.8 | - | - | 8192/0/8192/0 |
| e4 | I | 2 | 3,215.1 | 81.5 | 3,229.6 | 5,822.3 | 6,455.5 | 54.5 | 75.1 | 2048/0/2048/0 |
| e4 | P | 2 | 2,624.5 | 100.0 | 4,123.9 | 9,661.9 | 13,094.8 | 63.1 | 98.7 | 2048/0/2048/0 |

## P-vs-I median deltas (%)
| stream | tps | total_time | ttft_p50 | ttft_p95 | ttft_p99 | tpot_mean |
|---|---|---|---|---|---|---|
| e2 | -19.1% | +23.6% | +23.5% | +39.6% | +47.1% | - |
| e3 | -17.1% | +20.9% | +17.5% | +37.1% | +47.7% | - |
| e4 | -18.4% | +22.6% | +27.7% | +65.9% | +102.8% | +15.8% |

## Per-round raw (for variance)
| cell | round | T_s | tps | ttft_p95 | tpot_mean |
|---|---|---|---|---|---|
| e2-I | r1 | 96.0 | 42.5 | 6,777.2 | - |
| e2-I | r2 | 95.4 | 42.8 | 6,919.1 | - |
| e2-P | r1 | 116.9 | 34.9 | 9,987.8 | - |
| e2-P | r2 | 119.7 | 34.1 | 9,136.7 | - |
| e3-I | r1 | 184.0 | 44.5 | 6,822.8 | - |
| e3-I | r2 | 180.8 | 45.3 | 6,562.7 | - |
| e3-P | r1 | 215.3 | 38.1 | 8,604.0 | - |
| e3-P | r2 | 225.8 | 36.3 | 9,747.1 | - |
| e4-I | r1 | 81.2 | 3,227.1 | 5,840.5 | 54.1 |
| e4-I | r2 | 81.8 | 3,203.0 | 5,804.0 | 54.8 |
| e4-P | r1 | 97.2 | 2,696.6 | 8,736.1 | 62.1 |
| e4-P | r2 | 102.7 | 2,552.4 | 10,587.8 | 64.2 |

## P-arm controller (sp_run_end per cell)
- e2 r1: state=TRACK W=16 windows=28 swaps=25 transitions=0
- e2 r2: state=TRACK W=16 windows=24 swaps=23 transitions=0
- e3 r1: state=TRACK W=16 windows=46 swaps=45 transitions=0
- e3 r2: state=TRACK W=16 windows=46 swaps=45 transitions=0
- e4 r1: state=TRACK W=16 windows=75 swaps=19 transitions=0
- e4 r2: state=TRACK W=16 windows=75 swaps=19 transitions=0
