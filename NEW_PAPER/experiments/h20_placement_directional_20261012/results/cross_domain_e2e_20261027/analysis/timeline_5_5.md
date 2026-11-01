# cross_domain_e2e_20261027 -- sec.5.5 timeline material (auto-generated)

Stream: e3_multidomain_switch.jsonl (sha d55fe2f7...), 4 phases x 2048 req, boundaries idx 2048/4096/6144.
Windows: 32 x 256 req -> boundary windows 8/16/24. v2p1 rows carry no wall clock; mapping is count-based.

## T1 boundary crossing offsets (s from bench start, mean over 4 rounds)
| arm | b1 idx2048 send | b2 idx4096 send | b3 idx6144 send | wall mean s |
|---|---|---|---|---|
| I | 40.4 | 89.5 | 137.2 | 190.6 |
| ES | 41.2 | 89.4 | 137.6 | 192.1 |
| ED | 43.5 | 96.0 | 151.5 | 208.4 |
| MT | 39.3 | 88.5 | 135.6 | 188.2 |
| DF | 40.7 | 88.9 | 135.9 | 188.2 |
| P0 | 45.6 | 91.3 | 138.2 | 200.0 |

## T2 per-phase ttft p95 (s), mean over rounds -- load migration near boundaries
| arm | sharegpt | prover | book | code |
|---|---|---|---|---|
| I | 6.81 | 6.67 | 6.08 | 6.06 |
| ES | 6.50 | 6.59 | 6.78 | 6.81 |
| ED | 6.78 | 13.26 | 6.44 | 7.69 |
| MT | 5.61 | 8.02 | 5.98 | 5.99 |
| DF | 6.64 | 7.59 | 6.18 | 5.93 |
| P0 | 10.34 | 7.41 | 8.79 | 9.94 |

## T3 per-phase req/s (mean over rounds)
| arm | sharegpt | prover | book | code |
|---|---|---|---|---|
| I | 44.8 | 37.1 | 38.1 | 38.4 |
| ES | 43.8 | 37.8 | 37.9 | 37.6 |
| ED | 40.7 | 31.5 | 33.1 | 36.0 |
| MT | 45.8 | 37.4 | 38.7 | 38.9 |
| DF | 43.9 | 37.9 | 38.7 | 39.2 |
| P0 | 40.0 | 40.3 | 39.3 | 33.3 |

## T4 P0 controller activity per boot
| boot | windows | swaps_started | transitions | issued_windows | boundary-adjacent (9/17/25) events |
|---|---|---|---|---|---|
| P0_r1 | 32 | 13 | 0 | [1, 2, 9, 10, 17, 18, 25, 26, 27, 28, 29, 30, 31, 32] | [9, 17, 25] |
| P0_r2 | 32 | 13 | 0 | [1, 3, 9, 10, 17, 18, 25, 26, 27, 28, 29, 30, 31, 32] | [9, 17, 25] |
| P0_r3 | 32 | 13 | 0 | [1, 3, 9, 10, 17, 18, 25, 26, 27, 28, 29, 30, 31, 32] | [9, 17, 25] |
| P0_r4 | 27 | 12 | 4 | [1, 5, 9, 10, 15, 16, 17, 22, 23, 24, 25, 26, 27] | [9, 17, 25] |

## T5 ED true rebalance events (correction: 16 log rows = 2 events x 8 DP ranks)
| boot | ev1 off_s | ev1 nearest bnd (delta_s) | ev2 off_s | ev2 nearest bnd (delta_s) |
|---|---|---|---|---|
| ED_r1 | 95.3 | idx4096 (+0.1) | 199.3 | idx6144 (+50.9) |
| ED_r2 | 97.5 | idx4096 (-0.5) | 205.5 | idx6144 (+48.8) |
| ED_r3 | 95.9 | idx4096 (+0.8) | 196.9 | idx6144 (+50.8) |
| ED_r4 | 95.9 | idx4096 (+0.2) | 203.9 | idx6144 (+49.0) |

## T6 static arms layout-freeze evidence
| boot | banner | injection_evidence (trunc) | runtime rebalance rows |
|---|---|---|---|
| ES_r1 | False | ["[2026-10-27 06:44:28] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| ES_r2 | False | ["[2026-10-27 07:40:53] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| ES_r3 | False | ["[2026-10-27 08:07:29] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| ES_r4 | False | ["[2026-10-27 08:33:53] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| MT_r1 | False | ["[2026-10-27 06:54:54] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| MT_r2 | False | ["[2026-10-27 07:20:14] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| MT_r3 | False | ["[2026-10-27 07:45:59] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| MT_r4 | False | ["[2026-10-27 08:43:53] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| DF_r1 | False | ["[2026-10-27 06:59:50] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| DF_r2 | False | ["[2026-10-27 07:25:34] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| DF_r3 | False | ["[2026-10-27 07:51:08] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| DF_r4 | False | ["[2026-10-27 08:18:14] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| I_r1 | False | ["[2026-10-27 07:10:03] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| I_r2 | False | ["[2026-10-27 07:35:55] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| I_r3 | False | ["[2026-10-27 08:01:53] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |
| I_r4 | False | ["[2026-10-27 08:29:07] server_args=ServerArgs(model_path='/root/models/ms_cache | 0 |

