# pbd campaign descriptive analysis (doubled stream n=5344, n=2 rounds, NON-INFERENTIAL)

## 1. pbd 12-boot throughput (req/s). stage1=prover[0:3344), stage2=book[3344:5344), early[3344:4344), late[4344:5344)
| boot | total | prover | book | b_early | b_late | lat_ms | ttft_ms | dbook_vs_I(round) |
|---|---|---|---|---|---|---|---|---|
| I_r1 | 6.70 | 9.64 | 4.41 | 4.40 | 4.30 | 4751.1 | 4751.0 |  |
| ES_r1 | 6.80 | 10.25 | 4.35 | 4.33 | 4.25 | 4663.8 | 4663.8 | -1.4% |
| ED_r1 | 6.80 | 9.64 | 4.56 | 4.55 | 4.43 | 4666.4 | 4666.3 | +3.2% |
| MT_r1 | 7.10 | 11.08 | 4.39 | 4.37 | 4.28 | 4498.8 | 4498.7 | -0.6% |
| DF_r1 | 7.10 | 11.05 | 4.42 | 4.40 | 4.28 | 4481.8 | 4481.7 | +0.2% |
| P0_r1 | 7.10 | 10.96 | 4.43 | 4.75 | 4.01 | 4480.5 | 4480.4 | +0.4% |
| I_r2 | 6.80 | 9.80 | 4.41 | 4.39 | 4.30 | 4717.4 | 4717.3 |  |
| ES_r2 | 7.00 | 10.54 | 4.40 | 4.40 | 4.28 | 4578.5 | 4578.4 | -0.2% |
| ED_r2 | 7.00 | 10.04 | 4.60 | 4.57 | 4.49 | 4559.7 | 4559.7 | +4.2% |
| MT_r2 | 7.10 | 11.20 | 4.42 | 4.42 | 4.28 | 4462.3 | 4462.2 | +0.1% |
| DF_r2 | 7.20 | 11.25 | 4.49 | 4.48 | 4.36 | 4410.8 | 4410.8 | +1.7% |
| P0_r2 | 7.20 | 11.02 | 4.48 | 4.48 | 4.30 | 4454.5 | 4454.4 | +1.4% |

## 2. round means (2 reps, descriptive)
| arm | prover r1/r2 | book r1/r2 | book mean | b_late r1/r2 | prover mean |
|---|---|---|---|---|---|
| I | 9.64 / 9.80 | 4.41 / 4.41 | 4.41 | 4.30 / 4.30 | 9.72 |
| ES | 10.25 / 10.54 | 4.35 / 4.40 | 4.38 | 4.25 / 4.28 | 10.39 |
| ED | 9.64 / 10.04 | 4.56 / 4.60 | 4.58 | 4.43 / 4.49 | 9.84 |
| MT | 11.08 / 11.20 | 4.39 / 4.42 | 4.40 | 4.28 / 4.28 | 11.14 |
| DF | 11.05 / 11.25 | 4.42 / 4.49 | 4.45 | 4.28 / 4.36 | 11.15 |
| P0 | 10.96 / 11.02 | 4.43 / 4.48 | 4.45 | 4.01 / 4.30 | 10.99 |

## 3. pb (original 3048 stream) same arms, ordering-replication reference
| boot | total | prover | book | b_early | b_late |
|---|---|---|---|---|---|
| I_r1 | 7.00 | 9.70 | 4.37 | 4.24 | 4.28 |
| ES_r1 | 7.30 | 10.56 | 4.38 | 4.36 | 4.18 |
| ED_r1 | 7.20 | 9.90 | 4.59 | 4.58 | 4.38 |
| MT_r1 | 7.50 | 11.12 | 4.39 | 4.31 | 4.26 |
| DF_r1 | 7.60 | 11.27 | 4.50 | 4.46 | 4.31 |
| P0_r1 | 7.30 | 10.97 | 4.29 | 4.41 | 3.97 |
| I_r2 | 6.80 | 9.13 | 4.44 | 4.43 | 4.23 |
| ES_r2 | 7.20 | 10.42 | 4.37 | 4.30 | 4.22 |
| ED_r2 | 7.10 | 9.58 | 4.52 | 4.53 | 4.30 |
| MT_r2 | 7.50 | 11.19 | 4.47 | 4.46 | 4.26 |
| DF_r2 | 7.30 | 11.19 | 4.23 | 3.98 | 4.30 |
| P0_r2 | 7.60 | 10.61 | 4.72 | 4.68 | 4.55 |

## 3b. arm ordering comparison (book-mean and prover-mean, descending)
- pbd book: ED 4.58 > P0 4.45 > DF 4.45 > I 4.41 > MT 4.40 > ES 4.38
- pb  book: ED 4.56 > P0 4.50 > MT 4.43 > I 4.41 > ES 4.38 > DF 4.36
- pbd prover: DF 11.15 > MT 11.14 > P0 10.99 > ES 10.39 > ED 9.84 > I 9.72
- pb  prover: DF 11.23 > MT 11.15 > P0 10.79 > ES 10.49 > ED 9.74 > I 9.42

## 4. P0 boundary + swap timing (pbd vs pb)
- pb_P0_r1: windows=85 boundary_w=27 h_rate pre=0.01057 spike=0.04897 plateau_post=0.01181 z_spike=45.3 z_plateau=0.76 swaps_total=27 swap_windows=27 p2l_v=27 steps=1000 state=TRACK
    events: {"init": 1, "hold": 30, "confirmed_grow": 12, "grow_blocked_headroom": 40, "midband_hold": 2}
- pb_P0_r2: windows=56 boundary_w=30 h_rate pre=0.01032 spike=0.09043 plateau_post=0.01039 z_spike=17.1 z_plateau=0.80 swaps_total=6 swap_windows=6 p2l_v=6 steps=1008 state=TRACK
    events: {"init": 1, "hold": 31, "confirmed_grow": 15, "grow_blocked_headroom": 9}
- pbd_P0_r1: windows=109 boundary_w=48 h_rate pre=0.01022 spike=0.05394 plateau_post=0.01253 z_spike=109.0 z_plateau=1.98 swaps_total=23 swap_windows=23 p2l_v=23 steps=1824 state=TRACK
    events: {"init": 1, "hold": 48, "confirmed_grow": 21, "grow_blocked_headroom": 35, "midband_hold": 4}
- pbd_P0_r2: windows=109 boundary_w=48 h_rate pre=0.01022 spike=0.05394 plateau_post=0.01253 z_spike=109.0 z_plateau=1.98 swaps_total=23 swap_windows=23 p2l_v=23 steps=1824 state=TRACK
    events: {"init": 1, "hold": 48, "confirmed_grow": 21, "grow_blocked_headroom": 35, "midband_hold": 4}

## 5. ED rebalance lines (events x 8 DP ranks expected)
- pb_ED_r1: lines=169 first=2026-10-27 12:32:52 last=2026-10-27 12:40:56
- pb_ED_r2: lines=169 first=2026-10-27 13:17:39 last=2026-10-27 13:26:25
- pbd_ED_r1: lines=297 first=2026-10-27 18:15:05 last=2026-10-27 18:29:02
- pbd_ED_r2: lines=297 first=2026-10-27 19:29:42 last=2026-10-27 19:43:17

## 6. static arms periodic-rebalance scan (expect init-only, no periodic events)
- pbd_DF_r1: rebalance-lines=1 sample=[2026-10-27 18:44:56] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pbd_DF_r2: rebalance-lines=1 sample=[2026-10-27 19:59:22] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pbd_ES_r1: rebalance-lines=1 sample=[2026-10-27 17:59:48] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pbd_ES_r2: rebalance-lines=1 sample=[2026-10-27 20:43:59] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pbd_I_r1: rebalance-lines=1 sample=[2026-10-27 19:14:13] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pbd_I_r2: rebalance-lines=1 sample=[2026-10-27 20:28:15] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pbd_MT_r1: rebalance-lines=1 sample=[2026-10-27 18:30:04] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pbd_MT_r2: rebalance-lines=1 sample=[2026-10-27 19:44:28] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
