# pb campaign descriptive analysis (n=2 rounds, NON-INFERENTIAL, descriptive deltas only)

## 1. pb 12-boot throughput (req/s). stage1=prover[0:2048), stage2=book[2048:3048), early[2048:2560), late[2560:3048)
| boot | total | prover | book | b_early | b_late | dbook_vs_I(round) |
|---|---|---|---|---|---|---|
| I_r1 | 7.00 | 9.70 | 4.37 | 4.24 | 4.28 |  |
| ES_r1 | 7.30 | 10.56 | 4.38 | 4.36 | 4.18 | +0.4% |
| ED_r1 | 7.20 | 9.90 | 4.59 | 4.58 | 4.38 | +5.1% |
| MT_r1 | 7.50 | 11.12 | 4.39 | 4.31 | 4.26 | +0.5% |
| DF_r1 | 7.60 | 11.27 | 4.50 | 4.46 | 4.31 | +3.0% |
| P0_r1 | 7.30 | 10.97 | 4.29 | 4.41 | 3.97 | -1.8% |
| I_r2 | 6.80 | 9.13 | 4.44 | 4.43 | 4.23 |  |
| ES_r2 | 7.20 | 10.42 | 4.37 | 4.30 | 4.22 | -1.6% |
| ED_r2 | 7.10 | 9.58 | 4.52 | 4.53 | 4.30 | +1.8% |
| MT_r2 | 7.50 | 11.19 | 4.47 | 4.46 | 4.26 | +0.6% |
| DF_r2 | 7.30 | 11.19 | 4.23 | 3.98 | 4.30 | -4.8% |
| P0_r2 | 7.60 | 10.61 | 4.72 | 4.68 | 4.55 | +6.3% |

## 2. round means (2 reps, descriptive)
| arm | prover r1/r2 | book r1/r2 | book mean | b_late r1/r2 |
|---|---|---|---|---|
| I | 9.70 / 9.13 | 4.37 / 4.44 | 4.41 | 4.28 / 4.23 |
| ES | 10.56 / 10.42 | 4.38 / 4.37 | 4.38 | 4.18 / 4.22 |
| ED | 9.90 / 9.58 | 4.59 / 4.52 | 4.56 | 4.38 / 4.30 |
| MT | 11.12 / 11.19 | 4.39 / 4.47 | 4.43 | 4.26 / 4.26 |
| DF | 11.27 / 11.19 | 4.50 / 4.23 | 4.36 | 4.31 / 4.30 |
| P0 | 10.97 / 10.61 | 4.29 / 4.72 | 4.50 | 3.97 / 4.55 |

## 3. pc (code stage2) same arms for domain contrast
| boot | total | prover | code | c_early | c_late |
|---|---|---|---|---|---|
| DF_r1 | 16.90 | 11.19 | 33.21 | 28.71 | 34.33 |
| DF_r2 | 17.10 | 11.25 | 34.05 | 29.60 | 35.13 |
| DF_r3 | 16.50 | 10.93 | 31.86 | 27.77 | 33.04 |
| ED_r1 | 14.60 | 9.36 | 31.92 | 28.76 | 32.50 |
| ED_r2 | 14.70 | 9.49 | 31.27 | 28.18 | 31.77 |
| ES_r1 | 15.80 | 10.28 | 32.51 | 27.82 | 33.69 |
| ES_r2 | 16.00 | 10.55 | 31.78 | 27.57 | 32.86 |
| I_r1 | 15.30 | 9.78 | 33.27 | 29.15 | 34.24 |
| I_r2 | 15.50 | 9.88 | 34.25 | 29.82 | 35.27 |
| MT_r1 | 16.80 | 11.15 | 32.27 | 28.29 | 33.19 |
| MT_r2 | 17.10 | 11.23 | 34.35 | 30.21 | 35.24 |
| MT_r3 | 16.90 | 11.23 | 32.51 | 29.28 | 33.07 |
| P0_r1 | 16.00 | 10.91 | 28.76 | 28.21 | 28.51 |
| P0_r2 | 16.40 | 11.07 | 30.56 | 25.26 | 32.19 |

## 4. P0 boundary + swap timing
- pb_P0_r1: windows=85 boundary_w=27 h_rate pre=0.01057 spike=0.04897 plateau_post=0.01181 z_spike=45.3 z_plateau=0.76 swaps_total=27 swap_windows=27 p2l_v=27 steps=1000 state=TRACK
    events: {"init": 1, "hold": 30, "confirmed_grow": 12, "grow_blocked_headroom": 40, "midband_hold": 2}
- pb_P0_r2: windows=56 boundary_w=30 h_rate pre=0.01032 spike=0.09043 plateau_post=0.01039 z_spike=17.1 z_plateau=0.80 swaps_total=6 swap_windows=6 p2l_v=6 steps=1008 state=TRACK
    events: {"init": 1, "hold": 31, "confirmed_grow": 15, "grow_blocked_headroom": 9}
- pc_P0_r1: windows=39 boundary_w=30 h_rate pre=0.01032 spike=0.07530 plateau_post=0.01752 z_spike=15.5 z_plateau=3.59 swaps_total=11 swap_windows=11 p2l_v=11 steps=704 state=TRACK
    events: {"init": 1, "hold": 19, "confirmed_grow": 8, "grow_blocked_headroom": 11}
- pc_P0_r2: windows=39 boundary_w=30 h_rate pre=0.01032 spike=0.07424 plateau_post=0.01748 z_spike=12.6 z_plateau=3.96 swaps_total=11 swap_windows=11 p2l_v=11 steps=704 state=TRACK
    events: {"init": 1, "hold": 19, "confirmed_grow": 8, "grow_blocked_headroom": 11}
- pc_P0_r3: windows=25 boundary_w=None h_rate pre=0.02138 spike=-1.00000 plateau_post=0.01014 z_spike=-1.0 z_plateau=0.11 swaps_total=2 swap_windows=2 p2l_v=2 steps=504 state=RECOVER
    events: {"init": 1, "hold": 15, "confirmed_grow": 5, "grow_blocked_headroom": 4}

## 5. ED rebalance lines (events x 8 DP ranks expected)
- pb_ED_r1: lines=169 first=2026-10-27 12:32:52 last=2026-10-27 12:40:56
- pb_ED_r2: lines=169 first=2026-10-27 13:17:39 last=2026-10-27 13:26:25
- pc_ED_r1: lines=121 first=2026-10-27 10:46:23 last=2026-10-27 10:52:13
- pc_ED_r2: lines=121 first=2026-10-27 11:18:09 last=2026-10-27 11:24:13

## 6. static arms periodic-rebalance scan (expect init-only, no periodic events)
- pb_DF_r1: rebalance-lines=1 sample=[2026-10-27 12:50:54] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pb_DF_r2: rebalance-lines=1 sample=[2026-10-27 13:36:38] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pb_ES_r1: rebalance-lines=1 sample=[2026-10-27 12:23:53] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pb_ES_r2: rebalance-lines=1 sample=[2026-10-27 14:03:35] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pb_I_r1: rebalance-lines=1 sample=[2026-10-27 13:08:36] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pb_I_r2: rebalance-lines=1 sample=[2026-10-27 13:54:09] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pb_MT_r1: rebalance-lines=1 sample=[2026-10-27 12:41:56] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
- pb_MT_r2: rebalance-lines=1 sample=[2026-10-27 13:27:18] server_args=ServerArgs(model_path='/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8', tokenizer_path='/root/models/ms_ca
