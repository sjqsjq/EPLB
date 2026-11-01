# REV1 重放分类（规则行为检查；不预测吞吐/路由）

免责（REV1 §8）：v1 退出后即离开 S4，下一 QUIET 周期信号为不可观测反事实。本表仅回答：给定该边界已记录的信号与 rev1 语义下该边界之前的 exit_streak，rev1 在该边界会作出什么决定。

| run | h_id | h_rate | h_rate/θ_out | z | event | swaps | pending | streak_in | rev1 决定 | reason 首条 |
|---|---|---|---|---|---|---|---|---|---|---|
| W1_b1_A | 33 | 0.010916 | 0.958 | 0.376 | grow_blocked_headroom | 0 | False | 0 | **STAY_band** |  |
| W1_b1_A | 39 | 0.011489 | 1.008 | 0.456 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W1_b1_A | 51 | 0.012593 | 1.105 | 0.424 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W1_b1_A | 58 | 0.012255 | 1.075 | 0.498 | grow_blocked_headroom | 0 | False | 0 | **STAY_streak1** |  |
| W1_b1_A | 64 | 0.011788 | 1.034 | 0.462 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W1_b2_A | 33 | 0.010916 | 0.958 | 0.376 | grow_blocked_headroom | 0 | False | 0 | **STAY_band** |  |
| W1_b2_A | 39 | 0.011489 | 1.008 | 0.456 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W1_b2_A | 51 | 0.012593 | 1.105 | 0.424 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W1_b2_A | 58 | 0.012255 | 1.075 | 0.498 | grow_blocked_headroom | 0 | False | 0 | **STAY_streak1** |  |
| W1_b2_A | 64 | 0.011788 | 1.034 | 0.462 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b1_A | 9 | 0.011595 | 1.018 | 0.297 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b1_A | 24 | 0.011857 | 1.041 | 0.538 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b1_A | 31 | 0.012829 | 1.126 | 0.874 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b1_A | 38 | 0.011120 | 0.976 | 0.340 | grow_blocked_headroom | 0 | False | 0 | **STAY_band** |  |
| W2_b1_A | 44 | 0.010341 | 0.907 | 0.417 | hold | 0 | False | 0 | **STAY_band** |  |
| W2_b1_A | 54 | 0.010258 | 0.900 | 0.442 | hold | 0 | False | 0 | **STAY_band** |  |
| W2_b1_A | 63 | 0.011660 | 1.023 | 0.509 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b1_A | 71 | 0.010831 | 0.950 | 0.371 | hold | 0 | False | 0 | **STAY_band** |  |
| W2_b1_A | 79 | 0.011120 | 0.976 | 0.447 | hold | 0 | False | 0 | **STAY_band** |  |
| W2_b1_A | 119 | 0.012872 | 1.130 | 0.738 | grow_blocked_headroom | 0 | False | 0 | **STAY_streak1** |  |
| W2_b1_A | 160 | 0.010671 | 0.936 | 0.582 | hold | 0 | False | 0 | **STAY_band** |  |
| W2_b2_A | 33 | 0.010916 | 0.958 | 0.376 | grow_blocked_headroom | 0 | False | 0 | **STAY_band** |  |
| W2_b2_A | 39 | 0.011489 | 1.008 | 0.456 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 51 | 0.012593 | 1.105 | 0.424 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 58 | 0.012255 | 1.075 | 0.498 | grow_blocked_headroom | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 64 | 0.011788 | 1.034 | 0.462 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 73 | 0.042980 | 3.772 | 24.664 | midband_hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 87 | 0.013543 | 1.188 | 0.600 | grow_blocked_headroom | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 106 | 0.012262 | 1.076 | 0.735 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 118 | 0.012839 | 1.127 | 1.067 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 141 | 0.013372 | 1.173 | 0.813 | hold | 0 | False | 0 | **STAY_streak1** |  |
| W2_b2_A | 149 | 0.015339 | 1.346 | 1.185 | hold | 0 | False | 0 | **STAY_streak1** |  |

## 汇总

| rev1 决定 | 次数 |
|---|---|
| P1_RECOVER | 0 |
| P2_TRACK | 0 |
| P3_TRACK | 0 |
| STAY_band | 9 |
| STAY_streak1 | 23 |
| 合计 | 32 |
