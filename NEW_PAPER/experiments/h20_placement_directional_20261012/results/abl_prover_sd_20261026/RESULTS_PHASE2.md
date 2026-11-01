# RESULTS_PHASE2 — prover同域调度消融 正式阶段

预注册: PHASE2_PREREGISTRATION.md (§5 分析规则先于数据冻结)。主指标 = 每 boot 完整流 tps。
解释护栏: h_rate 为低余量带信号, 不表示还可以加速多少; 机制事实与性能分开陈述。

## 每臂 tps 描述统计 (USABLE boots)

- P: n=8 median=96.55 mean=95.71 min=87.8 max=99.4
- F_1_16_16: n=8 median=95.6 mean=95.17 min=90.5 max=99.4
- F_4_16_64: n=8 median=88.45 mean=89.84 min=86.0 max=95.4

## 预注册对比 (Welch 双侧 + Holm, alpha=0.05)

- P_vs_F_1_16_16: {"t": 0.3257, "df": 12.8783, "p": 0.7499, "mean_x": 95.7125, "mean_y": 95.175, "sd_x": 3.7563, "sd_y": 2.7712, "n_x": 8, "n_y": 8, "p_holm": 0.7499, "significant_alpha05": false}
- P_vs_F_4_16_64: {"t": 3.2723, "df": 13.8765, "p": 0.0056, "mean_x": 95.7125, "mean_y": 89.8375, "sd_x": 3.7563, "sd_y": 3.4172, "n_x": 8, "n_y": 8, "p_holm": 0.0112, "significant_alpha05": true}

## 机制终点汇总 (每臂)

- P: {"boots_usable": 8, "transitions_total": 56, "boots_with_transition": 8, "transition_kinds": {"TRACK->QUIET": 24, "QUIET->RECOVER": 16, "RECOVER->TRACK": 16}, "swaps_total": 16, "det_map_finals": [2], "stable_s_max_range": [2, 2], "forwards_set": [256], "planhash8_all_identical": true}
- F_1_16_16: {"boots_usable": 8, "transitions_total": 0, "boots_with_transition": 0, "transition_kinds": {}, "swaps_total": 16, "det_map_finals": [2], "stable_s_max_range": [2, 2], "forwards_set": [256], "planhash8_all_identical": true}
- F_4_16_64: {"boots_usable": 8, "transitions_total": 0, "boots_with_transition": 0, "transition_kinds": {}, "swaps_total": 16, "det_map_finals": [2], "stable_s_max_range": [7, 7], "forwards_set": [256], "planhash8_all_identical": true}

## 排除 boot 登记

- 无 (全部 USABLE)

## 每 boot 明细见 phase2_summary.json (per_boot)

