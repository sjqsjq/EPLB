# paperpicturetrace — 论文每张图的数据集映射

每张图 → 其数据文件位置（详见各子文件夹 README + NEW_PAPER/figures/FIGURES_GUIDE.md）。

## 动机/跨域图（fig1_2_3_4_7_9_13_motivation_crossdomain/）
- Fig 0/1/1b/2/2b/3/4/7/9/13 → counts*.json（离线 94×128 路由计数，MMLU/prover/book/57b/30b）

## PD 相关性图（fig5_6_8_14_PD_correlation/）
- Fig 5/6/8/14 → rt2_<dataset>/rank*_fwd_chunk*.npz（9 数据集 identity trace）+ short_pd_9dataset_samesession.json（ρ 结果）
- Fig 14 重画脚本：/tmp/plot_fig14.py

## 热点GPU/原型图（fig12_12b_12c_12d_hotgpu_pinned_volatile/）
- Fig 12/12b/12c/12d → rt2 traces（与 PD 文件夹共享）+ prover rank0 本目录

## OEPLB 在线图（fig15_15b_16_17_17b_oeplb_online/）
- Fig 15/15b/16/17/17b → rt_churn_A_adpt_rank0/ + server_churn_A_adpt__DIAG_ADW_TIMING.log

## 实验/对比图（figA_B_C_D_E_placement_overhead_eplb/）
- Fig A/B/C/D/E → perds_gain.json + oeplb_churn_results.json + server_churn_A_adpt__DIAG_TIMING_overhead.log + server_eplb_MMLU__EPLB_comparison.log

## 死区/增益上界图（figG_H_I_J_K_L_bound_theory/）
- Fig G/H → bound_curve.py（含 CFG/DS/MEAS 表）
- Fig I/J/K/L → counts json + T(r) 扫描日志 + bound_curve.py

## 消融图（figF_F2_M_N_ablation/）★ 本轮补全
- **Fig F2（α-sweep）** → figF2_alpha_sweep_throughput_reconstructed.json（从 server_d3/d4 的 200-OK 时序重建的 α=0/0.5/0.9 × 3 负载 tps）+ figF2_decay_sweep_hotGPU_arrays.json（α 的 per-forward 热点 GPU 数组）
- **Fig N（M 收敛/同 M 聚簇）** → _d31_g_M<W>a<α>.json（8 个，同 M 不同 (W,α)，带 tps）+ _d34_d34_M<a>.json（M-sweep with tps）
- Fig F2/N 重画脚本：/tmp/redraw_ablation.py
