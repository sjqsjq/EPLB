# paperpicturetrace — 论文图的 trace/数据集合

每个子文件夹对应一组图,文件名标注具体信息。复现见 NEW_PAPER/figures/FIGURES_GUIDE.md。

- fig1_2_3_4_7_9_13_motivation_crossdomain/: counts*.json(离线 94×128 路由计数,MMLU/prover/book/57b/30b)
- fig5_6_8_14_PD_correlation/: 9 数据集 rt2_*/rank*_fwd_chunk*.npz(identity trace)+ short_pd_*.json(ρ 结果)
- fig12_12b_12c_12d_hotgpu_pinned_volatile/: prover rank0 + README(其余与 fig5 共享)
- fig15_15b_16_17_17b_oeplb_online/: rt_churn_A_adpt rank0 + server_churn_*.log(DIAG/ADW/TIMING)
- figA_B_C_D_E_placement_overhead_eplb/: perds_gain.json + oeplb_churn_results.json + server logs(DIAG/TIMING/EPLB 对比)
- figG_H_I_J_K_L_bound_theory/: counts*.json + bound_curve.py
- figF_F2_M_N_ablation/: oeplb_fixed_adaptive_results.json(α sweep)
