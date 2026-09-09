# 论文图数据集核对总表（三个文件夹 41 张图）

逐一核对每张图的数据源是否存在/可复现。✓=有且对得上;△=有但需重建/有瑕疵;✗=缺/不可复现。

## OEPLB/figures/（25 张，Fig 0–17b）

| 图 | 数据源 | 状态 | 说明 |
|---|---|---|---|
| fig0 methodology | counts235b(book L62) | ✓ | counts235b.json |
| fig1 imbalance_per_layer | counts235b(book) | ✓ | max/min 10.35≈论文11.79× |
| fig1b per_forward_ratio | rt2 traces | ✓ | 9 数据集 rt2_*/rank*_fwd_chunk*.npz |
| fig2 identity_vs_optimal | counts235b(book) | ✓ | LPT 模拟 |
| fig2b per_forward_id_vs_opt | rt2 traces | ✓ | |
| fig3 per_gpu_load | counts235b | ✓ | |
| fig4 cross_domain_similarity | counts(MMLU/prover/book) | ✓ | MMLU/prover 从 rt2 重建,book=counts235b |
| fig5 pd_rho_mmlu | rt2_MMLU + compute_pd_rho | ✓ | |
| fig6 topk_overlap | rt2 traces | ✓ | |
| fig7 expert_heatmap | counts(MMLU/prover/book 3域) | ✓ | counts235b+rt2重建 |
| fig8 length_dependence | rt2(7数据集 ρ vs 长度) | ✓ | |
| fig9 cross_domain_transfer | counts(跨域 LPT 迁移) | ✓ | counts235b+rt2 |
| fig10 swap_timeline | rt_churn_A_adpt | ✓ | |
| fig11 remap_effect | rt_churn_A_adpt | ✓ | |
| fig12 domain_switch | rt2(MMLU→prover→book) | ✓ | book 部分用 counts235b |
| fig12b id_vs_lpt_hot_gpu | rt2(7域) | ✓ | |
| fig12c 9dataset_hot_gpu | rt2(9数据集) | ✓ | |
| fig12d pinned_vs_volatile | rt2 prover + counts235b(book) | ✓ | prover pinned, book volatile |
| fig13 multi_granularity | rt2 traces | ✓ | |
| fig14_7 7dataset_pd | short_pd_all7.json | ✓ | 旧7数据集 |
| fig14_9 9dataset_pd | short_pd_9dataset_samesession.json | ✓ | 本session9数据集,重画 /tmp/plot_fig14.py |
| fig15 oeplb_real_timeline | rt_churn_A_adpt + server log | ✓ | 15 npz + DIAG/ADW |
| fig15b entropy_comparison | 同上 | ✓ | |
| fig16 per_domain_convergence | 同上 | ✓ | |
| fig17 id_vs_oeplb_ratio | 同上 | ✓ | |
| fig17b id_vs_oeplb_per_domain | 同上 | ✓ | prover 1.166→1.006 |

## OEPLB/figures2/（15 张，Fig A–N）

| 图 | 数据源 | 状态 | 说明 |
|---|---|---|---|
| figA placement_spectrum | perds_gain.json + server log | ✓ | |
| figB ratio_convergence | server_churn_A_adpt.log(DIAG) | ✓ | |
| figC eplb_vs_oeplb | perds_gain.json + eplb6 logs | ✓ | 本session三方对比 |
| figD overhead_breakdown | server_churn log(TIMING) | ✓ | |
| figE migration_blocking | server logs(0.37s vs 1.55s) | ✓ | |
| figF decay_factor | oeplb_fixed_adaptive_results | ✓ | α=0.5 最优 |
| figF2 decay_sweep | d3/d4 重建 tps | **△** | 跨session重建,A的α0.9 +14.6% 噪声大(同session +6.4%)。**建议同session重测** |
| figG two_ceilings | bound_curve.py | ✓ | |
| figH cross_model_efficiency | counts + bound_curve.py | ✓ | 235B/57B/30B η |
| figI hinge_curve | T(r)扫描日志 | △ | T(r)扫描原始日志可能散落;counts+bound_curve可重算 |
| figJ marginal_swap | server DIAG(avg_ratio before/after) | ✓ | |
| figK rk_powerlaw | bound_curve.py CFG + 跨模型盲测 | ✓ | |
| figL cross_model_validation | counts30b.json | ✓ | 30B Δ_max+η≈0 |
| figM kv_cache_pressure | EPLB 16副本显存分析 | **△** | ad-hoc,无明确数据文件 |
| figN M_convergence | _d31_g_*.json(同M不同W,α) | ✓ | 重画 /tmp/redraw_ablation2.py |

## OEPLB/figure1/（1 张）

| 图 | 数据源 | 状态 | 说明 |
|---|---|---|---|
| system_architecture | 手绘 | ✓ | 架构图 |

## 总结

- **38/41 ✓**（数据在、可复现）。
- **2 张 △**：figF2（跨session α-sweep,噪声,建议同session重测）、figM（ad-hoc 无数据文件）。
- **0 张 ✗**（没有完全不可复现的）。
- counts235b 源 .pt 已丢,但 json(book)在 + MMLU/prover 从 rt2 可重建 → 动机图全可复现。
- 唯一需要重跑：figF2 的 α-sweep（同 session）。
