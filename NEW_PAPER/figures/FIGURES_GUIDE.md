# NEW_PAPER/figures 生成指南

本目录 41 张图复制自 `OEPLB/figures`、`OEPLB/figures2`、`OEPLB/figure1`。下表标注每张图的**数据来源**与**生成脚本**。脚本若不在 repo(ad-hoc 生成)标"ad-hoc";图的详细内容见 `OEPLB/figures/FIGURE_GUIDE.md`。

## 数据源总览

| 数据源 | 路径 | 用于图 |
|---|---|---|
| 离线路由计数矩阵(94×128 expert 频率) | `OEPLB/repro/counts235b.json`(MMLU/prover/book)、`counts57b.json`/`counts57b_l512.json`/`counts57b_multi.json`/`counts57b_share.json`、`counts30b.json` | Fig 1/1b/2/2b/3/4/7/9/13、Fig G/H/I/J/K/L |
| Routing trace npz(per-forward 直方图,identity 放置) | `/workspace/logs/rt2_<dataset>/rank*_fwd_chunk*.npz`(9 数据集:MMLU/ARC/ARC-E/CSQA/OBQA/GSM8K/prover/HumanEval/CMMLU) | Fig 5/6/8/14、Fig 12/12b/12c/12d |
| OEPLB 在线 trace(swap 决策+热点 GPU) | `/workspace/logs/rt_<run>/`+ `server_churn_*.log`/`server_sp_*.log`(DIAG/ADW/TIMING 行) | Fig 15/15b/16/17/17b、Fig A/B/C/D/E |
| T(r) 扫描(MoE 层时间 vs ratio) | server nsys/profile 日志 + `bound_curve.py` 内 CFG 表(235B:T=167/B=58.78/r_k=1.093) | Fig I/J/K、Fig G/H |
| 实验结果 json | `OEPLB/oeplb_*_results.json`、`short_pd_9dataset_samesession.json`、`perds_gain.json`、`oeplb_churn_results.json` | Fig A-N、Fig 14 |
| 架构图 | 手绘/`figure1/system_architecture.png`(已复制) | 架构图 |

## 逐图数据源 + 脚本

### 现象/motivation(数据:counts json + rt2 trace)
| 图 | 数据源 | 脚本 |
|---|---|---|
| fig0_methodology_placement | counts235b.json(book L62) | ad-hoc |
| fig1_imbalance_ratio_per_layer | counts235b.json(MMLU/prover/book) | ad-hoc(r_avg.py + matplotlib) |
| fig1b_per_forward_ratio | rt2_*/*.npz | ad-hoc |
| fig2_identity_vs_optimal | counts235b.json(book) | ad-hoc(LPT 贪心模拟) |
| fig2b_per_forward_identity_vs_optimal | rt2_*/*.npz | ad-hoc |
| fig3_per_gpu_load | counts235b.json | ad-hoc |
| fig4_cross_domain_similarity | counts235b.json(3 域 Spearman) | ad-hoc |
| fig7_expert_heatmap | counts235b.json(94×128 热力图) | ad-hoc |
| fig9_cross_domain_transfer | counts235b.json(跨域 LPT 迁移) | ad-hoc |
| fig13_multi_granularity | rt2_*/*.npz | ad-hoc |

### PD 相关性(数据:rt2 trace npz;脚本:compute_pd_rho.py)
| 图 | 数据源 | 脚本 |
|---|---|---|
| fig5_prefill_decode_rho_mmlu | rt2_MMLU_25tok_QA/*.npz | `benchmark/analysis/compute_pd_rho.py` |
| fig6_topk_overlap | rt2_*.npz | ad-hoc |
| fig8_length_dependence | rt2_*.npz(ρ vs prompt 长度) | ad-hoc |
| fig14_7dataset_pd_correlation | short_pd_all7.json(旧 7 数据集) | ad-hoc |
| fig14_9dataset_pd_correlation | short_pd_9dataset_samesession.json(+HumanEval/CMMLU) | `/tmp/plot_fig14.py`(本会话生成) |

### 路由原型/热点 GPU(数据:rt2 trace npz)
| 图 | 数据源 | 脚本 |
|---|---|---|
| fig12_domain_switch_timeline | rt2 trace(MMLU→prover→book) | ad-hoc |
| fig12b_identity_vs_lpt_hot_gpu | rt2_*.npz(7 域 identity vs LPT) | ad-hoc |
| fig12c_9dataset_hot_gpu | rt2_*.npz(9 数据集 entropy/switch_rate) | ad-hoc |
| fig12d_pinned_vs_volatile | rt2_prover + rt2_book | ad-hoc |

### OEPLB 在线运行(数据:rt_ 在线 trace + server log)
| 图 | 数据源 | 脚本 |
|---|---|---|
| fig15_oeplb_real_timeline | rt_churn_A_adpt/ + server_churn_A_adpt.log | ad-hoc |
| fig15b_oeplb_entropy_comparison | 同上 | ad-hoc |
| fig16_per_domain_convergence | 同上(逐域 ratio) | ad-hoc |
| fig17_identity_vs_oeplb_ratio | 同上 | ad-hoc |
| fig17b_identity_vs_oeplb_per_domain | 同上(prover 1.166→1.006) | ad-hoc |

### 死区/增益上界(数据:T(r) 扫描 + bound_curve.py)
| 图 | 数据源 | 脚本 |
|---|---|---|
| figI_hinge_curve | T(r) 扫描(7 布局×2 轮,57B+235B) | ad-hoc(铰链拟合 R²=0.998) |
| figJ_marginal_swap | OEPLB DIAG(avg_ratio_before/after) | ad-hoc |
| figK_rk_powerlaw | bound_curve.py CFG 表 + 跨模型盲测 | ad-hoc(幂律 0.00408·EP^1.52) |
| figG_two_ceilings | bound_curve.py(r_place vs r_k) | `OEPLB/repro/bound_curve.py` |
| figH_cross_model_efficiency | counts235b/57b/30b.json + bound_curve.py | `OEPLB/repro/bound_curve.py` |
| figL_cross_model_validation | counts30b.json(30B Δ_max+η≈0) | ad-hoc |

### 实验/对比(数据:结果 json + server log)
| 图 | 数据源 | 脚本 |
|---|---|---|
| figA_placement_spectrum | server_churn/oeplb logs(DIAG) | ad-hoc |
| figB_ratio_convergence | server_churn_A_adpt.log(DIAG 时序) | ad-hoc |
| figC_eplb_vs_oeplb | perds_gain.json + eplb6 results | ad-hoc |
| figD_overhead_breakdown | server_churn_*.log(TIMING begin()) | ad-hoc |
| figE_migration_blocking | server logs(0.37s vs EPLB 1.55s) | ad-hoc |
| figF_decay_factor / figF2_decay_sweep | oeplb_fixed_adaptive_results.json(α sweep) | ad-hoc |
| figM_kv_cache_pressure | EPLB 16 副本显存分析 | ad-hoc |
| figN_M_convergence | driver31 (W,α) sweep | ad-hoc |
| fig10_swap_timeline / fig11_remap_effect | OEPLB 在线 trace | ad-hoc |
| system_architecture | 手绘架构图(五组件+四观察) | figure1/ 本会话生成 |

## 复现要点

1. **离线图(counts)**:`SGLANG_OEPLB_ROUTING_TRACE=1` 在 identity 下录各数据集 → `rank*_fwd_chunk*.npz`;`r_avg.py` 聚合成 `counts*.json`;ad-hoc 脚本绘图。
2. **PD ρ**:`compute_pd_rho.py <rt2_dir> <tag>` 直接出 ρ。
3. **bound 图**:`OEPLB/repro/bound_curve.py` 含 CFG/DS/MEAS 表,直接跑出 Fig G/H。
4. **在线图**:`--enable-pb-oeplb` 跑 crossdomain,从 server log grep DIAG/ADW/TIMING 绘图。
5. **9-数据集 Fig14**:`/tmp/plot_fig14.py` 重新生成 `fig14_9dataset_pd_correlation.png`。
GUIDE_EOF
echo "written: $(wc -l < /workspace/EPLB/NEW_PAPER/figures/FIGURES_GUIDE.md) lines"
echo "=== also check patch-ON experiment progress ==="; cat /workspace/logs/run_patch_ab_on.runlog 2>/dev/null | tail -3; date +%T
## trace/数据归集

所有图的原始 trace 与数据已归集到 `/data/minghua/sjq/paperpicturetrace/`,按图组分 7 个子文件夹,每个含 README 标注图号/数据源/复现路径:
- fig1_2_3_4_7_9_13_motivation_crossdomain/(counts json)
- fig5_6_8_14_PD_correlation/(9 数据集 rt2 trace npz + ρ json)
- fig12_12b_12c_12d_hotgpu_pinned_volatile/(prover rank0 + README,余与上共享)
- fig15_15b_16_17_17b_oeplb_online/(OEPLB 在线 trace + server log DIAG/ADW/TIMING)
- figA_B_C_D_E_placement_overhead_eplb/(结果 json + server log)
- figG_H_I_J_K_L_bound_theory/(counts json + bound_curve.py)
- figF_F2_M_N_ablation/(α sweep 结果 json)

| fig_deepgemm_staircase (Fig DG) | `experiments/microbench_deepgemm/deepgemm_flat_0_256_clean.json` + `deepgemm_flat_dense.json` (0-256 high-rep + 256-1024) | `experiments/microbench_deepgemm/plot_key_figure.py` | §3.1.1 dead-zone operator root cause |
