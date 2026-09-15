# 论文实验、图、数据集与 trace 位置索引

本文档索引论文（`paper.md`）中每一张图、每个数据集、每条实验 trace 的物理位置，并在每项标注与基础配置的偏差。配合 `figures/FIGURES_GUIDE.md`（逐图数据源+脚本）与 `figures/FIGURE_DATA_VERIFICATION.md`（可复现性核验）使用。

## §1 基础配置（所有实验共享，除非该项标注偏差）

| 项 | 配置 |
|---|---|
| 硬件 | 8× NVIDIA H20（96GB HBM/卡），NVLink 互联 |
| 模型 | Qwen3-235B-A22B-FP8（94 MoE层, 128路由专家, top-8, shared expert） |
| 并行 | TP=DP=EP=8（每卡16专家），`--enable-dp-attention` |
| 通信/GEMM | DeepEP all-to-all（`--moe-a2a-backend deepep`）+ DeepGEMM FP8（`--moe-runner-backend deep_gemm`） |
| 精度 | bfloat16（权重FP8量化 `--quantization fp8`） |
| 软件栈 | SGLang 0.5.6.post2 + PB-OEPLB patch（6文件1949行）、DeepEP v1.2.1、DeepGEMM、torch 2.9.1+cu128 |
| 公共flag | `--disable-radix-cache --watchdog-timeout 600 --trust-remote-code` |
| 基线放置 | identity（`init_expert_location=trivial`） |

**两类口径**：
- **服务级/在线**（§5.2–§5.8, Fig 15/17b等）：`--cuda-graph-max-bs 128`（OEPLB兼容cuda-graph）或 fair-compare `--disable-cuda-graph`，`--mem-fraction-static 0.8`，conc=32/256，真实请求流（O=10/32/128）。
- **纯prefill microbench**（§5.3.1, §3.1.1 Fig DG）：`--disable-cuda-graph --mem-fraction-static 0.78`，O=1纯prefill，256并发，3次中位。

## §2 图索引（每图：内容 → 数据/trace 位置 → 脚本 → 偏差）

### 现象/motivation（数据：离线 counts json + rt2 trace npz）
| 图 | 内容 | 数据/trace | 脚本 | 偏差 |
|---|---|---|---|---|
| Fig 1 | 逐层max/min负载比 | `OEPLB/repro/counts235b.json`（MMLU/prover/book） | ad-hoc(r_avg.py) | — |
| Fig 1b | 逐forward ratio | `/workspace/logs/rt2_<ds>/rank*_fwd_chunk*.npz` | ad-hoc | — |
| Fig 2/2b | identity vs LPT最优 | counts235b.json(book) | ad-hoc(LPT贪心) | — |
| Fig 3 | per-GPU load | counts235b.json | ad-hoc | — |
| Fig 4 | 跨域Spearman矩阵 | counts235b.json(3域) | ad-hoc | — |
| Fig 7 | 94×128专家热力图 | counts235b.json | ad-hoc | — |
| Fig 9 | 跨域LPT迁移失败 | counts235b.json | ad-hoc | ratio 3.67>identity 3.51 |
| Fig 13 | 多粒度ratio | rt2_*.npz | ad-hoc | — |

### PD相关性（数据：rt2 trace npz）
| Fig 5 | MMLU逐层ρ | rt2_MMLU_25tok_QA/*.npz | `benchmark/analysis/compute_pd_rho.py` | — |
| Fig 6 | topk重叠 | rt2_*.npz | ad-hoc | — |
| Fig 8 | ρ vs prompt长度 | rt2_*.npz | ad-hoc | — |
| Fig 14 | 9数据集PD相关性 | `short_pd_9dataset_samesession.json` | `/tmp/plot_fig14.py` | +HumanEval/CMMLU |

### 热点GPU/原型（rt2 trace）
| Fig 12 | 域切换时间线 | rt2 trace(MMLU→prover→book) | ad-hoc | — |
| Fig 12b/c/d | pinned vs volatile | rt2_*.npz | ad-hoc | — |

### OEPLB在线（rt_ 在线trace + server log）
| Fig 15/15b/16/17/17b | swap决策/收敛/逐域 | `/workspace/logs/rt_<run>/` + `server_churn_*.log`(DIAG/ADW/TIMING) | ad-hoc | 在线trace |

### 死区/增益上界（T(r)扫描 + bound_curve.py）
| Fig I | T(r)铰链 | nsys/profile日志 + bound_curve.py CFG表(235B T=167/B=58.78/r_k=1.093) | ad-hoc | 7布局×2轮 |
| Fig J | 边际swap | OEPLB DIAG | ad-hoc | — |
| Fig K | r_k幂律 | bound_curve.py + 跨模型盲测 | ad-hoc | 0.00408·EP^1.52 |
| Fig G/H/L | 两ceiling/跨模型/30B | counts235b/57b/30b.json + bound_curve.py | `OEPLB/repro/bound_curve.py` | 3模型 |

### 实验对比（结果json + server log）
| Fig A/B | 放置谱/收敛 | server_churn_*.log(DIAG) | ad-hoc | — |
| Fig C | EPLB vs OEPLB | `perds_gain.json` + eplb6 results | ad-hoc | — |
| Fig D/E | 开销/阻塞 | server logs(TIMING) | ad-hoc | 0.37s vs 1.55s |
| Fig F/F2 | α sweep | `oeplb_fixed_adaptive_results.json` | ad-hoc | — |
| Fig M/N | KV压力/M收敛 | driver31 (W,α) sweep | ad-hoc | — |
| Fig DG | DeepGEMM T(M) staircase | `experiments/microbench_deepgemm/deepgemm_flat_0_256_clean.json`+`deepgemm_flat_dense.json` | `plot_key_figure.py` | **纯GPU-event microbench，非服务级；trace `/data/minghua/sjq/OEPLBdata/experiment_logs/microbench_deepgemm_20260914/`** |


### §5.3.1 DataForest基线5方对比（prover同分布 + freq6跨域）
| 表 | 数据集 | trace位置 | 偏差 |
|---|---|---|---|
| prover同分布5方 | `datasets/single_domain/prover_256tok_out1.jsonl`(256tok) | `experiment_logs/baseline_comparison_20260914/`（`datafore_prover_placement.json`+`rt_prover/`+bench脚本+`results.json`） | **纯prefill microbench**：`--disable-cuda-graph --mem 0.78`, O=1, 256并发, 3次中位 |
| freq6跨域5方 | 自拼6段book↔prover（`book_4438tok`+`prover_2048tok`, 6×100=600req, O=10, conc=32） | 同上（`freq6_bench.py`+`launch_oeplb_f6.sh`） | 同上；PB-OEPLB需warmup收敛(run2稳态5.1) |

## §3 数据集索引

| 数据集 | 路径 | 用于 | 特征 |
|---|---|---|---|
| prover_256tok_out1 | `OEPLBdata/datasets/single_domain/` | §5.3.1同分布表 | pinned, 10×per-layer不均衡, 256tok |
| prefill_heavy_universal | `OEPLBdata/datasets/multi_domain/` | §5.3.1跨域表 | 4域(sharegpt/prover/book/code)×4000, 507tok, O=1 |
| multidomain_v2_out1 | `OEPLBdata/datasets/multi_domain/` | (早期跨域探索) | 4段prover/book/中文/prover |
| 9域单域集 | `OEPLBdata/datasets/prefill_decode_correlation/` + rt2 trace | Fig 5/8/14, §3.4 | MMLU/ARC/CSQA/OBQA/GSM8K/prover/HumanEval/CMMLU/book |
| crossdomain_freq6 | `OEPLBdata/datasets/...`（bench脚本生成流） | §5.2主结果 | 6段book↔prover, 4438tok, conc=32 |

## §4 trace 归档总览（`/data/minghua/sjq/`）

| 归档目录 | 内容 | 对应图/表 |
|---|---|---|
| `OEPLBdata/experiment_logs/microbench_deepgemm_20260913/` | DeepGEMM T(M) bench脚本+json+图+README | Fig DG, §3.1 |
| `OEPLBdata/experiment_logs/baseline_comparison_20260914/` | DataForest/EPLB/PB-OEPLB基线对比：placement JSON+rt_prover/+bench脚本+launch脚本+results.json | §5.3.1两表 |
| `OEPLBdata/datasets/` | 所有benchmark数据集 | 全文 |
| `paperpicturetrace/` | 旧图(§2-§5既有图)的trace，7子文件夹+README | Fig 1-17b, A-N |
| `OEPLBdata/nsys_traces/` | 3×baseline+3×OEPLB的nsys profiling | 死区T(r)分析 |

## §5 复现要点（摘要，详见 FIGURES_GUIDE.md）
1. 路由录制：`SGLANG_OEPLB_ROUTING_TRACE=1 SGLANG_OEPLB_ROUTING_TRACE_DIR=<dir>`（identity下）→ `rank*_fwd_chunk*.npz`（layer_hists=logical_count）。
2. DataForest-Remap冻结：`--init-expert-location <logical_count.json>`（调rebalance_experts冻结，无--enable-eplb）。
3. PB-OEPLB：`--enable-pb-oeplb --pb-oeplb-adaptive-window --pb-oeplb-adaptive-decay`（+`--pb-oeplb-min-prefill-tokens 32 --pb-oeplb-sync-window 8` 用于短prompt/多域）。
4. EPLB动态：`--enable-eplb --ep-num-redundant-experts 16 --deepep-mode normal`；EPLB静态：+`--init-expert-location <lc.json>`。
5. bound图：`OEPLB/repro/bound_curve.py`（含CFG/DS/MEAS表）。
