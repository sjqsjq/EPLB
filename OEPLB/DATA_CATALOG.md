# PB-OEPLB 实验数据完整目录

> 所有数据分门别类，每类写明：**为什么测**、**数据完整路径**、**图表**。
> 你可以直接用这些数据画论文图，或引用已生成的图。

---

# 一、负载不均衡刻画（§2.1 问题形式化）

## 为什么测
证明MoE推理存在严重的专家负载不均衡——热点GPU负载是冷点的2-12倍，
导致straggler、计算资源浪费。这是负载均衡的基本动机。

## 数据路径

### 1.1 逐层不均衡度 + per-GPU负载 + 94×128热力图矩阵
```
/workspace/logs/imbalance_motivation_data.json
```
内容（每个数据集3个，共3个数据集）：
- `prefill_freq_94x128`: (94层 × 128专家) prefill频率矩阵——用于画热力图
- `gpu_loads_94x8`: (94层 × 8GPU) 每层每GPU负载——用于画straggler
- `total_gpu_load_8`: 8张GPU总负载（跨层求和）——用于画per-GPU柱状图
- `imbalance_ratio_per_layer`: 94层各层的max/min ratio——用于画逐层曲线
- `mean_ratio` / `max_ratio` / `min_ratio`: 汇总统计
- `top10_global_experts`: 全局top-10热点专家ID
- `top5_per_layer`: 每层top-5热点专家ID

### 1.2 Identity vs 最优放置对比
```
/workspace/logs/identity_vs_optimal_placement.json
```
内容（每个数据集）：
- `identity_ratio_per_layer`: 94层identity(连续)放置的max/min ratio
- `optimal_ratio_per_layer`: 94层LPT贪心最优放置的max/min ratio
- `identity_mean` / `optimal_mean`: 均值
- `reduction_pct`: 最优放置降低不均衡的百分比

## 数据摘要

| 数据集 | identity mean r | optimal mean r | 降低 | 最极端层(identity→optimal) |
|--------|----------------|---------------|------|---------------------------|
| MMLU (25tok, QA) | 2.264 | 1.001 | 53.3% | 4.04 → 1.01 |
| prover (1253tok, math) | 3.510 | 1.000 | 68.0% | 7.03 → 1.00 |
| book (4438tok, BookCorpus) | 4.380 | 1.000 | 73.4% | 11.79 → 1.00 |

## 图表

![Fig1](figures/fig1_imbalance_ratio_per_layer.png)
**Fig 1**: 94层identity放置的不均衡度曲线（3数据集）。y=max/min ratio。

![Fig2](figures/fig2_identity_vs_optimal.png)
**Fig 2**: Identity vs Optimal放置均值对比。绿色=最优放置降到~1.00。

![Fig3](figures/fig3_per_gpu_load.png)
**Fig 3**: 8张GPU的负载分布。虚线=完美均衡(12.5%)。不同数据集的热点GPU不同。

![Fig7](figures/fig7_expert_heatmap.png)
**Fig 7**: 94×128专家激活频率热力图（log₁₀）。3数据集并排——展示域间路由差异。

---

# 二、跨域路由差异（§2.3 观察1：域内稳定+域间切换）

## 为什么测
证明不同域(QA/数学/叙事)路由到不同专家(spearman≈0)，热点专家完全不重叠。
→ 静态最优放置对一域最优对另一域是错的→必须动态自适应。

## 数据路径
```
/workspace/logs/imbalance_motivation_data.json
```
（同一文件，使用以下字段）：
- `prefill_freq_94x128`: 跨域比较用——对两个数据集的128维全局频率算spearman
- `top10_global_experts`: 各数据集top-10热点专家——看是否重叠

## 数据摘要

### 跨域相似度
| 数据集对 | Spearman ρ | 含义 |
|---------|-----------|------|
| MMLU vs prover | 0.054 | 几乎正交 |
| MMLU vs book | -0.038 | 正交 |
| prover vs book | -0.216 | 反相关 |

### 热点专家跨域不重叠
| 数据集 | top-5热点专家 |
|--------|-------------|
| MMLU | 110, 64, 30, 91, 76 |
| prover | 99, 54, 94, 51, 80 |
| book | 34, 103, 67, 10, 69 |
→ top-5完全不重叠

## 图表

![Fig4](figures/fig4_cross_domain_similarity.png)
**Fig 4**: 跨域路由相似度矩阵(3×3 spearman)。对角线=1.0，非对角≈0→域间正交。

---

# 三、Prefill→Decode相关性（§2.3 观察3 + §3.6 充分性论证）

## 为什么测
证明prefill路由强预测decode路由(ρ≥0.7)，所以prefill-only recording是decode分布的
充分统计量——decode走CUDA graph零开销。这是PB-OEPLB §3.6设计选择的核心依据。
按DataFore(ISCA 2026) Ob3方法学：逐层Spearman ρ，聚合pooling，O=10。

## 数据路径

### 3.1 逐层ρ + top-K overlap（MMLU + sharegpt）
```
/workspace/logs/ob3_figure_data.json
```
内容：
- `mmlu.per_layer_rho`: 94层各层prefill→decode Spearman ρ数组
- `mmlu.mean_rho` = 0.833, `mmlu.layers_ge07` = 94/94
- `mmlu.top5_overlap` = 49%, `top10_overlap` = 58%
- `sharegpt.per_layer_rho`: sharegpt的94层ρ（对比组，ρ=0.689, 44/94层强）

### 3.2 top-K逐层数组 + 长度曲线 + 聚合窗口
```
/workspace/logs/ob3_extra_figure_data.json
```
内容：
- `mmlu_topk_per_layer`: {5:[94值], 10:[94值], 20:[94值], 40:[94值]} 逐层overlap
- `length_curve`: 7个长度点的ρ + layers≥0.7
- `aggregation_window`: W=1~50的ρ + std

### 3.3 O=10干净长度结果（6数据集）
```
/workspace/logs/length_O10_clean_results.json
```
内容（每个数据集）：
- `agg_rho`, `layers_ge07`, `top5/10/20`, `pf_sel_per_layer`, `dc_sel_per_layer`

## 数据摘要

### Ob3复现（MMLU, O=10, 对标DataFore）
| 指标 | 我的结果 | DataFore claim |
|------|---------|---------------|
| mean ρ | **0.833** | ≥0.7 |
| layers ≥0.7 | **94/94** | "most layers" |
| top-5 overlap | 49% | ~60% (Qwen3) |

### 长度依赖（O=10, 真实非拼接）
| 数据集 | prompt长度 | ρ | layers≥0.7 |
|--------|-----------|---|-----------|
| MMLU | 25 tok | 0.833 | 94/94 |
| prover | 1253 tok | 0.980 | 94/94 |
| book | 4438 tok | 0.967 | 94/94 |

## 图表

![Fig5](figures/fig5_prefill_decode_rho_mmlu.png)
**Fig 5**: 94层prefill→decode Spearman ρ柱状图(MMLU, O=10)。绿=强(≥0.7), 橙=中, 红=弱。

![Fig6](figures/fig6_topk_overlap.png)
**Fig 6**: Top-K overlap vs K。蓝=我们, 橙虚线=DataFore。

![Fig8](figures/fig8_length_dependence.png)
**Fig 8**: ρ vs prompt长度(log x)。实线=可靠点, 灰点=噪声(短prompt sparse decode)。

---

# 四、时间衰减（§3.5/§3.6 边界条件）

## 为什么测
证明prefill对early decode预测最好(ρ=0.62)、late decode衰减(ρ=0.47)。
→ PB在prefill边界记录+及时决策，正好捕获高ρ区。这是充分统计量的时间边界。

## 数据路径
```
/workspace/logs/ob3_extra_figure_data.json → aggregation_window
```
（时间衰减的5段数据在PREFILLBOUNDARY_DATA.md §5，具体值）：
| Decode段 | ρ |
|-----------|---|
| 0-20% (early) | 0.616 |
| 20-40% | 0.511 |
| 40-60% | 0.501 |
| 60-80% | 0.481 |
| 80-100% (late) | 0.467 |

---

# 五、聚合窗口收敛（§3.5 sync_window=16的依据）

## 为什么测
证明聚合足够多prefill batch后ρ收敛——单batch ρ=0.57(噪声)，W=16时ρ=0.75(收敛)。
→ sync_window=16是经验最优。这是PB-OEPLB自适应窗口的参数依据。

## 数据路径
```
/workspace/logs/ob3_extra_figure_data.json → aggregation_window
```
| W | ρ | std(ρ) |
|---|----|--------|
| 1 | 0.572 | 0.160 |
| 4 | 0.707 | 0.090 |
| 16 | 0.753 | 0.046 |
| 50 | 0.788 | 0.011 |

---

# 六、Prefill引导放置→MoE加速（DataFore Case-Study-2复现）

## 为什么测
证明基于prefill路由数据的放置确实加速decode MoE计算(+14.5%)。
→ 闭环证明：prefill预测decode(§三) → 基于prefill放置(§六) → decode加速。

## 数据路径
```
/workspace/EPLB/OEPLB/datafore_repro/placement_algo.py
/workspace/EPLB/OEPLB/datafore_repro/REPRODUCTION_REPORT.md
/workspace/EPLB/OEPLB/datafore_repro/placements/{default,remap,dup,best,worst}.json
```

## 数据摘要
| 放置 | MoE speedup | max/min ratio |
|------|-------------|-------------|
| Default (identity) | 1.000 | 2.525 |
| **Remap (prefill-guided)** | **1.145 (+14.5%)** | 1.736 |
| Best (oracle) | 1.469 (+46.9%) | 1.002 |
| Worst (adversarial) | 0.412 (-58.8%) | 81.4 |

---

# 七、3个干净数据集+Trace（所有实验的数据来源）

## 为什么留
所有动机/相关性实验的数据来源。真实非拼接，O=10，清晰命名，含复现文档。

## 路径
```
/data/minghua/sjq/OEPLBdata/datasets/prefill_decode_correlation/
├── mmlu_25tok_QA_O10.jsonl        (14042 req, MMLU, O=10)
├── prover_1253tok_math_O10.jsonl  (2048 req, prover math, O=10)
├── book_4438tok_O10.jsonl         (1000 req, BookCorpus, O=10)
├── README.md                       (复现文档：设置/指标/命令)
└── traces/
    ├── mmlu_25tok_QA_O10/         (8 rank × 5 chunk, rank{0-7}_fwd_chunk{0-4}.npz)
    ├── prover_1253tok_math_O10/   (8 rank × 4 chunk)
    └── book_4438tok_O10/          (8 rank × 4 chunk)
```

每个npz文件含：`is_prefill`(bool数组), `layer_hists`((94,128) int直方图), `forward_ids`

## 图表目录
```
/workspace/EPLB/OEPLB/figures/
├── fig1_imbalance_ratio_per_layer.png  (178KB)
├── fig2_identity_vs_optimal.png        (51KB)
├── fig3_per_gpu_load.png                (38KB)
├── fig4_cross_domain_similarity.png     (48KB)
├── fig5_prefill_decode_rho_mmlu.png     (52KB)
├── fig6_topk_overlap.png               (50KB)
├── fig7_expert_heatmap.png             (199KB)
└── fig8_length_dependence.png           (61KB)
```
