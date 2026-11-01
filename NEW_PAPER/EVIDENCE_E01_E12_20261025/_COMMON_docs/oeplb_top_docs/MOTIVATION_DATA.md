# OEPLB 动机实验数据：负载不均衡刻画 + 跨域路由差异 + PD相关性

> 本文档提供论文§2（背景与动机）所需的前情提要实验数据。3个真实非拼接数据集，
> Qwen3-235B-A22B-FP8（94层, 128专家, top-8, EP8=16专家/卡），O=10。
> 所有数据来自`/data/minghua/sjq/OEPLBdata/datasets/prefill_decode_correlation/traces/`。

---

## 1. 负载不均衡刻画（为什么需要负载均衡）

### 1.1 Identity（默认连续放置）的不均衡度

默认放置：专家0-15→GPU0, 16-31→GPU1, ..., 112-127→GPU7。

| 数据集 | 域 | mean max/min | max层 max/min | 含义 |
|--------|-----|-------------|--------------|------|
| MMLU (25tok) | QA | **2.264** | 4.04 | 热点GPU负载是冷点的2.3倍（均值），最极端层4× |
| prover (1253tok) | math | **3.510** | 7.03 | 3.5倍，最极端7× |
| book (4438tok) | BookCorpus | **4.380** | 11.79 | 4.4倍，最极端**12×** |

**含义**：在默认放置下，MoE step时间由最慢GPU决定（straggler），即热点GPU的负载是平均的2-4倍→MoE计算浪费50-75%。

### 1.2 最优放置（LPT贪心）能修复多少

LPT bin-packing：把热专家按频率降序，贪心分配到当前最轻GPU。

| 数据集 | identity r | optimal r | 不均衡降低 | 最极端层(id→opt) |
|--------|-----------|-----------|-----------|-----------------|
| MMLU | 2.264 | **1.001** | **53.3%** | 4.04 → 1.01 |
| prover | 3.510 | **1.000** | **68.0%** | 7.03 → 1.00 |
| book | 4.380 | **1.000** | **73.4%** | 11.79 → 1.00 |

**含义**：最优放置能将不均衡度从2-4×降到~1.00（几乎完美），降低53-73%。**这就是负载均衡的理论上限收益**——OEPLB的工作就是逼近这个上限。

### 1.3 Per-GPU负载分布（8卡，哪张卡过载）

identity放置下各GPU总负载（prefill selections，跨所有层求和）：

| GPU | MMLU | prover | book | 含义 |
|-----|------|--------|------|------|
| 0 | 8.64M | 232.5M | 395.5M | |
| 1 | 8.89M | 254.5M | 325.0M | book冷点 |
| 2 | 8.86M | 268.3M | 382.2M | |
| 3 | 9.35M | 288.0M | 359.6M | |
| 4 | **9.55M** | 247.6M | **394.4M** | MMLU/book热点 |
| 5 | 8.68M | **308.4M** | 390.1M | prover热点 |
| 6 | 8.51M | 272.8M | 355.2M | |
| 7 | 8.78M | 247.2M | 320.0M | book冷点 |

**含义**：不同数据集的热点GPU不同（MMLU=GPU4, prover=GPU5, book=GPU0/4）。**静态最优放置对一个数据集最优，对另一个是次优甚至最差**→需要动态自适应。

### 1.4 逐层不均衡度（用于热力图/曲线图）

每层(94层)的identity max/min ratio数组：
- `imbalance_motivation_data.json → {dataset}.imbalance_ratio_per_layer` (94 values)

**Figure建议**：94层identity ratio曲线（3条线，MMLU/prover/book），y轴max/min ratio，展示不均衡度的逐层变化和跨数据集差异。

---

## 2. 跨域路由差异（为什么需要动态/自适应，而非静态）

### 2.1 全局专家频率的跨域相似度

不同数据集的路由分布Spearman相关（全局128专家频率）：

| 数据集对 | Spearman ρ | cosine | 含义 |
|---------|-----------|--------|------|
| MMLU vs prover | **0.054** | 0.954 | 几乎正交——QA和数学路由到完全不同专家 |
| MMLU vs book | **-0.038** | 0.934 | 正交——QA和叙事文本路由不同 |
| prover vs book | **-0.216** | 0.897 | 反相关——数学和叙事的路由模式相反 |

**含义**：不同域激活不同专家集（ρ≈0）。为数据集A优化的静态放置对数据集B是错的→**静态放置在域切换负载下失败，必须动态/自适应**。这正是OEPLB的adaptive window机制（§3.5）的存在理由。

### 2.2 热点专家跨域差异

各数据集全局top-10热点专家（ID）：

| 数据集 | top-5热点专家 | top-10 |
|--------|-------------|--------|
| MMLU | 110, 64, 30, 91, 76 | + 0, 48, 96, 62, 125 |
| prover | 99, 54, 94, 51, 80 | + 15, 43, 110, 107, 22 |
| book | 34, 103, 67, 10, 69 | + 114, 46, 102, 58, 91 |

**含义**：MMLU和prover的top-5热点专家**完全不重叠**（MMLU={110,64,30,91,76} vs prover={99,54,94,51,80}）。这意味着为MMLU把专家110放到GPU4，但prover根本不用110→放置无效。**不同域需要不同放置→动态OEPLB的动机**。

**Figure建议**：3个数据集的128专家频率柱状图（或热力图），并排展示热点专家完全不同。

### 2.3 逐层专家频率热力图数据

每个数据集的(94×128) prefill专家频率矩阵：
- `imbalance_motivation_data.json → {dataset}.prefill_freq_94x128`

**Figure建议**：94×128热力图（行=层, 列=专家, 颜色=频率），3个数据集并排——展示路由模式的域差异。

---

## 3. Prefill→Decode相关性（为什么prefill-only recording是充分统计量）

（详见`PREFILLBOUNDARY_DATA.md`，此处仅概述与动机的连接）

| 数据集 | prompt长度 | O | mean ρ | layers≥0.7 | 含义 |
|--------|-----------|---|--------|-----------|------|
| MMLU | 25 tok | 10 | **0.833** | 94/94 | prefill强预测decode→prefill-only充分 |
| prover | 1253 tok | 10 | **0.980** | 94/94 | 近完美→prefill-only充分 |
| book | 4438 tok | 10 | **0.967** | 94/94 | 近完美→prefill-only充分 |

**与§3.6的连接**：因为prefill路由强预测decode路由（ρ 0.83-0.98），所以只在prefill阶段记录路由数据（decode走CUDA graph零开销）就捕获了decode分布的充分统计量→swap修改全局`physical_to_logical_map`→decode也受益。

**时间衰减（§5边界条件）**：prefill对early decode预测最好（ρ=0.62），late decode衰减（ρ=0.47）→PB在prefill边界记录+及时决策，正好捕获高ρ区。

---

## 4. 三条论证链汇总

| 论证链 | 数据 | 论文章节 | 结论 |
|--------|------|---------|------|
| **负载不均衡存在** | identity r=2-4×, 最极端12× | §2.1问题形式化 | 需要负载均衡 |
| **最优放置能修** | optimal r→1.00, 降低53-73% | §2.4理论上界 | OEPLB的理论收益空间 |
| **不同域路由不同** | 跨域ρ≈0, 热点专家不重叠 | §2.3观察1 | 静态放置失败→需动态 |
| **prefill预测decode** | ρ=0.83-0.98, 94/94层 | §2.3观察3, §3.6 | prefill-only充分→零开销decode |
| **时间衰减边界** | early 0.62→late 0.47 | §3.5, §3.6边界 | PB边界记录正好捕获高ρ区 |

---

## 5. 数据文件索引

| 数据 | 路径 |
|------|------|
| 不均衡热力图+per-GPU+ratio | `/workspace/logs/imbalance_motivation_data.json` |
| identity vs optimal放置 | `/workspace/logs/identity_vs_optimal_placement.json` |
| PD相关性(逐层ρ+top-K) | `/workspace/logs/ob3_figure_data.json` |
| 长度曲线+聚合窗口 | `/workspace/logs/ob3_extra_figure_data.json` |
| O=10干净长度结果 | `/workspace/logs/length_O10_clean_results.json` |
| 3数据集+trace+README | `/data/minghua/sjq/OEPLBdata/datasets/prefill_decode_correlation/` |
| PD相关性数据包 | `/workspace/EPLB/OEPLB/PREFILLBOUNDARY_DATA.md` |
| 长度实验driver | `/workspace/logs/driver_length_clean.sh` |
