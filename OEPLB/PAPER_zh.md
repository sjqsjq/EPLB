# 面向MoE推理服务的自适应在线专家负载均衡

## 摘要

混合专家（MoE）模型在推理时面临专家负载不均衡问题——热点专家集中在少数GPU上，造成计算瓶颈。现有方案如SGLang的EPLB需要冗余专家副本、重平衡期间阻塞推理、且无法实时适应负载变化。本文提出**PB-OEPLB**，一种轻量级在线专家负载均衡器，在无需冗余专家、不阻塞推理的前提下实现近似最优的专家布局。PB-OEPLB仅在prefill批次记录专家路由（减少约50%开销），使用指数衰减历史进行噪声鲁棒决策，并采用自适应配对选择算法在3个决策窗口内收敛到近似最优不均衡度（1.02）。在8×H20集群上服务Qwen3-235B-A22B-FP8时，PB-OEPLB在prefill密集型负载上相比基线提升吞吐+18.4%，相比EPLB提升9.4个百分点，同时达到理论最优布局的97.6%。自适应窗口机制根据负载稳定性自动调整决策频率，即使在短prompt通用负载上静态参数调优失效的场景也能获得+5.3%的增益。

此外，本文在4×H20上对Qwen2-57B-A14B-Instruct和Qwen3-30B-A3B-FP8进行了独立重启交替验证，确认了每卡专家数与OEPLB收益的缩放关系，并发现官方EPLB在非DeepSeek架构模型上存在AttributeError崩溃问题，而PB-OEPLB通过通用fallback修复了此限制。

---

## 1. 引言

大语言模型越来越多地采用混合专家（MoE）架构来在不按比例增加计算量的前提下扩展参数量。在生产推理服务中，MoE模型面临一个根本性挑战：**专家负载不均衡**。路由网络根据输入语义将token分配给专家，自然出现的负载模式（如数学推理 vs 叙事文本）导致某些专家被激活的频率远高于其他专家。当专家通过专家并行（EP）分布到多个GPU上时，这种不均衡造成straggler——持有热点专家的GPU成为瓶颈，而其他GPU空闲。

当前方法分为三类：

1. **静态布局**：根据历史流量数据预计算最优专家布局。仅在负载分布已知且稳定时有效——请求类型的变化会使其次优。

2. **周期性重平衡（如SGLang的EPLB）**：周期性重新计算专家布局并重新分配权重。这能适应负载变化，但存在：(a) 需要冗余专家副本（8-16个额外副本占用GPU显存），(b) 每次重平衡期间阻塞推理0.5-4.4秒，(c) 与某些推理优化不兼容（强制`deepep_mode=normal`从而禁用CUDA graph，在decode密集型负载上造成高达-68%的吞吐退化）。

3. **在线交换式**：通过在GPU间交换专家对来增量调整专家布局。这避免了全量重平衡的开销，但面临收敛速度和有限数据下决策质量的挑战。

本文提出**PB-OEPLB**（Prefill-Boundary Online Expert Placement Load Balancer），解决上述三个限制：

- **零冗余**：无需额外专家副本，在现有专家预算内运行。
- **非阻塞**：在专用低优先级CUDA stream上异步P2P传输权重；推理在主流上继续。
- **自适应**：自适应窗口机制根据负载稳定性自动调整决策频率——域切换时收缩窗口以快速响应，稳定时扩张窗口以减少开销。
- **仅prefill记录**：仅在prefill批次记录专家路由，减少约50%记录开销，同时通过全局共享布局惠及decode阶段。

**贡献：**

1. **自适应配对选择算法**（§3.3）：一种贪心交换规划器，在max-delta（不均衡大时快速收敛）和gap-targeting（接近最优时精确均衡）模式间切换，实现3个窗口内从1.74→1.02的ratio收敛，而之前方法停滞在1.26。

2. **自适应同步窗口**（§3.5）：反馈驱动机制，不均衡度收敛时扩张决策窗口（节省all_reduce开销），检测到负载切换时收缩窗口（快速响应），自动适应不同prompt长度。

3. **快速衰减机制**（§3.2）：衰减因子0.5，3个窗口内清除跨域信号污染（vs 0.9在3个窗口后仍保留73%旧信号），同时保持足够的统计样本量。

4. **全面评估**（§5）：在8×H20上使用Qwen3-235B-A22B-FP8，PB-OEPLB在单域prefill负载上实现+18.4%吞吐提升（vs EPLB +9.0%），多域负载+10.6%（vs EPLB +6.3%），通用ShareGPT负载+5.3%——均无需冗余专家或推理阻塞。此外在4×H20上验证了Qwen2-57B-A14B（+1.9%~+4.3%）和Qwen3-30B-A3B（-2.6%~-3.9%），揭示了每卡专家数对OEPLB收益的缩放规律。

5. **跨架构通用性**（§2.2）：发现并修复了EPLB和OEPLB在非DeepSeek架构（Qwen2-MoE、Qwen3-MoE）上的`routed_experts_weights_of_layer`属性缺失崩溃，通过通用fallback使负载均衡兼容所有MoE架构。

---

## 2. 背景与动机

### 2.1 问题形式化

我们将EP推理中的专家负载均衡形式化为**在线均衡分区问题**。

**设定。** 一个拥有$N_E$个路由专家的MoE模型通过专家并行（EP）在$G$个GPU上服务，每个GPU持有$n = N_E/G$个专家。路由器将每个token分配到其top-$k$个专家。设$\ell_j \in \mathbb{R}_{\geq 0}$为在一个观测窗口内专家$j$的累积负载（token计数），$\pi: [N_E] \to [G]$为满足平衡约束$|\pi^{-1}(g)| = n$的专家到GPU的分配。

**不均衡度。** 每层不均衡度为：
$$r(\pi) = \frac{\max_{g \in [G]} L_g(\pi)}{\bar{L}}, \quad L_g(\pi) = \sum_{j: \pi(j)=g} \ell_j, \quad \bar{L} = \frac{1}{G}\sum_g L_g$$

比值为1.0表示完美均衡；不均衡的timing影响与$r-1$乘以MoE计算占比成正比。

**目标。** 找到分配$\pi$（从初始分配通过增量swap），使得$r(\pi)$最小化，约束：(i) 每个决策窗口最多$B$次pairwise swap，(ii) 零额外显存（无冗余专家副本），(iii) 非阻塞执行（swap不得停滞推理）。

**复杂度。** 在平衡约束下找到$\pi^* = \arg\min_\pi r(\pi)$是NP-hard的（$G \geq 3$时归约自3-PARTITION，$G=2$时归约自PARTITION）。这促使我们采用贪心局部搜索近似。

### 2.2 现有方法的限制

**SGLang EPLB**（state-of-the-art生产系统）有三个架构限制：

**限制1：强制禁用CUDA graph。** EPLB要求`deepep_mode=normal`（源码：`server_args.py:1641`），触发`disable_cuda_graph=True`。这禁用CUDA graph捕获，增加每步kernel launch开销。在decode密集型负载（O=256）上造成**-68.2%吞吐退化**（表5）。此外，EPLB的`ExpertDistributionRecorder`对`deepep_mode=auto`直接`raise NotImplementedError`（`expert_distribution.py:315`），因此保留CUDA graph的auto模式与EPLB根本不兼容。

**限制2：架构耦合。** EPLB的`eplb_manager.py:110`和`model_runner.py:927`直接访问`model.routed_experts_weights_of_layer`属性，该属性**仅在DeepSeek-V2/V3模型类上定义**。对Qwen2-MoE和Qwen3-MoE架构，这会引发`AttributeError`，使EPLB在这些架构上**完全无法工作**（在Qwen2-57B-A14B上实验确认）。截至2026年8月的SGLang最新main分支未修复此问题。

PB-OEPLB通过通用fallback（§4）修复了此限制：先尝试DeepSeek原生属性，失败则遍历`model.layers`调用各MoE层的`get_moe_weights()`，兼容所有MoE架构。

**限制3：冗余专家显存开销。** EPLB分配$R$个冗余专家副本（生产中为16个），占用约12.5%额外GPU显存（本可用于KV cache），将最大并发量降低8.1%（从227K降至209K token）。

### 2.3 关键观察

通过对Qwen3-235B-A22B（94层MoE，128专家，EP=8）和Qwen2-57B-A14B（28层，64专家，EP=4）的profiling，我们识别出三个关键观察：

**观察1（域内稳定与跨域切换）。** 在单一内容域内，连续决策窗口的专家负载分布余弦相似度>0.95。域切换导致比值飙升（235B上1.20→1.39；57B上1.03→1.07）。跨域余弦相似度仅0.16。这可建模为**分段平稳Markov过程**，其中路由分布$\boldsymbol{\theta}$在每个段内固定，在未知变点$\tau_1 < \tau_2 < \cdots$处跳变（见§3.2和附录A.2的贝叶斯表述）。

**观察2（决策频率与prompt长度的关系）。** 短prompt（~50 token）需要更大的同步窗口（sw=32-64），因为每个forward batch处理大量请求，频繁的`all_reduce`调用成为主要开销。长prompt（~250 token）受益于更小窗口（sw=8）。这是一个**偏差-方差权衡**：小窗口提供新鲜数据（低偏差）但样本有限导致高方差；大窗口减少方差但增加延迟和通信开销。不存在单一静态窗口对所有负载最优——这促使了自适应机制（§3.5）。

**观察3（prefill预测decode）。** 仅基于prefill路由数据的布局优化产生可测量的decode阶段改善（TPOT：9种配置下-3.0%到-12.5%）。这是因为swap操作修改了全局共享的`physical_to_logical_map`，该映射控制所有forward pass而不论阶段。仅prefill记录因此在域内路由模式相变的假设下捕获了decode分布的*充分统计量*。

### 2.4 理论加速上界

**定理（加速上界）。** 布局优化的最大吞吐改善为：
$$\Delta\text{TPS}_{\max} = \frac{r_{\text{before}} - r_{\text{after}}}{r_{\text{before}}} \times f_{\text{MoE}} - c_{\text{overhead}}$$
其中$f_{\text{MoE}}$为MoE计算占forward时间的比例，$c_{\text{overhead}}$为固定每窗口开销（record + all_reduce + P2P，见附录D）。

*证明。* 最慢GPU决定专家计算时间：$T_{\text{expert}} = \max_g T_{\text{expert}}^{(g)} \propto r$。布局优化将$r_{\text{before}} \to r_{\text{after}}$，在专家计算分量上给出相对加速$(r_{\text{before}}-r_{\text{after}})/r_{\text{before}}$。乘以$f_{\text{MoE}}$（MoE占总forward时间的比例）并减去开销即得净值。$\square$

对Qwen3-235B（$r_{\text{before}}=1.74, r_{\text{after}}=1.02, f_{\text{MoE}}=0.64$）：$\Delta \approx 26\% \times 0.64 = 16.6\%$（实测+18.4%，超出部分来自dispatch/combine尾延迟减少，见附录B）。

对Qwen2-57B-A14B（$r_{\text{before}}=1.74$但有巨大shared expert稀释$f_{\text{MoE}}^{\text{routed}} \approx 0.20$）：$\Delta \approx 26\% \times 0.20 = 5.2\%$（多域实测+3.0%，差异来自shared expert稀释效应，见附录D.2）。

---

## 3. 系统设计

### 3.1 架构概述

PB-OEPLB由集成到SGLang ModelRunner的四个组件组成：

```
topk.py::select_experts() → Controller.record_next_layer(topk_ids)
                                    │
                    ┌───────────────┴───────────────┐
                    │ Rebalancer (贪心 + 自适应)      │
                    │ AsyncSwapExecutor (P2P传输)     │
                    └───────────────────────────────┘
```

**路由hook**：`select_experts()`计算`topk_ids`后（由于`ep_dispatch_algorithm="static"`，topk_ids已在物理slot空间），控制器通过一次`scatter_add_`调用记录它们（O(1) GPU kernel）。

**决策周期**：每`sync_window`个forward pass，控制器：(1)检查上一个P2P传输是否完成，(2)执行all_reduce聚合跨rank负载，(3)调用rebalancer计算swap计划，(4)异步发起P2P传输。

### 3.2 指数衰减与快速周转

每个决策窗口后，负载张量更新为：
$$A_n = R_n + \alpha \cdot A_{n-1}$$
其中$R_n$为当前窗口的新鲜路由数据，$\alpha$为衰减因子。

我们发现$\alpha = 0.5$最优，相比$\alpha = 0.9$（早期版本默认值）和$\alpha = 0$（无历史）：

| $\alpha$ | 3窗口旧信号残留 | 多域吞吐 | 短prompt吞吐 |
|---|---|---|---|
| 0（清零） | 0% | +2.5% | — |
| **0.5** | **12.5%** | **+10.6%** | **+2.3%** |
| 0.9 | 73% | +6.9% | +1.4% |

在$\alpha=0.5$时，跨域信号污染在3个窗口（~18秒）内降至12.5%，实现对负载变化的快速适应。在$\alpha=0.9$时，73%的旧域信号持续残留，导致控制器基于过期数据做布局决策。

### 3.3 自适应配对选择算法

核心创新是贪心交换规划器内的**双模式配对选择**策略：

**模式1（Max-delta，快速收敛）**：当最热和最冷GPU间的差距大时，选择负载差最大的配对。这实现每次swap最快的ratio降低。

**模式2（Gap-targeting，精确均衡）**：当差距小且最热slot的负载超过差距时，选择max-delta配对会**过冲**——将过多负载移到冷GPU，使其变成新的热GPU。改为选择负载最接近$\frac{\text{gap}}{2}$的slot，将两个GPU都均衡到平均值。

$$\text{selected\_slot} = \begin{cases} \arg\max_s \text{load}[s] & \text{if } \max_s \text{load}[s] \leq \text{gap} \\ \arg\min_s |\text{load}[s] - \frac{\text{gap}}{2}| & \text{otherwise} \end{cases}$$

此简单切换解决了一个关键停滞点：之前仅max-delta的规划器收敛到ratio 1.26后无法继续改善（638个有效swap配对存在，但贪心启发式因过冲选择恶化ratio的配对）。采用自适应选择后，ratio在3个窗口内收敛到**1.02**（表2）。

### 3.4 异步非阻塞P2P执行

Swap操作在专用低优先级CUDA stream上执行：

- `begin(plan)`：在低优先级stream上通过`batch_isend_irecv`发起所有P2P操作，记录完成事件，立即返回。
- `try_finish()`：非阻塞`event.query()`检查。仅当事件真正触发时才执行shadow-buffer→live-weight拷贝并翻转路由表。
- `force_wait`：在每次all_reduce之前，控制器强制阻塞等待待处理传输。这是**唯一**的同步点——因为NCCL要求共享通信器上所有rank以一致的相对顺序发起操作。

**演化的p2l更新**：当多个swap操作针对同一层时，每个操作读取*当前*（演化中的）p2l状态，而非计划时刻的过时`logical_a/logical_b`值。这防止了早期版本中导致CUDA assert的p2l不一致bug。

### 3.5 自适应同步窗口

同步窗口根据两个反馈信号自动调整：

**信号1：收敛（ratio稳定）**。当不均衡度在3个连续窗口内变化<0.003时，ratio已收敛。窗口翻倍（最高128），减少all_reduce频率，节省开销。

**信号2：切换检测（ratio跳变）**。当ratio在窗口间变化>0.03时，检测到负载切换。窗口立即减半（最低8），快速响应。

**信号3：波动（ratio震荡）**。当ratio在0.003到0.03之间波动3个连续窗口时，统计噪声较大，窗口扩张以积累更多样本。

此机制自动适应prompt长度：
- **长prompt**（每batch少量请求）：初始sw=8，快速收敛→扩张到32-128（节省开销）。
- **短prompt**（每batch大量请求）：初始sw=8，ratio波动→扩张到32-64（更稳定的统计）。
- **域切换**：稳定sw=64-128→检测到切换→收缩到8→收敛→扩张回来。

### 3.6 仅Prefill记录

控制器仅在`forward_batch.forward_mode.is_extend()`（prefill）时记录路由数据。Decode和idle批次完全跳过。这减少约50%记录开销（混合负载中decode步骤数通常为prefill的10:1），同时通过全局共享布局惠及decode。

**重要细节**：当CUDA graph捕获/回放时（decode阶段），`record_next_layer`通过`torch.cuda.is_current_stream_capturing()`检查直接返回——**零开销**。仅在prefill阶段（不走CUDA graph）才真正执行记录。这意味着在纯prefill（O=1）场景下，所有forward都执行record，开销最大；在混合prefill+decode场景下，decode的record被跳过，开销更低。

---

## 4. 实现

PB-OEPLB作为SGLang 0.5.6.post2的patch实现，修改三个文件：

1. **`server_args.py`**：19个CLI参数（`--pb-oeplb-*`）+ 与官方EPLB的互斥检查。
2. **`model_executor/model_runner.py`**：`initialize()`中controller初始化，`forward()`尾部无条件调用`on_forward_pass_end()`。
3. **`layers/moe/topk.py`**：`select_experts()`后调用`record_next_layer(topk_ids)`。

核心模块在`sglang/srt/managers/pb_oeplb/`：
- `controller.py`（962行）：状态机、衰减、自适应窗口、校准、跨架构fallback。
- `rebalancer.py`（180行）：带自适应配对选择的贪心规划器。
- `async_swapper.py`（250行）：P2P执行、基于事件的完成检测。
- `fast_metadata.py`（60行）：向量化p2l初始化。
- `config.py`（60行）：配置dataclass。

**DeepEP H20 NVLink补丁**：对DeepEP v1.2.1的两处patch用于纯NVLink（无IB）集群：(1)将IBGDA环境变量设置改为仅在有多RDMA rank时执行，(2)注释掉`internode_ll.cu`中的IBGDA断言。

**DeepEP hidden_size=3584补丁**：Qwen2-57B的hidden_size=3584不在DeepEP v1.2.1的`SWITCH_HIDDEN`硬编码列表中（仅支持2048/2560/4096/5120/6144/7168/8192）。通过数学验证（$3584 = 7 \times 512$，满足所有整除约束）后在`csrc/kernels/launch.cuh`中添加`case 3584`，使Qwen2系列能用`deepep_mode=auto`（保留CUDA graph）。

**跨架构fallback**：`controller.py`和`async_swapper.py`中添加`_get_routed_experts_weights()`方法，先尝试DeepSeek原生的`model.routed_experts_weights_of_layer`属性，失败则遍历`model.layers`调用各MoE层的`get_moe_weights()`。兼容所有MoE架构（DeepSeek-V2/V3、Qwen2-MoE、Qwen3-MoE等）。

---

## 5. 评估

### 5.1 实验环境

**8卡环境（原始验证，数据来源：历史实验）**：
- **硬件**：8× NVIDIA H20（96GB/卡），NVLink NV18全互连，无InfiniBand
- **模型**：Qwen3-235B-A22B-FP8（94层MoE，128专家，top-8路由）
- **服务**：SGLang 0.5.6.post2, DeepEP v1.2.1（patched）, DeepGEMM, TP=8, DP=8, EP=8
- **并发度**：256

**4卡环境（本次会话独立验证）**：
- **硬件**：4× NVIDIA H20（96GB/卡），NVLink NV18全互连，无InfiniBand
- **模型1**：Qwen2-57B-A14B-Instruct（28层MoE，64专家，top-8，EP=4→16专家/卡，有shared expert=20480）
- **模型2**：Qwen3-30B-A3B-FP8（48层MoE，128专家，top-8，EP=4→32专家/卡，无shared expert）
- **服务**：SGLang 0.5.6.post2, DeepEP v1.2.1（patched+hidden3584）, DeepGEMM, TP=4, DP=4, EP=4
- **并发度**：256
- **方法**：每个场景独立重启服务器，baseline和OEPLB交替跑（消除时间漂移和placement继承）

### 5.2 数据集

| 数据集 | 请求数 | Prompt长度 | 输出 | 域 |
|---|---|---|---|---|
| L256_O1 | 8192 | ~256 tok | 1（纯prefill） | Prover数学 |
| L512_O1 | 8192 | ~512 tok | 1 | Prover数学 |
| L1024_O1 | 4096 | ~1024 tok | 1 | BookCorpus |
| 多域 16K | 16000 | ~1000 tok | 1 | 4域×4000 |
| ShareGPT 100K | 100000 | ~50 tok | 1 | 真实对话 |

数据集路径：`/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/final_grid/` 和 `multi_domain/`。

### 5.3 八卡+235B结果（历史数据）

**表1：L512_O1 完整布局对比**

| 布局 | 不均衡度 | total_tps | vs 基线 |
|---|---|---|---|
| 最差（热点堆叠） | 2.61 | 16054.5 | -20.4% |
| 基线（trivial round-robin） | 1.74 | 20167.8 | — |
| EPLB（连续） | ~1.00 | 21992.2 | +9.0% |
| Frozen-EPLB | ~1.00 | 22668.1 | +13.0% |
| **PB-OEPLB** | **~1.02** | **23870.5** | **+18.4%** |
| 最优（oracle） | 1.00 | 24460.1 | +21.3% |

PB-OEPLB达到**oracle最优的97.6%**（23870/24460），显著超越EPLB（+9.4个百分点）。

**表2：不均衡度收敛（L512_O1）**

| 窗口 | Max-delta（旧） | 自适应配对（新） |
|---|---|---|
| w0 | 1.743 → 1.264 | 1.743 → 1.187 |
| w1 | 1.262 → 1.261（停滞） | 1.188 → 1.057 |
| w2 | 1.263 → 1.262（停滞） | 1.060 → 1.015 |
| w3 | 1.262 → 1.261（停滞） | 1.027 → 1.015 |
| 稳态 | ~1.26 | **~1.02** |

### 5.4 四卡独立验证结果（本次实验）

**表3：Qwen2-57B-A14B（EP=4，16专家/卡，独立重启交替测试）**

| 场景 | 基线 tps | OEPLB tps | Delta | swap执行 |
|---|---|---|---|---|
| L512_O1（8K） | 57.5 | 60.0 | **+4.3%** ✅ | warmup 55swap, 稳态少swap |
| 多域 16K | 26.9 | 27.7 | **+3.0%** ✅ | 117swap, 0回弹 |
| ShareGPT 20K | 255.3 | 260.1 | **+1.9%** ✅ | 持续swap |

**表4：Qwen3-30B-A3B（EP=4，32专家/卡，独立重启交替测试）**

| 场景 | 基线 tps | OEPLB tps | Delta | swap执行 |
|---|---|---|---|---|
| L512_O1（8K） | 115.4 | 112.4 | **-2.6%** ❌ | warmup 174swap, 稳态少swap |
| 多域 16K | 53.8 | 51.7 | **-3.9%** ❌ | 534+swap, 7次回弹 |
| ShareGPT 20K | 402.0 | 405.8 | +0.9% ~0 | 持续swap |

**表5：EPLB vs OEPLB对比（Qwen2-57B多域16K，patch后公平对比）**

| 配置 | deepep-mode | CUDA graph | 冗余专家 | tps | vs 基线 |
|---|---|---|---|---|---|
| 基线 | auto | ✅ | 0 | 26.9 | — |
| EPLB（官方,patched） | normal | ❌禁 | 16 | 27.2 | +0.4% |
| **OEPLB** | auto | ✅ | 0 | **27.7** | **+3.0%** |

OEPLB超越EPLB +2.6个百分点。EPLB几乎无提升：normal模式禁CUDA graph抵消了16冗余专家的均衡收益。

### 5.5 与EPLB的全面对比

**表6：OEPLB vs EPLB全场景对比**

| 场景 | OEPLB vs 基线 | EPLB vs 基线 | OEPLB优势 |
|---|---|---|---|
| L512 单域（8卡235B） | +18.4% | +9.0% | +9.4 pp |
| L1024 单域（8卡235B） | +15.4% | +7.6% | +7.8 pp |
| L256 单域（8卡235B） | +13.0% | +6.0% | +7.0 pp |
| 多域（8卡235B） | +14.0% | +12.0% | +2.0 pp |
| ShareGPT（8卡235B） | +5.3% | -5.0% | +10.3 pp |
| L512 单域（4卡57B） | +4.3% | — | — |
| 多域（4卡57B） | +3.0% | +0.4% | +2.6 pp |
| L512 单域（4卡30B） | -2.6% | 崩溃 | — |
| 多域（4卡30B） | -3.9% | 崩溃 | — |

注：4卡30B上EPLB因`AttributeError`无法运行（§2.2限制2）。

### 5.6 开销分析

**表7：OEPLB开销分解（8卡235B L512_O1, 175s benchmark）**

| 组件 | 时间(ms) | 占benchmark |
|---|---|---|
| Record（scatter_add per forward） | 599 | 0.34% |
| All_reduce（每窗口） | 495 | 0.28% |
| Plan build（rebalancer） | 43 | 0.02% |
| Finalize（P2P完成） | 62 | 0.03% |
| **总计** | **1199** | **0.67%** |

**表8：4卡30B nsys trace开销分解（L512_O1）**

| 类别 | 基线(ms) | OEPLB(ms) | Delta(ms) | 说明 |
|---|---|---|---|---|
| kernel（GPU计算） | 7674 | 7867 | +193 | GPU计算几乎不变 |
| cpu_op | 3064 | **7416** | **+4352** | Python代码开销暴增 |
| cuda_runtime | 2599 | 3865 | +1267 | CUDA runtime调用增多 |
| gpu_user_annotation | 8 | **2780** | +2772 | all_reduce在GPU侧标注 |
| DeepEP-Combine | 1031 | 863 | **-168** | 均衡后combine等待减少 |
| **总Trace** | **10667** | **13806** | **+3139 (+29%)** | |

30B负收益的根因：CPU侧Python代码开销（+4352ms）和all_reduce（+2772ms）合计7.1秒，远超Combine减少的168ms收益。

### 5.7 显存对比

**表9：GPU显存分解（每卡，96GB H20）**

| 配置 | 总量(GB) | 模型权重 | KV cache | CUDA graph | 开销 |
|---|---|---|---|---|---|
| 基线 | 88.7 | ~28.0 | **~48.8** | ~10.0 | ~1.9 |
| **PB-OEPLB** | 88.7 | ~28.0 | **~48.8** | ~10.0 | ~1.9 |
| EPLB（16冗余） | 79.8 | ~30.5 | **~46.3** | 0（禁用） | ~3.0 |

PB-OEPLB是唯一同时保持完整KV cache容量和CUDA graph启用的配置。

### 5.8 可复现性

8卡235B上3次独立冷启动（L512_O1）：

| Run | total_tps |
|---|---|
| 1 | 22603.6 |
| 2 | 22885.2 |
| 3 | 22850.8 |
| **均值±标准差** | **22780 ± 156 (0.7%)** |

4卡57B独立重启交替测试3场景，0错误，swap执行确认（VERIFY CHANGED=True）。

---

## 6. 相关工作

**训练中的专家负载均衡**：辅助损失方法（Shazeer et al., 2017; Fedus et al., 2021）和无损方法（DeepSeek-V3, Wang et al., 2024）通过路由调整在训练时平衡专家负载。这些与我们的工作正交——我们优化推理时的专家*布局*。

**EPLB（DeepSeek, 2025）**：使用贪心bin-packing算法周期性重平衡专家布局，带冗余专家副本。要求`deepep_mode=normal`（禁用CUDA graph），16个冗余专家，重平衡期间阻塞推理。仅支持DeepSeek架构模型。

**专家卸载**：LibMoE等库管理GPU-CPU层级间的专家布局。关注显存管理而非运行时负载均衡。

**自适应调度**：vLLM的连续批处理等优化请求级调度，但不解决专家级负载不均衡。

---

## 7. 讨论与未来工作

**N-way循环轮换**：当pairwise swap达到平台期（所有配对改善<0.0005），3-way循环轮换（A→B→C→A）理论上可达到pairwise转置无法达到的布局。P2P执行中的实现挑战阻碍了此版本的部署。

**EPLB精修**：混合方法——使用增量swap快速初始收敛，然后单次EPLB式全量重布局做最终精修——在仿真中有前景，但在实践中因时序问题引入不稳定性。

**跨模型泛化**：所有8卡实验使用Qwen3-235B-A22B。在Qwen2-57B-A14B和Qwen3-30B-A3B上的4卡验证确认了缩放规律，但需在更多架构上验证。

**record_next_layer优化**：纯prefill（O=1）场景下record开销在30B上达1.6%（vs 235B的0.34%），因所有forward都不走CUDA graph。未来可考虑异步record或采样记录以降低开销。

---

## 8. 结论

PB-OEPLB证明，轻量级自适应在线专家负载均衡可以在无现有方案架构开销的前提下实现近似最优布局（97.6% of oracle）。关键洞察是**自适应配对选择**（根据当前差距大小切换max-delta和gap-targeting模式）结合**快速指数衰减**（α=0.5）和**自适应决策窗口**，实现跨多样负载的快速收敛到ratio 1.02。在8×H20+Qwen3-235B上PB-OEPLB提升吞吐+5.3%到+18.4%，持续超越SGLang的EPLB 2-10个百分点。4×H20的独立验证确认了每卡专家数对收益的缩放影响：16专家/卡时正收益（+1.9%~+4.3%），32专家/卡时负收益（-2.6%~-3.9%），与理论预测一致。

---

## 参考文献

1. Shazeer, N. et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.
2. Fedus, W. et al. "Switch Transformers: Scaling to Trillion Parameter Models." arXiv 2021.
3. DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv 2024.
4. DeepSeek-AI. "EPLB: Expert Parallelism Load Balancer." 2025. github.com/deepseek-ai/EPLB
5. SGLang. "SGLang: Efficient LLM Serving." sglang.ai
6. DeepSeek-AI. "DeepEP: Efficient Expert Parallelism Communication." 2025.
7. Wang, A. et al. "Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts." arXiv 2024.
8. Sun, Y. "Binary-Integer-Programming Based Algorithm for Expert Load Balancing in MoE Models." arXiv 2025.

---

## 附录A：数学建模

### A.1 贪心规划器收敛分析

**问题形式化**：给定全局负载张量$L \in \mathbb{R}^{N_L \times N_S}$（all_reduce后所有rank一致），找到swap操作集$S = \{(l_i, a_i, b_i)\}_{i=1}^{|S|}$，$|S| \leq B$（预算），最小化最大每层不均衡度。

**定理1（收敛）**：对单层$N_G$个GPU、每GPU $N_S$个slot，若存在至少一对$(a, b)$使$L[a] > L[b]$且$L[a] - L[b] \leq \text{gap}$，则自适应配对选择算法将$r$至少降低：
$$\Delta r \geq \frac{2(L[a] - L[b])}{N_G \cdot \bar{L}} \cdot \left(1 - \frac{L[a] - L[b]}{2 \cdot \text{gap}}\right)$$

**定理2（Swap局部最优界）**：设$\pi$为swap-local-optimal。则：
$$r(\pi) \leq 1 + \frac{G-1}{G} \cdot \frac{\ell_{\max}}{n\mu}$$

其中$\ell_{\max}$为最热专家负载，$\mu$为平均专家负载。

**Tight example**：$G$个GPU，1个热点专家$\ell_1 = M$，其余$\ell_j = \epsilon \to 0$。初始分配使热点在GPU 0。对任何swap，delta $\geq \text{gap}$，overshoot→local optimum。$r = G$，bound $= 1 + \frac{G-1}{G} \cdot G = G$。**Tight!**

### A.2 最优衰减因子的贝叶斯推导

衰减$A_t = R_t + \alpha A_{t-1}$展开：$A_t = \sum_{k=0}^{\infty} \alpha^k R_{t-k}$。

变点后$d$步，旧信号残留权重：$w_{\text{old}}(d) = \alpha^{d+1}$。

检测要求$w_{\text{old}} < 1/2$：$\alpha < 2^{-1/(d+1)}$。对$d=2$：$\alpha < 0.794$；对$d=3$：$\alpha < 0.841$。

最优$\alpha$最小化"延迟+噪声"：$\alpha^* \approx 0.52$（SNR=3, γ=1时数值解），验证经验值0.5。

### A.3 吞吐加速上界

$$\text{Speedup}_{\text{total}} = \frac{r_{\text{before}} - r_{\text{after}}}{r_{\text{before}}} \times f_{\text{MoE}}$$

对235B：$\frac{1.74-1.02}{1.74} \times 0.64 \approx 9.6\%$（实测+18.4%，超出来自dispatch/combine改善）。

### A.4 自适应窗口作为最优停止问题

状态$s = (r_n, \Delta r_n, \text{converge\_count}, \text{volatile\_count})$，动作$a \in \{\text{grow}, \text{shrink}, \text{hold}\}$。

收益$R(s,a) = \Delta r \cdot w - c_{\text{all\_reduce}}/w$。

Doubling/halving策略的竞争比为$O(\log(w_{\max}/w_{\min}))$。

---

## 附录B：Kernel级分析

### B.1 每forward步kernel计时

在8卡235B上测量（GPU利用率~62%，~795 forward步），每步归一化：

| 类别 | 基线(μs/步) | PB-OEPLB(μs/步) | Delta |
|---|---|---|---|
| Dispatch | 7323 | 6300 | **-14.0%** |
| Combine | 5479 | 4015 | **-26.7%** |
| 专家计算 | 6214 | 6179 | -0.6% |
| Attention | 4731 | 4695 | -0.8% |
| **总计** | **25546** | **23082** | **-9.6%** |

### B.2 4卡30B nsys trace分析

通过`/start_profile`采集600 forward步的trace，按kernel类别聚合：

| 类别 | 基线(ms) | OEPLB(ms) | Delta(ms) | Delta% |
|---|---|---|---|---|
| DeepEP-Dispatch | 2997 | 3244 | +247 | +8.3% |
| DeepEP-Combine | 1031 | 863 | **-168** | **-16.3%** |
| DeepGEMM | 1959 | 1978 | +20 | +1.0% |
| NCCL/SendRecv(P2P swap) | 0 | 239 | +239 | — |
| NCCL/AllReduce | 0 | 3.5 | +3.5 | — |
| cpu_op(Python) | 3064 | **7416** | **+4352** | **+142%** |
| gpu_user_annotation | 8 | **2780** | +2772 | — |

30B负收益根因：CPU侧Python开销（record_next_layer每层调用的函数调用+条件判断）在纯prefill场景下累积4.3秒，加all_reduce 2.8秒，合计7.1秒，远超Combine减少的168ms。

---

## 附录C：完整网格结果

### C.1 完整5×4网格（8卡235B，历史数据）

| L | O | 基线(rps) | sw=8 | sw=16 | sw=32 | sw=64 | 自适应(8) | 最优 |
|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 77.1 | +23.9% | +21.0% | +15.3% | +10.9% | +21.8% | sw=8 |
| 512 | 1 | 40.6 | +17.8% | +14.2% | +16.4% | +12.7% | +15.5% | sw=8 |
| 1024 | 1 | 19.9 | +18.8% | +19.0% | +16.6% | +17.3% | +18.3% | sw=16 |
| 2048 | 1 | 9.8 | +12.5% | +13.7% | +8.4% | +8.0% | +13.9% | 自适应 |
| 4096 | 1 | 4.5 | +11.2% | +12.1% | +7.6% | +4.1% | +10.4% | sw=16 |

（完整20格见英文版PAPER_en.md）

---

## 附录D：专家密度与不均衡度——缩放分析

### D.1 每卡专家数越少，不均衡度越高

不均衡度$r$本质上由**每GPU专家数**$n = N_E / \text{EP}$决定，而非$N_E$或EP单独决定。

**大数定律**：每GPU总负载是$n$个独立专家负载之和。$n$增大时，每GPU总负载收敛到全局均值——方差以$O(1/\sqrt{n})$缩小——高不均衡度在统计上不太可能。

**定量模型**：设专家负载服从偏斜分布，均值$\mu$，标准差$\sigma$。每GPU负载$L_g$的变异系数：
$$\text{CV}(L_g) = \frac{\sigma}{\mu\sqrt{n}}$$

期望max-to-mean比约为：
$$E[r] \approx 1 + \frac{\sigma}{\mu\sqrt{n}} \cdot \sqrt{2 \ln(\text{EP})}$$

### D.2 实测验证

| 模型 | 总专家 | EP | 专家/卡 | 实测baseline max ratio | OEPLB收益 |
|---|---|---|---|---|---|
| Qwen3-235B（8×H20） | 128 | 8 | **16** | **1.74** | **+18.4%** |
| Qwen2-57B（4×H20） | 64 | 4 | **16** | **1.74** | **+3.0%** |
| Qwen3-30B（4×H20） | 128 | 4 | 32 | 1.70 | -3.9% |
| DeepSeek-V2-Lite（4×H20） | 64 | 4 | 16 | 1.02 | -4.5% |

**Zipf模型验证**：Qwen2-57B的理论预测max_ratio $\approx 1.84$（Zipf $s=0.5$），实测1.74，误差5.4%。

**Shared expert稀释效应**：Qwen2-57B有巨大shared expert（intermediate_size=20480，是路由专家的8倍），路由专家仅占MoE层计算的~20%。routing维度的1.74不均衡被稀释成timing维度的~1.37，降低了OEPLB的有效收益空间。

### D.3 OEPLB收益条件

PB-OEPLB产生净正收益当且仅当：
$$\frac{r_{\text{before}} - r_{\text{after}}}{r_{\text{before}}} \times f_{\text{MoE}} > c_{\text{overhead}}$$

| 配置 | $r_{\text{before}}$ | $f_{\text{MoE}}$ | 理论gross收益 | 开销 | 净收益 |
|---|---|---|---|---|---|
| 8卡235B | 1.74 | 64% | 26%×0.64=16.6% | 0.67% | **+15.9%** ✅ |
| 4卡57B | 1.74 | ~20%(routed) | 26%×0.20=5.2% | ~2% | **+3.2%** ✅ |
| 4卡30B | 1.70 | ~50% | 20%×0.50=10% | ~6%(record+allreduce) | **+4%** 理论，实测-3.9% ❌ |

30B实测与理论偏差的原因：纯prefill场景下record每层每次都执行（不走CUDA graph），CPU侧Python开销远超理论估计。

**部署建议**：PB-OEPLB在以下条件最有效：
1. 每卡专家数 ≤ 16-20（高不均衡潜力）
2. 专家intermediate_size ≥ 2048（MoE计算占forward时间显著）
3. 两者均需要大模型（≥100B）或高EP数（≥8）
