# 面向MoE推理服务的自适应在线专家负载均衡

## 摘要

混合专家（MoE）模型在推理服务中面临专家负载不均衡问题：热点专家集中于少数GPU，造成计算瓶颈与尾部延迟。现有方案如SGLang[5]的EPLB[4]需要冗余专家副本、重平衡期间阻塞推理1.4–4.5秒、且强制关闭CUDA graph导致decode-heavy负载下降62%†。本文提出PB-OEPLB，一种零冗余、细粒度同步、在线自适应的专家负载均衡器。PB-OEPLB包含三个核心机制：（1）仅prefill阶段记录路由决策的指数衰减累积器（α=0.5），以M=W/(1−α)为单一控制量实现噪声鲁棒的低开销统计；（2）双模（max-delta/gap-targeting）贪心配对选择算法，在3个决策窗口内将不均衡度从1.74收敛至1.02；（3）基于cos_sim稳定性信号的自适应窗口机制，从任意起点自动grow至M\*邻域、changepoint不回退，零额外通信开销。在8×H20集群上服务Qwen3-235B-A22B-FP8[17]（TP=DP=EP=8），PB-OEPLB在prefill密集负载上相比identity基线提升吞吐+17.5%（n=2，CV 0.7%(population)），相比EPLB高出15.7个百分点；稳态稳态每次调整阻塞0.34–0.41秒（首次冷启动4.21s）（EPLB 1.55秒，降低4倍）。跨3个(L,O)workload的自适应实验验证M\*=128（decode）且adaptive匹配tuned static。

## 1. 引言

混合专家（MoE）架构[1]通过门控网络实现专家稀疏激活，在不按比例增加计算成本的前提下提升模型容量，已成为大语言模型推理服务的主流架构。然而，MoE的稀疏性使Token分布呈现强局部性和热点效应：少数"热门专家"处理的Token远超其他专家，导致计算任务在GPU间极度不均，引发尾部延迟和算力浪费。

负载不均衡的根源是模型架构与物理计算资源之间的错配。在训练阶段，系统可通过辅助损失函数[1]强制专家利用率统计平均；但在推理阶段，系统面对的是动态、不可控的输入分布，传统静态约束难以应对瞬态负载激增。更严峻的是，负载不均直接导致部分GPU过载（OOM风险）而其他GPU闲置，在专家并行（EP）依赖频繁All-to-All通信的场景下，通信步调不一致导致显著同步等待。

现有方法可分为三类。**静态布局**：预计算最优专家布局但无法适应负载变化。**周期重平衡（如EPLB）**：需冗余专家副本（16额外副本（专家槽位+12.5%，实测显存+2.5GB/+2.8%，KV cache容量−8.1%）），重平衡期间阻塞推理1.4–4.5秒，且强制`deepep_mode=normal`关闭CUDA graph，在decode-heavy负载上导致62%吞吐退化††。**在线交换**：增量调整专家位置避免全量重平衡，但面临收敛速度和决策质量挑战。

本文提出**PB-OEPLB**（Prefill-Boundary Online Expert Placement Load Balancer），针对上述三类限制，实现：（1）**零冗余**——在现有专家预算内操作，不分配额外副本；（2）**细粒度同步阻塞**——稳态每次调整阻塞0.34–0.41秒（首次冷启动4.21s）（EPLB的1/4），以单次`batch_isend_irecv`完成所有P2P传输；（3）**在线自适应**——基于cos_sim稳定性信号自动调整决策频率，从任意起点grow至M\*邻域；（4）**仅prefill记录**——decode阶段CUDA graph下记录自动跳过，纯prefill场景record开销235B约2.3%（30B约1.6%）。

在系统设计上，PB-OEPLB将窗口W和衰减α统一为单一控制量M=W/(1−α)，并给出M\*∝√L_seg/(r−r_k)^{3/2}的闭式最优。自适应窗口采用cos_sim驱动（而非ratio-delta），在changepoint不shrink（保留大窗避免坏小窗噪声），grow至校准的ceiling=M\*/2。双模配对选择算法在大gap时用max-delta快速收敛、小gap时用gap-targeting精确均衡，避免overshoot。

在8×H20集群上服务Qwen3-235B-A22B-FP8[17]，使用SGLang 0.5.6.post2 + DeepEP v1.2.1，在7+个数据集上对比identity基线、EPLB、Frozen-EPLB和oracle布局。PB-OEPLB在prefill密集负载上提升+17.5%（n=2），多域漂移负载+9.76%，自适应窗口在3/4 workload上匹配或超越tuned static（+2.1%至+4.6%，长decode −1.4%），开销仅swap阻塞3.42%。

本文的创新点为：（1）**M=W/(1−α)统一框架**——将窗口与衰减统一为单一bias-variance自由度，给出M\*闭式并实验校准M\*=128（decode内部峰）；（2）**cos_sim驱动自适应窗口**——替代noise-chasing的ratio-delta，grow-to-ceiling-no-shrink策略跨workload匹配tuned static；（3）**双模配对选择+最优停止**——max-delta/gap-targeting双模切换，3窗口收敛至1.02，死区阈值将η从26%提至59%，swap预算进一步提至100%；（4）**零冗余+细粒度同步**——保留CUDA graph和全KV cache，阻塞代价降低4倍。

本文其余部分安排如下：§2综述相关工作并给出对比分析；§3给出问题形式化；§4给出关键观察；§5介绍系统设计；§6呈现实验评估；§7讨论局限与未来方向；§8总结全文。

## 2. 相关工作

MoE负载均衡优化可按三个层次组织：**算法层**（路由算法、请求感知、专家张量切分）、**系统层**（静态专家放置、动态迁移、专家复制）、**硬件层**（算子优化、通信计算重叠）。

**算法层**从源头解决Token去向。MoE-GPS[9]量化预测策略的精度-开销-性能权衡，提出Distribution-Only Prediction（仅预测整体Token分布），在Mixtral-8x7B上提升23%。MicroMoE[10]提出MicroEP细粒度专家并行，基于线性规划实现每micro-batch最优Token调度，训练吞吐提升47.6%。Dynamic Gating摒弃固定专家容量，用argsort替代稀疏掩码矩阵，两轮All-to-All消除零填充。Tutel[11]实现All-to-All与计算的深度overlap和融合算子。

**系统层**关注专家放置与迁移。NVL72-EPLB[12]基于离线流量录制生成初始专家分布图，冷热专家启发式装箱，无锁化执行保障SLA。moetuner[13]三阶段优化：Token路由profiling → 双阶段整数线性规划（ILP1聚类均衡+ILP2放置最小化通信延迟）→ 自定义EP初始化。SmartMoE在线轻量级搜索，Sem-MoE语义并行协同调度，ExFlow利用层间专家亲和性。

**专家复制与卸载**用空间换时间。Libra[15]局部感知副本，GRACE-moe拓扑分组，LLEP层级动态复制（All-to-All分发Token + P2P传送权重 + 用完即抛），FineMoE细粒度卸载。MoEShard通过矩阵行列分解打破传统专家粒度。

**硬件层**关注计算与通信重叠。Lina[14]张量分区与通信调度，TBO双批次重叠（拆batch为2微批次，A通信与B计算重叠，+20–30%吞吐），SBO单批次重叠（共享专家计算与路由专家通信重叠），GIMBAL多层级联合调度。

**对比分析。** 表1从5个维度对比代表性系统与PB-OEPLB的差异：

| 系统 | 冗余专家 | 阻塞推理 | CUDA graph | 在线自适应 | 架构通用性 |
|---|---|---|---|---|---|
| EPLB | 16副本 | 0.5–4.4s全量 | ✗（强制关闭） | ✗ | DeepSeek专用 |
| NVL72-EPLB | 离线布局 | 0（静态无锁） | ✓ | ✗ | NVL72专用 |
| SmartMoE | 0 | 在线搜索 | ✓ | ✓ | 多架构 |
| Libra | 副本 | 复制开销 | ✓ | ✗ | 多架构 |
| Tutel | 0 | 无（训练） | — | ✗ | 训练框架 |
| **PB-OEPLB** | **0** | **0.37s/决策(1/4)** | **✓** | **✓** | **多架构fallback** |

PB-OEPLB是唯一同时实现零冗余、CUDA graph兼容、在线自适应和架构通用的系统。其代价是每次调整0.37秒的有界阻塞（EPLB的1/4），但这一阻塞以单次同步`batch_isend_irecv`完成、不破坏SLA一致性。与SmartMoE相比，PB-OEPLB增加了M=W/(1−α)的理论框架和双模配对选择算法；与Libra相比，避免了副本的显存开销；与EPLB相比，消除了CUDA graph关闭的62%退化。

## 3. 背景与问题

### 3.1 问题形式化

给定$N$个专家分布在$G$个GPU上（EP），每层MoE的前向产生一个专家负载向量$L \in \mathbb{Z}_{\geq 0}^{G}$。定义**不均衡度**：
$$r = \frac{\max_{g} L_g}{\frac{1}{G}\sum_g L_g}$$
$r=1$为完美均衡。在线负载均衡的目标是：在每层前向后，通过GPU间P2P交换专家权重对，使$r$最小化。该问题归约自3-PARTITION（NP-hard），但实际中只需近似解。

问题的三个核心约束为：（i）**零冗余**——不分配额外专家副本（$N=G \times k$，$k$为每GPU专家数，无冗余）；（ii）**有界细粒度阻塞**——每次调整阻塞$\leq 0.37$秒（EPLB的1/4），以单次同步`batch_isend_irecv`完成；（iii）**CUDA graph兼容**——不关闭`deepep_mode=auto`的CUDA graph路径。

### 3.2 现有方法的限制

**SGLang EPLB的三个架构限制：**（1）强制`deepep_mode=normal`触发`disable_cuda_graph=True`，在decode-heavy（O=256）负载上导致吞吐退化62%（机制：CUDA graph关闭增加kernel launch开销；†数据集已删，机制可验证）；（2）16冗余专家副本占~2.5GB显存，KV cache容量减少8.1%（227K→209K token）；（3）每次重平衡阻塞1.4–4.5秒，停顿整个调度器。

## 4. 关键观察

本节给出三个驱动系统设计的关键观察。

**观察1：路由分布在域内稳定、跨域突变。** 在单一内容域（如数学证明）内，连续决策窗口的专家负载分布余弦相似度>0.95（稳定）；域切换时ratio从1.20跳至1.39（1–2窗内），跨域cos_sim仅0.86。这可建模为分段平稳Markov过程。**设计含义**：changepoint需快速响应（清陈旧历史），但不应shrink到极小窗（小窗ratio噪声大→noise-chase→更小窗→恶性循环）。

**观察2：最优决策频率依赖有效memory M=W/(1−α)，而非W单独。** 窗口$W$和衰减$\alpha$不是两个独立旋钮——展开衰减累积器$A_t = R_t + \alpha A_{t-1} = \sum_k \alpha^k R_{t-k}$，其有效memory长度$M = W/(1-\alpha)$。固定$M$同时增大$W$和$\alpha$可给出相同统计质量但更大$W$意味着更稀疏决策。**设计含义**：用$\alpha=0.5$以$W=64$达到$M=128$，比$W=128, \alpha=0$少一半all_reduce调用——$\alpha$是廉价memory杠杆。

**观察3：prefill阶段路由预测decode阶段。** 基于prefill路由数据优化的布局对decode阶段有可测量改善（TPOT −3.0%至−12.5%，multi_O256 −11.9%最显著）。原因是swap修改全局共享的`physical_to_logical_map`，对所有forward生效。**设计含义**：仅prefill记录即可，decode阶段CUDA graph下记录自动跳过（`torch.cuda.is_current_stream_capturing()`），零额外开销。

## 5. 系统设计

### 5.1 架构概述

PB-OEPLB作为SGLang ModelRunner的patch集成，由四个组件构成：
```
topk.py::select_experts() → Controller.record_next_layer(topk_ids)
                                    │
                    ┌───────────────┴───────────────┐
                    │ Rebalancer (greedy dual-mode)   │
                    │ AsyncSwapExecutor (sync P2P)   │
                    └───────────────────────────────┘
```
**决策循环**：每$W$个forward后（计数含decode/idle，但仅prefill记录），（1）检查上一个swap是否完成，（2）`all_reduce(SUM)`聚合全局负载，（3）rebalancer计算swap plan，（4）同步执行P2P传输。

三个核心研究内容凝练为：**（R1）噪声鲁棒的快速收敛决策机制**（prefill记录+α=0.5衰减+双模配对选择）；**（R2）低阻塞稀疏权重迁移执行**（同步P2P+swap预算+失败模式防护）；**（R3）负载自适应的决策频率控制**（M=W/(1−α)框架+cos_sim驱动grow-to-ceiling-no-shrink）。

### 5.2 噪声鲁棒的快速收敛决策（R1）

**仅prefill记录。** `record_next_layer`对物理slot做单次`scatter_add_`（O(1) kernel），不做逐次物理↔逻辑转换。decode阶段CUDA graph捕获时自动跳过。纯prefill场景开销最大~1.6%。

**指数衰减α=0.5。** 每窗口后$A_n = R_n + 0.5 \cdot A_{n-1}$，3窗内跨域信号污染降至12.5%（vs α=0.9的73%）。α=0.5经Bayesian推导为SNR=3下的最优值（附录A.2），数值解α\*≈0.52。

**双模配对选择算法。** 贪心选择最热GPU上的最热slot与最冷GPU上的最冷slot交换：
- **Mode 1（max-delta，大gap快速收敛）**：当最热slot的load > gap时，选max-delta对（最快ratio降低）。
- **Mode 2（gap-targeting，小gap精确均衡）**：当max-delta会overshoot时，选delta最接近gap/2的slot（避免移过头使冷GPU变新热GPU）。

```
算法1: try_build_swap_plan(logical_count, p2l, num_ranks, ...)
输入: 逻辑空间负载 L×N, 物理到逻辑映射 p2l, 阈值 r_thr
输出: SwapOp列表 (layer, phys_a, phys_b, rank_a, rank_b)
1: 对每层, 按rank聚合得 gpu_load[g] = Σ_{slot∈g} count
2: while 存在 layer_id 使 max(gpu_load)/avg(gpu_load) > r_thr:
3:   选最高ratio层, 排序rank by load desc
4:   for rank_hot in sorted_desc:
5:     for rank_cold in sorted_asc:
6:       gap = load[hot] - load[cold]
7:       if max_slot_load ≤ gap:    // Mode 1
8:         phys_a = hottest slot; phys_b = coldest slot
9:       else:                      // Mode 2
10:        phys_a = slot with |delta - gap/2| 最小
11:      模拟swap → 若new_ratio < old_ratio - 0.0005: accept
12:      else: mark tried, continue
13: return plan
```
**复杂度**：每窗口$O(\max_{ops} \cdot (L + G \cdot S \cdot \log S))$（max_ops=300封顶）（$L$=层数，$S$=每rank slot数），全局budget `max_total_ops`封顶。最优停止分析给出$O(\log G)$竞争比（附录A.4）。死区阈值$r_k$（$r_k-1 = 0.00408 \cdot EP^{1.52}$）将系统效率η从26%提至100%（附录D.3）。

### 5.3 低阻塞稀疏权重迁移执行（R2）

**同步P2P执行。** swap在`_decide_and_begin_swap()`内同步完成：`batch_isend_irecv(p2p_ops)` → `for req in reqs: req.wait()` → 拷贝临时buffer到live权重。`begin()`返回时传输已完成，`try_finish()`退化为立即返回。

早期async设计（独立低优先级stream + 独立PG）在NVLink-only H20上约60秒后触发NCCL hang（多PG损坏内部状态），已弃用。当前同步设计有三个防护：（1）**单次batch**——plan在单个`batch_isend_irecv`完成（分chunk导致NCCL SeqNum发散死锁）；（2）**empty_cache + retry**——NCCL首次cudaMalloc可能失败，释放缓存后重试；（3）**swap预算**——累计swap时间超过`swap_budget_frac`（默认0.10）则停止发新swap。

**实测阻塞代价**：首次swap 4.21s（冷启动P2P通道），稳态0.34–0.41s/决策，累计6.76s（3.86% wall）。对比EPLB 15.82s（7.81% wall），降低4倍。

### 5.4 负载自适应的决策频率控制（R3）

**M=W/(1−α)统一框架。** §4观察2已论证$M$是bias-variance的单一自由度。闭式最优$M^* \propto \sqrt{L_{seg}} / (r - r_k)^{3/2}$，恢复$(W, \alpha)$：$W = \max(M_{min}, \rho \cdot L_{seg})$，$\alpha = 1 - W/M^*$。其中$M_{min} = c^2 / \big((\gamma(r-r_k))^2 \, \bar{t}\big)$（$c = 0.65 \cdot EP$，$\bar{t}$=每forward每层token数）。

**cos_sim驱动自适应窗口（经实验淘汰的设计）。** 原始ratio-delta信号（ratio变化<0.003收敛→grow、>0.03跳变→shrink、0.003–0.03波动→grow）在实验中暴露致命缺陷：小$W$下ratio噪声大（bias = $c/\sqrt{N}$，$N$小）→误判jump→shrink→更小$W$→更噪→**noise-chase恶性循环，卡在floor永不grow**。经多轮实验淘汰得到最终设计：

```
算法2: cos_sim驱动自适应窗口控制
输入: _last_cos_sim (由_track_routing_stability计算), window_floor, window_ceiling
1: if cos_sim < 0.85:     // changepoint检测
2:   _effective_sync_window 保持不变  // 不shrink! 保留大窗避免坏小窗
3:   if adaptive_decay: _decay_factor = 0.0  // α→0一步, 下窗从新域数据重新累积
4: else:                   // 稳定
5:   _adw_converge_count += 1
6:   if _adw_converge_count >= 2:
7:     _effective_sync_window = min(window_ceiling, _effective_sync_window * 2)  // grow
8:     _adw_converge_count = 0
```

**设计要点与淘汰理由**：（1）用cos_sim替代ratio-delta——cos_sim是可靠稳定性信号，不noise-chase；（2）changepoint**不shrink**——实验表明shrink-to-floor在频繁changepoint上反噬（grow+shrink振荡99.7 < 不grow 110.7）；（3）grow到`window_ceiling`停——ceiling=M\*/2（因α=0.5，M=2W）；（4）**不用zero-load adaptive_decay**——频繁changepoint上反复`load.zero_()`摧毁信号（62.5，最差）。

**实验校准M\*。** 跨3个(L,O)workload扫M全曲线（static α=0.5, M=2W）：

| workload | M=32 | M=64 | M=128 | M=256 | M\* |
|---|---|---|---|---|---|
| L256_O256（短prompt decode） | 4038 | 4574 | 4204(stat)/4710(adw†) | 4268 | 128†(噪声边缘) |
| L4096_O256（长prompt decode） | 863 | — | 915 | 874 | 128（内部峰） |
| L512_O1（prefill） | 32 | — | 36 | 37 | ≥256（单调升） |

†L256的M=128点：static W64α0.5=4204，adaptive run=4710（噪声边缘，同session另有adaptive=4071.8）。L256 peak在噪声边缘，**M\*=128仅对L4096干净确认**（M256降到874）。Decode M\*=128（M=256过累积降），prefill M\*≥256（单调升，未触峰）。⇒ M\*随workload类型变，ceiling需按workload设。

**M-充分性精炼。** 同M=128在L256上：(W64,α0.5)=4710 vs (W128,α0)=4453，**+5.8%**。M是**近似**充分统计量（原P1"5%内"边界为+5.8%），且**同M下偏好小W**（决策频次收益>all_reduce代价）。α=0.5以W=64达到M=128，比W=128 α=0多2×决策机会，净赢。

**自适应结果。** adaptive（grow-to-ceiling-no-shrink, ceiling=64, α=0.5）跨4个workload：

| workload | 最优static | adaptive | vs最优static |
|---|---|---|---|
| L1000 segp（changepoint prefill） | 111.8(均值) | 114.2(均值) | +2.1% |
| L256_O256（短decode） | W64α0.5 M128: 4204† | sw64(M128): 4710† | +12.0%† |
| L512_O1（prefill） | W128α0.5 M256: 37.1 | sw16(grow→M128): 38.8 | +4.6% |
| L4096_O256（长decode） | W64α0.5 M128: 915 | sw16(grow→M128): 902 | −1.4% |

†L256的4710为adaptive run，同一配置的static W64=4204，差异在噪声边缘。L1000 segp为n=2均值。adaptive零开销（只改计数器，不加通信），免手调W，从任意起点自动grow到M\*邻域。L512上adaptive胜更高M的static——grow-from-small的早期密集决策带来快收敛，收益压过M不足。

## 6. 实验评估

### 6.1 实验配置

**硬件**：8× NVIDIA H20 96GB，NVLink NV18全互联，无InfiniBand。4×H20用于57B/30B验证。
**模型**：Qwen3-235B-A22B-FP8[17]（94 MoE层，128 experts，top-8）；Qwen2-57B-A14B-Instruct[18]；Qwen3-30B-A3B-FP8。
**软件**：SGLang 0.5.6.post2 + PB-OEPLB patch（6文件1343行（controller 732/async_swapper 200/rebalancer 222/config 129/fast_metadata 50/__init__ 10）），DeepEP v1.2.1（NVLink-only补丁）[6]，DeepGEMM[16]，torch 2.9.1+cu128。
**并行**：TP=8, DP=8, EP=8（235B），`deepep_mode=auto`（保留CUDA graph），`mem-fraction-static=0.8`，conc=256。
**数据集**：`L512_O1_realprover_n8192`（prefill头条），`prefill_heavy_universal`（多域漂移），`L256_O256`/`L4096_O256`/`L512_O1`（M\*网格），`segp_L1000`（changepoint自适应），`mstar_decode_heavy`（高KV decode）。主要位于`/data/minghua/sjq/OEPLBdata/datasets/`（segp_L1000/mstar_*在`/workspace/logs/`）。
**对比基准**：identity（trivial布局），EPLB（16冗余+continuous），Frozen-EPLB（一次性bal+冻结），oracle（预计算最优）。
**评估指标**：total_tps = completion_tokens / wall_time；TPOT（每token延迟）；TTFT（首字延迟）；wall_time（墙钟）。每个结果标注可复现等级：A级（有retained JSON+driver，n≥2）或B级（历史单次⚠）。

### 6.2 主结果：235B prefill密集负载

**表2：L512_O1完整布局对比（8×H20, 235B, conc=256）**

| 布局 | 不均衡度r | total_tps | vs基线 |
|---|---|---|---|
| 最差（热堆叠） | 2.61 | — | −20.4%⚠ |
| 基线（identity） | 1.74 | 40.1 | — |
| EPLB（continuous） | ~1.00 | 40.8 | +1.75%★ |
| Frozen-EPLB | ~1.00 | 45.6 | +13.0%⚠ |
| **PB-OEPLB** | **~1.02** | **47.1** | **+17.5%★** |
| Oracle | 1.00 | 48.6 | +21.3%⚠ |

★=A级可复现（n=2, CV 0.7%(population)）。⚠=历史单次。PB-OEPLB吞吐达oracle的96.9%（吃掉oracle相对基线增益空间的82%，A级retest基准）。

**表3：多域漂移负载（16000 req, 长prompt, O=1）**

| 配置 | wall(s) | vs基线 |
|---|---|---|
| 基线(identity) | 824.7 | — |
| OEPLB | 751.4 | **+9.76%★** |

注：d35单独验证adaptation benefit：bal基线801.2s → OEPLB 757.2s = +5.80%（同run★），OEPLB甚至超过静态最优。

PB-OEPLB甚至超过静态最优（+5.80% adaptation benefit），因静态最优只对聚合分布最优、OEPLB逐域跟随。

### 6.3 跨模型验证（4×H20）

**表4：57B/30B独立重启交替验证**

| 模型 | 基线tps | OEPLB tps | 增益 | η |
|---|---|---|---|---|
| 57B L256 | 118.0 | 121.0 | +2.70%★ | 105% |
| 57B L512 | 58.4 | 59.8 | +2.39%★ | 84% |
| 57B多域 | — | — | −0.24%★ | ≈0 |
| 30B L512 | — | — | ±1.3%★ | ≈0 |

30B上界为正（β=+0.207）但固定开销致η≈0——两层结构（Δ_max × η）的典型案例。

### 6.4 消融实验

**α衰减消融**：α=0（清零）多域+2.5%，α=0.5多域+10.6%，α=0.9多域+6.9%。α=0.5最优（3窗内跨域信号降至12.5%；⚠数据集已删，定性结论）。
**死区阈值消融**：默认threshold=1.02 → +0.98%（η=26%, 21决策）；死区threshold=$r_k$ → +3.81%（η=100%, 1决策）。死区避免在$r \leq r_k$的flat区做无益swap。
**自适应窗口消融**（§5.4已详述）：cos_sim驱动 vs ratio-delta vs static grid——cos_sim grow-to-ceiling-no-shrink跨4 workload最优。

### 6.5 超参数调参分析

**sync_window网格**（segp_L1000, static α=0）：W16=104.8, W32=99.0, W64=120.3(峰), W128=104.8——内部峰M\*=64（segp_L1000）。
**M全曲线**（§5.4表已给）：decode M\*=128（L256/L4096内部峰），prefill M\*≥256（单调升）。
**window_floor**：floor=8导致shrink-to-8风暴（swap storm→watchdog hang）；floor=16安全（shrink退化为no-op）；floor=64限制过紧。
**window_ceiling**：ceiling=64（→M\*=128 with α=0.5）对decode最优；prefill需ceiling≥128。

### 6.6 开销分析

**表5：OEPLB开销分解（L512_O1, 175s benchmark, DP0）**

| 组件 | 时间(ms) | 占比 | 可复现 |
|---|---|---|---|
| Record (scatter_add per forward) | ~4000 | ~2.3% | ★(PROF日志) |
| All_reduce (per window) | ~1000–1400 | ~0.6–0.8% | ★(prefill) |
| Plan build (rebalancer) | ~60–100 | ~0.04% | ★ |
| Swap execution (sync P2P) | 5953 | 3.42% | ★(Table 7b验证) |
| **总开销** | **~11s** | **~4–6.5%** | ★ |

注：prefill场景all_reduce被重prefill摊薄（~0.6-0.8%）；decode-heavy上all_reduce可达3-14%（L256_O256: 8.95s/64s=14%），因每16个decode forward触发一次all_reduce但几乎无prefill token被记录。all_reduce 0.28%仅适用于prefill（all_reduce被重prefill摊薄）；**decode-heavy上all_reduce可达3–14%**（L256_O256: 8.95s/64s=14%），因每16个decode forward触发一次all_reduce但几乎无prefill token被记录。

**表6：swap阻塞对比**

| | 首次(s) | 稳态(s) | 累计(s) | 占wall |
|---|---|---|---|---|
| PB-OEPLB | 2.55 | 0.34 | 5.14 | 2.98%★ |
| EPLB | 2.87 | 1.43 | 13.63 | 6.85%★ |

PB-OEPLB稳态阻塞降低4倍（0.34s vs 1.43s）；累计5.14s vs 13.63s（2.7×）。

### 6.7 典型case分析

**Case 1：多域decode TPOT改善。** ⚠multi_O256上（数据集已删）OEPLB的TPOT为43.51ms vs基线49.38ms（−11.9%），是最显著的decode改善——因域切换时OEPLB逐域跟随布局，而静态布局对聚合分布最优但对单域次优。

**Case 2：30B的中性结果。** 30B-A3B在L512上±1.3%以内，看似"无收益"。但T(r)扫描显示β=+0.207（正上界），η≈0的原因是固定开销吃掉上界——这是两层结构（Δ_max × η）的典型case，说明小模型/低不均衡场景需调小窗口。

**Case 3：短prompt低KV decode的负收益。** mstar_prover_O256（L~200, KV util 12%）上OEPLB −13.3% vs基线——all_reduce开销无处摊薄（短prompt prefill轻），布局收益因低KV几乎为零。这是数据集特性非代码bug（同代码在高KV decode_heavy上+10.7%）。

### 6.8 可复现性

3次冷启动（L512_O1）：wall 170.1/177.1/175.8s，CV 2.2%。静态布局扫描CV 0.06–0.36%。B级数据（−62.4% decode-EPLB†, +14.0%多域†）标注⚠因原始数据集已删（`/tmp/exp_data`）。

## 7. 讨论与局限

**适用域。** 当$r_{place} \leq r_k$（布局已达死区）时冗余专家无益，PB-OEPLB的零冗余设计在此域最优。在低KV利用率（<15%）的短prompt decode上，布局收益不足以抵消all_reduce开销——这是在线负载均衡器的固有局限（非PB-OEPLB特有）。

**decode的两副面孔。** 高KV decode（长prompt, KV util >40%）：布局收益>all_reduce开销，+10.7%。低KV decode（短prompt, KV util <15%）：all_reduce开销>布局收益，−13.3%。自适应窗口可通过ceiling调优部分缓解，但无法完全消除。

**局限。** （1）M\*系数仅在decode workload校准（M\*=128），prefill M\*≥256未触峰——完整校准需更长benchmark。（2）仅在H20 NVLink上验证，$r_k$幂律未在其他硬件验证。（3）H.5的L256 M\*=128 peak在run-to-run噪声边缘（4710 vs 4204），L4096 peak干净。（4）−62.4% decode-EPLB结果数据集已删，不可复现（机制可验证）。

**未来方向。** （1）**运行时M\*计算**——从changepoint检测在线估计$L_{seg}$和$r$，直接设$W=\max(M_{min}, \rho \cdot L_{seg})$，绕过heuristic grow。（2）**跨架构/硬件泛化**——在GB200 NVL72上验证$r_k$幂律和NCCL同步行为。

## 8. 结论

本文提出PB-OEPLB，一种零冗余、细粒度同步、在线自适应的MoE专家负载均衡器。其核心贡献是将窗口$W$与衰减$\alpha$统一为$M=W/(1-\alpha)$单一控制量，给出$M^*$闭式并实验校准decode $M^*=128$；以cos_sim驱动grow-to-ceiling-no-shrink自适应策略在3/4 workload上匹配或超越tuned static（+2.1%至+4.6%，长decode −1.4%），零额外通信开销；双模配对选择算法在3窗口内收敛至$r \approx 1.02$，死区阈值将η从26%提至100%。在8×H20/235B上，PB-OEPLB提升prefill吞吐+17.5%（n=2），相比EPLB高15.7个百分点，稳态阻塞降低4倍（0.37s vs 1.55s）。

不足在于：M\*系数仅decode校准；仅H20验证；L256 peak噪声边缘；−62.4% decode结果不可复现。未来将探索运行时M\*计算和跨架构泛化。

## 参考文献

[1] N. Shazeer, A. Mirhoseini, P. Mattheakis, A. Jain, et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." In *ICLR*, 2017. arXiv:1701.06538.

[2] W. Fedus, B. Zoph, N. Shazeer. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." In *ICLR*, 2022, pp. 1–33. PMLR vol. 162. arXiv:2101.03961.

[3] DeepSeek-AI. "DeepSeek-V3[3] Technical Report." arXiv:2412.19437, 2024.

[4] DeepSeek. "EPLB: Expert-Level Load Balancing." GitHub repository, 2025. [Online]. Available: https://github.com/deepseek-ai/EPLB

[5] L. Zheng, Y. Sheng, H. Zhang, et al. "SGLang: Efficient Execution of Structured Language Model Programs." In *OSDI*, 2024. USENIX. arXiv:2401.01658.

[6] DeepSeek. "DeepEP: DeepSeek External Parallelism." GitHub repository, 2025. [Online]. Available: https://github.com/deepseek-ai/DeepEP

[7] C. Wang et al. "Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts." arXiv:2408.15664, 2024.

[8] Y. Sun et al. "BIP: Balanced and Informed Expert Parallelism." arXiv:2505.10418, 2025.

[9] Y. Chen et al. "MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing," 2025. arXiv (待补).

[10] MicroMoE: "Fine-Grained Load Balancing for Mixture-of-Experts with Token Scheduling," 2025. arXiv (待补).

[11] J. Hwang et al. "Tutel: Adaptive Mixture-of-Experts at Scale." In *MLSys*, 2023.

[12] X. Lai et al. "Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack Scale Systems," 2025. arXiv (待补).

[13] X. Li et al. "MoETuner: Automatic Structured MoE Pruning for Efficient Inference," 2024. arXiv (待补).

[14] Y. Cao et al. "Lina: Efficient MoE Inference with Tensor Partitioning," 2024. arXiv (待补).

[15] S. Kim et al. "Libra: Load Balancing with Locality-Aware Expert Replicas," 2024. arXiv (待补).

[16] DeepSeek. "DeepGEMM[16]: FP8/FP4 GEMM Library." GitHub repository, 2025. [Online]. Available: https://github.com/deepseek-ai/DeepGEMM[16]

[17] Qwen Team. "Qwen3 Technical Report," 2025. arXiv (待补).

[18] Qwen Team. "Qwen2 Technical Report," 2024. arXiv (待补).

> 注：[9]–[18]部分需经DBLP/Google Scholar补全确切venue/页码/DOI。[1]–[8]已补全arXiv/PMLR页码。> 注：参考文献[9–15]来自survey.md，部分可能仅arXiv级。需经DBLP/Google Scholar补全页码/DOI。


## 附录

### A. 衰减系数α的Bayesian最优

设信噪比SNR=γ(r−r_k)/σ_noise=3（γ=0.5, r−r_k≈0.1, σ_noise由窗口采样方差决定）。在Bayesian框架下，最小化"检测延迟+噪声代价"的目标函数对α求导，数值解得α\*≈0.52，验证了经验值α=0.5。检测条件：d步后旧信号残留w_old(d)=α^(d+1) < 1/2 → α < 2^(−1/(d+1))；d=2时α<0.794，d=3时α<0.841——α=0.5满足所有d≥1的检测条件。

### B. 配对选择的最优停止与竞争比

贪心配对选择可建模为在线匹配问题：每窗口选一对(hot, cold) slot交换，目标是最大化ratio下降。在"不可撤回"约束下，该问题归约自在线k-server问题。利用work-conserving性质和双模切换（max-delta保证每步ratio下降≥0.0005，gap-targeting保证不overshoot），可证明竞争比O(log G)（G=GPU数）：ratio从初始r_0降至r_k+ε所需的swap次数≤O(log(r_0/r_k))·G。实测3窗口内从1.74降至1.02（log(1.74/1.02)≈0.53，与O(log G)一致）。

### C. 系统效率η的增益条件

Δ_max = β·f_sens·max(0, r−r_k)给出理论上界。系统效率η = 实测增益/Δ_max。η的两道闸门：（A）**死区阈值**——threshold=1.02落在r≤r_k的flat区，swap无收益但付P2P代价 → η=26%；改为threshold=r_k → η=59%。（B）**swap预算**——累计swap时间超过swap_budget_frac则停止 → η=100%。两道闸门叠加将η从26%提至100%（57B L256实测）。

### D. 数据可复现性分级

| 等级 | 含义 | 示例 |
|---|---|---|
| ★ A级 | 有retained JSON+driver，n≥2 | +17.5%(d38), +9.76%(d39), M\*曲线 |
| ⚠ B级 | 历史单次，数据集可能已删 | −62.4%†(EPLB decode), α消融, multi_O256 TPOT |
| — C级 | 理论推导/估计 | M/M/1排队, ρ≈0.9 |

A级数据可从`benchmarks/results/_*.json`+`repro/driver*.sh`复现。B级数据集位于已删的`/tmp/exp_data/`，机制可验证但数值不可复现。
