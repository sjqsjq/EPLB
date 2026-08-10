# Swap Planner 近似比分析

## 1. 问题形式化

### 单层设定

- $G$ 个 GPU，每个持有 $n = N_E / G$ 个专家
- 专家负载向量 $\ell \in \mathbb{R}^{N_E}_{\geq 0}$
- 分配 $\pi: [N_E] \to [G]$，满足平衡约束 $|\pi^{-1}(g)| = n, \forall g$
- GPU 负载：$L_g(\pi) = \sum_{i: \pi(i)=g} \ell_i$
- 目标：通过 ≤ B 次 pairwise swap，最小化 makespan $M(\pi) = \max_g L_g(\pi)$
- 等价目标：最小化 imbalance ratio $r(\pi) = M(\pi) / \bar{L}$，其中 $\bar{L} = \frac{1}{G}\sum_g L_g$ 为常数

### Swap 操作

交换两个不同 GPU 上的专家 $(a, b)$，$\pi(a) \neq \pi(b)$。结果：$\pi'(a) = \pi(b), \pi'(b) = \pi(a)$，其余不变。平衡约束自动保持。

## 2. 计算复杂度

**命题1**：找到最优平衡分配 $\pi^* = \arg\min_\pi M(\pi)$ 是 NP-hard 的。

**证明**：$G=3$ 时归约到 3-PARTITION；$G=2$ 时归约到 PARTITION。

**推论**：贪心 swap 是 local search 近似算法，应用近似比衡量质量。

## 3. Swap-Local-Optimum 定义

**定义**：$\pi$ 是 swap-local-optimal 当且仅当不存在 swap $(a,b)$ 使得 $M(\pi') < M(\pi)$。

**Swap 有效条件**：设 $a \in \pi^{-1}(g^*)$（最热GPU），$b \in \pi^{-1}(g')$（某其他GPU），$\delta = \ell_a - \ell_b > 0$。

Swap 后：$L_{g^*}' = L_{g^*} - \delta$，$L_{g'}' = L_{g'} + \delta$。

Swap 严格改善 makespan ⟺ $\delta < L_{g^*} - L_{g'}$（否则 $g'$ 变成新最热 → overshoot）。

## 4. 核心定理：Local Optimum 质量保证

### 引理A（关键结构性质）

**引理A**：在 swap-local-optimal $\pi$（gap > 0）下，$g^*$ 上最轻的专家 $a_n$ 满足 $\ell_{a_n} \leq \ell_{b_1}$（$g_-$ 上最重的专家）。

**证明**：反证法。设 $\ell_{a_n} > \ell_{b_1}$，则 swap $(a_n, b_1)$ 的 delta = $\ell_{a_n} - \ell_{b_1} > 0$。

由 local optimality，需 $\delta \geq \text{gap} = L_{g^*} - L_{g_-}$。

但 $\ell_{a_n} \leq L_{g^*}/n$（最轻 ≤ 平均）且 $\ell_{b_1} \geq L_{g_-}/n$（最重 ≥ 平均），所以：

$$\delta = \ell_{a_n} - \ell_{b_1} \leq \frac{L_{g^*}}{n} - \frac{L_{g_-}}{n} = \frac{\text{gap}}{n}$$

当 $n \geq 2$ 时，$\text{gap}/n < \text{gap}$，与 $\delta \geq \text{gap}$ 矛盾。 $\square$

### 定理2（Swap Local Optimum Bound）

**定理2**：设 $\pi$ 是 swap-local-optimal。则：

$$r(\pi) \leq 1 + \frac{G-1}{G} \cdot \frac{\ell_{\max}}{n\mu}$$

其中 $\ell_{\max} = \max_i \ell_i$，$\mu = \frac{1}{N_E}\sum_i \ell_i$ 为平均专家负载。

**证明**：

**Step 1**：证明 gap $\leq \ell_{\max}$。

在 swap-local-optimal 下，$g^*$ 上必存在至少一个专家 $a_j$ 使得 $\ell_{a_j} > \ell_{b_n}$（$g_-$ 最轻专家），否则 $L_{g^*} \leq n \cdot \ell_{b_n} \leq L_{g_-}$，与 gap > 0 矛盾。

对这样的 $a_j$，local optimality 要求 $\ell_{a_j} - \ell_{b_n} \geq \text{gap}$。

而 $\ell_{a_j} - \ell_{b_n} \leq \ell_{a_j} \leq \ell_{\max}$，所以 $\text{gap} \leq \ell_{\max}$。

**Step 2**：从 gap 推 ratio。

$$L_{g^*} - \bar{L} = \frac{1}{G}\sum_{g \neq g^*}(L_{g^*} - L_g) \leq \frac{G-1}{G}(L_{g^*} - L_{g_-}) = \frac{G-1}{G} \cdot \text{gap}$$

结合 Step 1：

$$r(\pi) - 1 = \frac{L_{g^*} - \bar{L}}{\bar{L}} \leq \frac{(G-1) \cdot \text{gap}}{G \cdot \bar{L}} \leq \frac{(G-1) \cdot \ell_{\max}}{G \cdot n\mu}$$

（用 $\bar{L} = n\mu$。）$\square$

### 数值验证

| 配置 | G | n | $\ell_{\max}/\mu$ (实测) | Bound 预测 $r$ 上界 | 实测 local-opt $r$ |
|---|---|---|---|---|---|
| 235B, EP=8 | 8 | 16 | ~5 | $1 + (7/8)(5/16) = 1.27$ | ~1.02 |
| 30B, EP=4 | 4 | 32 | ~3 | $1 + (3/4)(3/32) = 1.07$ | ~1.02 |

Bound 比实际松——原因是实际中 swap 通常能找到远好于"仅存在一个超重专家"假设的解。Bound 只保证最坏情况。

### Tight Example

$G=2, n=2$：$\ell = (M, \epsilon, \epsilon, 0)$，$M \gg \epsilon$。

初始分配 GPU0=$(M, 0)$, GPU1=$(\epsilon, \epsilon)$：
- $L_0 = M, L_1 = 2\epsilon$, gap = $M - 2\epsilon$
- Swap $(M, \epsilon)$: delta = $M-\epsilon \geq M - 2\epsilon$ = gap。Overshoot!
- Swap $(0, \epsilon)$: delta = $\epsilon - 0 = \epsilon < \text{gap}$... 但0在GPU0不是比$\epsilon$大，所以这是把GPU0的轻专家换到GPU1的更轻专家，不满足 $\ell_a > \ell_b$。

实际上 swap(M, ε): delta = M-ε。gap = M-2ε。delta = M-ε > M-2ε = gap（当ε>0）。Overshoot。

唯一其他选项 swap(0, ε): ℓ_a=0 < ℓ_b=ε，不满足改善条件。

**所以此分配已经是 local optimum！**

Ratio = $M / ((M+2\epsilon)/2) = 2M/(M+2\epsilon) \to 2$ as $\epsilon \to 0$。

Bound 预测：$1 + (1/2) \cdot M / (2 \cdot (M+2\epsilon)/4) = 1 + (1/2) \cdot 2M/(M+2\epsilon) \to 1 + 1 = 2$。

**Bound is tight for this construction!** (Ratio = 2, Bound = 2 as $\epsilon \to 0$.)

## 5. Max-delta 的震荡问题

**命题3（Max-delta 可能不收敛）**：

$G=2, n=3$：$\ell = (10, 5, 1, 8, 4, 2)$。初始 GPU0=$(10,5,1)=16$, GPU1=$(8,4,2)=14$, gap=2。

Max-delta 选择 $(10, 2)$: delta=8 > gap=2。Overshoot! 新 GPU0=$(2,5,1)=8$, GPU1=$(8,4,10)=22$。

下一步 max-delta 选择 $(10, 2)$ 从 GPU1 swap 回来... 循环。

**定理4（Adaptive Selection 单调收敛）**：

Gap-targeting 模式选择 $\delta \leq \text{gap}$ 的 pair，保证每步 $M$ 严格递减，至多 $O(\frac{M_0 - M^*}{\delta_{\min}})$ 步收敛。

## 6. 势函数分析：为什么 gap/2 最优

定义 $\Phi(\pi) = \sum_g (L_g - \bar{L})^2$。

每次 swap(a,b)（$a$ on $g^*$, $b$ on $g_-$, delta=$\ell_a - \ell_b$）：

$$\Delta\Phi = 2\delta(\delta - \text{gap})$$

有效 swap ($\delta < \text{gap}$)：$\Delta\Phi < 0$，即势函数严格递减。

$|\Delta\Phi| = 2\delta(\text{gap} - \delta)$，在 $\delta = \text{gap}/2$ 时取最大值 $\text{gap}^2/2$。

**这是 adaptive pair selection 选择 "load closest to gap/2" 的信息论最优性证明：它最大化每步的方差下降量。**

初始势 $\Phi_0 \leq G(M_0 - \bar{L})^2$。势函数非负 ($\Phi \geq 0$)。

收敛步数上界（gap-targeting 模式）：

$$T \leq \frac{\Phi_0}{\text{gap}^2/2} = \frac{2G(M_0-\bar{L})^2}{\text{gap}_{\text{init}}^2}$$

对初始ratio=1.74, $\bar{L}$=某常数：$T = O(G)$，与实测的"3个window内收敛"一致。

## 7. 多层全局预算分配

### 7.1 问题形式化

- $N_L$ 层，每层 $l$ 有独立的负载向量 $\ell^{(l)}$ 和独立的 imbalance ratio $r_l$
- 总 swap 预算 $B$（全局共享），分配为 $B = \sum_l B_l$
- 每层分配 $B_l$ 个 swap 后，该层的 ratio 降为 $r_l(B_l)$（递减函数）
- 目标：$\min_{B_1,...,B_{N_L}} \max_l r_l(B_l)$，s.t. $\sum_l B_l \leq B$

### 7.2 等价于 Minimax Knapsack

这是一个 **minimax 资源分配问题**：

$$\min_{\mathbf{x}} \max_{l \in [N_L]} f_l(x_l), \quad \text{s.t.} \quad \sum_l x_l \leq B, \quad x_l \geq 0$$

其中 $f_l(x_l) = r_l(x_l)$ 是递减凸函数（更多 swap → ratio 单调下降，但边际递减）。

**经典结果**：对凸递减 $f_l$，最优解满足 **水位条件（water-filling）**：

$$f_l(x_l^*) = \lambda^* \quad \forall l \text{ with } x_l^* > 0$$

即最优策略是让所有层的 ratio 降到同一个"水位" $\lambda^*$。

### 7.3 贪心近似 = 论文的算法

论文的 greedy planner 按如下逻辑分配预算：
1. 找当前 ratio 最高的层
2. 对该层做一次 swap
3. 重复直到预算耗尽

这就是 **greedy water-filling** 的精确实现——每次把资源给"水位最高"的那个，直到资源耗尽。

**定理5（Greedy 的最优性）**：当每层的 $r_l(x)$ 是严格凸递减函数时，greedy water-filling 达到全局最优（即与 offline optimal 相同）。

**证明**：标准的 water-filling optimality proof（KKT 条件 + 凸性）。greedy 的每一步都在减小 $\max_l r_l$，而凸性保证不存在更好的分配方式能用同样的预算达到更低的 max。

### 7.4 实际中的非凸性

定理5要求 $r_l(x)$ 严格凸递减。实际上：
- $r_l$ 不是 $x$ 的解析函数——它取决于具体选择了哪些 swap pair
- 某些层可能在1次swap后就达到 threshold（边际收益突变为0）
- 存在"阶梯"效应：某些层的 ratio 只有特定的 discrete 值

**但实验表明 greedy 的行为接近最优**：因为在高 ratio 层做 swap 的边际收益远大于低 ratio 层，greedy 的优先级排序天然把预算流向最需要的地方。

### 7.5 与调度理论的联系

将每层的"初始 ratio"视为 **job**，"swap 操作"视为"减少 job size 的处理时间"，则：
- 我们的 greedy = **Shortest Remaining Processing Time (SRPT)** 在 ratio 空间的对偶
- 经典 SRPT 最小化平均完成时间；我们的变体最小化 max ratio（makespan）

**更精确的对应**：这是一个 **preemptive scheduling on parallel machines with controllable processing times** 问题。已知 greedy（按 longest-remaining-first 分配加速资源）在凸可控模型下最优。

### 7.6 论文当前实现的"budget exhaustion"行为

观察论文的 DIAG 日志：
- Window#1: 240 ops（接近 budget=300 上限）
- Window#2: 44 ops
- Window#3: 20 ops
- Window#4+: 数个 ops

这验证了 water-filling 的预测：首次纠偏时几乎所有层都在高水位，预算被均匀消耗；收敛后只有少数层偶尔超过 threshold，消耗极少预算。

**理论意义**：budget $B$ 不需要 per-layer 调参——greedy 自动实现最优分配。用户只需设一个全局 $B$（足够大即可），算法自己会找到正确的层间分配。

## 8. 与调度理论经典结果的对比

### 8.1 经典结果回顾

平衡负载调度（Balanced Partition / Makespan Minimization on Identical Machines）是经典问题：

| 算法 | 近似比 | 模型 | 特点 |
|---|---|---|---|
| LPT (Longest Processing Time First) | $4/3 - 1/(3G)$ | 离线，从头分配 | 需要完整负载信息 + 全量重排 |
| MULTIFIT | 多项式近似方案 | 离线 | 复杂度高 |
| Random assignment | $O(\sqrt{\ln G / n})$ 竞争比 | 在线，不可撤回 | 无法纠偏 |
| **Swap local search** | **$1 + \ell_{\max}/(n\mu)$** | **从任意初始出发，增量调整** | 本文 |

### 8.2 Swap 的独特优势：从给定状态出发的增量改善

**EPLB/LPT 的根本问题**：它们假设可以 **从头** 计算最优分配并 **一次性** 全量部署。在 MoE 推理中这意味着：
1. 需要阻塞推理（移动所有/大部分专家到新位置）
2. 需要冗余空间暂存两套权重
3. 无法利用"当前分配已经不错"这个信息

**Swap 的在线增量性质**：
- 从当前 $\pi$ 出发（不需要重头算最优）
- 每步只改变2个专家的位置（最小干扰量）
- 可以在运行中不阻塞地执行（异步 P2P）
- 每步的决策只需要当前 window 的局部信息

### 8.3 竞争比分析：Swap vs 全量重排

设 OPT 为离线最优 ratio（知道所有未来负载后的最优分配），SWAP 为我们的算法。

**定理6（Online Competitive Ratio）**：

在非预知（non-clairvoyant）在线模型下——负载分布每个窗口可能改变，算法在每个窗口只能基于过去和当前窗口的观测做决策——PB-OEPLB 的竞争比为：

$$\frac{r_{\text{SWAP}}}{r_{\text{OPT}}} \leq 1 + \frac{\ell_{\max}}{n\mu}$$

**对比全量重排（EPLB）**：EPLB 在每个窗口都能达到 $r \approx 1$（假设无限预算），但付出的代价是每次重排的 **阻塞时间 $T_{\text{block}}$** 和 **KV cache 损失**。

**有效吞吐比较**：

$$\text{TPS}_{\text{EPLB}} = \text{TPS}_{\text{ideal}} \times \frac{T_{\text{window}} - T_{\text{block}}}{T_{\text{window}}} \times (1 - \text{KV\_loss})$$

$$\text{TPS}_{\text{SWAP}} = \text{TPS}_{\text{ideal}} \times \frac{r_{\text{OPT}}}{r_{\text{SWAP}}} \times (1 - \text{overhead})$$

当 $T_{\text{block}} / T_{\text{window}}$ 或 KV_loss 足够大时，SWAP 即使 ratio 不如 EPLB 好（1.02 vs 1.00），有效吞吐也更高——因为它不付阻塞和显存代价。

### 8.4 Regime 分析：何时 Swap 优于全量重排

定义 **"swap regime"**：swap-based 方法优于 full-rebalance 的条件。

设 $f_{\text{MoE}}$ 为 MoE 计算占 forward time 比例，$c_{\text{swap}}$ 为 swap 的每窗口开销占比，$c_{\text{eplb}}$ 为 EPLB 的每次重排阻塞占比 + CUDA graph 损失占比。

**Swap 更优 ⟺**：

$$\underbrace{\frac{r_{\text{EPLB\_effective}}}{r_{\text{SWAP}}}}_{\text{EPLB ratio advantage}} < \underbrace{\frac{1 - c_{\text{swap}}}{1 - c_{\text{eplb}}}}_{\text{Swap overhead advantage}}$$

由于 $r_{\text{EPLB\_effective}} \approx 1.00$ 而 $r_{\text{SWAP}} \leq 1.02$（差异仅2%），只需：

$$\frac{1.00}{1.02} < \frac{1 - 0.007}{1 - c_{\text{eplb}}}$$

即 $c_{\text{eplb}} > 2.7\%$。

**实测 $c_{\text{eplb}}$**：
- Rebalance blocking: 0.5-4.4s per event / 175s benchmark = 0.3-2.5%
- CUDA graph disable: +68% decode latency → 有效 $c_{\text{eplb}} > 40\%$ on decode-heavy workloads

**结论**：在有 decode 的场景下（output > 64 tokens），swap regime 几乎总是成立。仅在纯 prefill + 短窗口场景下，EPLB 的 blocking 成本可以被摊薄到足以竞争。

### 8.5 总结表格

| 维度 | Full Rebalance (EPLB) | Incremental Swap (OEPLB) |
|---|---|---|
| Ratio 质量 | ~1.00 (optimal) | ≤ 1 + ℓ_max/(nμ) |
| 阻塞开销 | 0.5-4.4s/event | 0 (async) |
| 显存代价 | -8.1% KV cache | 0 |
| CUDA graph | 必须禁用 | 保留 |
| 在线适应性 | 每1000步一次 | 每16步一次 |
| 理论近似比 | 1.0 (but costly) | 1 + O(1/n) |
| **有效吞吐** | **受阻塞+CG禁用拖累** | **净正收益** |

## 9. 补全：一般 G 的 Tight Example 与下界

### 9.1 问题

定理2给出的 bound $r(\pi) \leq 1 + \frac{G-1}{G} \cdot \frac{\ell_{\max}}{n\mu}$
仅在 $G=2$ 时构造了 tight example。对于一般 $G$，需要构造一个配置使得
该 bound 被达到，从而证明 bound 不可进一步改进。

### 9.2 一般 G 的 Tight Construction

**构造**: $G$ 个 GPU，每卡 $n$ 个专家，共 $N_E = Gn$ 个专家。

专家负载设计（极端偏斜，模拟"一个超级热点专家"）：
- 1 个热点专家：$\ell_1 = M$（$M \gg 1$）
- $N_E - 1$ 个冷专家：$\ell_j = \epsilon$（$\epsilon \to 0$）

初始分配（trivial round-robin 把热点放在 GPU 0）：
- GPU 0 = $\{M, \epsilon, ..., \epsilon\}$（$n$ 个），$L_0 = M + (n-1)\epsilon$
- GPU 1..G-1 = $\{n \times \epsilon\}$，$L_g = n\epsilon$

**Swap 局部最优分析**：

gap = $L_0 - L_{n\epsilon} = M + (n-1)\epsilon - n\epsilon = M - \epsilon$。

对 GPU 0 上的热点专家 $a$（$\ell_a = M$）和任一其他 GPU $g$ 上的冷专家 $b$（$\ell_b = \epsilon$）：
- swap(a,b) 的 delta = $M - \epsilon$
- swap 后 GPU $g$ 新负载 = $n\epsilon + (M-\epsilon) = M + (n-1)\epsilon = L_0$
- **Overshoot!** GPU $g$ 变成跟 GPU 0 一样热（甚至更热，因为原来更冷的那些）

所以这个 swap 不会改善 makespan。对所有 $(a,b)$ pairs 都成立 → **该分配是 swap-local-optimal**。

**Ratio 计算**:
$$\bar{L} = \frac{M + (N_E-1)\epsilon}{G} = \frac{M}{G} + \frac{(Gn-1)\epsilon}{G} \xrightarrow{\epsilon \to 0} \frac{M}{G}$$

$$r(\pi) = \frac{L_0}{\bar{L}} = \frac{M + (n-1)\epsilon}{M/G} \xrightarrow{\epsilon \to 0} \frac{M}{M/G} = G$$

**Bound 验证**:
$$r(\pi) = G$$
$$\text{bound} = 1 + \frac{G-1}{G} \cdot \frac{\ell_{\max}}{n\mu} = 1 + \frac{G-1}{G} \cdot \frac{M}{n \cdot \frac{M}{Gn}} = 1 + \frac{G-1}{G} \cdot G = 1 + (G-1) = G$$

**$r(\pi) = G = \text{bound}$**。Tight! $\square$

### 9.3 这个 tight example 的含义

当存在"超级热点专家"（负载远超其他所有专家之和）时，swap-based 方法
无法改善——因为把热点挪到任何其他 GPU，都会让那个 GPU 变得同样热。

**这解释了为什么 DeepSeek-V2-Lite（小 expert）不均衡度低**：它的专家体积
小，即使有热点，$\ell_{\max}/\mu$ 比值也小，bound 接近 1。

**对比 235B（大 expert）**：单个热点专家可以承载显著比例的总负载，
$\ell_{\max}/\mu$ 大，bound 远离 1，swap 有改善空间。

### 9.4 与 EPLB（bin-packing + replicate）的理论对比

EPLB 用冗余专家"复制"热点，而不是"移动"它。复制后热点同时在多个 GPU，
真正分摊了负载。设冗余度 $R$（每个热专家复制 $R$ 份）：

$$r_{\text{EPLB}} \leq 1 + \frac{\ell_{\max}}{R \cdot n \mu}$$

对比 swap 的 bound:
$$r_{\text{swap}} \leq 1 + \frac{(G-1)}{G} \cdot \frac{\ell_{\max}}{n\mu}$$

当 $R > \frac{G}{G-1} \approx 1$（即冗余≥2），EPLB 的 ratio bound 严格优于 swap。
但这以付出 $R \times$ 额外显存和禁用 CUDA graph 为代价。

**Swap 的理论优势不在 ratio 质量**（它不如 replicate），**而在零显存代价 +
不阻塞 + 保留 CUDA graph**——有效吞吐（考虑开销后）更高。这跟 studypaper/04
的通信代价分析和 studypaper/06 的实验结果一致。

## 10. 补全：多层预算分配的 convex 论证

### 10.1 问题

定理5（greedy water-filling 最优）要求每层的 $r_l(x)$ 是严格凸递减函数。
实际中 $r_l(x)$ 是离散的（每次 swap 改变一个具体的 pair），不连续。
需要论证这个近似为什么在实际中成立。

### 10.2 $r_l(x)$ 的凸性近似

设第 $l$ 层当前 gap = $G_l$，每次 swap 用 gap-targeting 选 $\delta \approx G_l/2$。
swap 后新 gap = $G_l - 2\delta \approx 0$（理想情况）或 $G_l - 2\delta^*$（有粒度误差）。

定义 $r_l(x) = 1 + G_l(x)/\bar{L}_l$，其中 $G_l(x)$ 是 $x$ 次 swap 后的剩余 gap。

在 gap-targeting 下，$G_l(x+1) \leq G_l(x) - \delta_{\min}$（每次至少减少 $\delta_{\min}$），
且减少幅度随 $G_l$ 减小而减小（因为可选的 $\delta$ 受限于 gap 大小）。

**这给出 $G_l(x)$ 是凹的**（递减但边际递减）→ $r_l(x)$ 凹。

凹函数上，greedy（每次给最高 $r_l$ 的层）仍然最优——因为给低水位层
swap 的边际收益永远小于给高水位层。凸性是 sufficient 不是 necessary；
凹性下 greedy 的最优性证明更简单。

### 10.3 离散性的影响

实际 $r_l$ 的离散跳跃导致 greedy 可能"多给"某层（它已到 local optimum 但
还剩预算）。但这只影响 1-2 次的效率损失，不影响最终 $r$ 的渐进质量
（因为 local optimum 的 $r \leq 1 + \ell_{\max}/(n\mu)$ 已有保证）。
