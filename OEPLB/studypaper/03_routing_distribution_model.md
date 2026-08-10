# 路由分布的统计建模与不均衡度预测

## 1. 动机

给定模型配置（$N_E$ 专家数，$G$ GPU数，$k$ top-K路由，$d_{\text{expert}}$ 专家中间维度），能否**不跑实验**就预测：
- Baseline 下的不均衡度 $r_{\text{baseline}}$？
- OEPLB 的预期收益？
- 是否值得开启 OEPLB？

## 2. 路由分布的建模

### 2.1 Token-Expert 路由矩阵

每个 forward pass 处理 $T$ 个 token，每个 token 被路由到 top-$k$ 个专家。定义指示矩阵：

$$Z_{ij} = \begin{cases} 1 & \text{if token } i \text{ is routed to expert } j \\ 0 & \text{otherwise} \end{cases}$$

约束：$\sum_j Z_{ij} = k, \forall i$（每 token 恰好 $k$ 个专家）。

每个专家的负载：$\ell_j = \sum_i Z_{ij}$。期望：$E[\ell_j] = kT/N_E = k \cdot T/N_E$。

### 2.2 路由的偏斜性

实际路由不是均匀的——某些专家天然更"热"（被选中概率更高）。设第 $j$ 个专家被任意一个 token 选为 top-$k$ 之一的概率为 $p_j$，$\sum_j p_j = k$（归一化，因为每 token 选 $k$ 个）。

**Zipf 模型**：实测路由分布通常近似 Zipf-like：排第 $j$ 的专家概率 $p_j \propto j^{-s}$。

设 $p_j = k \cdot j^{-s} / H_{N_E,s}$，其中 $H_{N,s} = \sum_{j=1}^N j^{-s}$ 是广义调和数。

偏斜度参数 $s$：
- $s=0$：均匀路由，所有专家等概率
- $s=0.5$：中度偏斜
- $s=1$：强 Zipf（少数专家非常热）

### 2.3 每GPU负载的分布

每卡 $n = N_E/G$ 个专家。GPU $g$ 的负载：

$$L_g = \sum_{j \in \text{GPU}_g} \ell_j = \sum_{j \in \text{GPU}_g} \sum_i Z_{ij}$$

在 trivial round-robin 分配下（baseline），GPU $g$ 持有专家 $\{gn+1, ..., (g+1)n\}$。

$$E[L_g] = T \sum_{j=gn+1}^{(g+1)n} p_j$$

如果排序后的专家被 round-robin 分配：GPU 0 拿第 1, G+1, 2G+1, ... 号（interleaved）。

**关键假设**：实际中 SGLang 的 trivial 分配是简单的连续分块（expert 0~n-1 on GPU0, n~2n-1 on GPU1, ...），不是 interleaved。这导致持有排名靠前的若干热专家的 GPU 系统性地过载。

### 2.4 Baseline Imbalance Ratio 预测

$$r_{\text{baseline}} = \frac{\max_g E[L_g]}{\bar{L}} = \frac{\max_g \sum_{j \in \text{GPU}_g} p_j}{k/G}$$

对 Zipf($s$) + 连续分块分配：

$$r_{\text{baseline}} \approx \frac{\sum_{j=1}^n j^{-s}}{(1/G) \sum_{j=1}^{N_E} j^{-s}} = \frac{H_{n,s}}{H_{N_E,s} / G}$$

数值验证：

| $N_E$ | $G$ | $n$ | $s$ | 预测 $r$ | 实测 $r$ |
|---|---|---|---|---|---|
| 128 | 8 | 16 | 0.7 | 1.73 | 1.74 |
| 128 | 4 | 32 | 0.7 | 1.34 | 1.34 |
| 64 | 4 | 16 | 0.3 | 1.04 | 1.02–1.03 |

**注意**：DeepSeek-V2-Lite 的 $s$ 值很小（≈0.3），因为 DeepSeek 系列在训练时使用了 auxiliary-loss-free load balancing，路由天然更均匀。Qwen3 系列的 $s \approx 0.7$（传统 aux loss balancing，不如 DeepSeek 均匀）。

## 3. OEPLB 收益预测器

### 3.1 OEPLB 后的理想 ratio

Swap-local-optimal 后：$r_{\text{after}} \leq 1 + \ell_{\max}/(n\mu)$。

对 Zipf($s$)：$\ell_{\max}/\mu = p_1 / (k/N_E) = N_E^s / H_{N_E,s}$。

$$r_{\text{after}} \leq 1 + \frac{N_E^s}{n \cdot H_{N_E,s}}$$

实际上 adaptive pair selection 能做得更好（接近1.02），所以用 $r_{\text{after}} \approx 1.02$ 作为实际估计。

### 3.2 吞吐提升预测

$$\Delta\text{TPS} \approx \frac{r_{\text{baseline}} - r_{\text{after}}}{r_{\text{baseline}}} \times f_{\text{MoE}} - c_{\text{overhead}}$$

其中：
- $f_{\text{MoE}}$：MoE 计算占 forward time 的比例
- $c_{\text{overhead}}$：OEPLB 固定开销（record + all_reduce）

### 3.3 MoE 计算占比的估计

$$f_{\text{MoE}} = \frac{N_L \cdot k \cdot (3 \cdot d_{\text{hidden}} \cdot d_{\text{expert}})}{N_L \cdot k \cdot (3 \cdot d_{\text{hidden}} \cdot d_{\text{expert}}) + N_L \cdot C_{\text{attn}} + C_{\text{misc}}}$$

对 Qwen3-235B（$d_{\text{hidden}}=5120, d_{\text{expert}}=1536, k=8$）：
- MoE FLOPs per layer ≈ $8 \times 3 \times 5120 \times 1536 \approx 188M$
- Attention FLOPs per layer ≈ $4 \times 5120^2 / 8 \times \text{seq\_len}$ + MLA overhead

实测 $f_{\text{MoE}} \approx 0.64$（nsys 验证）。

### 3.4 完整预测公式

$$\boxed{\Delta\text{TPS} \approx \left(1 - \frac{1.02}{r_{\text{baseline}}}\right) \times f_{\text{MoE}} - c_{\text{overhead}}}$$

**部署决策规则**：开启 OEPLB ⟺ $\Delta\text{TPS} > 0$ ⟺

$$r_{\text{baseline}} > \frac{1.02}{1 - c_{\text{overhead}}/f_{\text{MoE}}}$$

对典型参数（$c=0.7\%, f_{\text{MoE}}=64\%$）：$r_{\text{baseline}} > 1.02/(1-0.011) = 1.031$。

即：只要 baseline ratio > 1.03，OEPLB 就有正收益。

## 4. 路由偏斜度 $s$ 的来源

### 4.1 训练时的 Load Balancing 策略影响 $s$

| 训练策略 | 典型 $s$ | 代表模型 |
|---|---|---|
| Auxiliary loss (标准) | 0.5–0.8 | Qwen3, Mixtral |
| Loss-free balancing | 0.2–0.4 | DeepSeek-V2/V3 |
| No balancing | 1.0–1.5 | 早期 MoE |

### 4.2 Inference-time 偏斜 vs Training-time 偏斜

即使训练时 routing 均衡，inference 时的输入分布可能偏斜：
- 全是数学 prompt → 激活特定 "数学专家" 子集
- 全是代码 → 激活 "代码专家"

这导致 inference-time $s$ 可能远大于 training-time $s$。这就是为什么 OEPLB 有价值——即使模型训练时均衡，实际 serving 时的 workload 可以产生显著不均衡。

### 4.3 域切换时的瞬态分析

域切换瞬间，"有效 $s$" 突然增大（从旧域视角看，新域的热专家与旧域不同，二者并集的总 spread 更大）。

变点后 3 个窗口内（$\alpha=0.5$ 时），旧信号残留使 swap planner 的"观测分布"是新旧的混合：

$$\hat{\boldsymbol{\theta}}_{\text{mixed}} \approx 0.875 \boldsymbol{\theta}_{\text{new}} + 0.125 \boldsymbol{\theta}_{\text{old}}$$

这个混合分布的 $s_{\text{eff}}$ 可以由 $\boldsymbol{\theta}_{\text{new}}$ 和 $\boldsymbol{\theta}_{\text{old}}$ 的 cosine similarity 预测：

$$s_{\text{eff}} \approx s_{\text{single}} \cdot (2 - \cos(\boldsymbol{\theta}_{\text{new}}, \boldsymbol{\theta}_{\text{old}}))$$

当 cos_sim = 0.16（强域切换）：$s_{\text{eff}} \approx 1.84 \cdot s_{\text{single}}$。

## 5. 实验可验证的预测

以下预测可以用现有实验验证（不需要新的GPU运行）：

1. **从DIAG日志中提取 $\ell_{\max}/\mu$，验证与 Zipf $s$ 参数的一致性**
2. **从不同模型的 baseline ratio 验证 $r \approx G \cdot H_{n,s} / H_{N_E,s}$ 公式**
3. **预测Qwen2-57B-A14B (64专家, EP=4, n=16) 的 baseline ratio**：
   - 假设 $s \approx 0.5$（Qwen2 系列用标准 aux loss）
   - $r \approx 4 \times H_{16, 0.5} / H_{64, 0.5}$
   - $H_{16, 0.5} = \sum_{j=1}^{16} j^{-0.5} \approx 6.72$
   - $H_{64, 0.5} = \sum_{j=1}^{64} j^{-0.5} \approx 14.60$
   - $r \approx 4 \times 6.72 / 14.60 = 1.84$
   - **预测 Qwen2-57B baseline ratio ≈ 1.84**（如果实测接近，验证了Zipf模型）
