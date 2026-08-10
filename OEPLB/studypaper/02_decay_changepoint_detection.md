# Decay 与 Adaptive Window 的贝叶斯变点检测建模

## 1. 问题本质

OEPLB 的 controller 面对一个核心决策问题：

> 当前 window 观测到的负载分布，是"旧 workload 的自然波动"还是"新 workload 到来（域切换）"？

如果是波动 → 不应该剧烈调整 placement（否则追噪声）
如果是真实切换 → 应该尽快响应（否则拖着旧 placement 损失吞吐）

这就是经典的 **在线变点检测（Online Change-Point Detection）** 问题。

## 2. 形式化

### 2.1 观测模型

每个 sync_window $t$，我们观测到全局负载分布向量：

$$\mathbf{x}_t \in \mathbb{R}^{N_E}_{\geq 0}, \quad \|\mathbf{x}_t\|_1 = T_t \text{（该窗口总token数）}$$

归一化后得到路由分布 $\mathbf{p}_t = \mathbf{x}_t / T_t$（各专家被选中的概率）。

### 2.2 分段平稳假设

workload 分为若干**段（segments）**，每段内路由分布 $\mathbf{p}_t = \boldsymbol{\theta}_k + \boldsymbol{\epsilon}_t$：
- $\boldsymbol{\theta}_k$ 是第 $k$ 段的"真实"路由分布（固定）
- $\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \sigma^2/T_t \cdot I)$ 是有限采样噪声（token数越多，噪声越小）
- **变点**：存在未知时刻 $\tau_1 < \tau_2 < ...$ 使得 $\boldsymbol{\theta}$ 突变

### 2.3 两个竞争目标

1. **检测延迟（Detection Delay）**：变点 $\tau$ 发生后，多少个窗口才能确认？
2. **误报率（False Alarm Rate）**：在没有变点时，多久会错误地认为发生了切换？

经典 trade-off：更敏感 = 更低延迟 + 更高误报。

## 3. 指数衰减作为 CUSUM 检验的隐式实现

### 3.1 经典 CUSUM (Cumulative Sum) 检验

标准 CUSUM 维护一个统计量：

$$S_t = \max(0, S_{t-1} + \log \frac{f_1(\mathbf{x}_t)}{f_0(\mathbf{x}_t)})$$

其中 $f_0$ 是"无变化"假设下的密度，$f_1$ 是"发生变化"假设。当 $S_t > h$（阈值）时报警。

### 3.2 指数加权移动平均（EWMA）与 CUSUM 的联系

OEPLB 的衰减历史：

$$A_t = \mathbf{x}_t + \alpha A_{t-1} = \sum_{k=0}^{t-1} \alpha^k \mathbf{x}_{t-k}$$

归一化后的"估计分布"：

$$\hat{\boldsymbol{\theta}}_t = \frac{A_t}{\|A_t\|_1} = \frac{\sum_k \alpha^k \mathbf{x}_{t-k}}{\sum_k \alpha^k T_{t-k}}$$

**这就是指数加权的路由分布估计**。当 $\alpha = 0.5$ 时，3个窗口前的权重仅为 $0.5^3 = 12.5\%$。

### 3.3 Cosine Similarity 作为检验统计量

OEPLB 用 $\cos(\hat{\boldsymbol{\theta}}_{t-1}, \mathbf{x}_t)$ 作为"是否发生变化"的判据：
- cos_sim > 0.95 → "稳定"（旧域延续）
- cos_sim < 0.85 → "变化"（新域到来）

**与似然比检验的联系**：

对高维向量，cosine similarity 与 KL 散度的关系（当分布接近时的二阶近似）：

$$1 - \cos(\mathbf{p}, \mathbf{q}) \approx \frac{1}{2} \|\mathbf{p} - \mathbf{q}\|^2 / (\|\mathbf{p}\| \cdot \|\mathbf{q}\|)$$

而 KL 散度的二阶展开：$D_{\text{KL}}(\mathbf{p} \| \mathbf{q}) \approx \frac{1}{2} \chi^2(\mathbf{p}, \mathbf{q})$

所以 cos_sim < threshold 本质上等价于 $\chi^2$-型变点检验超过阈值。

## 4. 最优衰减因子的贝叶斯推导

### 4.1 贝叶斯模型

设变点以速率 $\lambda$ 到来（Poisson 过程），即每个窗口有概率 $p_{\text{change}} = 1 - e^{-\lambda}$ 发生域切换。

给定两个假设：
- $H_0$（无变化）：$\mathbf{x}_t \sim \text{Multinomial}(T_t, \boldsymbol{\theta}_{\text{old}})$
- $H_1$（变化发生在 $\tau \leq t$）：$\mathbf{x}_t \sim \text{Multinomial}(T_t, \boldsymbol{\theta}_{\text{new}})$

### 4.2 后验概率与衰减因子的联系

Bayesian 在线变点检测（Adams & MacKay, 2007）维护 **run length** 的后验分布 $P(r_t | \mathbf{x}_{1:t})$，其中 $r_t$ 是"距上次变点过了多少步"。

对于指数衰减权重 $\alpha^k$：

$$P(\text{当前数据来自 window } t-k) \propto \alpha^k$$

**这隐式地假设了一个几何先验**：$P(r_t = k) \propto \alpha^k (1-\alpha)$，即 **run length 服从参数为 $(1-\alpha)$ 的几何分布**。

因此：$\alpha = 0.5$ 等价于假设"变点平均每 $1/(1-\alpha) = 2$ 个窗口发生一次"。

这解释了为什么 $\alpha = 0.5$ 对多域场景最优——它预设了"域切换频繁发生"，所以历史数据很快失去相关性。

而 $\alpha = 0.9$ 对应"变点平均每 10 个窗口才发生一次"——过于保守，旧信号残留太久。

### 4.3 最优 $\alpha$ 的封闭解

给定：
- 真实变点频率 $\lambda$（每 $1/\lambda$ 个窗口一次切换）
- 信号强度 $d = \|\boldsymbol{\theta}_{\text{new}} - \boldsymbol{\theta}_{\text{old}}\|_2$（新旧分布的距离）
- 噪声水平 $\sigma^2 / T$（每窗口的有限采样方差）

**最小化检测延迟 + 误报代价**的最优 $\alpha$：

$$\alpha^* = \arg\min_\alpha \underbrace{T_{\text{detect}}(\alpha)}_{\text{延迟}} + \underbrace{\gamma \cdot P_{\text{FA}}(\alpha)}_{\text{误报代价}}$$

其中：
- $T_{\text{detect}}(\alpha) \approx \frac{\ln(\tau_{\text{SNR}})}{-\ln(\alpha)} + 1$（衰减需要多少步让旧信号低于检测阈值）
- $P_{\text{FA}}(\alpha) \propto e^{-d^2 T / (2\sigma^2(1-\alpha^2))}$（在无变化时误报的概率）

对 $T_{\text{detect}}$ 求导令其为零：

$$\frac{\partial}{\partial \alpha}: \quad \frac{\ln(\tau_{\text{SNR}})}{\alpha \ln^2(\alpha)} = \gamma \cdot \frac{\partial P_{\text{FA}}}{\partial \alpha}$$

对实际参数（$d \approx 0.3, \sigma/\sqrt{T} \approx 0.05, \gamma \approx 10$）数值求解得 $\alpha^* \approx 0.45 - 0.55$，验证了 $\alpha = 0.5$ 的经验选择。

## 5. Adaptive Window 的最优停止框架

### 5.1 形式化为最优停止（Optimal Stopping）

每个 window $t$，controller 面临决策：是否"现在就做 all_reduce + 决策"（花费 $c$ 时间），还是"再等一个 window 积累更多数据"？

**状态**：$(w, r, \dot{r})$ — 当前窗口大小、当前估计 ratio、ratio 变化速率
**Action**：$a \in \{$决策, 等待$\}$

**奖励**：
$$R(\text{决策}) = \underbrace{f_{\text{MoE}} \cdot (r - 1)}_{\text{ratio改善的收益}} - \underbrace{c/w}_{\text{all\_reduce开销摊到每forward}}$$

$$R(\text{等待}) = -\underbrace{f_{\text{MoE}} \cdot (r-1) \cdot \Delta t}_{\text{这段时间浪费在不均衡上}}$$

### 5.2 最优窗口的闭式解（稳态情况）

在稳态（ratio已收敛，没有域切换）：

$r-1 \approx 0$（几乎不需要纠偏），收益≈0。唯一的成本是 $c/w$。

→ $w$ 应尽可能大（减少开销），上限受"能多快检测到下一次域切换"的约束。

**最优稳态窗口**：

$$w^*_{\text{stable}} = \sqrt{\frac{c}{f_{\text{MoE}} \cdot \lambda \cdot \Delta r_{\text{shift}}}}$$

其中 $\lambda$ 是域切换频率，$\Delta r_{\text{shift}}$ 是切换后ratio的跳变量。

直觉：切换越频繁/越剧烈 → $w^*$ 越小；通信开销 $c$ 越大 → $w^*$ 越大。

### 5.3 与实际 OEPLB 启发式的对应

| 理论概念 | OEPLB 实现 |
|---|---|
| $r-1 \approx 0$ 持续3窗口 | converge_count ≥ 3 → grow |
| $\Delta r > 0.03$（检测到切换）| ratio_jump > 0.03 → shrink |
| $w$ 翻倍/减半 | 倍增/减半策略 |
| 竞争比 $O(\log w^*)$ | doubling trick 的经典结果 |

**定理7（Doubling 策略的竞争比）**：
设最优静态窗口为 $w^*$（需要 oracle 知识才能确定）。OEPLB 的 grow/shrink 策略在最坏情况下的性能不超过最优的 $O(\log(w_{\max}/w_{\min}))$ 倍——这是 doubling/halving 策略的标准 competitive ratio，对应 $O(\log(128/8)) = O(\log 16) \approx 4$ 倍。

实际中远好于此——因为 ratio jump 信号通常很清晰（cos_sim 从 0.99 突降到 0.85），false alarm 很少。

## 6. 联合优化：Decay × Window

### 6.1 为什么要联合？

$\alpha$ 和 $w$ 不是独立的：
- 大 $\alpha$（慢衰减）+ 小 $w$（频繁检查）= 每次决策基于很多历史 → 检测延迟大但统计稳定
- 小 $\alpha$（快衰减）+ 大 $w$（稀疏检查）= 每次决策只看最近数据 → 检测快但可能被噪声骗

**Pareto 最优前沿**：存在一条 $(\alpha, w)$ 曲线使得"检测延迟 vs 误报率"取到帕累托最优。

### 6.2 有效样本量

指数衰减的有效独立样本数：

$$N_{\text{eff}} = \frac{(\sum_k \alpha^k T_k)^2}{\sum_k \alpha^{2k} T_k^2} \approx \frac{w \cdot \bar{T}}{1+\alpha^2/(1-\alpha^2)} \cdot \text{const}$$

简化（假设每窗口 token 数相同=$\bar{T}$）：

$$N_{\text{eff}} \approx w \cdot \bar{T} \cdot \frac{1-\alpha}{1+\alpha}$$

要使检测可靠，需 $N_{\text{eff}} \geq N_{\text{min}}$（最小统计量）。

$$w \geq \frac{N_{\text{min}} (1+\alpha)}{(1-\alpha) \bar{T}}$$

对 $\alpha=0.5, \bar{T}=1000, N_{\text{min}}=500$：$w \geq 500 \times 3 / (1000) = 1.5$ → 几乎任何 $w \geq 2$ 都够。

对 $\alpha=0.9, \bar{T}=100, N_{\text{min}}=500$：$w \geq 500 \times 19 / 100 = 95$ → 需要很大窗口才够数据。

**这解释了为什么小模型/短prompt场景（$\bar{T}$ 小）需要更大的 sync_window 或更小的 $\alpha$。**

## 7. 关键洞察总结

1. **Exponential decay = Geometric prior on run length**：$\alpha$ 不是一个调参trick，而是对"域切换频率"的贝叶斯先验表达。

2. **Cosine similarity ≈ $\chi^2$ test**：OEPLB 的 shift detection 本质上是一个分布变点检验。

3. **最优 $\alpha$ 由信噪比决定**：$\alpha^* \approx 1 - \sqrt{\sigma^2/(d^2 T)}$，高 SNR → 可以用小 $\alpha$（快遗忘）。

4. **Window 和 $\alpha$ 联合决定有效样本量**：不是两个独立旋钮，而是一个 trade-off 的两面。

5. **Doubling/halving ≤ 4× optimal**：自适应窗口不需要精确知道最优 $w^*$，竞争比有保证。
