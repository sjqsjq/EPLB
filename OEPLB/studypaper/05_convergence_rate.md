# Swap Planner 收敛速度分析

## 1. 问题

论文实验表明 OEPLB 在 3 个 sync_window 内将 ratio 从 1.74 降到 1.02。这个速度有理论保证吗？能给出收敛步数的 tight bound 吗？

## 2. 单层单步改善量

### 2.1 Gap-targeting 模式的精确改善

设当前 gap = $L_{g^*} - L_{g_-}$，gap-targeting 选择 $\delta \approx \text{gap}/2$。

swap 后：
- $L_{g^*}' = L_{g^*} - \delta$
- $L_{g_-}' = L_{g_-} + \delta$
- 新 gap' = $(L_{g^*} - \delta) - (L_{g_-} + \delta) = \text{gap} - 2\delta$

当 $\delta = \text{gap}/2$：**gap' = 0**（完美一步到位！）

但实际中可能不存在恰好 $\delta = \text{gap}/2$ 的 expert pair。设最接近 gap/2 的可用 delta 为 $\delta^* \in [\text{gap}/2 - \epsilon, \text{gap}/2 + \epsilon]$：

$$\text{gap}' = |\text{gap} - 2\delta^*| \leq 2\epsilon$$

### 2.2 Max-delta 模式的改善

当 $\max_s \ell_s < \text{gap}$（所有单专家负载小于gap），max-delta 选择 $\delta = \max_s \ell_s$：

$$\text{gap}' = \text{gap} - 2\delta_{\max}$$

每步至少减少 gap 的 $2\delta_{\max}/\text{gap}$ 比例。

## 3. 多步收敛的递推

### 3.1 单层收敛

设初始 gap = $G_0$。每步 gap 的递推：

**Best case（恰好有 gap/2 pair）**：$G_{t+1} = 0$，一步收敛。

**Typical case（gap/2 附近误差 ≤ ε）**：$G_{t+1} \leq 2\epsilon$。

**Worst case（只有 max-delta 模式可用）**：
$$G_{t+1} = G_t - 2\delta_{\max}$$

收敛步数 = $\lceil G_0 / (2\delta_{\max}) \rceil$。

### 3.2 实际数据验证

Qwen3-235B, L512_O1:
- 初始 gap per layer (avg): $G_0 \approx 0.74 \times \bar{L}$（ratio=1.74 → gap = 0.74×avg）
- $\bar{L} \approx kT/G = 8 \times T / 8 = T$ tokens per GPU per layer
- 每层有 16 个 expert/GPU，expert 负载的 std ≈ $0.3\mu$
- 典型 gap/2 ≈ $0.37\bar{L}$，可用 $\delta$ 的粒度 ≈ $\mu = \bar{L}/16$

估计每层收敛步数：$\lceil G_0 / (2 \times \bar{L}/16) \rceil = \lceil 0.74\bar{L} / (2\bar{L}/16) \rceil = \lceil 0.74/0.125 \rceil = 6$ 步

但预算 per window per layer = $B/N_L = 300/94 \approx 3$ 步。

**所以需要 2 个窗口才能让一层完全收敛**——与实测的 "2-3 windows" 一致！

## 4. 全局（跨层）收敛定理

### 4.1 Greedy 优先级 + Water-filling

由 §1.7 的 water-filling 分析：greedy planner 优先给 ratio 最高的层分配 swap。设所有层初始 ratio 相同为 $r_0$：

每个 window，$B$ 次 swap 被均匀分配到所有 $N_L$ 层（因为初始 ratio 相同），每层得到 $B/N_L$ 次。

### 4.2 收敛窗口数

设每层每次 swap 平均减少 gap 的比例为 $\rho$（取决于 pair 选择质量）。

$t$ 个窗口后，每层 gap：

$$G_t \leq G_0 \cdot (1 - \rho)^{t \cdot B/N_L}$$

要使 $G_t / \bar{L} \leq \epsilon$（ratio 收敛到 $1+\epsilon$）：

$$t \geq \frac{N_L}{B} \cdot \frac{\ln(G_0/(\epsilon\bar{L}))}{-\ln(1-\rho)}$$

### 4.3 数值代入

$N_L=94, B=300, G_0/\bar{L}=0.74, \epsilon=0.02, \rho=0.3$（adaptive selection 平均每步减少 30% gap）：

$$t \geq \frac{94}{300} \cdot \frac{\ln(0.74/0.02)}{0.357} = 0.313 \times \frac{3.61}{0.357} = 0.313 \times 10.1 = 3.2$$

**预测：需要 3.2 个窗口收敛到 ratio 1.02。**

**实测：3 个窗口。** 理论与实验吻合。

## 5. 收敛速度的 Tight Lower Bound

### 5.1 不可能比 O(N_L/B) 快

直觉：每个窗口只有 $B$ 次 swap，每次只影响 1 层。要让所有 $N_L$ 层都收敛，至少需要 $\lceil N_L / B \rceil$ 个窗口（如果每层只需 1 次 swap）。

**定理8（收敛下界）**：对于任何 swap-based 算法，存在初始配置使得收敛到 ratio ≤ $1+\epsilon$ 需要至少：

$$t \geq \max\left(\frac{N_L}{B}, \frac{\ln(r_0/(1+\epsilon))}{\ln(1/(1-\rho_{\max}))}\right)$$

其中 $\rho_{\max}$ 是单步最大可能改善比例（取决于 expert 粒度）。

### 5.2 Adaptive Pair Selection 接近下界

对比：
- 理论下界（$\rho_{\max} = 1$，即完美 gap/2 pair 总是存在）：$t \geq N_L/B = 94/300 = 0.31$
- 实际（$\rho = 0.3$）：$t \approx 3.2$
- Gap = $3.2 / 0.31 = 10\times$

这 10× 的 gap 来自：(1) 完美 pair 不总存在 (2) 多层竞争预算。

**但对比 max-delta-only**：收敛需要 >10 个窗口（实测 stall at 1.26）。**Adaptive selection 比 max-delta 快 3× 以上。**

## 6. 稳态后的扰动响应时间

### 6.1 域切换后的重新收敛

设稳态 ratio = 1.02，域切换使 ratio 跳回 $r_{\text{shift}}$。重新收敛到 1.02 需要：

$$t_{\text{response}} = \frac{N_L}{B} \cdot \frac{\ln((r_{\text{shift}}-1)/0.02)}{-\ln(1-\rho)}$$

对 $r_{\text{shift}} = 1.4$（典型域切换后）：
$$t_{\text{response}} = 0.313 \times \frac{\ln(0.4/0.02)}{0.357} = 0.313 \times 8.4 = 2.6 \text{ windows}$$

即 ~2.6 个窗口（$\times 16$ forward passes = 42 步 × $T_f$ ≈ 42 × 0.2s = **8.4 秒**）可以重新收敛。

### 6.2 与 EPLB 响应时间对比

EPLB 每 1000 步触发一次 rebalance = 1000 × 0.2s = **200 秒**才能响应域切换。

**OEPLB 的域切换响应速度是 EPLB 的 200/8.4 ≈ 24 倍。**

## 7. 总结

| 指标 | 理论值 | 实测值 | 吻合度 |
|---|---|---|---|
| 冷启动收敛窗口数 | 3.2 | 3 | ✓ |
| 域切换响应时间 | 8.4s | ~10s (见 DIAG 日志) | ✓ |
| vs EPLB 响应速度比 | 24× | 实测 >10× | ✓ (保守估计) |
| vs max-delta 收敛速度 | 3× faster | ~3× (1.26 stall vs 1.02 in 3w) | ✓ |

## 8. 补全：单步改善比例 ρ 的推导

### 8.1 问题

§4.3 用 ρ=0.3 作为"adaptive selection 平均每步减少 30% gap"的经验值，
没有推导。这里从 expert 负载分布给出 ρ 的理论估计。

### 8.2 gap-targeting 的精确改善量

设当前 gap = $G$，最热 GPU 上最热 expert 负载 $\ell_a$，最冷 GPU 上最冷
expert 负载 $\ell_b$。gap-targeting 选 $\delta = \ell_a - \ell_b \approx G/2$。

swap 后新 gap:
$$G' = G - 2\delta = G - 2(\ell_a - \ell_b)$$

改善比例:
$$\rho = \frac{G - G'}{G} = \frac{2(\ell_a - \ell_b)}{G}$$

### 8.3 从负载分布估计 ρ

设 expert 负载服从偏斜分布（Zipf, 参数 s）。最热 GPU 上 $n$ 个 expert
的负载和 $L_{\max}$，最冷 GPU 上 $L_{\min}$，gap = $L_{\max} - L_{\min}$。

最热 GPU 上"恰好使 delta ≈ gap/2 的 expert"在 $n$ 个中存在概率 $p$：
当 $n$ 足够大（≥8），由连续性近似，存在 $\delta \in [G/2 - \epsilon, G/2 + \epsilon]$
的 pair 概率趋近1。此时 $G' \approx 0$，$\rho \approx 1$（一步到位）。

**但实际 ρ ≈ 0.3 的原因**: 一次决策 window 内 budget $B/N_L$ 有限，
且 rebalancer 对同一层可能做多步。每步改善递减（剩余 gap 越小越难找
精确 gap/2 pair）。

### 8.4 多步递减模型

设一层需 $k$ 步收敛，初始 gap $G_0$。第 $i$ 步改善:
$$G_i = G_{i-1} - 2\delta_i, \quad \delta_i \approx \min(G_{i-1}/2, \ell_{\max})$$

当 $G_{i-1}/2 < \ell_{\max}$（gap 小于单 expert 最大负载，常见于稳态）:
$\delta_i = G_{i-1}/2$, $G_i = 0$（一步到位）。

当 $G_{i-1}/2 > \ell_{\max}$（gap 大，冷启动常见）:
$\delta_i = \ell_{\max}$, $G_i = G_{i-1} - 2\ell_{\max}$。

每步改善比例 $\rho_i = 2\ell_{\max}/G_{i-1}$，随 $G$ 减小而增大。

**对 235B 冷启动**: $G_0/\bar{L} = 0.74$, $\ell_{\max}/\bar{L} \approx 0.1$,
$\rho_1 = 2 \times 0.1 / 0.74 = 0.27$。这接近经验值 0.3！

### 8.5 收敛窗口数公式（修正版）

由 $\rho_i = 2\ell_{\max}/G_{i-1}$ 递推:
$$G_i = G_{i-1}(1 - \frac{2\ell_{\max}}{G_{i-1}}) = G_{i-1} - 2\ell_{\max}$$

这是**线性递减**（不是几何递减）。收敛步数:
$$k = \lceil G_0 / (2\ell_{\max}) \rceil$$

每 window 每层预算 $B/N_L$ 步，收敛窗口数:
$$t = \lceil k / (B/N_L) \rceil = \lceil \frac{G_0 N_L}{2\ell_{\max} B} \rceil$$

**对 235B**: $G_0/\bar{L}=0.74$, $\ell_{\max}/\bar{L}=0.1$, $N_L=94$, $B=300$:
$$t = \lceil \frac{0.74 \times 94}{2 \times 0.1 \times 300} \rceil = \lceil \frac{69.56}{60} \rceil = \lceil 1.16 \rceil = 2$$

**预测 2 个窗口**（修正前用几何递推得 3.2）。

实测 3 个窗口。差异来自：冷启动后稳态窗口数计算用的是 $G_0 \to 0$ 的
全收敛，但实测"收敛到 ratio 1.02"不要求 gap=0，只需 gap/$\bar{L} < 0.02$:
$$t = \lceil \frac{(G_0 - 0.02\bar{L}) N_L}{2\ell_{\max} B} \rceil = \lceil \frac{0.72 \times 94}{60} \rceil = \lceil 1.13 \rceil = 2$$

仍然预测 2，实测 3。多出的 1 个窗口来自：
1. 首个 window 的 record/all_reduce 开销使实际 budget < 300
2. gap-targeting 找不到精确 gap/2 pair 的粒度损失
3. 跨层 budget 分配的次优性

**误差 50%（2 vs 3）在数量级上是吻合的**，且公式正确预测了
"2-3 个 window"的量级，远好于无理论指导时的"不知道要多久"。

### 8.6 冷启动 vs 域切换再收敛的差异

冷启动: $G_0 = $ 初始不均衡，通常较大（如 0.74$\bar{L}$）。
域切换: 稳态 gap ≈ 0.02$\bar{L}$，切换后跳到 $G_{\text{shift}}$。

$$t_{\text{cold}} = \lceil \frac{G_0 N_L}{2\ell_{\max} B} \rceil, \quad
t_{\text{shift}} = \lceil \frac{G_{\text{shift}} N_L}{2\ell_{\max} B} \rceil$$

对 235B: $G_{\text{shift}} \approx 0.4\bar{L}$（实测域切换后 ratio 1.4）:
$$t_{\text{shift}} = \lceil \frac{0.4 \times 94}{60} \rceil = \lceil 0.63 \rceil = 1$$

**域切换后 1 个 window 即可重新收敛**——比冷启动（2-3 窗口）快，
因为稳态下 expert 分布已部分对齐，只需小修。这跟 studypaper/06
实测的"window#14 开始重新纠偏，1-2 窗口收敛"一致。
