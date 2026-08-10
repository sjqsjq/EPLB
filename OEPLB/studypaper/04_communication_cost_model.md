# Swap 通信代价模型：与全量重排的数量级优势

## 1. 动机

论文声称"swap 是轻量的、非阻塞的"，但没有定量分析：
- 一次 swap 到底移动多少数据？
- $B$ 次 swap 的总通信量 vs 一次 EPLB 全量重排的通信量？
- 异步 P2P 的延迟隐藏率是多少？

## 2. 单次 Swap 的通信量

### 2.1 Expert 权重结构

MoE 中每个 expert 通常包含 3 个矩阵（gate/up/down proj）：
- Gate: $(d_{\text{hidden}}, d_{\text{expert}})$
- Up: $(d_{\text{hidden}}, d_{\text{expert}})$
- Down: $(d_{\text{expert}}, d_{\text{hidden}})$

总参数量 per expert：$3 \cdot d_{\text{hidden}} \cdot d_{\text{expert}}$

数据量（FP8）：$W_{\text{expert}} = 3 \cdot d_{\text{hidden}} \cdot d_{\text{expert}} \cdot 1 \text{ byte}$

| 模型 | $d_{\text{hidden}}$ | $d_{\text{expert}}$ | $W_{\text{expert}}$ (FP8) | $W_{\text{expert}}$ (BF16) |
|---|---|---|---|---|
| Qwen3-235B | 5120 | 1536 | 23.6 MB | 47.2 MB |
| Qwen3-30B | 2048 | 1536 | 9.4 MB | 18.9 MB |
| Qwen2-57B | 2048 | 2560 | 15.7 MB | 31.5 MB |
| DeepSeek-V2-Lite | 2048 | 1408 | 8.6 MB | 17.3 MB |

### 2.2 单次 Swap 通信

Swap(a, b) 在两个 rank 间交换两个 expert 的权重：
- Rank A → Rank B：发送 expert $a$ 的权重 = $W_{\text{expert}}$
- Rank B → Rank A：发送 expert $b$ 的权重 = $W_{\text{expert}}$

**总通信量 per swap = $2 \cdot W_{\text{expert}}$**

对 Qwen3-235B (FP8)：$2 \times 23.6 = 47.2$ MB/swap。

### 2.3 一个 Window 的总 Swap 通信

典型 cold-start window：$B \approx 240$ ops（实测 Window#1）。
$$C_{\text{swap}} = 240 \times 47.2 \text{ MB} = 11.3 \text{ GB}$$

但注意：这 240 个 swap 分布在 94 层上，每层约 2-3 个 swap，且跨不同 rank pair。**NVLink 的双向带宽为 ~450 GB/s**，所以：

$$T_{\text{swap}} = \frac{11.3 \text{ GB}}{450 \text{ GB/s}} \approx 25 \text{ ms}$$

实测 Window#1 的 P2P 传输耗时：~255 ms。差异来自：
- batch_isend_irecv 的 setup overhead（每次 NCCL P2P 调用的固定开销）
- 多个 rank-pair 之间的串行化（NCCL comm 只有一个）
- GPU kernel launch 和 event synchronization

## 3. 全量重排（EPLB）的通信量

### 3.1 EPLB 重排的最坏情况

EPLB 可能改变所有层所有 expert 的位置。最坏情况：每层每个 expert 都被移动到不同的 GPU。

$$C_{\text{EPLB\_worst}} = N_L \times N_E \times W_{\text{expert}} = 94 \times 128 \times 23.6 \text{ MB} = 283.8 \text{ GB}$$

### 3.2 EPLB 实际通信量

实际上 EPLB 不会移动所有 expert（很多已经在正确位置）。设实际变动比例为 $f_{\text{move}}$：

$$C_{\text{EPLB}} = f_{\text{move}} \times N_L \times N_E \times W_{\text{expert}}$$

实测（从 EPLB 日志）：$f_{\text{move}} \approx 0.3$（约30%的 expert 被移动），
$$C_{\text{EPLB}} \approx 0.3 \times 283.8 = 85.1 \text{ GB}$$

### 3.3 通信量对比

| 方法 | 单次通信量 | 频率 | 阻塞 |
|---|---|---|---|
| OEPLB cold-start | 11.3 GB | 1次（首窗口） | 异步（~255ms后完成） |
| OEPLB steady-state | ~0.5 GB/window | 每16步 | 异步 |
| EPLB rebalance | ~85 GB | 每1000步 | **阻塞 0.5-4.4s** |

**比例**：OEPLB 首次通信量仅为 EPLB 的 13%，稳态更是低2个数量级。

## 4. 异步隐藏的效率分析

### 4.1 延迟隐藏率

设 sync_window = $w$ 步，每步 forward time = $T_f$。Window 周期 = $w \cdot T_f$。

P2P 传输在低优先级 stream 上执行，与主 stream 的 forward pass 并行：

$$\text{隐藏率} = \min\left(1, \frac{w \cdot T_f}{T_{\text{P2P}}}\right)$$

对 Qwen3-235B (sw=16, $T_f \approx 200$ms per forward, $T_{\text{P2P}} \approx 255$ms)：
$$\text{隐藏率} = \min(1, 16 \times 200 / 255) = 1.0$$

即 P2P 传输完全被后续的 forward passes 隐藏——零额外延迟！

### 4.2 唯一的同步点

唯一阻塞发生在下一个 window 的 all_reduce 之前（`force_wait`），此时 P2P 必须已完成。

条件：$T_{\text{P2P}} < w \cdot T_f$（传输在下一个 window 到来前完成）。

对冷启动（240 ops, 255ms）：只要 window ≥ 2 步（$2 \times 200 = 400 > 255$ms），条件就满足。

**对稳态（8-20 ops, ~20ms）**：任何 $w \geq 1$ 都满足。

## 5. 通信复杂度的渐近分析

### 5.1 Swap 的摊还通信复杂度

设总 serving time = $S$，域切换次数 = $K$。每次切换需要 ~200 ops cold-start + 逐渐收敛。

$$C_{\text{OEPLB\_total}} = K \times B_{\text{cold}} \times 2W + (S/w - K) \times B_{\text{steady}} \times 2W$$

$$\approx K \times 240 \times 47\text{MB} + (S/w) \times 10 \times 47\text{MB}$$

$$= K \times 11\text{GB} + (S/w) \times 0.47\text{GB}$$

### 5.2 EPLB 的摊还通信复杂度

$$C_{\text{EPLB\_total}} = \lfloor S / T_{\text{rebalance}} \rfloor \times C_{\text{EPLB}} = (S/1000T_f) \times 85\text{GB}$$

### 5.3 比较

对 $S = 750$s（标准benchmark），$w=16$, $T_f=0.2$s, $K=3$（3次域切换）：

$$C_{\text{OEPLB}} = 3 \times 11 + (750/3.2) \times 0.47 = 33 + 110 = 143 \text{ GB}$$
$$C_{\text{EPLB}} = (750/200) \times 85 = 3.75 \times 85 = 319 \text{ GB}$$

**OEPLB 总通信量仅为 EPLB 的 45%**，且全部异步执行不阻塞。

## 6. 关键结论

1. **单次 swap 通信量 = 2×expert权重** = O(10MB)量级，NVLink下<1ms
2. **OEPLB 总通信量 ≈ EPLB 的 45%**，且100%异步不阻塞
3. **完全隐藏条件**：$w \geq \lceil T_{\text{P2P}} / T_f \rceil$（通常2步就够）
4. **EPLB的通信量不可避免地包含全量Expert重分配 + KV cache压力**——swap仅搬2个expert
5. **异步P2P的正确性**由NCCL操作顺序保证：唯一同步点是all_reduce前的force_wait

## 7. 补全：域切换次数 K 与数据集特征的联系

### 7.1 问题

§5.1 摊还分析假设 $K$（域切换次数）已知，没说怎么估计。
这里从数据集结构推导 K。

### 7.2 数据集的域结构模型

设 benchmark 有 $S$ 个不同的域段（segment），每个段长度 $L_s$ 条请求。
总请求数 $N = \sum L_s$。

域切换次数 $K = S - 1$（段间切换）。对:
- L512_O1（单域 Prover）: $S=1, K=0$
- multidomain_v2（4域）: $S=4, K=3$
- ShareGPT 100K（混合真实对话）: 域数难定义，但每条请求近似独立域 → $K \approx N$

### 7.3 切换频率 λ

每条请求触发切换的概率 $p_{\text{switch}} = K/N$。
每 window 处理 $w \cdot T_{\text{batch}}$ 条请求，所以每 window 的预期切换:
$$E[\text{switches per window}] = w \cdot T_{\text{batch}} \cdot K/N$$

对 OEPLB，切换检测由 cos_sim 触发（非逐条）。但宏观上:
$$\lambda = \frac{K}{N / (w \cdot T_{\text{batch}})} = \frac{K \cdot w \cdot T_{\text{batch}}}{N}$$

### 7.4 总通信量（修正版）

冷启动通信 + 稳态通信 + 域切换重纠偏通信:
$$C_{\text{total}} = B_{\text{cold}} \cdot 2W + \frac{S_{\text{bench}}}{w \cdot T_f} \cdot B_{\text{steady}} \cdot 2W + K \cdot B_{\text{shift}} \cdot 2W$$

其中 $B_{\text{cold}} \approx 240$（冷启动），$B_{\text{steady}} \approx 5$（稳态），
$B_{\text{shift}} \approx 20$（域切换后重纠偏）。

代入 235B multi-domain（$S_{\text{bench}}=750$s, $w=16, T_f=0.2$s, $K=3$）:
$$C = 240 \times 47 + \frac{750}{3.2} \times 5 \times 47 + 3 \times 20 \times 47$$
$$= 11280 + 55430 + 2820 = 69530 \text{ MB} \approx 68 \text{ GB}$$

对比 EPLB（$S_{\text{bench}}/1000T_f$ 次 rebalance）:
$$C_{\text{EPLB}} = \frac{750}{200} \times 85 = 319 \text{ GB}$$

**OEPLB 总通信仅 EPLB 的 21%**（修正前估计 45%，加上 $B_{\text{shift}}$
后更优，因为稳态开销远低于 EPLB 的每次 85GB）。

### 7.5 P2P 与 forward 的带宽竞争

§4.2 假设 P2P 完全隐藏在 forward 期间。但 NVLink 带宽是共享的:
forward 的 DeepEP dispatch/combine 也用 NVLink。P2P swap 跟它们竞争。

**有效隐藏条件修正**:
$$T_{\text{P2P}}^{\text{eff}} = \max\left(\frac{C_{\text{P2P}}}{B_{\text{NVLink}} - B_{\text{MoE comm}}}, T_{\text{P2P}}\right)$$

当 MoE 通信已占满带宽（$B_{\text{MoE comm}} \to B_{\text{NVLink}}$）时，
P2P 的有效传输时间趋无穷 → 不能完全隐藏。

**4卡 vs 8卡**: 4卡 NVLink 带宽 450 GB/s，MoE dispatch/combine 在
4卡下数据量小（约8卡的1/4），$B_{\text{MoE comm}}$ 占比低，P2P 有充足
隐藏空间。8卡下 MoE 通信更大，P2P 隐藏更紧——这解释了8卡 swap 开销
占比反而更低的现象（studypaper/06）。
