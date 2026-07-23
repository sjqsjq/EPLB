# PB-OEPLB: Prefill-Boundary Online Expert Load Balancing for MoE Inference

## 论文定位

面向MoE大模型在线推理场景的**轻量级、自适应、零冗余**专家负载均衡系统。核心差异化：不需要冗余专家、不阻塞推理、只用prefill信号就能同时优化prefill和decode、能实时感知workload shift并自适应调整。

---

## Observations & Insights

### Observation 1: 相同类型请求的专家激活高度稳定，但请求类型切换时热点剧烈漂移

相同类型的请求（如数学证明、小说续写），使用的专家分布几乎不变——我们观察到同一domain内连续多个sync_window的cos_sim稳定在0.95+。但当请求类型发生切换时（如Prover→BookCorpus），专家热点分布会在1-2个window内剧烈改变，avg_ratio_before从稳定的1.19-1.20突然飙升到1.30-1.39。

**Insight**: 这意味着(1)静态专家放置方案只能针对一种workload优化,无法适应真实生产环境中的请求类型变化；(2)动态方案必须能够**快速感知**这种变化并**及时纠偏**，而不是按固定周期慢慢调整。

**实验证据**:
- fixeddata实验（1024×Prover + 2048×BookCorpus）: domain切换点swap收敛轨迹清晰可见（window4-5 ratio冲到1.30+, ops=124+216）
- triplefixed实验（+2048×HellaSwag）: 证明"不是所有domain切换都会触发大幅漂移"——BookCorpus→HellaSwag的切换几乎没有冲击（两者都是英文自然语言），只有Prover→BookCorpus（结构化数学语法 vs 自然语言）才有显著漂移

### Observation 2: 不同负载特征（输入/输出长度、batch组成）需要不同的均衡策略参数

- 输入越长，OEPLB收益越大（short +5-12% < medium +8-16% < long +10-16%），因为prefill阶段在总时长中占比更大
- 输出越长，最优sync_window越大（256tok最优sw=64，2048tok最优sw=32），因为长输出意味着更长的decode阶段，更频繁的检查带来的all_reduce同步开销需要被更大的纠偏收益盖过
- deepep-mode选择直接影响收益方向：normal模式下长输出会变成负收益，auto模式下保持正收益——因为auto模式让decode走low_latency kernel，效率提升一倍

**Insight**: 固定参数的负载均衡策略不可能在所有场景下都最优。系统需要**根据当前workload的特征自适应调整**检查频率和决策阈值。

**实验证据**:
- 9种负载网格（3输入×3输出）× baseline/OEPLB，3轮重复，独立重启验证
- sw32/64/128参数扫描 × 256tok/2048tok数据集
- auto vs normal模式下的medium×out256/1024/2048对比

### Observation 3: Prefill batch的token路由能够有效反映decode batch的路由分布

在MoE推理中，大多数方案都只关注prefill阶段的负载均衡（因为prefill阶段不均衡更严重、收益更明显）。但decode阶段同样存在负载不均问题。我们的实验发现：只基于prefill batch记录的专家激活分布做的placement优化，**对decode阶段同样产生正收益**。

**Insight**: Prefill阶段处理的token数量更多（整个prompt vs 逐token），对专家激活的采样更充分、统计更可靠。只记录prefill batch可以(1)大幅减少record开销（decode step远多于prefill step，全记录的话开销线性增长）；(2)避免decode路由的噪声干扰决策质量；(3)一次优化同时惠及两个阶段。这跟"Patterns behind Chaos"论文中"prefill路由能预测decode路由"的结论相互印证。

**实验证据**:
- 9种负载网格中TPOT（decode指标）全面改善（-3%~-15%），跟TTFT（prefill指标）的改善方向一致
- auto模式下out=1024/2048的TPOT改善（-1.4%/-3.2%），直接证明placement优化对decode有效

---

## System Design

### D1: Prefill-Boundary记录机制

- `record_next_layer(topk_ids)` 只在scheduler判定当前batch为prefill时执行
- 直接在physical slot空间做scatter_add（一次kernel调用，无p2l转换开销），跟EPLB的`on_select_experts`对齐
- Decode/idle batch完全跳过，零开销
- 累积到`self.load`张量，按sync_window周期进行决策

### D2: 指数衰减历史 + 窗口化决策

当前累积值：$A_n = R_n + \alpha R_{n-1} + \alpha^2 R_{n-2} + \cdots$

- $\alpha = 0.9$（decay_factor），当前window权重 $(1-\alpha) = 10\%$，历史总权重 $\alpha = 90\%$
- 半衰期 $\approx \frac{\ln 0.5}{\ln \alpha} \approx 6.6$ 个窗口
- 比"每次清零重新统计"抗噪声能力更强：单个异常batch不会导致过激反应
- 比"无限累积不衰减"响应速度更快：真实的workload shift在6-7个窗口内就能反映到决策中

### D3: Greedy Global-Budget Swap Planner

**问题形式化**：给定全局负载张量 $L \in \mathbb{R}^{N_{layers} \times N_{slots}}$（all_reduce后所有rank一致），找到一组swap操作集合 $S = \{(l_i, a_i, b_i)\}_{i=1}^{|S|}$，其中 $|S| \leq B$（全局预算），使得所有层的max GPU负载比最小化：

$$\min_{S} \max_{l \in [N_{layers}]} \frac{\max_{g \in [N_{gpus}]} \sum_{s \in GPU_g} L[l][s]'}{\frac{1}{N_{gpus}} \sum_{s} L[l][s]}$$

**贪心算法**：
1. 计算所有层的当前ratio，按ratio从高到低排序
2. 取ratio最高的层，在该层内找最热GPU的最热slot和最冷GPU的最冷slot
3. 模拟swap（交换load计数和p2l映射），重新计算该层ratio
4. 如果swap没有改善ratio → 标记该层完成，不再参与竞争
5. 如果ratio已降到threshold以下 → 该层达标，不再参与
6. 回到步骤1（所有未完成的层重新竞争预算），直到预算耗尽

**vs EPLB策略的核心区别**：
- EPLB：每层独立做固定数量的复制操作（layers × num_replicate），不管该层是否真的不均衡 → **过度治疗**（我们的V1实验数据：mean imbalance只有1.13-1.17时仍在全量swap，收益极小但P2P开销全额发生）
- OEPLB：全局预算贪心分配，只治最严重的层，每次swap后立刻重新排序 → 预算自动流向最需要的地方

### D4: 异步非阻塞P2P Swap执行

- 在独立的低优先级CUDA stream上执行P2P权重交换
- `begin(plan)`: 发起所有P2P ops，立刻返回（非阻塞）
- `try_finish()`: 非阻塞event.query()检查完成状态
- 只在下一个window的all_reduce之前做一次`force_wait`（防死锁）
- 单次swap只涉及两个slot（两个rank之间），耗时微秒级
- **对比EPLB**: EPLB的rebalance需要重新计算整个physical_to_logical映射然后一次性更新，首次耗时4.4s，后续0.5s/次，期间推理完全阻塞

### D5: Adaptive Window（自适应检查频率）

基于routing distribution的cos_sim检测workload shift：

$$cos\_sim = \frac{\sum_l \vec{L}_n^{(l)} \cdot \vec{L}_{n-1}^{(l)}}{\sum_l \|\vec{L}_n^{(l)}\| \cdot \|\vec{L}_{n-1}^{(l)}\|}$$

- 当 $cos\_sim < \theta_{shift}$ (0.85)：请求类型发生了变化 → **立刻收缩**sync_window（64→32），加快检查频率
- 当 $cos\_sim > \theta_{stable}$ (0.95) 且**连续2个窗口**都稳定：恢复原始sync_window（32→64），减少开销
- **非对称确认设计**：收缩方向单窗口触发（收缩只改变检查频率，不直接触发swap，误判代价低）；恢复方向多窗口确认（保守一点无坏处）

### D6: Swap收敛性分析

Swap行为天然是收敛的：
- 首个窗口（冷启动大修正）：ops通常在200-240（接近预算上限250）
- 第2个窗口：ops降到40-50
- 第3个窗口及以后：ops降到个位数或0
- 收敛终点：avg_ratio稳定在1.19-1.20左右

当遇到极端不均衡（某个专家在所有GPU上都是热点）时，swap只能把热点从一个rank挪到另一个rank，无法真正"分摊"——这是swap-only机制相对replication机制的固有限制。面对不同EP规模（不同的GPU数量），均衡的理论下限不同，threshold参数应该相应调整。

---

## Evaluation Plan

### 已完成的实验

| 编号 | 实验 | 状态 |
|---|---|---|
| E1 | 9种负载网格（3输入×3输出）× baseline/OEPLB，3轮 | ✅ |
| E2 | 冷启动公平性验证（独立 vs 链式） | ✅ |
| E3 | domain-switch数据集（fixeddata, triplefixed） | ✅ |
| E4 | Baseline vs EPLB(iter=32/64, red=8/16) vs OEPLB(sw=32/64/128)，纯prefill | ✅ |
| E5 | 隔离实验（冗余专家alone ≈ baseline） | ✅ |
| E6 | domain-switch场景三方对比（OEPLB+7.9% vs EPLB+2.8%） | ✅ |
| E7 | 含decode场景（out=8/64/128/256/1024/2048，normal vs auto模式） | ✅ |
| E8 | adaptive_window验证（fixeddata上v1对称 vs v2非对称 vs 静态） | ✅ |
| E9 | 最优sw与输入长度关系（256tok vs 2048tok × sw32/64） | ✅ |
| E10 | EPLB rebalance阻塞开销量化 + CUDA error crash记录 | ✅ |

### 需要补充的实验

| 编号 | 实验 | 目的 | 优先级 |
|---|---|---|---|
| E11 | **多种domain拼接的长混合数据集**（3-4种不同domain交替出现，每段500-1000条） | 验证Obs1"请求类型突变时快速感知"的adaptive_window机制在多次切换场景下的表现，目前只验证了1-2次切换 | 高 |
| E12 | **EPLB在domain-switch场景的swap/rebalance记录对比** | 目前只有OEPLB的逐window DIAG数据，缺少EPLB在同一数据集上的rebalance时间线对比，无法直接展示"EPLB反应慢" | 高 |
| E13 | **不同EP规模（EP=4 vs EP=8）下的收益对比** | 验证Obs2提到的"面对不同EP应该设计不同的均衡阈值"，以及swap在小EP下是否更容易遇到"热点搬来搬去"的极端情况 | 中 |
| E14 | **record开销的精确对比**（OEPLB只记prefill vs 假设全记prefill+decode） | 量化Obs3"只记录prefill能大幅减少record开销"的具体数字，目前有PROF数据（record总耗时），但缺少"如果也记录decode会多花多少"的对照 | 中 |
| E15 | **收益的理论上界推导验证** | 从实测的avg_ratio_before/after计算理论加速上界（ratio改善×MoE占比），跟实测req/s对比，验证模型的预测精度 | 中 |
| E16 | **真实生产流量模拟**（混合短长请求、随机到达、EOS早停） | 目前所有实验都是"统一max_tokens+conc=1024的burst模式"，跟真实生产流量差别很大 | 低（工作量大） |
| E17 | **不同模型的泛化性**（除Qwen3-235B外，至少再测一个MoE模型如DeepSeek-V3） | 验证方案不是针对单一模型调参调出来的 | 低（需要新模型/新环境） |

### 建议的优先级排序

**必须补的**（直接支撑论文核心claim）：E11（多domain拼接）、E12（EPLB对比时间线）、E15（理论上界验证）

**最好有的**（让论文更完整）：E13（不同EP规模）、E14（record开销量化）

**锦上添花的**（有时间再做）：E16（真实流量）、E17（多模型）

---

## 数学建模机会

### M1: Greedy Planner的竞争比分析

定义离线最优解为：知道整个benchmark的完整路由分布后，一次性算出的最优placement。我们的在线greedy planner每个sync_window只能看到局部信息，做出的placement可能不是全局最优的。可以分析：
- 在什么条件下（比如路由分布的变化速率有界），greedy planner能在多少个窗口内收敛到跟离线最优解的差距小于ε？
- 我们的实验数据显示通常3-4个窗口就能收敛到avg_ratio ≈ 1.19-1.20，这个收敛速率跟理论分析是否吻合？

### M2: 指数衰减参数的最优选择

$\alpha$太大（如0.99）：对shift反应太慢
$\alpha$太小（如0.5）：对噪声太敏感

最优的$\alpha$取决于两个competing objectives的trade-off：
- **噪声抗性**：希望$\alpha$大，多个窗口平均后噪声被smoothed out
- **响应速度**：希望$\alpha$小，新workload的信号能快速"盖过"旧历史

可以用信号检测理论（SNR分析）来推导：给定workload shift的"信号强度"（前后两种workload的routing distribution距离）和每个窗口的"噪声水平"（单窗口内routing统计的方差），最优的$\alpha$应该使得shift被检测到的expected latency最小化，同时false positive rate（误把噪声当成shift）控制在可接受水平以下。

### M3: 收益上界模型

假设MoE层耗时 $T_{MoE} = T_{dispatch} + T_{expert} + T_{combine}$

其中 $T_{expert} = \max_{gpu} T_{expert}^{(gpu)}$（最慢的GPU决定这一步的耗时）

设均衡前max/avg ratio为$r_{before}$，均衡后为$r_{after}$：

$$speedup_{MoE} = \frac{r_{before} - r_{after}}{r_{before}} \times \frac{T_{expert}}{T_{MoE}}$$

$$speedup_{total} = speedup_{MoE} \times \frac{T_{MoE}}{T_{total}}$$

用我们实测的参数代入（$r_{before}=1.71, r_{after}=1.20, T_{expert}/T_{MoE} \approx 0.36, T_{MoE}/T_{total} \approx 0.64$）：

$$speedup_{total} \approx \frac{1.71-1.20}{1.71} \times 0.36 \times 0.64 \approx 6.9\%$$

跟实测的short/medium输入（+8-16%，还有prefill调度优化的额外收益）在同一量级，偏保守但方向正确。

---

## 与现有工作的差异总结

| 维度 | 静态放置 | EPLB (SGLang官方) | PB-OEPLB (本工作) |
|---|---|---|---|
| 适应workload变化 | ❌ 无法适应 | △ 周期性rebalance，反应慢 | ✅ 在线感知+快速swap |
| 冗余专家 | 不需要 | **需要（8-16个，占显存）** | **不需要** |
| 推理阻塞 | 无 | **每次rebalance阻塞0.5-4.4s** | **异步P2P，不阻塞** |
| 决策信号 | 离线历史数据 | prefill+decode全记录 | **只记prefill（更干净、开销更低）** |
| Decode阶段收益 | 无 | 正收益但rebalance开销抵消 | **正收益（placement优化惠及decode）** |
| Domain-switch响应 | 无法响应 | 反应慢（+2.8%） | **快速响应（+7.9%）** |
| 参数自适应 | 无 | 固定迭代周期 | **cos_sim驱动的adaptive window** |
