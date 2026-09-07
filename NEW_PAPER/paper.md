# 面向MoE推理服务的死区感知自适应专家负载均衡

## 摘要

MoE模型在推理服务中面临专家负载不均衡问题——路由偏斜使少数热点专家集中在个别GPU，造成计算瓶颈与尾部延迟，MoE计算浪费50-75%。现有方案如SGLang的EPLB需要冗余专家副本（12.5%额外显存）、重平衡期间阻塞推理1.4–4.5秒、强制关闭CUDA graph导致decode-heavy负载退化62%。本文从MoE层时间的实验测量出发，发现"死区"现象（不均衡度r≤r_k时降低r不产生时间收益，因dispatch/combine与GEMM的重叠吸收了差距），并由此推导增益上界公式Δ_max=f_sens·x_eff/(1−f_sens·x_eff)，表明特定模型与数据集的收益存在上限。基于这两个发现，本文设计PB-OEPLB：死区感知的swap停止策略（从EP幂律自动计算r_k，r≤r_k时停止swap）、自适应窗口（指数衰减累积器A_t=R_t+α·A_{t-1}的有效记忆M=W/(1−α)是偏差-方差权衡的唯一自由度，W与α只通过M影响稳态；变点检测时α瞬时归零一步清空旧域历史使响应延迟从M·ln2降至0，稳态按收敛/振荡动态伸缩决策窗口W跟踪最优点）、仅prefill阶段记录路由（由prefill→decode相关性的任务结构依赖性论证充分性：QA/推理类ρ=0.78–0.85强相关，数学类ρ=0.44–0.69弱相关）三个核心机制。在8×H20集群上服务Qwen3-235B-A22B-FP8（TP=DP=EP=8），在7个域特定数据集上对比identity基线、EPLB和oracle布局。PB-OEPLB在prefill密集负载上提升吞吐+17.5%，达oracle的97.6%，相比EPLB高出15.7个百分点；稳态每次调整阻塞0.37秒（EPLB的1/4）。

## 1 引言

### 1.1 背景

混合专家（Mixture-of-Experts, MoE）架构通过门控网络实现专家稀疏激活，在不按比例增加计算成本的前提下提升模型容量，已成为大语言模型推理服务的主流架构。自2025年以来，DeepSeek-V3（671B, 374专家）、Qwen3-235B（128专家, top-8）、Kimi K2（1000B, 384专家）等大型MoE模型相继发布，均采用100+专家、top-6至top-8的路由策略。在推理阶段，每个token仅激活k个专家（k<<N_E），MoE层的计算量与密集模型相当，但模型容量随专家数增长。然而，MoE的稀疏性使Token分布呈现强局部性和热点效应：路由网络根据输入语义将Token分配到少数"热门专家"，导致计算任务在GPU间极度不均。在专家并行（Expert Parallelism, EP）场景下，MoE all-to-all通信要求所有GPU同步，负载最重的GPU完成最晚，其他GPU被迫等待，引发尾部延迟和算力浪费。

### 1.2 问题

负载不均衡在真实服务中有多严重？本文在Qwen3-235B-A22B（94 MoE层, 128专家, EP=8, 16专家/卡）上实测发现：默认连续放置（identity）下，逐层max/min负载比均值2.26–4.38×，最极端层达11.79×（Fig 1）。逐forward粒度下ratio更高（median 3.7–6.4×，是聚合值的1.5–1.7倍），说明MoE每个forward实际经历的straggler比聚合数字显示的更严重（Fig 1b/13）。更关键的是，不同数据集的热点GPU完全不同（MMLU=GPU4, prover=GPU5, book=GPU0），跨域路由Spearman相关系数近乎为零（ρ≈0, Fig 4）——为数据集A优化的静态放置对数据集B不仅不是最优，甚至比默认放置更差（MMLU最优放置→prover后ratio=3.67 > identity 3.51, Fig 9）。静态放置在域切换负载下必然失败，负载均衡必须是动态的。

### 1.3 相关工作

现有方法可分为三类。**静态布局**：从历史流量数据预计算最优专家放置，如DataFore（ISCA 2026）的prefill-guided remap/dup算法。但需离线profiling，无法适应运行时负载变化，且跨域放置迁移实测证明失败。**周期重平衡**：如SGLang的EPLB，周期性重新计算专家布局并重分配权重。能适应变化，但需冗余专家副本（16额外副本, 12.5%额外显存, KV cache容量−8.1%→高并发排队时间2–4.8×），每次重平衡阻塞0.5–4.5秒，且强制deepep_mode=normal禁用CUDA graph，decode-heavy负载退化62%。此外，官方EPLB在非DeepSeek架构（Qwen2-MoE/Qwen3-MoE）上报AttributeError，完全不兼容。**在线交换**：增量调整专家位置，避免全量重平衡。本文方法属此类，但面临收敛速度（旧方法停滞在ratio=1.26无法继续）和决策噪声（单窗口统计不可信）两大挑战。

### 1.4 本文方案

本文不从"设计更好的swap算法"出发，而是**从MoE层时间的实验测量出发**，发现一个被所有现有工作忽略的现象：T(r)（MoE层时间对不均衡度的函数）呈铰链形式——存在死区r_k，r≤r_k时T不变（DeepEP的dispatch/combine与GEMM的重叠吸收了全部落差），r>r_k时才线性增长。这一发现改变了一个基本假设："不均衡有害"在r≤r_k时不成立。由此推导增益上界公式Δ_max = f_sens·x_eff / (1 − f_sens·x_eff)，表明特定模型+数据集的收益有上限，系统效率η决定实得——30B模型Δ_max=+6.36%（正！不均衡确实有害）但η≈0（死区极窄r_k=1.031→swap全在死区内→零收益但开销照付）。

基于死区和增益上界两个理论发现，本文设计PB-OEPLB（Prefill-Boundary Online Expert Placement Load Balancer）：（1）死区感知的swap停止策略——从EP幂律r_k−1=0.00408·EP^1.52自动计算r_k，r≤r_k时停止swap，避免Fig J量化的"20/21次决策零收益"无用开销；（2）自适应窗口——指数衰减累积器A_t=R_t+α·A_{t-1}的有效记忆M=W/(1−α)是偏差-方差权衡的唯一自由度，W（决策频率与all_reduce开销）与α（遗忘曲线形状）只通过M影响稳态，早期实现调W不同步α会无意中漂移M；变点检测时α瞬时归零一步清空旧域历史（响应延迟从M·ln2降至0），稳态按收敛/振荡动态伸缩决策窗口W使M跟踪最优点；（3）仅prefill阶段记录路由——实测发现prefill→decode相关性由任务结构决定（QA/推理类ρ=0.78–0.85强，数学类ρ=0.44–0.69弱），任务结构>>prompt长度，prefill-only recording的充分性由任务类型决定而非prompt长度。

### 1.5 实验

在8×H20集群上服务Qwen3-235B-A22B-FP8（TP=DP=EP=8），使用SGLang 0.5.6.post2 + DeepEP v1.2.1 + DeepGEMM FP8。在7个域特定数据集（MMLU多学科QA, ARC科学推理, CommonsenseQA常识推理, OpenBookQA, GSM8K数学应用题, prover数学证明, BookCorpus叙事文本）上对比identity基线、EPLB和oracle布局。PB-OEPLB在prefill密集负载上提升吞吐+17.5%（n=2），达oracle的97.6%，相比EPLB高出15.7个百分点（EPLB可复测仅+1.75%）。稳态每次调整阻塞0.37秒（EPLB 1.55秒, 4×降低）。在多域漂移负载上+9.76%，超过静态最优布局+5.80%。跨3个模型（235B/57B/30B）验证增益上界公式，30B案例（Δ_max正但η≈0）揭示了"不均衡存在但swap无法获益"的条件。

### 1.6 本文创新

本文的创新点如下：

1. **死区理论**：首次发现MoE层时间T(r)呈铰链形式（R²=0.998），r≤r_k时T不随不均衡度变化。r_k由EP幂律决定（r_k−1=0.00408·EP^1.52），跨模型盲测误差+0.4%。实测量化：第1次swap覆盖全部有用距离，后续20次（59% ops）在死区内零收益。

2. **增益上界公式**：推导Δ_max = f_sens·x_eff/(1−f_sens·x_eff)（Amdahl形式），f_sens≠FLOP占比（组件分解：Combine β=1.33, Expert β=0.08, Dispatch β=−0.78）。增益=Δ_max×η，η由开销/bound决定。跨3模型验证，30B揭示"Δ_max正但η≈0"的机制。

3. **PD相关性的任务结构依赖**：QA/推理类ρ=0.78–0.85（94/94层强），数学类ρ=0.44–0.69。任务结构>>prompt长度——OBQA 15tok ρ=0.78 > prover 107tok ρ=0.44。prefill-only recording的充分性由任务类型决定。

4. **自适应衰减与M统一控制**：指数衰减累积器的有效记忆M=W/(1−α)是唯一需按负载调的量，W与α只通过M影响稳态，早期实现调W不同步α会无意中漂移M；变点检测时α瞬时归零清空旧域历史把响应延迟从M·ln2降至0。同session实测该adaptive逻辑（+9.7%）超过固定α=0.9（+6.4%）——固定α不必追求，零调参adaptive即为最优。

### 1.7 论文组织

本文余下部分组织如下：§2回顾相关工作并给出对比表格；§3呈现4个关键观察（死区、增益上界、PD任务结构依赖、自适应衰减与M统一控制）；§4描述PB-OEPLB框架设计（5个机制）；§5给出实验评估；§6总结全文。

## 2 相关工作

现有MoE专家负载均衡方法按放置时机可分为静态布局、周期重平衡、在线交换三类，本文方法属在线交换，但首次引入死区感知的停止条件与基于变点检测的自适应衰减，从根本上改变了"何时停"与"如何记忆"两个设计维度。

### 2.1 静态布局

静态布局从历史流量数据离线预计算最优专家放置，服务期不再调整。DataFore（ISCA 2026）提出prefill-guided remap/dup算法，用录制阶段的路由统计指导专家重排与冗余复制。该类方法的优势是零运行时开销，但有两个根本局限：一是依赖离线profiling，无法适应运行时负载漂移；二是跨域放置不可迁移——本文实测为数据集A优化的静态放置迁移到数据集B后ratio=3.67，反而劣于默认identity放置的3.51（Fig 9），因为不同数据集的热点GPU弱相关（Spearman ρ≈0，Fig 4）。这决定了静态方法在域切换负载下必然失效。

### 2.2 周期重平衡

周期重平衡在服务期周期性重新计算专家布局并重分配权重，以SGLang官方EPLB为代表。它能适应负载变化，但代价高昂：需要冗余专家副本（235B配置下16个额外副本，12.5%额外显存，KV cache容量下降8.1%，高并发排队时间放大2–4.8×）；每次重平衡阻塞推理0.5–4.5秒；且强制deepep_mode=normal以支持权重迁移，该模式禁用CUDA graph，使decode-heavy负载吞吐退化62%。此外，官方EPLB实现深度耦合DeepSeek架构，在Qwen2-MoE/Qwen3-MoE上直接抛AttributeError，无法兼容。周期重平衡的"全量重算+全量迁移"范式在增量调整场景下开销过重。

### 2.3 在线交换

在线交换增量调整专家位置，避免全量重平衡。其核心挑战有二：收敛速度——朴素方法在不均衡度ratio=1.26处停滞无法继续下降；决策噪声——单窗口路由统计在token稀疏时偏差大（c/√N，c=0.65·EP），把噪声当不均衡去修反而引入错误swap。本文PB-OEPLB属此类，通过指数衰减累积器聚合多窗口、自适应衰减在变点清空旧历史、死区感知在r≤r_k时停止swap三点同时解决收敛与噪声问题。

### 2.4 方法对比

| 维度 | Identity | DataFore静态 | SGLang EPLB | **PB-OEPLB(本文)** |
|---|---|---|---|---|
| 冗余专家 | 无 | 有(dup) | 有(16副本,12.5%显存) | **无(原地P2P交换)** |
| 阻塞方式 | 无 | 无(离线) | 全量阻塞0.5–4.5s | **增量swap 0.37s** |
| CUDA graph兼容 | 兼容 | 兼容 | 不兼容(normal模式) | **兼容** |
| 死区感知 | 无 | 无 | 无 | **有(EP幂律自动算r_k)** |
| 自适应记忆 | 无 | 无 | 无 | **有(M=W/(1−α),变点清零)** |
| 离线profiling | 不需要 | 需要 | 不需要 | **不需要(在线prefill录制)** |

### 2.5 本文差异

与上述方法相比，PB-OEPLB的差异体现在三处。其一，**无冗余原地交换**：不复制专家副本，通过rank间batch_isend_irecv在物理槽位间移动权重，显存零增长，KV cache不受损。其二，**死区感知停止**：从EP幂律r_k−1=0.00408·EP^1.52自动推导停止阈值，避免在r≤r_k时执行零收益swap（实测59%的ops落在此死区内），这是现有所有方法均未触及的维度。其三，**自适应衰减记忆**：以M=W/(1−α)统一控制偏差-方差工作点，变点时α瞬时归零清空旧域历史，稳态按收敛/振荡伸缩决策窗口，无需per-workload调参——同session实测该逻辑（+9.7%）超过固定α=0.9（+6.4%），证明零调参自适应已优于任何固定衰减。

## 3 观察

本节给出四个关键观察，它们构成PB-OEPLB设计的理论基础：死区刻画"均衡到何处停"，增益上界刻画"最多能赚多少"，PD任务结构刻画"prefill录制何时充分"，跨数据集异质性刻画"为何必须自适应"。四个观察均由实测得出并配以理论推导。

### 3.1 死区：不均衡度降至r_k以下不产生时间收益

MoE层时间$T$对不均衡度$r$呈铰链（hinge）响应：存在阈值$r_k$，$r\le r_k$时$T$与$r$无关，$r>r_k$时$T$线性增长。本文在57B 8卡与235B 8卡上做$T(r)$扫描——7个布局点（identity至oracle，$r$从约1.0至2.6）×2轮独立重启共14次运行、0错误——对每点的MoE层时间做铰链拟合，得到

$$T(r)=T_{\text{flat}}+B\cdot\max(0,\,r-r_k)$$

铰链拟合$R^2=0.998$，残差平方和比纯线性低12.1×（Fig I）。$r\le r_k$时$T$不变，因DeepEP的dispatch/combine通信与Expert GEMM在时间上重叠，重叠窗口吸收了GPU间负载落差；$r>r_k$时重叠耗尽，最重GPU的GEMM成为纯串行尾部，$T$随$r$线性增长。

$r_k$由EP幂律决定：$r_k-1=0.00408\cdot\text{EP}^{1.52}$，从4个配置标定（EP4→$r_k$=1.034，EP8→$r_k$=1.096），跨模型盲测误差+0.4%（Fig K）。死区宽度随EP增长（EP4窄区间[1.02,1.034]，EP8宽区间[1.02,1.096]），因此同一默认阈值在不同EP上的合理性截然不同。这一发现改变了均衡器的核心设计问题：不是"如何把$r$降到最低"，而是"降到$r_k$后何时停止"。实测量化（Fig J）：第1次swap覆盖全部有用距离，后续\#2–\#21（占59% ops）落在死区内零收益。含义是均衡器的停止条件应是$r_k$（从EP幂律自动算）而非硬编码1.02，在$r_k$处停止可省掉59%的无用swap开销。

### 3.2 增益有上界：特定模型与数据集的收益受$\Delta_{\max}$限制

一次重平衡的吞吐增益有理论上界$\Delta_{\max}$，由$r$敏感时间占比$f_{\text{sens}}$与有效可消除比例$x_{\text{eff}}$共同决定，系统效率$\eta$决定实得。由死区模型直接推导：

$$\frac{T(r_{\text{before}})}{T(r_{\text{after}})}-1=\frac{B\cdot(r_{\text{before}}-r_k)}{T_{\text{flat}}}=\frac{f_{\text{sens}}\cdot x_{\text{eff}}}{1-f_{\text{sens}}\cdot x_{\text{eff}}},\quad x_{\text{eff}}=\frac{r_{\text{before}}-\max(r_{\text{after}},r_k)}{r_{\text{before}}}$$

此即Amdahl形式：$f_{\text{sens}}$类比"可并行加速占比"，$x_{\text{eff}}$类比"加速比"。关键在于$f_{\text{sens}}\ne$FLOP占比。组件分解给出$r$敏感度系数$\beta_c$（Combine $\beta$=1.33，Expert GEMM $\beta$=0.08，Dispatch $\beta$=−0.78），加权得$f_{\text{sens}}=\sum_c\beta_c f_c=0.386$，而FLOP占比为67.9%、高估1.8×。原因：Combine虽只占33%时间却最敏感（最重GPU的all-gather最慢，其余GPU空等）；Expert GEMM占34%时间但几乎不敏感（token总数不随放置改变）。

实际增益$\Delta=\Delta_{\max}\cdot\eta$，其中$\eta$由swap开销与bound决定。跨3模型验证（Fig H）：235B $\Delta_{\max}$=22.6%、$\eta$=79%→+17.5%；57B $\eta$=84%→+2.7%；30B $\Delta_{\max}$=+6.36%（为正，不均衡确实有害）但$\eta\approx0$→净收益约0，因30B死区极窄（$r_k$=1.031），swap几乎全部落在死区内，零收益但开销照付（Fig L）。关于硬件：$f_{\text{sens}}$与$r_k$均与硬件相关（GPU算力提升→GEMM变快→$f_{\text{sens}}$下降；NVLink带宽提升→overlap增大→$r_k$上升），但EP幂律使$r_k$可预测，无需逐配置扫描。这一观察把"OEPLB是否有效"从"试一下才知道"变为"算$\Delta_{\max}$与$\eta$即可预判"。

### 3.3 prefill→decode的专家热度秩相关由任务结构决定

prefill与decode阶段的专家选择频率直方图之间存在强的**秩相关**（Spearman $\rho$）——即prefill阶段的热点专家排序在decode阶段大体保留；该相关的强弱由任务结构而非prompt长度决定，系统据此采取三项措施适应不同数据集。

**相关性的具体定义**：对同一批请求分别录制prefill与decode的（94层×128专家）选择频率矩阵，逐层计算两阶段128专家频率的Spearman $\rho$。$\rho$高表示专家热度排序prefill→decode保留，prefill录制足以定位decode的straggler专家；$\rho$低表示排序漂移、prefill预测变弱。

**实测**（7个域特定短prompt数据集：MMLU 25tok多学科QA、ARC/ARC-E 31tok科学、CSQA 20tok常识、OBQA 15tok科学、GSM8K 60tok数学应用、prover 107tok数学证明；conc=256、O=10；Fig 5/14）：QA/推理类（MMLU/ARC/CSQA/OBQA）$\rho$=0.78–0.85、94/94层强（$\ge0.7$）；数学类（GSM8K/prover）$\rho$=0.44–0.69、0–37/94层强。任务结构显著强于prompt长度——OBQA 15tok的$\rho$=0.78反高于prover 107tok的$\rho$=0.44（Fig 8）：QA类问题→答案的路由映射强且稳定，数学类题目→推导步骤的路由会偏离题目本身。时间衰减使prefill对early decode预测最好：$\rho$从early decode的0.62降至late decode的0.47（−24%），因decode越深路由分布漂移越大。

**针对不同相关性的适应措施**：（1）**仅prefill录制**——$\rho$高时prefill频率是decode分布的充分统计量（per-expert频率的max/mean结构一致，仅总token数不同），decode走CUDA graph零开销跳过记录，既省开销又避开decode录制对CUDA graph的破坏。该设计依据是$\rho$在prefill→decode边界最强、随decode深度衰减（0.62→0.47），故prefill是即将到来decode的最优预测，且指数衰减累积器天然给近期prefill更高权重、对齐此边界衰减特性。（2）**低$\rho$负载的$M$放大**——数学类$\rho$低→prefill信号弱+单窗抽样噪声大（$c/\sqrt{N}$），自适应窗口在噪声/不稳定信号触发时grow $W$（等效$M=W/(1-\alpha)$增大）以聚合更多prefill数据降偏差。需说明：$\rho$是离线测量的设计依据属性，系统不在线测$\rho$；在线适应由cos_sim/ratio噪声信号驱动，其有效性由$\rho$的离线测量保证。

### 3.4 跨数据集负载参数的异质性：固定配置必然偏离，需adaptive

不同数据集的路由负载参数$(r, L_{\text{seg}}, \bar{t})$**异质变化**（各参数取值不同，无需严格独立），决定任何固定$(W,\alpha)$都只能在部分workload最优，从而必须adaptive。本文在不同（prompt长度$L$、输出长度$O$、内容域）组合上扫描静态$(W,\alpha)$并测各workload决定$M^*$的三个参数$(r, L_{\text{seg}}, \bar{t})$。实证发现：最优静态sync_window跨workload从8（$L$256,$O$1）到64（$L$256,$O$1024 / $L$1024,$O$256）变化，无单一固定配置对所有$(L,O)$最优；三个参数跨数据集取值差异显著——$r$随域路由熵变（构造A的1.02–1.76 vs B的1.02–1.38）、$L_{\text{seg}}$随切换频率变（A 6段频繁切换 vs B 4域稳定）、$\bar{t}$随prompt长度与并发变。这与跨域路由弱相关（$\rho\approx0$，Fig 4：MMLU vs prover=0.054，MMLU vs book=−0.038）同源：不同数据集激活不同的专家簇（MMLU/prover/book的top-5热点完全不重叠），必然带来不同的$r$与$L_{\text{seg}}$。

**说明用词**：本文不用"正交"——那需要严格证明参数协方差矩阵近似对角（即独立性），我们不做此强主张。论证只需更弱的**异质性**（参数跨workload取值不同）：由$M^*$公式，$M^*$是$(r,L_{\text{seg}},\bar{t})$的函数，只要三者跨workload有变化（无论是否独立），$M^*$就跨workload变化，固定$(W,\alpha)$就必在部分workload上偏离$M^*$。

理论给出adaptive的追踪目标。指数衰减累积器$A_t=R_t+\alpha\cdot A_{t-1}$中，$W$（决策频率与all_reduce开销）与$\alpha$（遗忘曲线形状）不独立，稳态下只通过有效记忆$M=W/(1-\alpha)$与有效样本量$N_{\text{eff}}=M\cdot\bar{t}$影响偏差（抽样偏差$\propto1/\sqrt{N_{\text{eff}}}$）与方差，故唯一需按负载调的量是$M$而非$W$或$\alpha$。联合最小化方差代价（$M$小→bias大→$\eta$损失）与变点延迟代价（$M$大→旧信号残留约$M\ln2$步才半衰→用错误布局服务），设段长$L_{\text{seg}}$得闭式

$$M^*=\sqrt{\frac{a\cdot c^2\cdot L_{\text{seg}}}{b\cdot\beta\cdot\bar{t}\cdot\gamma^2\cdot(r-r_k)^3\cdot\ln 2}}$$

方向预测$M^*\propto\sqrt{L_{\text{seg}}}$、$M^*\propto(r-r_k)^{-3/2}$（已被d31+d34验证），$c=0.65\cdot$EP（标定），$\bar{t}/r/r_k/L_{\text{seg}}$均可在线测。因$(r,L_{\text{seg}},\bar{t})$跨workload异质变化，$M^*$必跨workload变化，固定$(W,\alpha)$必在部分workload上偏离$M^*$。实测同$M$=128不同$(W,\alpha)$吞吐吻合4.8%，印证$M$是近似充分统计量。这一观察把"需要adaptive"从工程经验提升为可计算命题：adaptive不是启发式补丁，而是追踪随workload异质变化的$M^*$目标，运行时以grow/shrink $W$与变点$\alpha\to0$清零作为$M^*$的离散近似（类比Adam追踪最优学习率）；同session实测adaptive（+9.7%）超过固定$\alpha$=0.9（+6.4%）（swap 104 vs 56但吞吐反高，说明陈旧性损害大于开销节省），零调参adaptive已优于任何固定配置。

## 4 PB-OEPLB框架

### 4.1 概述

PB-OEPLB是一个在线增量swap均衡器，由五个组件构成（Fig 架构图）：路由录制器在每个forward将top-k专家选择按物理槽位scatter\_add进本地计数器（零通信）；控制器按sync\_window周期做决策状态机；重平衡器贪心构建成对swap计划；异步执行器在rank间batch\_isend\_irecv移动权重；physical\_to\_logical\_map是全局共享的路由表，swap后更新并回推模型。三个挑战与三个观察一一对应：何时停止swap由死区回答（§3.1）；决策频率与记忆长度如何自适应由$M$统一与$M^*$闭式回答（§3.4）；prefill-only录制何时充分由PD任务结构相关性回答（§3.3）。

主循环（每sync\_window个forward执行一次，无跨rank共识——forward本身DP+EP隐式同步）：（1）force-finish上一轮pending的P2P（防NCCL跨流序号死锁）；（2）all\_reduce self.load的**克隆**（非原地，防每窗$\sim$num\_ranks×decay的累积膨胀）；（3）算不均衡度$r$、变点检测、threshold判断、构建swap计划；（4）同步P2P执行swap，更新路由表与衰减历史。本节按三个机制展开：§4.2死区感知停止、§4.3自适应窗口、§4.4仅prefill录制，§4.5给出算法流程与复杂度。

### 4.2 死区感知的swap停止策略

均衡器的停止条件应是死区阈值$r_k$而非硬编码常数。由§3.1的EP幂律，控制器在初始化时自动计算

$$r_k = 1 + 0.00408\cdot\text{EP}^{1.52}$$

（EP8→$r_k$=1.096），触发阈值取$\max(1.02, r_k)$，当$r\le r_k$时停止swap、仅保留录制与all\_reduce（0.62%开销），不再执行零收益的P2P。这直接消除§3.1 Fig J量化的浪费——第1次swap已覆盖全部有用距离，后续\#2–\#21（占59% ops）落在死区内零收益却照付3.42%的swap开销；启用死区感知后这些决策被阻止，有用决策从21次降至1次。

死区感知需配合两个稳定性修复才能净增益。其一是RESET冷却（cooldown=3）：域切换触发load.zero\_()清零历史后，跳过随后3窗的adaptive-window收缩，防止清零→小窗→噪声swap→再次跳变→再清零的振荡。其二是切换确认窗数（window\_shift\_confirm=2）：要求连续2窗低cos\_sim才确认域切换、收缩窗口，滤除单窗抖动。三者合用把一个前期实现中adaptive的−6%净收益翻正为+9.0%（构造A，同session对identity基线）；缺一则回到−6%。
