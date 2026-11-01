# 论文Outline（详细版）

## 一、摘要（200-300字）

**背景与问题**（1-2句）：MoE模型在推理服务中面临专家负载不均衡问题——路由偏斜使少数热点专家集中在个别GPU，造成计算瓶颈与尾部延迟，MoE计算浪费50-75%。

**现有工作不足**（1-2句）：现有方案如SGLang的EPLB需要冗余专家副本（12.5%额外显存）、重平衡期间阻塞推理1.4-4.5秒、强制关闭CUDA graph导致decode-heavy负载退化62%。在线swap方案面临收敛速度慢和决策噪声大的挑战。

**本文方案**（1句）：本文从MoE层时间的实验测量出发，发现死区现象（r≤r_k时降低不均衡度不产生时间收益），并由此推导增益上界公式，设计死区感知的自适应在线专家放置均衡器PB-OEPLB。

**方案内容**（2-3句）：PB-OEPLB包含四个核心机制：（1）从EP幂律自动计算死区阈值r_k，r≤r_k时停止swap，避免无用开销；（2）自适应窗口根据路由稳定性（cos_sim）和实时ratio自动调整决策频率，在域切换时收缩快速响应、稳态时扩张减少开销；（3）仅在prefill阶段记录路由数据（decode走CUDA graph零开销），并基于prefill→decode相关性的任务结构依赖性论证充分性；（4）将窗口W和衰减α统一为单一控制量M=W/(1-α)，给出M*的闭式最优。

**实验配置与效果**（2-3句）：在8×H20集群上服务Qwen3-235B-A22B-FP8（TP=DP=EP=8），使用DeepEP all-to-all和DeepGEMM FP8，在7+个域特定数据集上对比identity基线、EPLB、Frozen-EPLB和oracle布局。PB-OEPLB在prefill密集负载上提升吞吐+17.5%（n=2），相比EPLB高出15.7个百分点，稳态每次调整阻塞0.37秒（EPLB的1/4）。跨3个模型（235B/57B/30B）验证增益上界公式的预测能力。

---

## 二、Introduction

### 第1段：背景

MoE架构通过门控网络实现专家稀疏激活，在不按比例增加计算成本的前提下提升模型容量，已成为大语言模型推理服务的主流架构（DeepSeek-V3、Qwen3、Kimi K2、Mixtral等2025年发布的大型MoE模型均采用128-384专家、top-6至top-8路由）。然而，MoE的稀疏性使Token分布呈现强局部性和热点效应：少数"热门专家"处理的Token远超其他专家，导致计算任务在GPU间极度不均，引发尾部延迟和算力浪费。在专家并行（EP）场景下，MoE all-to-all通信步调不一致导致显著同步等待。

### 第2段：问题

负载不均衡在真实服务中有多严重？本文实测（Fig 1/2）：在Qwen3-235B上用identity放置服务3个真实数据集，逐层max/min ratio均值2.26-4.38×，最极端层达11.79×。逐forward粒度下ratio更高（median 3.7-6.4×，是聚合的1.5-1.7倍，Fig 13）。不同数据集的热点GPU完全不同（MMLU=GPU4, prover=GPU5, book=GPU0/4），跨域路由Spearman ρ≈0（Fig 4）——为数据集A优化的静态放置对数据集B不仅不是最优、甚至比默认identity更差（MMLU最优→prover ratio=3.67 > identity 3.51，Fig 9）。静态放置在域切换负载下必然失败。

### 第3段：相关工作

现有方法分三类。**静态布局**（如DataFore Case Study 2的remap/dup算法）：从历史流量预计算最优放置，但无法适应负载变化，跨域失败。**周期重平衡**（如SGLang的EPLB）：周期性重新计算专家布局并重分配权重，能适应变化，但需要冗余专家副本（8-16额外副本，12.5%额外显存，KV cache容量-8.1%→高并发排队时间2-4.8×）、每次重平衡阻塞1.4-4.5秒、强制deepep_mode=normal禁用CUDA graph（decode-heavy负载-62%）。**在线交换**（如OEPLB）：增量调整专家位置避免全量重平衡，但面临收敛速度（旧方法停滞在ratio=1.26）和决策噪声（单窗口统计不可信）的挑战。

### 第4-5段：本文方案

本文不是直接设计一个更好的swap算法，而是**从测量出发发现MoE时间对不均衡度的铰链响应**——存在死区r_k，r≤r_k时降低ratio不产生时间收益。这一发现改变了均衡器的核心设计问题：不是"如何把r降到最低"，而是"降到r_k后何时停止"。由此推导增益上界公式Δ_max=f_sens·x_eff/(1-f_sens·x_eff)，表明特定模型+数据集的收益有上限，η（系统效率）决定实得。基于这些理论，本文设计PB-OEPLB：死区感知的swap停止（auto-dead-zone从EP幂律自动算r_k）、自适应窗口（以M=W/(1-α)统一偏差-方差单一自由度为理论依据，变点α→0清零+grow/shrink W追踪M*闭式目标）、prefill-only recording（PD相关性由任务结构决定，QA类ρ=0.78-0.85充分、数学类ρ=0.44-0.69需补偿）。

### 第6段：实验

在8×H20集群上服务Qwen3-235B-A22B-FP8，使用SGLang 0.5.6.post2 + DeepEP v1.2.1 + DeepGEMM FP8。在7+个域特定数据集（MMLU/ARC/CommonsenseQA/OpenBookQA/GSM8K/prover/book）上对比identity基线、EPLB、Frozen-EPLB和oracle布局。PB-OEPLB在prefill密集负载上提升+17.5%（n=2，CV 0.7%），达oracle的97.6%，相比EPLB高出15.7个百分点（EPLB可复测仅+1.75%）。稳态每次调整阻塞0.37秒（EPLB 1.55秒，4×降低）。在多域漂移负载上+9.76%，超过静态最优布局+5.80%。跨3个模型（235B/57B/30B）验证增益上界公式：235B η=79%、57B η=84%、30B η≈0%（Δ_max正但死区极窄→swap全在死区内→零收益）。

### 第7段：创新4点

1. **死区理论**：首次发现MoE层时间T(r)呈铰链形式（R²=0.998），r≤r_k时T不变。r_k由EP幂律决定（r_k-1=0.00408·EP^1.52），跨模型盲测误差+0.4%。第1次swap覆盖全部有用距离，后续20次（59% ops）在死区内零收益。
2. **增益上界公式**：Δ_max=f_sens·x_eff/(1-f_sens·x_eff)，Amdahl形式。f_sens≠FLOP占比（组件分解：Combine β=1.33, Expert β=0.08, Dispatch β=-0.78）。增益=Δ_max×η，η由开销/bound决定。
3. **PD相关性的任务结构依赖**：QA/推理类ρ=0.78-0.85（94/94层强），数学类ρ=0.44-0.69。任务结构>>prompt长度——OBQA 15tok ρ=0.78 > prover 107tok ρ=0.44。prefill-only recording的充分性由任务类型决定。
4. **M统一与自适应衰减**：指数衰减累积器A_t=R_t+α·A_{t-1}的有效记忆M=W/(1-α)是偏差-方差权衡的唯一自由度，W与α只通过M影响稳态，早期实现调W不同步α会无意漂移M；变点检测时α瞬时归零清空旧域历史把响应延迟从M·ln2降至0，稳态按收敛/振荡伸缩决策窗口W追踪M*目标。同session实测该adaptive逻辑（+9.7%）超过固定α=0.9（+6.4%），固定α不必追求。

### 第8段：章节安排

§2相关工作，§3观察（4个insight：死区、增益上界、PD任务结构、M统一与M*闭式），§4框架设计（3个机制：死区感知停止、自适应窗口、仅prefill录制），§5实验评估，§6总结。

---

## 三、相关工作

### 3.1 静态布局

DataFore (ISCA 2026)提出prefill-guided专家放置算法（remap_based和dup_based），从prefill路由频率计算放置。但需离线profiling，无法适应运行时负载变化。本文实测（Fig 9）：MMLU的最优放置apply到prover后ratio=3.67，比默认identity(3.51)还差——跨域迁移必然失败。

### 3.2 周期重平衡

SGLang的EPLB每3000+步重平衡一次，需冗余专家副本（16额外副本+12.5%显存+KV cache-8.1%），每次重平衡阻塞0.5-4.5秒，强制deepep_mode=normal禁用CUDA graph（decode-heavy -62%至-68%）。且官方EPLB在非DeepSeek架构（Qwen2-MoE/Qwen3-MoE）上报AttributeError。

### 3.3 在线交换

本文PB-OEPLB：增量pairwise swap，零冗余，稳态阻塞0.37秒（EPLB的1/4），保留CUDA graph（deepep_mode=auto）。核心区别：基于死区理论感知何时停止，自适应窗口调整决策频率。

### 对比表格

| 维度 | EPLB | DataFore | PB-OEPLB (本文) |
|------|------|----------|----------------|
| 冗余专家 | 16副本 | 0 | 0 |
| 阻塞方式 | 全局重排 | 离线计算 | 稀疏pairwise swap |
| CUDA graph | 禁用 | N/A | 保留 |
| 死区感知 | 无 | 无 | auto r_k from EP幂律 |
| 自适应 | 固定周期 | 无 | cos_sim+ratio驱动 |
| 跨架构 | 仅DeepSeek | N/A | 通用fallback |

### 不同之处

本文的核心区别不在"swap vs重排"，而在**理论驱动的设计**：死区理论告诉"何时停"，增益上界告诉"最多能赚多少"，PD任务结构告诉"prefill-only何时充分"，M统一与M*闭式告诉"记忆长度如何自适应"（M*给目标，运行时启发式追踪之）。

---

## 四、观察（4个Insight）

### 观察1：死区——不均衡降低到r_k以下不产生时间收益

**场景**：在57B 8卡和235B 8卡上做T(r)扫描——7个布局点（从identity到oracle，ratio从~1.0到~2.6）×2轮独立重启，共14次运行0错误。对每个布局点的MoE层时间做铰链拟合。

**发现**：T(r) = T_flat + B·max(0, r-r_k)。r≤r_k时T=T_flat不变（DeepEP的dispatch/combine与GEMM的重叠吸收了全部落差）；r>r_k时T线性增长。铰链拟合R²=0.998，残差平方和比纯线性低12.1×。

**独特角度**：不是"不均衡有害"——而是"不均衡在r_k以下无害"。这改变了均衡器的停止条件：目标应是r→r_k而非r→1.02。默认threshold=1.02落在死区内[1.02, r_k]，为gap付出swap开销但换不回时间。

**r_k的可预测性**：r_k-1=0.00408·EP^1.52，4点拟合（57B EP2/4/8 + 235B EP8跨模型盲测，误差+0.4%）。指数1.52可分解为T_GEMM∝EP^{-0.99}（理论-1）和slack∝EP^{+0.53}（通信overlap增长）。

**量化**（Fig J）：57B 8卡21次决策中，第1次139 ops覆盖全部有用距离（r 1.216→1.017，Δr_eff=0.117），后续20次204 ops（59%总量）Δr_eff=0.000——全在死区内零收益。

### 观察2：增益有上界——特定模型+数据集的收益是有限的

**场景**：从死区推导Δ_max公式，跨3模型验证。

**公式推导**：T(r_before)/T(r_after) - 1 = [B·(r_before-r_k)] / T_flat = f_sens·x_eff / (1-f_sens·x_eff)，其中f_sens=B·r_before/T(r_before)（r敏感时间占比），x_eff=(r_before-max(r_after,r_k))/r_before（有效可消除比例）。

**独特角度**：30B不是"不均衡不存在"——是Δ_max=+6.36%（正！不均衡确实有害）但η≈0（r_k=1.031极窄→r_before=1.03→swap把r从1.03降到1.02全在死区内→零收益但开销record+all_reduce+swap照付→净≈0）。增益=f_sens·x_eff/(1-f_sens·x_eff) × η，η由swap开销/bound决定。

**f_sens≠FLOP占比**：235B FLOP占比67.9%，但实测f_sens=0.496（高估1.4×）。组件分解：Combine β=1.33（高度敏感，最热GPU的all-gather最慢→其他GPU等待），Expert β=0.08（几乎不受placement影响，token总数不变），Dispatch β=-0.78（负，OEPLB的all_reduce抢NVLink带宽）。

**硬件关系**：f_sens取决于GPU算力vs NVLink带宽（决定GEMM和dispatch/combine的overlap程度）；r_k取决于EP（幂律编码了T_GEMM∝EP^{-1}和slack∝EP^{+0.53}）。两者都与硬件有关，但幂律使r_k可跨配置预测。

### 观察3：prefill→decode相关性由任务结构决定，不是prompt长度

**场景**：7个域特定短prompt数据集（MMLU 25tok QA, ARC 31tok science, ARC-E 31tok, CSQA 20tok commonsense, OBQA 15tok science, GSM8K 60tok math, prover 107tok math），conc=256, O=10, Qwen3-235B-A22B-FP8。

**发现**：
- QA/推理类：ρ=0.776-0.849，83-94/94层强相关——即使15-31tok也强
- 数学类：ρ=0.443-0.686，0-37/94层强——即使60-107tok也弱

**独特角度**：OBQA 15tok ρ=0.776 > prover 107tok ρ=0.443——任务类型 >> prompt长度。QA类question→answer路由一致（同域同任务用同样专家），数学类题目→推导路由偏移（读题用A专家，算数用B专家）。

**充分性论证**：ρ高→prefill-only recording是decode分布的充分统计量（prefill的per-expert频率与decode的max/mean结构相同，只是总token数不同）。ρ低→需增大M补偿。时间衰减：ρ从early decode 0.62降到late 0.47（-24%），prefill对early decode预测最好——这正是PB-OEPLB在prefill边界记录的设计依据。

**聚合收敛**：W=16时ρ=0.753, std=0.046（93% of ceiling）→sync_window=16的经验依据。

**需补充实验**：代码生成类（HumanEval/MBPP）、多语言类（Global-MMLU多语言）、长文档理解类（LongBench/RACE）——验证PD相关性的任务结构规律是否跨任务类型成立。

### 观察4：跨数据集正交性 → 需 adaptive，M* 闭式给目标

**场景**：在不同（prompt长度L, 输出长度O, 内容域）组合上扫静态(W,α)，测每个workload的最优静态配置；同时测各workload决定M*的三个参数(r, L_seg, t̄)，并用同session adaptive vs 固定α对比。

**实证发现（正交性）**：
- **最优静态(W,α)跨workload不同**：oracle表显示最优sync_window从8（L256,O1）到64（L256,O1024 / L1024,O256），无单一固定配置对所有(L,O)最优。
- **决定M*的(r, L_seg, t̄)跨数据集近似正交变化**：r随域路由熵变（A 1.02-1.76, B 1.02-1.38）、L_seg随切换频率变（A 6段频繁、B 4域稳定）、t̄随prompt长度/并发变。三者独立漂移→M*跨workload差异大。
- 与跨域路由ρ≈0（Fig 4）同源：不同数据集激活不同专家→不同r/L_seg→不同最优M。

**理论（M*闭式）**：W与α不独立，只通过M=W/(1-α)影响稳态偏差-方差；联合最小化方差代价（M小→bias大）与变点延迟代价（M大→旧信号残留~M·ln2步），设段长L_seg：
M* = √( a·c²·L_seg / (b·β·t̄·γ²·(r−r_k)³·ln2) )
方向预测：M*∝√L_seg、∝(r−r_k)^{-3/2}。因(r,L_seg,t̄)跨workload正交变化，**M*必跨workload变化→固定(W,α)必在部分workload上偏离M*→需adaptive**。

**设计动机**：adaptive不是启发式补丁，而是追踪跨workload变化的M*目标。运行时用grow/shrink W + 变点α→0清零作为M*的离散近似（类比Adam追踪最优lr）。**同session实测**：adaptive（+9.7%）超过固定α=0.9（+6.4%）（swap 104 vs 56但吞吐反高→陈旧性损害>开销节省），零调参adaptive已优于任何固定配置。

**独特角度**：把"需要adaptive"从工程经验提升为可计算命题——不是"固定不好"的模糊断言，而是"M*随workload参数正交变化、固定配置必然偏离"的定量论证，并由M*闭式给出adaptive的追踪目标。

**结论**：跨数据集正交性→固定配置必然偏离→需adaptive；M*闭式给adaptive的目标。

---

## 五、框架/方案

### 5.1 概述

**挑战**：（1）何时停止swap？（2）决策频率与记忆长度如何按负载自适应设置？（3）prefill-only recording何时充分？**思想**：死区理论→auto-dead-zone自动算停止条件；M统一与M*闭式（观察4）→自适应窗口追踪M*目标；PD任务结构→prefill-only充分性边界。**框架**（Fig架构图），含routing tracer→controller→rebalancer→async executor→physical_to_logical_map 全局共享。

### 5.2 死区感知swap停止策略

- auto-dead-zone从EP幂律自动算r_k=1+0.00408·EP^1.52（不硬编码）
- r≤r_k时threshold=max(1.02, r_k)→阻止死区内swap
- Fig J的20/21次无用swap→启用后只剩1次有用决策→开销降95%
- 修复效果：unfixed adaptive -6% → fixed adaptive +9%（auto-dead-zone + cooldown + confirm_windows=2）

### 5.3 自适应窗口

- **理论依据（观察4）**：W与α只通过M=W/(1-α)影响稳态；本机制调W作为M*闭式（见§3观察4）的离散近似追踪目标，变点α→0清零把响应延迟从M·ln2降至0。同session实测+9.7%超固定α=0.9 +6.4%。
- 设计节奏：域切换（cos_sim<0.95）→shrink window→一次决定性swap→r≤r_k停止→grow window
- adaptive_decay：域切换时α→0一次（load.zero_()清零旧域信号），稳态恢复α按M*增长
- M*=f(L_seg, r, r_k)闭式：M*∝√L_seg/(r-r_k)^{3/2}，随段长增长、随信噪比下降
- 实测（Fig 15）：97次决策，域切换处ratio spike(1.35-1.72)→swap→稳态1.01-1.05
- Fig 16：逐域收敛（第1决策降幅最大-24%~-33%）
- Fig 17b：identity vs OEPLB逐域对比（prover 1.166±0.006→1.006±0.002，-14%，entropy 0→2.82）

### 5.4 prefill-only recording

- 仅is_extend()时记录，decode走CUDA graph零开销（torch.cuda.is_current_stream_capturing()检查直接返回）
- 充分性（观察3支撑）：ρ高→prefill是decode的充分统计量；ρ低→M增大补偿
- 边界条件：域切换时prefill作为decode的充分统计量不成立→域切换→窗口收缩→用新域数据覆盖旧域残留

### 算法流程图+复杂度分析

- 每sync_window forwards: O(1) all_reduce（94×128 int64=96KB）+ O(E) plan build + O(plan) swap P2P
- record_next_layer: O(E) scatter_add（E=128，单次GPU kernel）

---

## 六、实验

### 6.1 配置

- 硬件：8×H20（96GB/卡, NVLink NV18）
- 模型：Qwen3-235B-A22B-FP8（94 MoE层, 128专家, top-8, EP8=16/GPU）
- 软件：SGLang 0.5.6.post2, DeepEP v1.2.1, DeepGEMM FP8
- 数据集：7+域特定（MMLU/ARC/CSQA/OBQA/GSM8K/prover/book）
- 指标：tps = completion_tokens / elapsed_time
- 基线：identity(连续放置), EPLB(官方, patched), Frozen-EPLB, oracle

### 6.2 主结果

- Fig A：放置谱系 Worst→Baseline→EPLB→PB-OEPLB→Oracle，PB-OEPLB=23870(+18.4%), 达oracle 97.6%
- Fig B：ratio收敛 max-delta停滞1.26 vs adaptive pairing 3窗口到1.02

### 6.3 EPLB对比

- Fig C：全场景PB-OEPLB vs EPLB，PB-OEPLB每场景都赢，EPLB在ShareGPT负(-5%)
- Fig D：开销饼图 swap 3.42%主导
- Fig E：阻塞对比 稳态0.37s vs 1.55s = 4×快

### 6.4 消融

- Fig F2：α=0/0.5/0.9 sweep——α=0.5始终在最优点2pp内（鲁棒默认值而非最优值）；同session实测adaptive(+9.7%)超过固定α=0.9(+6.4%)，固定α非天花板
- Fig N：M*收敛（M≥32饱和，同M吻合4.8%）

### 6.5 OEPLB在线运行

- Fig 15：97次决策时间线+热点GPU（3面板）
- Fig 16：逐域收敛
- Fig 17b：identity vs OEPLB逐域对比

### 6.6 跨模型验证

- Fig H：Δ_max vs实际，η决定实得
- Fig L：57B(+2.7%, η=84%) + 30B(≈0%, η≈0)
- Fig K：r_k幂律跨模型盲测

### 6.7 典型case

- prover：identity 1.166±0.006（永远GPU5, entropy=0）→OEPLB 1.006±0.002（近完美, entropy=2.82），-14%（所有域中降幅最大）

---

## 七、总结

### 凝练4点

1. 死区：MoE时间T(r)铰链函数，r_k由EP幂律决定
2. 增益上界：Δ_max=f_sens·x_eff/(1-f_sens·x_eff)，η决定实得
3. PD任务结构依赖：QA强(0.78-0.85)数学弱(0.44-0.69)
4. M统一与自适应：W与α只通过M=W/(1-α)影响稳态，M*闭式给目标，运行时grow/shrink W+变点α→0清零追踪之；同session adaptive超固定α=0.9，零调参即最优

### 不足

- M*精确数值未定标（长benchmark无峰，短benchmark太噪）
- 小模型η噪声大（30B n=1，CV 5-9%）
- 未在>8 GPU测试（r_k幂律在EP≥16外推不确定度增大）

### 2个未来方向

1. 更大EP的r_k外推验证 + 硬件变化（B300/GB200）对f_sens的影响
2. 基于观察3的PD相关性，设计workload-aware的M*自适应（QA类用小M，数学类用大M）

---

## 八、参考文献

- 格式：作者, 标题, 会议/期刊(缩写), 页码, 年, DOI
- 来源：ACM/IEEE/DBLP/Google Scholar
- 关键引用：EPLB(SGLang), DataFore(ISCA 2026), DeepSeek-V3, Qwen3, Kimi K2, MoEntwine, WSC-LLM
