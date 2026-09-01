# 论文Outline

## 4个核心观察（Insight）

### 观察1：死区——不均衡降低到r_k以下不产生时间收益

**场景**：T(r)铰链曲线扫描（7布局点×2轮，14次运行0错误），发现T在r≤r_k时flat不变。

**独特角度**：不是"不均衡有害"——而是"不均衡在r_k以下无害"。这改变了均衡器的停止条件：目标应是r→r_k而非r→1.02。

**关键数据**：
- T(r) = T_flat + B·max(0, r-r_k)，铰链拟合R²=0.998，比线性拟合RSS低12.1×
- r_k幂律：r_k-1 = 0.00408·EP^1.52，跨模型盲测（235B predicted 1.097 vs measured 1.093，误差+0.4%）
- Fig J：第1次swap 139 ops=100%有用距离（r 1.216→1.017），#2-#21 204 ops=0%收益（全在死区内）
- 指数分解：T_GEMM∝EP^{-0.99}（理论-1）/ slack∝EP^{+0.53} = EP^1.52

**图表**：Fig I（T(r)铰链曲线）、Fig K（r_k幂律）、Fig J（边际swap效率）

### 观察2：增益有上界——特定模型+数据集的收益是有限的

**场景**：从死区推导Δ_max公式，跨3模型验证（235B +17.5%/η=79%，57B +2.7%/η=84%，30B ≈0%/η≈0）。

**独特角度**：30B不是"不均衡不存在"——是Δ_max正(6.36%)但η≈0（死区极窄r_k=1.031→swap全在死区内→零收益但开销照付）。

**关键公式**：
- Δ_max = f_sens·x_eff / (1 - f_sens·x_eff)（Amdahl形式）
- f_sens = B·r_before / T(r_before)≠FLOP占比（组件分解：Combine β=1.33, Expert β=0.08, Dispatch β=-0.78）
- x_eff = (r_before - max(r_after, r_k)) / r_before
- Δ_actual = Δ_max × η（η由swap开销/bound决定）

**硬件关系**：f_sens和r_k都与硬件有关（f_sens=routed GEMM在临界路径占比，取决于GPU算力vs NVLink带宽；r_k=overlap/GEMM比，取决于EP）。但幂律使r_k可预测，不需每次扫描。

**图表**：Fig H（Δ_max vs实际，η决定实得）、Fig L（30B案例：Δ_max正但η≈0）

### 观察3：prefill→decode相关性由任务结构决定，不是prompt长度

**场景**：7个域特定数据集（ARC/MMLU/CSQA/OBQA/GSM8K/prover×2），conc=256, O=10, Qwen3-235B-A22B-FP8。

**独特角度**：OBQA 15tok ρ=0.78 > prover 107tok ρ=0.44——任务类型 >> prompt长度。QA类question→answer路由一致（同域同任务），数学类题目→推导路由偏移（读题用A专家，算数用B专家）。

**关键数据**：
- QA/推理类（ARC 0.849, ARC-E 0.841, MMLU 0.788, CSQA 0.804, OBQA 0.776）：83-94/94层强
- 数学类（GSM8K 0.686, prover 0.443）：0-37/94层强
- 时间衰减：ρ从early decode 0.62降到late 0.47（-24%），prefill对early decode预测最好
- 聚合收敛：W=16时ρ=0.753, std=0.046（93% of ceiling），sync_window=16的经验依据

**需补充实验**：代码生成类、多语言类、长文档理解类数据集

**图表**：Fig 5（逐层ρ柱状图）、Fig 14（7数据集PD bar图）、Fig 8（ρ vs prompt长度）

### 观察4：α最优值是U形——固定α只能在一个区域最优

**场景**：4组实验（长prompt conc32, 长prompt conc32×6域, 中prompt conc32, 中prompt conc256），每组测α=0/0.5/0.9。

**独特角度**：α=0.5从未最优但始终在2pp内——是"鲁棒的默认值"而非"最优值"。最优取决于benefit-per-swap：高r→少swap够用（α=0.9赢），中r→多swap最大化收益（α=0赢），低r→少swap少亏（α=0.9赢）。

**关键数据**：
- 长prompt(4438tok) conc32: α=0/0.5/0.9 = +5.4%/+6.1%/**+9.9%**
- 长prompt(4438tok) conc32 6域: α=0/0.5/0.9 = +5.6%/+5.3%/**+12.6%**
- 中prompt(1000tok) conc32: α=0/0.5/0.9 = -7.4%/-5.0%/**+0.1%**
- 中prompt(1000tok) conc256: α=0/0.5/0.9 = **+7.7%**/+5.7%/+6.7%

**结论**：固定α必然只在一个区域最优→需adaptive自动切换策略。

**图表**：Fig F2（α sweep 3面板决策时间线）

---

## 论文结构

### 一、摘要（200-300字）

1-2句背景+问题；1-2句现有不足；1句引出方案；2-3句方案内容；2-3句实验配置+效果。

### 二、Introduction

- 第1段：背景（MoE主流+路由偏斜+straggler）
- 第2段：问题（不均衡2-12×, 跨域失败, Fig1/2数据）
- 第3段：相关工作（EPLB/DataFore/在线swap）
- 第4-5段：本文方案（4个insight→OEPLB系统）
- 第6段：实验（235B/57B/30B, 7+数据集, +17.5%）
- 第7段：创新4点
- 第8段：章节安排

### 三、相关工作

- 3.1 静态布局（DataFore Case Study 2）
- 3.2 周期重平衡（EPLB：冗余+阻塞+CUDA graph禁用-68%）
- 3.3 在线swap（OEPLB：本文）
- 对比表格：冗余专家？| 阻塞方式 | CUDA graph兼容 | 死区感知 | 自适应
- 一段描述本文不同之处

### 四、观察

- 观察1：死区（Fig I, Fig K, Fig J）
- 观察2：增益上界（Fig H, Fig L）
- 观察3：PD任务结构依赖（Fig 5, Fig 14, Fig 8）+补充实验
- 观察4：α U形（Fig F2）

### 五、框架/方案

- 5.1 概述：挑战→思想→框架图（figure1/）
- 5.2 死区感知swap停止（auto-dead-zone from EP幂律, Fig J→阻止无用swap）
- 5.3 自适应窗口（α U形→adaptive, 设计节奏, M*闭式, Fig 15/16/17b）
- 5.4 prefill-only recording（观察3支撑, CUDA graph零开销, 充分性边界）
- 5.5 M=W/(1-α)统一控制（bias-variance, M*闭式, Fig N）
- 算法流程图+复杂度分析

### 六、实验

- 6.1 配置（235B/57B/30B, 8×H20/4×H20, 7+数据集, 指标定义）
- 6.2 主结果（Fig A放置谱系+18.4%, Fig B收敛3窗口）
- 6.3 EPLB对比（Fig C全场景, Fig D/E开销+阻塞4×）
- 6.4 消融（α sweep Fig F2, M*收敛 Fig N）
- 6.5 OEPLB在线运行（Fig 15 97次决策, Fig 16逐域收敛, Fig 17b identity vs OEPLB）
- 6.6 跨模型验证（Fig H/L η两段结构, 30B案例）
- 6.7 典型case（prover: 1.166→1.006, entropy 0→2.82）

### 七、总结

- 凝练4点insight
- 不足：精确M*未定标；小模型η噪声；未在>8 GPU测试
- 2个未来方向：更大EP的r_k外推；硬件变化对f_sens的影响

### 八、参考文献

- ACM/IEEE/DBLP格式
- 作者+标题+会议/期刊+页码+年+DOI
