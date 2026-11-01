# 交付物 G — 理论指导的自适应窗口 v2：论文可用方法描述（含范围限定）

日期：2026-10-16 · 依据：交付物 B（机制规范）、D（校准行为+成本）、E（正式协议）、F（冻结后正式结果）
本文件是规划三项全部执行完毕后的最终方法描述。所有数字取自 F/handoff（程序化读出）。

---

## 0. 规划符合性审计（三项规划 → 已执行证据）

### 第一项：在 calibration 上完成窗口 v2（不先跑新的大规模服务战役）

| 规划要求 | 实现 | 证据 | 状态 |
|---|---|---|---|
| M1 可靠变化检测：新鲜逻辑路由统计+实际样本量，避免窗口缩小后把增大的统计波动当变化 | z = s_change/(κ/n_layer)：新鲜样本归一化余弦距离 vs EWMA 参考（β=0.5）；κ 校准拟合；检测/规划累计器分离；不以 \|Δr\| 单独判断 | 机制规范 §1.2；V4 schema（z, s_change, n_fresh, cos_legacy, r_legacy 全记录）；单元测试 T0–T10 | ✓ |
| M2 收益与成本感知调整：不仅看是否变化，还判断是否存在值得及时处理的损失、提前检查是否划算 | h_w = Σ_l max(0, max_rank_load_l − max(total_l/EP, max_expert_l))（含不可分专家下界）；收缩需 h_w ≥ θ_act（有可回收损失才动）；增长需 h_w < θ_H_low（余量大则不值得提前检查） | 规范 §2.1；W0 s1 实测 3 次 shrink_rejected_low_headroom + 14 次 grow_blocked_headroom；M2off 对照（§8） | ✓ |
| 边界、迟滞、冷却、多 rank 一致性为统一实现约束 | 网格 {8,16,32,64}；m_shrink=2/m_grow=3 确认；dwell 2/2；urgent episode cap 3；每 16 窗口决策 hash allreduce，desync → 冻结 w_init | 规范 §3.1；24 个控制器启动零 desync；T7b desync→确定性冻结单元测试；smoke C run1 假 desync 事件保留在案（安全模式实证） | ✓ |
| 每次改 W 可回答：为何现在改/减少什么损失/增加什么成本/证据不足怎么办 | 每变更日志含 z（为何）、h_w（可回收损失代理）、reasons 字符串（规则级归因）、c_window_ms（成本）；低样本 → UNCERTAIN 显式状态（保持 W、冻结 ref、carry≤2、u_max=4 强制缩到中位） | v2_rank0.jsonl 逐窗口记录；F 报告 T3 全变更证据链 | ✓ |
| 初始窗口和允许范围由 calibration 确定，不把 16 写成普适答案 | W_init=16 与全部阈值/网格边界在 W0（calibration.jsonl）上盲拟合（R5–R9），冻结记录在案；固定基线同等享有校准（F-local = 网格 argmin） | FREEZE_record_v2_knobs.md；F §7 | ✓ |
| 固定基础 α 和历史策略，只改窗口 | α=0、D、planner/acceptance/executor 全部不变；V5 验证 remapping/planner/rebalancer/executor 零 diff | step4_verification_record V3/V5 | ✓ |
| 时间预测器有系统偏差 → 只辅助判断量级，不当严格上界/精确门控 | J(W) ≈ c_check/W + λ·ℓ·(W−1)/2 + E_est 仅作离线设计分析，从不在线求最小；在线门控全部用 h_w（赋值单位保守代理）+ 校准阈值；h_w 的"经验 μs 换算"仅日志标注 | 规范 §3.2；V4（labeled empirical us conversion, log-only） | ✓ |

### 第二项：小型机制验证（calibration 请求，四场景）

| 场景 | 要验证的行为 | 实测证据 | 结论 |
|---|---|---|---|
| 相对稳定、候选剩余收益低 | 减少无益检查而不长期失去观察能力 | W1_V2 s2：全程 **1 次变更**，87% 步数驻留 W=32（检查频率减半）；仍每窗检查 + 每 16 窗 hash 心跳；grow_blocked_headroom 挡住 W=64（观察能力保留在中位而非极端） | ✓ 符合声称 |
| 分布变化且有明显布局机会 | 及时重新评估、采取有用调整 | W2 切换：36 步内 urgent 32→16（z=271.8/244.0，两启动逐窗口一致）；W0 B1/B2：36–70 步检出，均伴随 h_w ≥ θ_act（确有机会才动） | ✓ 符合声称 |
| 样本少、统计不确定 | 避免连续误触发和窗口反复变化 | UNCERTAIN 规则经单元测试 T6（hold→carry≤2→u_max=4 强制缩中位→样本恢复即重置）；**诚实注记：服务运行中从未触发**——饱和 chunked prefill 下 n_fresh≈49.3M/窗 ≫ n_min=12.3M，该路径仅逻辑级验证，未经生产负载行使 | ✓（逻辑级）／⚠（未经生产行使） |
| 变化存在但收益很小 | 避免仅因变化就频繁调整 | W0：3 次 shrink_rejected_low_headroom（z=34.7/55.4/37.6 ≥ z_hi 但 h_w < θ_act → 不动作）；边界 B3（轻微漂移）全 5 臂静默（z_max 1.95 ≪ z_lo） | ✓ 符合声称 |

必过项：多 rank 检查时刻一致（hash allreduce 每 16 窗，24 启动零 desync）✓；窗口上下界（网格强制，越界不可能）✓；开关独立（V3 v1 字节等价 + V5 每启动读版本、无中途切换）✓；输出正确性（V4 schema 完整 + 全部运行 ok=N/errors=0）✓。
记录字段齐全：z/s_change（变化信号）、n_fresh（样本量）、W before/after、h_w（收益代理）、c_window_ms（检查成本）、event+reasons（实际动作）✓。

### 第三项：冻结后有限净效果验证 + 内嵌消融

| 规划矩阵 | 实际执行 | 状态 |
|---|---|---|
| 校准选出的强固定窗口 × {稳定, 变化} | FL（W1→8, W2→32）× {W1, W2}，另加 W0 与 FG（全局 W16） | ✓（超出） |
| 完整窗口 v2 × {稳定, 变化} | V2 × {W1, W2}，另加 W0 | ✓（超出） |
| v2 − M1 不确定保护 × 1 条件 | M1p（仅 β=0）× {W0, W2} | ⚠ 部分消融（新鲜归一化与 UNCERTAIN 无法经配置关闭）——限制已预声明并写入 F |
| v2 − M2 收益/成本依据 × 同一条件 | M2off × {W0, W2}，另加 M3off（成本/滞回关闭） | ✓（超出） |
| 工作负载事先选定、变化尺度不按 evaluation 结果挑选 | W1/W2 数据集 sha256 冻结于任何正式运行之前；W0 声明为已知回归基准 | ✓ |
| 7963 条已见请求只用于回归；确认性结论用未参与开发的真实请求 | 旋钮仅在 W0 盲拟合；W1（稳定 holdout）/W2（预定义切换）承担确认性角色 | ✓ |
| 固定基线也允许 calibration；基础 α/planner/executor 一致 | 同一网格、同一基座、同一并发 | ✓ |
| 建议预算：5 配对启动/条件（20+10=30） | **偏差**：Protocol E 预注册 36 启动 = 12 网格 + 18 正式（**n=2 配对**）+ 6 消融（n=1）——同量级预算换了更宽的矩阵、更浅的配对深度；冻结后按"预算冻结→结果不确定就报告不确定，不根据显著性追加"规则**不补跑** | ⚠ 已登记偏差（§10-L1） |
| 小效应无法分辨就如实报告，不追加到显著为止 | F：时间维度差值一律"与零相容/符号一致"语言，无 CI 声称，无追加 | ✓ |
| 单模块消融只改对应机制 | ABL_* 环境变量仅覆盖对应旋钮，其余冻结值原样；ACTIVE 日志逐启动核验 | ✓ |

## 1. 论文方法描述（可直接改写为正文）

### 1.1 问题与控制对象

在线专家重均衡系统（PB-OEPLB）以 W（forward 步数）为窗口聚合逻辑路由统计并周期性重规划。W 是新鲜度—成本的单一旋钮：小 W 提高对路由变化的响应速度但放大统计波动与每步检查成本；大 W 相反。基线系统使用固定 W（A00 = 固定 W 固定 α 的在线 PB-OEPLB，非静态 identity）。本文方法将 W 的控制替换为理论指导的自适应控制器（v2），其余组件（planner、acceptance、executor、历史策略 D、α）严格不变，以隔离窗口控制这一单一因素。

### 1.2 从理论到控制量的映射

三条理论观察被翻译为两个在线信号，而非一个固定窗口公式：

1. **就绪裕量（readiness slack）**：并非所有路由分布变化都产生完成时间损失——只有当变化使某 rank 的最大就绪时刻超过不可分下界 max(total/EP, max_single_expert) 时，才存在可回收的时延。据此定义**可回收余量** h_w = Σ_l max(0, max_rank_load_l − max(total_l/EP, max_expert_l))（赋值单位，保守代理；经验时间换算仅离线标注）。h_w 回答"现在最多还能回收多少、值不值得提前检查"。
2. **有限样本下的检测**：窗口缩小后统计波动天然增大，直接比较分布会把波动误判为变化。据此定义**噪声归一化变化分** z = s_change/(κ/n_layer)，其中 s_change 为新鲜归一化赋值统计对 EWMA 参考（β=0.5）的余弦距离，κ 由校准负载拟合，n_layer 归一消除层数尺度；检测累计器与规划累计器分离；样本量 n_fresh < n_min 时进入显式 UNCERTAIN 状态（保持 W、冻结参考、跨窗累计证据至多 2 窗、连续 4 窗强制缩到中位网格点）。z 回答"这是真变化还是估计噪声"。
3. **成本—延迟权衡 J(W) ≈ c_check/W + λ·ℓ·(W−1)/2 + E_est(W,N,α)** 仅作为**设计分析**用于离线解释网格与 dwell 边界的合理性（c_check 实测、λ 来自负载切换频率、ℓ 由校准运行离线估计），从不在线求最小、从不作为未经验证的在线定律。时间预测器存在系统偏差，故只用于量级判断，不作为严格上界或精确小收益门控。

### 1.3 控制律（M1/M2/M3）

单一权威状态机（每窗口一次，全 rank 相同输入 → 相同决策）：
- **紧急收缩（M1+M2 合取）**：z ≥ z_urgent ∧ h_w ≥ θ_act → 立即降一格（绕过冷却；每事件上限 3 次）——强证据且有可回收损失时才允许最快响应。
- **确认收缩**：连续 m_shrink=2 窗 z ≥ z_hi ∧ h_w ≥ θ_act。
- **确认增长（M2 成本侧）**：连续 m_grow=3 窗 z < z_lo ∧ h_w < θ_H_low ∧ 稳定计数 ≥3——只有"没变化且剩余收益低"才延长窗口降低检查成本；h_w ≥ θ_H_low 时拒绝增长（grow_blocked_headroom），防止在潜在损失未回收时钝化观察。
- **负决策（M2 核心）**：z 高但 h_w < θ_act → shrink_rejected_low_headroom，不动作——"高原始信号 ≠ 时间收益"。
- **M3 统一约束**：网格边界 {8,16,32,64}、双向 dwell、确认计数、紧急帽、每 16 窗决策 hash 一致性校验（不一致 → 全体冻结于 W_init）。系统不声称回退路径"不慢于固定"——它只保证有界与一致。

每次 W 变更日志回答四问：为什么现在改（z、规则归因 reasons）、预计减少什么损失（h_w）、增加什么成本（c_window_ms、dwell 状态）、证据不足怎么办（UNCERTAIN 路径）。

### 1.4 校准与冻结

全部旋钮（κ、z_lo/z_hi/z_urgent、θ_act/θ_H_low、n_min、m、dwell、cap、β、W_init）在校准负载上盲拟合后冻结（sha256 代码锁 + 冻结记录 + 冻结规则：不事后调参、缺陷记录不打补丁、修改需新预注册规则并重冻结）。固定基线与 v2 同等享有校准（F-local 取网格 argmin）。

### 1.5 验证设计与主要结果（详见交付物 E/F）

3 条件（已知回归 W0 / 稳定 holdout W1 / 预定义一次切换 W2）× 3 臂（V2 / F-global W16 / F-local argmin）× 2 配对启动 + 3 消融 × 2 条件，共 36 个预注册计时启动，全部 PASS（ok=N、errors=0、零 desync）。

**行为结果（机制按声称工作）**：
- 预定义真实变化检出延迟 32–74 步（5 臂无漏检）；非事件边界全臂正确静默；相同输入下轨迹逐窗口比特一致（确定性）。
- 稳定负载：单变更、87% 步数驻留 W=32、零抖动。
- 监控成本 0.54–0.66 ms/窗（中位），全部含在报告时间内。
- **M2 门的因果证据（本战役最清晰的消融分离）**：关闭余量门（M2off）后控制器过扩展至 W=64，恰逢切换 → 检出延迟 72 步 = 2× Full（36 步），z 爆至 876；Full 靠 14 次 grow_blocked_headroom 从未到 64。
- M3 关闭：单窗口即动作、回长更快（更多抖动暴露），此余量下无时间后果。
- M1（β=0 部分消融）：边界检出与 Full 相当；无启动瞬态收缩；切换后 z 回落更慢、回长推迟。

**时间结果（诚实）**：n=2 配对下，无任何条件显示净服务时间胜利——W0 两启动均慢于 F-global（+4.76s/+1.05s，原样保留）；W1 对 FG 与零相容、对 FL 的负差有网格噪声混杂；W2 差值符号在启动离散内翻转（与零相容）。**根因是测量出来的**：固定 W 网格本身只跨 0.5–2.3%（W0 0.50%、W1 1.44%、W2 2.32%）——α=0 下窗口大小的时间余量小于启动噪声，任何控制器都无法在此设置下把行为质量换算成墙钟收益。

### 1.6 "理论指导了自适应窗口优化"的证据链（规划要求的闭合条件）

规划规定：只有当理论量**实际影响了代码决策**，并通过机制验证与净效果比较，才能作此声称。闭合情况：
1. 理论量进入决策：h_w（就绪裕量+不可分下界）直接门控每一次收缩/增长（θ_act/θ_H_low）；z（有限样本检测）直接门控每一次动作；c_window（成本）实测并进入设计分析。——**是**（代码级，非事后解释）。
2. 机制验证：四场景全部按声称行为（§0 第二项表），负决策与阻断事件逐条有日志。——**通过**。
3. 净效果比较：预注册、冻结、双基线、消融内嵌，负结果原样保留。——**完成**；结论为"行为稳健性成立、净时间收益在窗口余量 ≤2.3% 的机制下不可分辨"。
4. 因此论文可声称：**理论量被翻译为在线控制量并因果地改变了决策与行为（消融分离证实），系统在真实负载上按设计运行**；不可声称：在本机制下获得了端到端时间收益。

### 1.7 适用范围限定（规划的分叉条款："如果新版仍没有优势，就进一步限定它的适用范围"）

**v2 适用的机制**（预期能把行为优势换算成时间的场景，均为外推、未经本战役验证）：窗口余量更大的设置——更大 W 网格或更长的稳定段（检查成本项 c_check/W 更陡）、α>0 时陈旧统计直接损害重均衡质量、路由变化更频繁/更剧烈（λ·ℓ 项主导）、更长的运行（dwell/确认的一次性成本被摊薄）。
**v2 不适用的场景**（本战役实证）：α=0 且窗口余量 ≤2.3% 的负载——固定 W 的任意选择都近似等效，自适应只增加复杂度不产生可测收益；此时应直接用校准选出的固定 W。
**明确不声称**：回退/冻结路径不慢于固定 W；UNCERTAIN 路径在生产饱和负载下有效（未经行使）；M1 整体有效（仅 β 可消融）。

## 2. 建议的论文表述（英文摘要句，可直接改写）

> We translate three theoretical observations — readiness slack under an indivisible-expert lower bound, finite-sample detectability of routing shifts, and the check-cost/delay-loss trade-off — into two online control signals (a noise-normalized change score z and a recoverable-headroom proxy h_w) that jointly gate every window adjustment of an online expert-rebalancing planner. The controller is calibrated once on a development workload, frozen, and evaluated against equally-calibrated fixed-window baselines under a pre-registered protocol on 8×H20 with Qwen3-235B-A22B-FP8. It detects predefined real routing changes within 32–74 engine steps with bit-identical trajectories across replicate runs, correctly remains silent on non-events, refuses changes whose recoverable headroom does not justify them, and halves the worst-case detection latency relative to an ablation without the headroom gate (36 vs 72 steps), while its per-window monitoring cost (≈0.6 ms) is included in all reported service times. Because the fixed-window time headroom on these workloads is itself only 0.5–2.3% at α=0, no net end-to-end time win is claimed or demonstrated; we characterize precisely the regime (larger window headroom, staleness-sensitive rebalancing) in which the mechanism's behavioral guarantees are expected to convert into time, and the regime in which a calibrated fixed window remains the right choice.

## 3. 限制登记（L 编号，论文 limitation 段素材）

- **L1 配对深度偏差**：规划建议 5 配对启动/条件（30 启动）；预注册 Protocol E 改为 36 启动 = 12 网格 + 18 正式（n=2 配对）+ 6 消融（n=1）——预算同量级、矩阵更宽、配对更浅。冻结后依"预算冻结→不确定就报告不确定，不根据显著性追加"规则未补跑。时间维度结论因此只能是"与零相容"，不能排除 ≤1–2% 量级的真实效应。
- **L2 M1 部分消融**：仅 β（EWMA 参考）可经配置消融；新鲜归一化与 UNCERTAIN 路径不可关。M1p ≠ M1-off。
- **L3 UNCERTAIN 未经生产行使**：饱和负载 n_fresh≈49.3M/窗 ≫ n_min=12.3M，该路径仅单元测试（T6）验证。
- **L4 时间预测器系统偏差**：h_w 的经验 μs 换算仅日志标注；一切门控用赋值单位+校准阈值。
- **L5 W0 双重角色**：既是旋钮拟合来源又是已知回归条件；其结果只重述、不作验证证据。确认性角色由 W1/W2（sha256 冻结于正式运行前）承担。
- **L6 单模型/单硬件/单并发**：8×H20、EP=8、Qwen3-235B-A22B-FP8、并发 256；不外推到其他 EP/硬件。
- **L7 FL 网格 argmin 噪声**：每格 1 启动，W1 已实证噪声抽中风险（网格 351.05 vs 正式 357.98/354.36）；事后最优固定格只作带标签参考。
- **L8 n=1 消融**：消融臂各 1 启动，行为差异（检出延迟、过扩展、抖动）为单样本观测；其方向与机制解释一致但无重复。
- **L9 prompt 池共享（内容级分离有限）**：W1 的 125 个唯一 prompt（各×32 重复）全部出现在 calibration 与 evaluation 的 code 段；W2 的 125 个 code prompt 同样共享，但其 125 个 chinese prompt 不在 calibration/evaluation 任何段中（逐段交集当日验证）。决策级分离仍成立（阈值只在 W0 拟合、W1/W2 预注册冻结、三臂收同一文件故无臂间优势），但"未见真实请求"仅在文件/序列层面与 W2 切换后域的内容层面成立，W1 不构成内容级新颖。系有限真实 prompt 池的固有结果，详见 handoff_addendum2_workloads.md §3。

## 4. 复现锚点

- 冻结记录：FREEZE_record_v2_knobs.md（旋钮+代码 sha256+冻结规则）
- 机制规范：adw_v2_mechanism_spec.md（§1.2 z、§2.1 h_w、§1.3 UNCERTAIN、§3.1 状态机、§3.2 J(W) 离线地位）
- 验证记录：step4_verification_record.md（V1 单元 T0–T10+T7b、V2 多 rank、V3 v1 等价、V4 schema、V5 开关不变量；假 desync 失败运行保留）
- 行为+成本：calib_step5/behavior_report_D.md
- 协议：step6_formal_protocol_E.md；数据集 hash：step6_dataset_hashes.txt
- 结果：results/step6/deliverable_F_report.md + deliverable_F_analysis.json + phaseA|B|C 原始 bench.json/server.log/v2_rank0.jsonl
- 交接：handoff_step6_full_data.md（外部 AI 分析用自包含文档）
- git（本地，未 push）：de917bf → ff08daf → 2edb810 → 088723a → a63b1ef → 411572f → c634941 → 8231f4a → 563871a
