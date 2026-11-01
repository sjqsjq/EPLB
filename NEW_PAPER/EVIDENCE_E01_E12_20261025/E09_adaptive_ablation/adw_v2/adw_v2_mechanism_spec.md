# ADW v2 机制规格（第二步交付：M1/M2/M3 设计，实现前冻结草案）

status: design_spec_draft_before_implementation
对应缺陷：audit_v1_window_control.md F1-F12
版本开关：`--pb-oeplb-window-controller v1|v2`（默认 v1 = 与锁定代码逐字节行为一致；
v2 关闭全部 5 个旧变更点，由唯一 `_window_control_v2()` 写 `_effective_sync_window`）。
第一阶段冻结：base α=0.0（校准 argmin，FACT-09）、adaptive_decay=OFF（两臂相同；α=0 下
RESET 的唯一实际作用是整窗丢弃规划，属有害副作用，见 F8；D 的历史策略研究留待第二阶段，
届时同 α 隔离并面对校准选出的强固定策略）、规划器/接受规则/预算/执行器/钩子不变。
注意：两臂都保留 decide 路径的 94 层 ratio 循环（F12 不做，避免给 v2 臂引入
非窗口来源的每窗 ~20ms 优势；F12 留作单独标记的后续优化）。

## 0. 在线量 vs 离线解释（防止理论与代码脱节）

在线实际使用（每检查窗，全部来自同一 all_reduce 的逐位一致数据）：
  N_fresh（本窗新鲜记录 token 数，标量）、S_change（新鲜分布变化分）、z（噪声归一化变化分）、
  H_w（不可分下界之上的剩余 excess，assignment 单位）、ĉ（每窗检查成本 EWMA，ms，仅日志/离线）、
  状态机 {STABLE, ALERT, UNCERTAIN, DWELL}、W 网格 {8,16,32,64}。
仅离线解释/校准（不进在线决策）：
  J(W) ≈ c_check/W + λ·ℓ·(W−1)/2 + E_est(W,N,α) —— 用于在开发负载上离线分析网格与 dwell
  边界的合理性（c_check≈23-28ms 已测；λ=域切换频率，单位：切换/forward step，本负载
  ≈3/1072≈0.0028；ℓ=切换后未调整时每 forward 执行损失，只能由 calibration 运行离线估计，
  带误差报告）；κ（噪声系数）由 calibration 平稳段拟合；assignment→µs 换算
  （E-T 斜率 0.135491µs/行，带角色偏差 +52.9~+115.1µs，FACT-07）只进日志与离线分析，
  在线决策一律用 H_w 的 assignment 单位相对比较。不宣称 J(W) 给出"最优窗口"；
  未知参数不设为任意常数后反推理论最优。

## 1. M1：区分真实变化与估计不确定性

### 1.1 新鲜检测统计（与规划历史分离）
- 第一阶段 base α=0.0：现有 `self.load` 每窗末清零 → all-reduced `global_load` 本身就是
  本窗新鲜统计，检测无需第二累计器、零额外热路径成本（此等价性写入代码断言：
  v2 要求 cfg.decay_factor==0.0，否则启动报错拒绝——α>0 的新鲜向量需要第二 scatter，
  记录成本 ~2×(+9s/轮)，留给 D 阶段并计入成本）。
- 样本量：`total_tokens`（每 rank 本窗 layer-0 新鲜记录数，现有变量）作为第 94×128+1 个
  元素拼进同一个 all_reduce（零新增 collective）；N_fresh = 全 rank 和。每窗记录
  steps_actual、prefill_batches、N_fresh——不再假设每窗等样本。

### 1.2 变化分与噪声归一化
- p_w = 逐层归一化新鲜负载（[94,128] 行归一），检测参考 ref = 检测专用 EWMA：
  ref ← β·ref + (1−β)·p_w（β=0.5，GPU 上一个 [94,128] float 张量；与规划历史无关）。
- S_change = mean_over_layers( 1 − cos(p_w[l], ref_prev[l]) )（valid 层：双方范数>0）。
- 噪声基线：平稳状态下两独立多项式样本的 1−cos 期望 ∝ κ/N_layer_fresh
  （N_layer_fresh = N_fresh×top_k/94 的每层 assignment 数）。z = S_change / (κ/N_layer_fresh)。
  κ 在 calibration 平稳段离线拟合（记录拟合残差与适用 N 范围）。
- 效果：W 收缩→N_fresh↓→噪声基线↑→z 对纯噪声不升——从结构上消除 F2 正反馈；
  Δr 不再单独作为域变化判据（placement 改善同样改 r，F3）；cos_sim 不再跨 α 语义漂移（F10）。

### 1.3 UNCERTAIN 状态（低样本显式规则）
- N_fresh < N_min（calibration 定）→ 状态 UNCERTAIN：W 保持不变（双向都不动），
  检测 EWMA 冻结不更新（防止小样本污染 ref），新鲜统计跨窗累计至多 2 窗
  （carry≤2：超过则丢弃最旧窗并计数 uncertain_drops）；连续 uncertain ≥ U_max 窗
  → 强制 W←min(W, W_mid)（保守缩到中位网格点，保证响应上界），事件记 forced_shrink_uncertain。
- uncertain 不是拒绝：证据照常累计，规则确定、离线可复演。

## 2. M2：剩余收益空间影响窗口

### 2.1 headroom 代理 H_w（保守、明确局限）
物理槽→rank 即 slot//16（load 为物理槽索引，无需 p2l 反查）。逐层（向量化，无 .item()）：
  rank_loads[l,r] = Σ_{slot∈r} global_load[l,slot]
  floor_l = max( total_l/EP , max_expert_l )        # 不可分专家工作下界（count 空间）
  excess_l = max(0, max_r rank_loads[l,r] − floor_l)
  H_w = Σ_l excess_l                                  # 单位：assignments/窗
- 这是"当前最坏 rank 超出不可分下界的可回收工作量"的代理。明确局限（写入代码注释与论文）：
  (i) 窗累计 count ≠ 单 forward 批形状，padding/对齐阶梯未在线定价；
  (ii) count→时间换算存在角色系统偏差（FACT-07），故在线只用 H_w 与校准阈值的相对比较；
  (iii) H_w 低不证明收益为零——模型定价能力不足时按保守规则行动并记录，不当成已验证预测器。
- raw r 不进入 v2 决策（只进日志对照）；dead_zone 1.096 继续只门 swap（C6 不动）。

### 2.2 headroom 进入窗口决策的规则
- 允许延长（STABLE 中）：需同时 z < z_lo 连续 m_grow 窗 且 H_w < θ_H_low（剩余收益低）；
  H_w ≥ θ_H_low 时即使分布平稳也不延长（还有可回收空间，保持检查频率），事件 grow_blocked_headroom。
- 允许缩短：z ≥ z_hi（可信变化）且 H_w ≥ θ_H_act（变化后存在可回收损失）；
  z 高但 H_w 低 → ALERT 不收缩（变化真实但无处可赚），事件 shrink_rejected_low_headroom。
- θ_H_low/θ_H_act 只在 calibration 上定；不把一个固定 deadzone ratio 当跨域定律——
  θ_H 是本网格/本负载族的校准常数，换负载须重校（写明）。

## 3. M3：单一权威、成本、迟滞与有界响应

### 3.1 状态机（唯一写 W 的地方）
状态：STABLE / ALERT(k) / UNCERTAIN / DWELL(j)。每窗转移（伪代码）：

  on_window(g, N_fresh, ĉ):
    S, z, H = compute(g, N_fresh)            # §1.2, §2.1
    log_window(...)                          # §5 行为日志
    if N_fresh < N_min: -> UNCERTAIN 规则(§1.3); return
    if z >= z_urgent and H >= θ_H_act:       # 强确认变化：立即收缩一次（每事件限 1 级）
        W <- grid_down(W); -> DWELL(dwell_shrink); event urgent_shrink; return
    if z >= z_hi and H >= θ_H_act: ALERT(k+1)
        if k+1 >= m_shrink: W <- grid_down(W); -> DWELL(dwell_shrink); event confirmed_shrink; k=0
    elif z < z_lo:
        STABLE 计数 s+1；ALERT 清零
        if s >= m_grow and H < θ_H_low and not in DWELL:
            W <- grid_up(W); -> DWELL(dwell_grow); event confirmed_grow; s=0
        elif s >= m_grow and H >= θ_H_low: event grow_blocked_headroom
    else: 中间带 -> STABLE 计数保持、ALERT 清零（迟滞带，不双清零）
    DWELL(j)：j 窗内禁止同方向再次变更（反方向按各自确认规则；urgent 例外）

- 边界：W ∈ {8,16,32,64}（复用已有实现与校准资料的网格，FACT-09；不新造网格点）；
  W_init = calibration 选定固定点（本负载族 W*=16；若后续负载校准出不同 W*，用各自 W*，
  这正是"不同负载是否有不同合适固定窗口"的检验点）。W_max=64 → 响应上界：任何变化最迟
  64 step + m_shrink×64 step 内被检查（有界响应，不宣称瞬时发现）。
- 无心跳/探针新增：v2 每窗额外成本 = 1 个标量入现有 all_reduce + 向量化 z/H 计算
  （GPU 上 O(94×128)，µs 级）；ĉ 用现有 prof 计数 EWMA，只读不写决策。
- 一致性：z、H、N_fresh 全部由 all-reduced 逐位一致张量派生，状态机无 rank 本地输入
  → 各 rank 决策序列必然一致；测试断言各 rank 决策事件哈希相同（§验证）。
  不宣称"永不慢于固定策略"：检测、试探、dwell 都有成本，净效果由正式比较回答。
- 旧路径处置：v1 的 S1（cos shift/stable）与 S2（jump/converged/volatile）在 v2 下全部
  短路（一处 `if controller_version==v1` 包裹，v1 行为逐字节保留）；RESET/tier 维持
  第一阶段 OFF 不可达。

### 3.2 与 J(W) 的对应（离线设计分析）
  c_check/W ←→ ĉ 与网格（23-28ms/窗 ÷ W step，W 加倍检查频率减半）；
  λ·ℓ·(W−1)/2 ←→ urgent/confirmed shrink 的价值上界（离线用 calibration 域切换段估计 ℓ 的量级，
  报告区间；变化位置在窗内近似均匀的假设显式写出）；
  E_est(W,N,α) ←→ z 归一化 + UNCERTAIN + m 确认（有限样本决策损失的在线保守化）。
  J(W) 不在线求最小；它只用于事后解释 v2 驻留分布是否落在合理区间（第八节行为证据）。

## 4. 初始窗口与公平性（第四节）
- W_init、W 网格、z/θ/N_min/m/dwell 全部来自 development/calibration 负载；
  不使用未来域标签、切换时刻或 evaluation 事后最优窗口；无在线初始化探测（若将来加入，
  其耗时计入端到端指标）。固定基线同样允许用 calibration 选点（F-local=网格 argmin，
  F-global=跨负载单一选点）——双方信息集对等。

## 5. 行为日志 schema（第八节证据链的数据基础）
每窗一行 JSONL（rank0 写；各 rank 决策哈希每 16 窗比对一次）：
  {window_id, steps_actual, N_fresh, S_change, z, H_w_assignments, H_w_us_empirical(标记换算),
   r_legacy(对照), cos_legacy(对照), c_window_ms, W_before, W_after, state, event,
   reason_codes[], dwell_remaining, uncertain_carry}
每次 W 变更附加：{trigger_z, trigger_H, N_fresh_at_trigger, confirm_windows_used, expected_effect}
——直接回答"窗口为何改变；改变是否产生预期行为；预期行为是否改善净性能"。

## 6. 验证计划（第四步的前置清单）
V1 单 rank 状态机单测（合成 z/H/N 序列→确定性 W 轨迹，含 dwell/uncertain/urgent 全路径）；
V2 多 rank 一致性：8-rank 短跑，断言各 rank 事件哈希序列逐位相同；
V3 边界：W 永不越 {8,64}；v1 flag 下与锁定版日志签名逐行一致（C5-dynamic 签名器复用）；
V4 输出正确性：v2 服务短跑输出与 v1 同请求逐 token 一致性抽查（窗口只影响 swap 时机，
   不影响单 forward 数学；抽查路由/输出合理性而非逐位相等——swap 时机不同属预期）；
V5 开关不变量：v2 flag OFF 时新增代码零执行（覆盖率断言）；load 历史随 swap 重映射保持。

## 7. 待定校准量（第五步在 calibration 上定，先于此不填数）
κ, z_hi, z_lo, z_urgent, m_shrink, m_grow, N_min, U_max, θ_H_low, θ_H_act, dwell_shrink,
dwell_grow, β, W_init（=网格 argmin 复核）。全部冻结后进入正式比较协议（第六步），
evaluation(7963) 只作已知 benchmark 回归，不作未见验证（用户第六节）。
