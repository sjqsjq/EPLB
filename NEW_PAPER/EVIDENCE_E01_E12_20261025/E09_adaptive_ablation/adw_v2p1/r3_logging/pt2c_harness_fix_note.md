# PT2-C 测试台修复记录（冻结前，harness-only，零参数改动）

时间：2026-10-17，PT2-C 首次真实数据运行 FAIL 之后、冻结之前。
性质：**验证 harness 的缺陷修复**。knob、拟合输入、拟合结果、门槛、
解释带均未触碰；fit_results_v2p1.json 未重算（sha 不变）。

## 1. 现象（v1，已归档 pt2c_results_v1_harnessbug.json）

- 决策成对缩放部分：0 个 F2_DEFECT（181/182 恒等；1 个失配被正确分类为
  n_min_absolute_gate）。
- 状态重构保真检查：182/182 全部失配 → gate FAIL → 按纪律停下排查。

## 2. 根因（三处 harness bug，均有代码级证据）

**B1 保真检查跨 knob 比较**：v1 用**冻结** knob 重放 logging run 的事件，
但 live 事件产生于**惰性** knob（LOCK §5：κ=1.0、z 三门 1e12/2e12/3e12、
θ_r_low=1e-12、θ_r_act=2.0、n_min=0）。跨 knob 变更本就不可能复现。
证据：失配的 live 事件全部 ∈ {hold, grow_blocked_headroom}（惰性 knob 的
唯一可能事件谱，LOCK §7 已预注册），replay 事件全部为冻结 knob 下的
urgent/shrink_rejected/hold_alert 等。修复：保真重放改用惰性 knob。

**B2 状态重构错位一窗**：controller.py 的 JSONL 记录在 `ctrl.step()`
**之后**构建（phase 4；非提交窗先 `restore(snap)` 再记录），因此记录中的
alert_k/stable_s/dwell/dwell_dir/uncertain_run/episode_shrinks 是该窗
**结束时**的有效状态 = **下一窗**的 pre-state；v1 却用同窗记录当 pre-state。
修复：pre-state 取自前一窗记录；并加强为同时校验重放的 post-state 与
本窗记录的全部状态字段（全状态链校验，强于 v1）。

**B3 z 基准未换算**：v1 把 κ=1.0 基准的 z（数值 ~1e3–1e4）直接喂给冻结
决策面（z_hi=314.14、z_urgent=942.42），所有窗落入 alert/urgent 带，
行为验证 run 实际会走的 quiet/grow/midband 路径完全未被探测。修复：
成对恒等两臂统一使用部署版 `v2p1_signals.z_from_signals(s_change_logical,
n_fresh, 94, κ_frozen)` 重算的 z_frozen（与线上公式逐字一致；同一值喂
两臂——成对主张只关于 (h, n) 缩放，按用户范围**不**主张也不检验
"重算 z 不变"）。

## 3. v2 结果（pt2c_results.json）

- 保真：243/243 合格窗（182 静止 + 61 变化）**全部**复现 live 事件、
  W_after 及全部 post-state 字段，0 失配 → 状态重构链与 step() 确定性
  在真实数据上验证成立；4 个 run 全部窗 committed∧prop_valid=1（链前提
  断言通过）；logging runs 无任何 uncertain/carry 事件（carry_windows=0
  假设成立）。
- 成对恒等（冻结 knob、冻结基准 z）：243 决策中 0 个 F2_DEFECT；唯一
  失配 = W8 窗 132（尾部不完整窗，n=6,294,992 < n_min=12,320,768 →
  原臂 uncertain_carry；缩放臂 n≈3.94e8 ≥ n_min → grow_blocked_headroom），
  分类 n_min_absolute_gate（绝对计数门的预期语义，用户范围明示不要求
  其不变）。
- h_rate：241/243 位相同，最大绝对差 1.73e-18（≤1 ulp，与 PT2 A/B 的
  精确有理恒等结论一致）。
- GATE：PASS（decision_identity 99.59%，f2_defects=0，fidelity_ok=true）。

## 4. 纪律说明

- 修复仅涉及 scripts/pt2c_v2p1_v2.py（新文件）；v1 脚本与其输出原样保留。
- 本修复不改变任何预注册门槛、解释带、拟合规则或六参数值。
- 如实报告：v1 的"0 F2_DEFECT"在错误 z 基准下探测面有限，不能单独作为
  F2 验收；验收以 v2（正确基准 + 全状态链保真）为准。
