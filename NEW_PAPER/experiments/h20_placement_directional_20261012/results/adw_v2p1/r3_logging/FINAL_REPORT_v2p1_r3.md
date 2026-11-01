# ADW v2.1 r3 校准与行为验证 — 最终报告（本轮唯一交付）

日期：2026-10-17。范围声明：本轮仅做校准与行为验证。全程 α=0、D 关闭、
planner 及其他服务配置不变；冻结 v2 与全部历史结果未触碰；共 5 次启动
（预算内），全部一次成功。数据仅为 W0（calibration.jsonl，8037 请求）。
**本报告不宣称 adaptive 优于固定窗口，不扣除任何实测成本，不将 W0 称为
未见验证集**（W0 既是校准数据源，也是行为验证数据源）。

## 0. 结论速览

- **校准完成**：4 次固定窗口 logging → 一次盲拟合 → 样本门全过 → 六参数
  冻结（FREEZE_record_v2p1.md，2026-10-17T07:25:26+00:00，stop_flags=NONE）。
- **行为验证通过**：第 5 次启动（冻结 knob、W 受控自适应）全部检查项通过：
  32/32 窗确定性重放匹配、θ 门 0 违例、8037/8037 请求 0 错误、
  零 desync/freeze/fatal、8 rank knob 与决策哈希完全一致。
- **失败记录**：0 次致命失败、0 次重启、0 次废弃启动。两处离线 harness
  缺陷（PT2-C v1、分析脚本 str-key）均在冻结前/分析阶段发现并修复，
  **零参数改动**，全部归档（§8）。
- **判定**：具备进入独立正式性能评价的条件（附 §9.4 注意事项）。

## 1. 版本锁（步骤 1）

- git：/workspace/EPLB main @ **51b83e6**（仅本地，未推送）。
- 部署 sha256（$PB=site-packages/sglang/srt/managers/pb_oeplb，与 $SRC 逐字
  一致，冻结前后均实测核验，见 FREEZE_record §4 代码锁）：
  controller.py `24ccf304…`、config.py `66b0760d…`、window_v2p1.py
  `d20d340f…`、v2p1_signals.py `98a13608…`、window_v2.py（冻结 v2，未触碰）
  `810c54da…`。完整 9 行清单：deployed_shas.txt（驱动脚本每次启动前
  `sha256sum -c` 强制核验，5 次启动全部通过）。
- 数据集：calibration.jsonl sha256 `7d53aa0b…`，8037 行（未改动）。
- 启动链：launch235b_eacampaign.sh `05b5c32f…`、run_grid_bench.py
  `2d56c977…`、env_235b.sh `26242c03…`。
- 日志协议 ↔ 过滤器一致性：控制器写 `protocol_rev="r3"` ↔ 合格窗过滤器
  要求 r3；event 枚举与 window_v2p1/window_v2 常量一一对应（LOCK §6）。
- **两种模式分别锁定**：校准模式（惰性 knob：κ=1.0、z 门 1e12/2e12/3e12、
  θ_r 1e-12/2.0、n_min=0、W 固定；LOCK §5，含 §7 预记录可执行性偏差）；
  行为验证模式（冻结 knob + n_min=12320768、W_init=16、W 受控；LOCK §8.1
  追加于 07:26:45 —— 晚于冻结 07:25:26、早于第 5 次启动 07:26:59）。

## 2. 五次启动清单（步骤 1/2/5）

| # | run | 模式 | boot | bench 墙钟 | ok/err | 记录窗 | forward 数(max calls/94) | total_steps | 尝试 | 致命/重启 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | W8  | 校准（W=8 固定）  | 92s  | 382.12s | 8037/0 | 132 | 1050 | 1056 | 1 | 0/0 |
| 2 | W16 | 校准（W=16 固定） | 100s | 388.43s | 8037/0 | 66  | 1020 | 1056 | 1 | 0/0 |
| 3 | W32 | 校准（W=32 固定） | 80s  | 379.75s | 8037/0 | 33  | 1020 | 1056 | 1 | 0/0 |
| 4 | W64 | 校准（W=64 固定） | 88s  | 398.55s | 8037/0 | 16  | 1020 | 1024* | 1 | 0/0 |
| 5 | behavior | 行为验证（冻结 knob，W_init=16，W 受控） | 68s | 361.26s | 8037/0 | 32 | 1050 | 1056 | 1 | 0/0 |

\* W64：16 满窗 ×64=1024 步，尾部 32 步不足一窗未触发边界（无窗丢失，
run 正常结束）。所有 run：prop_valid=1、h_min==h_max、v_min==1 全窗成立；
fatal_collective=0；jsonl 行数 = 记录窗数（无缺行）。
bench 墙钟仅如实记录：5 次 run 的 knob 状态不同（W 固定 vs 受控），单次
行为验证不构成任何性能比较（见 §9.5 声明）。

## 3. 样本门与合格窗（步骤 2）

- 合并池：**Z_s / H_rate_s = 182**（门 ≥150 → **PASS**）；**Z_c = 61**
  （门 ≥20 → **PASS**）。分位数组件 window_list_sha256
  `6087350edec8bf36…`。
- 逐 run（记录/合格/合格静止/合格变化；唯一排除 = init×1）：
  W8 132/131/101/30，W16 66/65/49/16，W32 33/32/23/9，W64 16/15/9/6。
- 合格静止 vs 计划 §2.2 预期：W8 101（104–107）、W16 49（51–53）、
  W32 23（25–26）、W64 9（12）——均在 30% 短缺规则内，未触发停止；
  如实记录，未降低门槛、未自动追加。

## 4. PT2 验证（步骤 3；只验证它能证明的性质）

- **PT2 A/B（数据驱动成对缩放，pre-fit，182 合格静止窗）**：精确有理恒等
  **182/182**（门=100% → PASS）；float64 层逐位相同 180/182，最大偏差
  1 ulp（纯舍入，分类记录）；θ 布尔不变扫描失败 0。
- **PT2-C（冻结 knob 完整状态机决策恒等，post-fit pre-freeze）**：
  - v1 首次真实数据运行 gate FAIL（0 个 F2 缺陷，但保真 182/182 失配）→
    按纪律停下排查 → 定位为 **3 处 harness 缺陷**（跨 knob 变更重放、
    状态重构错位一窗、z 基准未换算），修复过程零 knob/参数/拟合改动，
    v1 脚本与输出原样归档（pt2c_harness_fix_note.md、
    pt2c_results_v1_harnessbug.json）。
  - **v2 结果：PASS**。保真：243/243 合格窗（182 静止 + 61 变化补充覆盖）
    用惰性 knob 从前一窗状态重放，**全部**复现 live 事件 + W_after +
    全部 post-state 字段（0 失配）→ 状态重构链与 step() 确定性在真实
    数据上成立。决策恒等：冻结 knob、同一状态、同一 z（冻结基准，由部署
    版 z_from_signals 重算，两臂同值），仅 (h_w, n) 成对缩放 →
    **0 个 F2_DEFECT**；唯一失配为 W8 w#132 尾部窗（n=6,294,992 <
    n_min=12,320,768，原臂 uncertain_carry、缩放臂越过绝对计数门）→
    分类 n_min_absolute_gate（预期语义，非缺陷）。h_rate 逐位相同
    241/243，最大绝对差 1.73e-18（≤1 ulp）。
- 范围声明（逐字遵守）：复制计数不作为新增独立样本（PT2 不新增数据）；
  不要求重算后的 z、UNCERTAIN 判断天然不变（z 为固定输入，n_min 为绝对
  计数门单独分类）；完整状态机比较时其他输入与初始状态完全相同。

## 5. 一次拟合六参数（步骤 4；冻结规则，执行一次，未反复调参）

- 合并池权重：**每窗等权**；有效样本数 = 池长度（Z_s/H_rate_s 182、
  Z_c 61）；分位数方法：**numpy.percentile(method="linear")**。
- 边界分类：伪切换 #2009/#4018/#6027，±250 请求 guard，cumulative
  steps 比例 × 8037 对齐（计划 §2 逐字执行）。

| knob | 冻结值 | 规则 |
|---|---|---|
| KAPPA (κ) | 3875.1768798828125 | median(Z_s) |
| Z_LO | 22.643858727592757 | p90(Z_s/κ) |
| Z_HI | 314.14010272527486 | p99(Z_s/κ) |
| Z_URGENT | 942.4203081758246 | max(3·Z_HI, p50(Z_c/κ)) |
| THETA_R_LOW | 0.010095552687949321 | p30(H_rate_s) |
| THETA_R_ACT | 0.011395643112507271 | p60(H_rate_s) |

沿用不拟合：N_MIN=12320768、M_SHRINK=2、M_GROW=3、DWELL_SHRINK/GROW=2/2、
U_MAX=4、W_UNCERTAIN_TARGET=16、URGENT_EPISODE_CAP=3、BETA=0.5、W_init=16。

校准诊断（G1-D/D1 仅用于解释与排查，未用于调参）：
- **G1-D：解释带 ≤1.5 → COMPATIBLE**。h_rate 中位数逐 W =
  0.01101/0.01085/0.00996/0.01257，max/min=**1.262**；对照 raw h_w
  中位数比 **9.11**（543954/1063211/1925018.5/4955858）→ F2 的尺度
  不变性在真实数据上成立（修复前后对照存档）。
- **G2 分离度：NOT separable（报告项）**：p50(Z_c/κ)=0.691 ≤ Z_HI=314。
  结构性原因：±250 宽 guard 使 Z_c 池含大量基线窗；v2 addendum 曾收窄
  Z_c，但 v2.1 计划按 r3 预注册逐字执行，未事后更改。
- **G3 重尾（报告项）**：p90/p50=22.6（ref 再收敛瞬态 w2–9 + W64 基线窗
  折算后 ~92 所致）→ z_lo/z_hi 显著高于 v2 终值（3.7/19.9），系冻结规则
  的确定性后果，未重拟合。
- **G5（报告项）**：κ_W/κ_pool = W8 0.87、W16 1.00、W32 2.30、W64 91.6
  （W64 静止合格窗仅 9 个，小样本仅列示）。
- **D1：R_swap=6.17 > 2 → 触发冻结前排查（§6）**。
- **D2（分层报告，非判据）**：swap 窗 3 窗内 z 回落至 ≤z_lo 共 90.3%
  （静止段 50/53，变化段 6/9）。

## 6. D1 排查与冻结（步骤 4/5a）

**D1 排查（investigation_d1_rswap.md，sha256 `2ee4ce6c…`，冻结前归档）**：
- 逐 W 分解：R_swap = W8 3.08、W16 18.13、W32 70.70、W64 176.85；
  合并值 6.17 由选择效应与跨 W 组成效应合成。
- **F1 泄漏的三条判别签名全部不成立**：(1) 对齐 z 轨迹峰值在 **k=−1**
  （映射未变的窗；p2l_version 升位发生在 k=−1→0 边界）——泄漏只能在
  commit 之后注入，无法解释 swap 之前的抬升；k=0 反而回落（7.09→4.24）。
  (2) swap 窗 s_change 反推 0.0154(W8)→0.35(W64)，超出 n_layer 的 8×
  缩放 ~23×——与"窗越长、真实漂移与 EWMA ref 滞后积累越多"一致，与
  计数成比例的重映射伪影矛盾。(3) 衰减比 0.502/0.564/0.500/0.681 ≫
  β^k=0.5/0.25/0.125/0.0625——持续真实信号，非一次性污染。
- 佐证：det_map_version==p2l_version **全部 247 窗 0 失配**（tripwire
  干净）；prop_valid=1 全窗；W8 有 4/30 swap 窗处于本 W 基线（硬性泄漏
  应影响每个 swap 窗）；D2 90.3% 回落；机制级 T2/T3 oracle 测试属 39 项
  套件，冻结前对最终源码重跑全过。
- 结论：planner 选择效应（swap 由真实漂移触发，z 本应高）+ 真实漂移随
  窗长积累 + 跨 W 组成效应；**未发现 F1 缺陷**。解释带未移动，任何 knob
  未因排查改动，排查记录归档而非静默通过。
- 局限如实记录：基于 rank0 日志离线重构与合成 oracle，无在线逐层 det
  对照（需新采集 run，超本轮预算）。

**冻结（FREEZE_record_v2p1.md，07:25:26）**：验收门全过 —— 39 项离线套件
对最终源码重跑 ALL_V2P1_TESTS_PASS、PT2 A/B PASS、PT2-C v2 PASS、
G1-D 带内、D1 排查归档 → **stop_flags=NONE → 六参数冻结**；代码锁：
4 文件 src==deployed sha 断言通过 + window_v2.py 冻结 sha 断言通过。

## 7. 行为验证 run（步骤 5b；第 5 次启动，冻结后执行）

**一致性（用户核对项 1：各 rank 窗口、事件和状态一致）**：
- 8 rank 全部打出 ACTIVE banner，knob 字符串**唯一**（跨 rank 逐字一致，
  含 κ=3875.1768798828125、θ、n_min=12320768、grid=(8,16,32,64)、β=0.5、
  w_init=16）。
- 32/32 窗 committed=True：h_min==h_max（决策哈希 8 rank 全同 → 事件、
  W_after、状态全同，这是协议内置的逐窗跨 rank 一致性证明）、v_min==1、
  confirm_fail_streak max=0、desync=0、freeze 事件 0、fatal_collective 0。
- PROF：8 rank 每窗汇报齐全（run 末 8 份 window#32 记录）；forward 数
  max calls 98700/94 ≈ **1050**；total_steps=1056。

**轨迹（核对项 2：稳定段、变化附近的 z、h_rate 与窗口动作）**：
- W_after 序列（全在网格上）：16×13 → **32**(w14) → 32×5 → **64**(w20) →
  64×3 → **32**(w24 shrink) → 32×3 → **64**(w28) → 64×4。
- 边界上下文（cumulative steps 比例映射，与拟合同法）：
  - **边界 2009**（→窗 idx15）：w12–13 z=109.4/32.7 → midband_hold
    （[z_lo,z_hi) 迟滞带行使）；w14 z=5.1、hr=0.00949<θ_r_low →
    confirmed_grow 16→32。
  - **边界 4018**（→窗 idx21）：w23 **z=771.5**（≥z_hi、<z_urgent=942）
    hr=0.0548 → hold_alert（alert_k=1/2）；w24 z=334.5≥z_hi ∧
    hr=0.0144≥θ_r_act=0.0114 → **confirmed_shrink 64→32**（M_SHRINK=2
    确认 + dwell 行使；**latency=3 窗**；episode_shrinks=1）；w25 z=34.5
    midband_hold；w26–27 z=7.8/2.0 回静；w28 z=0.78、hr=0.00927 →
    confirmed_grow 32→64（episode 复位）。
  - **边界 6027**（→窗 idx27）：**无告警、无收缩**——w28–32 z=0.57–1.20
    全静。如实记录：第三个伪切换在本 run 未产生可检测 z 抬升（与 v2
    校准中"switch#3 不可见"发现一致），该边界的检测/收缩路径未被行使。

**事件谱与原因（核对项 3：每个 grow/shrink/hold 的原因）**：
- 事件总计：init 1、hold 12、grow_blocked_headroom 8、midband_hold 6、
  confirmed_grow 3、hold_alert 1、confirmed_shrink 1（proposed 与 event
  完全一致 → 无回滚窗）。
- 离线顺序重放（冻结状态机，r3 日志无 reasons 字段 → 重放重构）：
  **32/32 all_match，0 失配** → live 事件流 = 冻结逻辑的确定性输出。
  action 窗 reasons 摘录：w14/w20/w28 confirmed_grow = "stable_s>=3 and
  h_w<theta_h_low"（rate 空间）；w23 hold_alert = "alert_k=1/2 z=771.46"；
  w24 confirmed_shrink = "alert_k>=2 and h_w>=theta_h_act"；w12/13/17/18/
  19/25 midband_hold = "z in [z_lo,z_hi)"。完整逐窗 reasons 存
  behavior_analysis_v2p1.json → replay.detail。
- grow_blocked_headroom 8/32：hr∈[θ_r_low, θ_r_act) 或 ≥θ_r_low 且
  stable_s≥3 → 余量门按设计频繁行使（θ 带窄系 p30/p60 构造的确定性
  后果，预注册于计划 §3）。
- 装饰性备注：reason 字符串沿用 v2 的 %.0f 模板，rate 值显示为
  "h_w=0"；判决逻辑使用 rate 空间真实值（θ 门核验与重放双重证实），
  无行为影响，仅日志可读性问题（如实记录）。

**成本（核对项 4：检查、记录、规划和迁移的实际成本；只记录，不扣除；
8 rank 并行，时间不相加）**：
- 窗边界决策 c_window_ms（rank0，含 det SUM + confirm MIN/MAX 两次
  collective + 本地判决）：median **1.54ms**、mean 1.56、p90 1.87、
  max 2.95；32 窗合计 ≈ 49ms。
- PROF 每 rank 全程累计（window#32 终报，8 rank 范围）：record
  **11.11–11.50s**（avg 114–117µs/call，含 F1 第二次 scatter_add）、
  allreduce **0.79–0.82s**、planbuild **0.068–0.074s**、finalize
  **0.28–0.32s**、total 12.25–12.71s。各 rank 并行发生，**不相加**，
  不从任何数字中扣除。
- 迁移（planner，配置未变）：13 个窗 issue swap，共 **2079** 次专家
  交换；单窗最大 300（w23——z=771 的真实需求变化窗，与 D1 排查的
  选择效应图景一致）；w1 初始整置 228 次。
- bench：361.26s 墙钟、tps 22.2、lat_p50 8824ms（仅记录，见 §9.5）。

**正确性（核对项 5：错误、异常、输出正确性）**：
- bench ok=**8037**、errors=**0**；z=None 窗仅 init；uncertain 窗 0
  （全部 n≥n_min，最小 N=68.5M ≫ 12.32M）；θ 门核验 **0 违例**
  （confirmed_grow ⇒ hr<θ_r_low；confirmed_shrink ⇒ hr≥θ_r_act；
  urgent 未发生）；轨迹全在网格；无 w_desync_freeze/confirm_fail_freeze/
  fatal_collective；server.log 无 FATAL。

## 8. 全部失败与偏差记录

| # | 类别 | 内容 | 处置 |
|---|---|---|---|
| 1 | 运行失败 | 无（5/5 一次成功；0 致命、0 重启、0 废弃启动） | — |
| 2 | 预记录偏差 | 计划惰性 knob 文本 Z_LO=Z_HI=Z_URGENT=1e12 被构造器严格不等式拒绝 | 启动前改用 1e12/2e12/3e12（不可达性不变），LOCK §7 预记录 |
| 3 | harness 缺陷 | PT2-C v1：跨 knob 重放保真 / 状态错位一窗 / z 基准未换算 → 首次真实数据 gate FAIL | 冻结前停下排查修复（v2 PASS）；零参数改动；pt2c_harness_fix_note.md + v1 输出归档 |
| 4 | harness 缺陷 | analyze_behavior_v2p1.py 边界上下文 str-key TypeError | 修复后重跑（仅离线分析；数据未动） |
| 5 | 报告项 | G2 非分离（结构性）、G3 重尾 22.6、G5（W64 91.6，小样本）、W64 静止窗 9 vs 预期 12（30% 规则内）、D1 超带（排查归档）、边界 6027 未行使、θ 带窄 → grow_blocked_headroom 8/32 | 全部如实列入 §5/§6/§7/§9，不静默、不挪带、不重拟合 |

## 9. 最终判定（步骤 6）

**9.1 校准是否完成：是。** 按 r3 预注册规则一次拟合，样本门全过
（Z_s=182≥150、Z_c=61≥20），六参数 + 代码于 07:25:26 冻结
（stop_flags=NONE），时间戳链合规（冻结 < LOCK §8.1 追加 < 第 5 次启动），
拟合仅执行一次，无任何事后调参。

**9.2 真正观察到的预期行为**（均有工件级证据）：
1. **F2 尺度不变性**：PT2 A/B 精确有理恒等 182/182；PT2-C v2 冻结 knob
   决策恒等 0 缺陷；G1-D 带内 1.262（对照 raw h_w 9.11）。
2. **F1 逻辑身份**：39 项套件（含 T2/T3 同状态跨 swap oracle）对最终
   源码全过；det_map tripwire 247/247 窗零失配；D1 排查排除泄漏签名。
3. **F3 confirm-then-commit**：32/32 窗 8-rank 决策哈希一致
   （h_min==h_max∧v_min==1），零 confirm_hold、零 freeze。
4. **受控自适应**：网格内双向调整（16→32→64→32→64）、midband 迟滞、
   M_SHRINK=2 确认 + dwell + episode 计数、θ 余量门（grow 与 shrink
   两侧）、边界 4018 检测（z=771→alert，z=334→confirmed_shrink，
   latency 3 窗）与随后恢复性 grow。
5. **确定性可复现**：离线顺序重放 32/32 精确复现 live 事件流并重构
   每窗 reasons。

**9.3 未被行使的机制**（如实列举，正式评价时需知其未验证）：
urgent_shrink / urgent_shrink_capped（本 run max z=771.5 < z_urgent=942.4）；
uncertain_carry / uncertain_hold / forced_shrink_uncertain（全窗 n≥n_min）；
confirm_hold / confirm_fail_freeze / w_desync_freeze / fatal_collective
恢复路径（零故障则无从行使）；grow_blocked_dwell / grow_blocked_max
（u_max=4 未触及）；shrink_blocked_dwell / shrink_blocked_cap；W=8 地板
（轨迹最低 16）；边界 6027 的检测/收缩（z 未抬升）。

**9.4 是否具备进入独立正式性能评价的条件：具备（机制面）。**
参数与代码已冻结锁 sha、协议在真实 8×H20 服务上端到端验证、正确性与
一致性检查全过、成本已量化记录。进入正式评价前须将下列事项写入其
独立预注册：(a) G2 非分离 + G3 重尾 → z_hi=314 偏高，部分真实切换
（如本 run 边界 6027）可能不触发收缩——评价协议需显式覆盖此风险；
(b) θ 带窄（0.0101–0.0114）→ grow 频繁被余量门阻断属预期行为而非故障；
(c) urgent/uncertain/故障恢复路径仅有套件级与协议级证据，无 live 行使
记录；(d) W64 校准样本小（静止 9 窗）。正式评价须为独立预注册协议
（计划 §5.3），本轮结果不得作为其性能先验。

**9.5 声明**：本报告不据单次行为验证宣称 adaptive 优于固定窗口（§2 的
墙钟差异不构成比较证据）；所有实测成本仅记录、未从任何数字扣除；W0 不
是未见验证集。本轮到此为止：校准 + 行为验证完成，停止，不追加运行。

## 10. 工件清单（$OUTROOT=results/adw_v2p1/r3_logging/）

LOCK_record_r3_logging.md（§8.1/§10 已追加）、deployed_shas.txt、
driver.log、driver_boot.log、W{8,16,32,64}/{v2p1_rank0.jsonl, server_a1.log,
bench_a1.{json,log}, run.log}、fit_results_v2p1.json、
window_inventory_v2p1.json、investigation_d1_rswap.md、
pt2c_results.json（v2 PASS）、pt2c_results_v1_harnessbug.json、
pt2c_harness_fix_note.md、test_v2p1_prefreeze_raw_output.txt
（ALL_V2P1_TESTS_PASS，39 PASS/0 FAIL）、FREEZE_record_v2p1.md、
behavior/{v2p1_behavior_rank0.jsonl, server.log, bench.log,
v21behavior.json, behavior_analysis_v2p1.json}、
behavior_driver_console.log、本报告。
脚本（$EXP/scripts/）：run_v2p1_logging.sh `f3cad763…`、
fit_v2p1_calibration.py、pt2c_v2p1.py（v1，留档）、pt2c_v2p1_v2.py、
analyze_behavior_v2p1.py、gen_freeze_v2p1.py、
run_v2p1_behavior.sh `876c71b9…`、test_window_v2p1_logic.py。
