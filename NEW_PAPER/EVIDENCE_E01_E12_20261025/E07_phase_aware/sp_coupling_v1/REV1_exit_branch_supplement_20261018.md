# REV1 退出分支只读补充：实测条件、目标状态、confirmed_grow 语义勘误、冻结阈值行号

日期：2026-10-18　性质：**只读核查**（应用户要求；未改代码、未跑 GPU、封存工件未动）。
用途：论文流程图与已测实现对齐。代码行号以 SRC=/workspace/EPLB/OEPLB/src 为准；部署副本经
deployed_shas_sp_coupling_rev1.txt（14 路径 sha256）与 SRC 位一致，行号相同。

## 1. QUIET 三条退出分支的实测条件与目标状态（sp_state.py，rev=1 分支 L267-319）

同窗优先级 P1a > P1b > P1c/P1d > P2 > P3（if/elif 链，每周期至多一条触发）：

| 分支 | 行号 | 条件（原文语义） | 目标状态 | 档位变化 |
|---|---|---|---|---|
| P1a | L279-286 | z ≥ z_hi（=314.14010272527486，首警报，既有紧急条件） | **RECOVER** | S4→S1, H16→H8, W64→W8 |
| P1b | L287-290 | event ∈ SP_GUARDRAIL_EVENTS（L60-63：uncertain_carry/uncertain_hold/forced_shrink_uncertain/w_desync_freeze/confirm_hold/confirm_fail_freeze） | **RECOVER** | 同上 |
| P1c/P1d | L291-298（change_ev 定义 L276-278） | event ∈ SP_CHANGE_EVENTS（L72-79）**或** swap_pending **或** swaps_started>0 | **RECOVER** | 同上 |
| P2 | L299-306 | n_fresh < n_min_eff（覆盖不足=证据不足） | **TRACK**（非 RECOVER） | S4→S1, H16, W64→W16 |
| P3 | L307-319 | h_rate ≥ θ_out **连续** SP_QUIET_EXIT_CONFIRM=2（L81）个 QUIET H 周期；任一周期 <θ_out 即 streak 清零（L308-309） | **TRACK**（L312） | 同上 |
| 带内驻留 | L307-309 else | h_rate < θ_out（含滞回带 [θ_in,θ_out) 与更低值） | QUIET（streak=0） | 不变 |

**P3 确认回 TRACK**：L310 `if exit_after >= SP_QUIET_EXIT_CONFIRM:` → L312 `state_after = "TRACK"`，
reason 串含 "-> TRACK (rev1 R1/R3)"（L314-318）；转换时 exit_after 清零（L319 + apply 的 PD-5 重置）。
P3 仅在 P1a/P1b/P1c/P1d/P2 全不成立时评估；P2 触发同样回 TRACK 并清 streak（L306）。

## 2. P1c/P1d 实测为何全部记为 confirmed_grow —— 语义勘误（关键）

**DEV_R1 全部 8 次 P1 退出（A_rev1 4 次运行）实测**：event='confirmed_grow'、
swap_pending=False、swaps_started=0（即全部经 P1c 事件项触发，P1a/P1b/P1d 零次独立触发）；
同行 W_before→W_after = **16→32**（EA 窗扩容提交）；z=0.36–0.69 ≪ z_lo=22.64；
h_rate=0.0095–0.0100 < θ_in。W2 b1/b2 的 3 次（h40/h52/h58）全部发生在**相变之前**的平稳段；
真实相变漂移（h67–h70：h_rate 0.0118→0.0212→0.0704，z 峰值 44.38 < z_hi，故 P1a 未触发）
经 **P3→TRACK（h70）** 退出——REV1 文档 §2 预告的结构性后果按预期发生。

**confirmed_grow 的冻结语义（window_v2.py，不可改）**：仅在**平稳带**分支发出——
L227 `elif z < k["z_lo"]:`（平稳）→ L230 stable_s+=1 → L231 `stable_s >= m_grow(=3, L55)` →
L232 若 h_w ≥ theta_h_low 则 grow_blocked_headroom（L235，机会性、不改 W）；dwell 阻塞则
grow_blocked_dwell（L239）；W 到顶则 hold；**否则 L248 `W = grid_up(W)` + dwell_grow → L253
`event = EV_CONFIRMED_GROW`**。即 confirmed_grow = **“分布已连续 3 窗平稳且余量压力低 → 规划器提交扩窗”**，
是稳定扩窗提交事件，**不是分布变化证据**。（对照：confirmed_shrink 走 L203-224 警报确认路径
alert_k≥m_shrink，才是漂移确认输出。）

**勘误（仅文档层面；代码 reason 串与封存日志按原样保留，不改）**：
1. sp_state.py L295 reason 串 "P1c/P1d confirmed change evidence" 与 L65-71 注释
   "confirmed_grow/shrink are v21's own two-window confirmed drift detectors" 对 confirmed_grow **误标**；
2. REV1_quiet_exit_revision_20261018.md L24（"confirmed_grow/shrink = v21 双窗确认漂移检测器输出"）、
   §2 T4 行按语、§7 C2 行（"真实变化 C2: event=confirmed_grow"）继承同一误标；
3. RESULTS_dev_r1_20261018.md §3 Q1/Q3 中 "P1c/P1d 已确认变化证据×8（confirmed_grow）" 应读作
   "P1c 规划器窗口变更提交×8（confirmed_grow 稳定扩窗，W 16→32）"。

**P1c 的正确流程图语义**：RECOVER 退出由三类证据授权——(i) 压力/警报族（hold_alert、urgent_shrink、
urgent_shrink_capped、shrink_blocked_dwell、shrink_rejected_low_headroom）与漂移确认收缩
（confirmed_shrink）；(ii) **规划器窗口变更提交**（confirmed_grow：W 扩档 + dwell 启动）；(iii) 迁移活动
（P1d：swap_pending ∨ swaps_started>0）。设计理由：EA 窗 W 或布局刚被规划器变更时，证据基与采样制度
随之改变，恢复 S1 全采样以重建证据——而非断言分布已变化。grow_blocked_*（未变更 W 的机会性尝试）
不在集合内；实测一致：9 次 P3 退出行中 6 次 event='grow_blocked_headroom'，均未误触 P1c。

**论文流程图必须体现**：rev1 下 QUIET→RECOVER 在全部 12 次 DEV_R1 运行中**只**由稳定扩窗提交触发；
真实漂移（z<z_hi）经 P3→TRACK（S1/H16/W16=F_dense 密度）响应，RECOVER（H8）仅由 z≥z_hi/guardrail/
规划器提交/迁移活动进入。W2 响应未受损的机制路径是 P3→TRACK，不是 P1→RECOVER。

## 3. 冻结 θ_in / θ_out：数值、来源与代码行号

| 量 | 值（全精度） | 别名 | 来源 |
|---|---|---|---|
| θ_in | 0.010095552687949321 | theta_h_low = v21_theta_r_low | 校准战役一次冻结（失活边界），v1 起用于进入规则，rev1 兼作滞回带下沿 |
| θ_out | 0.011395643112507271 | theta_h_act = v21_theta_r_act | 同一冻结校准对的激活边界；θ_out/θ_in=1.1288；rev1 零新数值参数 |
| m_exit | 2 | SP_QUIET_EXIT_CONFIRM | sp_state.py L81 模块常量（匹配冻结确认惯例 window_shift_confirm_windows=2、v21_dwell 2/2），非在线旋钮 |

数值不以字面量存在于代码，经环境→配置→控制器→状态机注入：
- config.py **L137-138**（字段 v21_theta_r_low/act 定义）、**L312-313**（from_env：OEPLB_V21_THETA_R_LOW / OEPLB_V21_THETA_R_ACT）；
- controller.py **L1632-1633**（SpController 实例化：theta_h_low=cfg.v21_theta_r_low, theta_h_act=cfg.v21_theta_r_act）；
- sp_state.py **L117-118**（构造签名）、**L128/L130**（赋值）、**L138-142**（rev1 校验 θ_out>θ_in>0，违反即 RuntimeError）；
- 比较点：进入（v1 不变，PD-2）sp_state.py **L246**（quiet_cond: h_rate<θ_in）、**L249**（RECOVER clean 同条件）；v1 退出 **L329**（h_rate≥θ_in 单周期）；rev1 P3 退出 **L309**（h_rate≥θ_out）。
- 冻结数值的运行工件：dev_r1/sp_r1_*/meta.txt 环境行（OEPLB_V21_THETA_R_LOW=0.010095552687949321、OEPLB_V21_THETA_R_ACT=0.011395643112507271，已对 A_rev1 运行核验）；REV1_quiet_exit_revision_20261018.md L21-22（provenance 表）。
- 相关常量：z_lo=22.643858727592757、z_hi=314.14010272527486、z_urgent=942.4203081758246、n_min=12320768（n_min_eff=n_min//S，S4 时 3080192）、SP_TABLE（sp_state.py **L48-52**）：TRACK=(S1,H16,W16)、QUIET=(S4,H16,W64)、RECOVER=(S1,H8,W8)。

## 4. 实测触发计数（12 次 DEV_R1 运行，A_rev1 4 次合计）

P1a=0，P1b=0，P1c=8（全部 confirmed_grow 稳定扩窗，W16→32），P1d=0（独立触发），P2=0，
P3=9（h_rate 0.011749/0.012385/0.012044/0.012119/0.012905 各 1 次 + 0.070414 一次=真实相变 + W1b1 0.011749、0.012385）；
对照 A_v1 4 次运行：裸 θ_in 单周期退出 26/26。
（P3 九次明细：W1b1 h35/h59，W1b2 h10，W2b1 h35/h70/h111，W2b2 h35/h70/h111。）
