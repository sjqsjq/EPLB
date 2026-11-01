# REV1：QUIET 退出路由与滞回修订（最小规则修订）
日期 2026-10-18 ｜ 作者指令：有限策略修订，保留封存结果，不启动正式性能战役
分支 sp_coupling_dev ｜ 门控 OEPLB_SP_REV=1（默认 0 = v1 冻结行为逐位保留）

## 0. 修订范围（对应作者指令二）

| 指令 | 落实 |
|---|---|
| R1 低余量失效/证据不足退出 → TRACK，不直接 RECOVER | QUIET 退出 P2（覆盖不足）与 P3（h_rate 持续越 θ_out）均转 TRACK |
| R2 RECOVER 仅由已确认变化证据或既有紧急条件授权；同窗优先级明确 | P1 族（z≥z_hi / guardrail / v21 变化事件 / 已接受迁移）→ RECOVER；优先级 P1 > P2 > P3（§2） |
| R3 进入/退出滞回 + 持续确认；稀疏采样影响说明；参数仅用校准材料 | 滞回带 [θ_r_low, θ_r_act]（校准冻结对）；退出需连续 2 周期（冻结确认惯例）；§4 出处；§5 影响说明 |
| R4 planner/α/迁移接受/S、H、W 档位/一致性协议不变 | §6 不变项清单；diff 仅触及 3 文件、QUIET 退出分支及其计数器 |

**不修改**：SP_TABLE 档位、PD-2 进入条件（θ_in=θ_r_low，3 连续周期）、PD-4 RECOVER 退出、
PD-5 streak 复位、PD-7 seal、PD-8 W_after 中和、PD-9 门、联合提交协议、全部冻结 v21 阈值、seed。

## 1. 参数与出处（先于任何 rev1 运行确定；不根据 W1/W2 服务时间反选）

| 参数 | 值 | 出处 |
|---|---|---|
| θ_in（进入，不变） | theta_h_low = v21_theta_r_low = 0.010095552687949321 | 校准战役一次冻结（失活边界） |
| θ_out（退出，rev1 新用） | v21_theta_r_act = 0.011395643112507271 | 同一冻结校准对的激活边界；θ_out/θ_in = 1.1288。**零新数值参数**：校准材料本身已把 h_rate 尺度冻结为 (low, act) 一对，v1 只用了 low（PD-3 无滞回），rev1 采用校准自己的配对 |
| m_exit（持续确认，rev1 新用） | SP_QUIET_EXIT_CONFIRM = 2（模块常量，非在线旋钮） | 与部署系统既有冻结确认惯例一致：window_shift_confirm_windows=2、v21_dwell_shrink/grow=2/2 |
| SP_CHANGE_EVENTS（已确认变化/紧急事件族） | {hold_alert, urgent_shrink, urgent_shrink_capped, confirmed_shrink, shrink_blocked_dwell, shrink_rejected_low_headroom, confirmed_grow} | 全部取自 window_v2.py 冻结事件词表：hold_alert/shrink 族 = 警报带（z≥z_hi 于窗边界）或紧急（z≥z_urgent）事件，即 v1 源码注释所称"ordinary change evidence"；confirmed_grow/shrink = v21 双窗确认漂移检测器输出。**无新检测器** |

排除的事件：init/hold/midband_hold（无变化证据）；grow_blocked_headroom/grow_blocked_dwell
（机会性扩张尝试，非确认变化，非紧急）；uncertain_*/w_desync_freeze/confirm_*_freeze
（已是 guardrail，走 P1b）。

## 2. 状态转移表（rev1 全表；加粗 = 相对 v1 的变化）

同窗多条件优先级：**P1 > P2 > P3**（已确认变化/紧急 > 覆盖不足 > 低余量持续）。
P1 内部各条同目的地（RECOVER），reasons 按 P1a→P1d 顺序记录。

| # | 当前态 | 条件 | 次态 | 备注 |
|---|---|---|---|---|
| T1 | TRACK | quiet_cond 连续 3 有效周期（valid ∧ z<z_lo ∧ h_rate<θ_in ∧ ¬swap_pending ∧ swaps_started==0） | QUIET | v1 不变（PD-2） |
| T2 | QUIET | P1a：z ≥ z_hi | RECOVER | 既有紧急条件（首警报），v1 不变 |
| T3 | QUIET | P1b：event ∈ SP_GUARDRAIL_EVENTS | RECOVER | v1 不变（PD-3 guardrail） |
| T4 | QUIET | P1c：event ∈ SP_CHANGE_EVENTS | RECOVER | **rev1 新增**（R2：已确认变化证据/紧急事件族） |
| T5 | QUIET | P1d：swap_pending ∨ swaps_started>0 | RECOVER | **rev1 新增**（迁移接受规则的输出 = 最强确认变化证据；接受规则本身不变） |
| T6 | QUIET | P2：n_fresh < n_min_eff（=n_min//S，覆盖不足 = 证据不足） | **TRACK**（v1 为 RECOVER） | **rev1 改变**（R1）：恢复 S1 全采样即可补足证据；无确认变化不进 RECOVER |
| T7 | QUIET | P3：h_rate ≥ θ_out 连续 m_exit=2 个 QUIET H 周期 | **TRACK**（v1 为 h_rate≥θ_in 单周期 → RECOVER） | **rev1 改变**（R1+R3）：低余量失效回 TRACK，滞回+持续确认 |
| T8 | QUIET | h_rate ∈ [θ_in, θ_out)（滞回带内），或 ≥θ_out 仅 1 周期（exit_streak=1） | QUIET | **rev1 新增**（R3）：带内驻留；单周期尖峰只累计 streak，次周期 <θ_out 即清零 |
| T9 | RECOVER | ≥2 个完整 clean H8 周期（valid ∧ z<z_lo ∧ h_rate<θ_in） | TRACK | v1 不变（PD-4，无超时收敛） |
| T10 | RECOVER | 其他 | RECOVER | v1 不变 |

计数器语义：exit_streak = 连续 h_rate≥θ_out 的 QUIET H 周期数；任何转移复位（PD-5 扩展）；
仅 QUIET 态内累计；输入 h_rate 为全 rank 归约值 → exit_streak 全 rank 一致 → 折叠进
联合提交哈希（版本标签 "sp1r1"），任何跨 rank 分歧照旧降级为联合 HOLD。

**结构后果（如实陈述）**：rev1 下 RECOVER 仅可从 QUIET 进入（v1 同）；QUIET 内真实漂移
若未达 z_hi/事件/迁移证据，将经 T7 回 TRACK（S1/H16/W16 = F_dense 密度）而非 RECOVER
（S1/H8/W8）——检测周期 16 vs 8 的差异是否损害 W2 变化响应，是开发协议的预声明问题 Q3，
不在此先验断定。

## 3. 最小 diff（3 文件；rev 门控，rev=0 路径逐字符不变）

### 3.1 sp_state.py

```diff
@@ SP_GUARDRAIL_EVENTS 之后新增 @@
+# rev1 (2026-10-18, author R2): confirmed-change / pre-existing-urgent event
+# family, taken VERBATIM from the frozen window_v2.py vocabulary -- no new
+# detector. hold_alert/shrink-family events are alert-band (z>=z_hi at a W
+# boundary inside the cycle) or urgent (z>=z_urgent); confirmed_grow/shrink
+# are v21's own two-window confirmed drift detectors. grow_blocked_* are
+# opportunistic (not confirmed change) and stay OUT; uncertain_*/freeze are
+# already guardrails (P1b).
+SP_CHANGE_EVENTS = frozenset({
+    "hold_alert", "urgent_shrink", "urgent_shrink_capped",
+    "confirmed_shrink", "shrink_blocked_dwell",
+    "shrink_rejected_low_headroom", "confirmed_grow",
+})
+
+# rev1 (author R3): sustained exit confirmation count. Matches the frozen
+# confirmation conventions already deployed (window_shift_confirm_windows=2,
+# v21_dwell_shrink/grow=2/2). Module constant, NOT an online knob.
+SP_QUIET_EXIT_CONFIRM = 2

@@ __init__ 签名与体 @@
-                 z_lo=0.0, z_hi=0.0, theta_h_low=0.0, n_min=0):
+                 z_lo=0.0, z_hi=0.0, theta_h_low=0.0, n_min=0,
+                 theta_h_act=0.0, rev=0):
         ...
+        self.theta_h_act = float(theta_h_act)
+        self.rev = int(rev)
+        if self.rev not in (0, 1):
+            raise ValueError("sp rev must be 0|1, got %r" % (rev,))
+        if self.rev == 1 and mode == "adaptive":
+            # hysteresis band must be ordered; both edges come from the
+            # frozen calibration pair (theta_r_low, theta_r_act).
+            if not (self.theta_h_act > self.theta_h_low > 0.0):
+                raise ValueError(
+                    "rev1 requires frozen theta_h_low < theta_h_act "
+                    "(calibration pair); got %r / %r"
+                    % (self.theta_h_low, self.theta_h_act))
         ...
         self.streak = 0
         self.recover_cycles = 0
+        self.exit_streak = 0       # rev1: consecutive QUIET cycles h_rate>=theta_out

@@ propose() 非 adaptive 早退分支：返回 dict 增加 @@
+                    "exit_streak_after": self.exit_streak,

@@ propose() QUIET 分支 @@
         elif self.state == "QUIET":
-            # PD-3: exit on ANY of these; low h_rate NEVER blocks recovery.
-            <v1 原分支整体保留为 else 侧，逐字符不动>
+            exit_after = self.exit_streak
+            if self.rev:
+                # rev1 (R1/R2/R3): P1 RECOVER (confirmed change / pre-existing
+                # urgent) > P2 TRACK (insufficient evidence) > P3 TRACK
+                # (sustained low-margin failure; hysteresis band [theta_low,
+                # theta_act) + SP_QUIET_EXIT_CONFIRM consecutive cycles).
+                change_ev = (ev in SP_CHANGE_EVENTS
+                             or bool(sig.get("swap_pending"))
+                             or int(sig.get("swaps_started") or 0) > 0)
+                if z is not None and z >= self.z_hi:
+                    transition = True
+                    state_after = "RECOVER"
+                    reasons.append(
+                        "P1a z=%.3f>=z_hi=%.3f first alert (pre-existing "
+                        "urgent condition, unchanged)" % (z, self.z_hi))
+                elif guardrail:
+                    transition = True
+                    state_after = "RECOVER"
+                    reasons.append("P1b guardrail event %r" % ev)
+                elif change_ev:
+                    transition = True
+                    state_after = "RECOVER"
+                    reasons.append(
+                        "P1c/P1d confirmed change evidence: event=%r "
+                        "swap_pending=%r swaps_started=%d"
+                        % (ev, bool(sig.get("swap_pending")),
+                           int(sig.get("swaps_started") or 0)))
+                elif not coverage:
+                    transition = True
+                    state_after = "TRACK"
+                    reasons.append(
+                        "P2 n_fresh=%d<n_min_eff=%d coverage failure = "
+                        "insufficient evidence -> TRACK (rev1 R1; NOT "
+                        "RECOVER)" % (n_fresh, nme))
+                    exit_after = 0
+                else:
+                    exit_after = (self.exit_streak + 1
+                                  if h_rate >= self.theta_h_act else 0)
+                    if exit_after >= SP_QUIET_EXIT_CONFIRM:
+                        transition = True
+                        state_after = "TRACK"
+                        reasons.append(
+                            "P3 h_rate=%.6f>=theta_out=%.6f for %d "
+                            "consecutive QUIET cycles (sustained low-margin "
+                            "failure, hysteresis band) -> TRACK (rev1 R1/R3)"
+                            % (h_rate, self.theta_h_act, exit_after))
+                        exit_after = 0
+            else:
+                <v1 原分支（PD-3），逐字符不动>
+            exit_streak_after = 0 if transition else exit_after

@@ propose() 返回 dict @@
+                "exit_streak_after": (0 if transition else
+                                      (exit_after if self.state == "QUIET"
+                                       else self.exit_streak)),

@@ proposal_hash @@
-        payload = ["sp1", ...]
+        if self.rev:
+            payload = ["sp1r1", self.mode, self.state, self.streak,
+                       self.recover_cycles, self.exit_streak,
+                       bool(prop.get("transition")),
+                       prop.get("state_after", self.state),
+                       int(prop.get("streak_after", 0)),
+                       int(prop.get("recover_cycles_after", 0)),
+                       int(prop.get("exit_streak_after", 0))]
+        else:
+            payload = ["sp1", ...原样...]

@@ apply() @@
         if transition:
             ...
             self.recover_cycles = 0
+            self.exit_streak = 0     # PD-5 extended: any transition resets
         else:
             ...
+            self.exit_streak = int(
+                prop.get("exit_streak_after", self.exit_streak))

@@ snapshot()/restore() @@
+                "exit_streak": self.exit_streak,          # snapshot
+        self.exit_streak = snap["exit_streak"]            # restore
```

### 3.2 config.py

```diff
     sp_recover_min_cycles: int = 2
+    sp_rev: int = 0            # 0 = v1 frozen; 1 = rev1 QUIET-exit revision
@@ validate_sp_coupling @@
+        if self.sp_rev not in (0, 1):
+            problems.append("OEPLB_SP_REV must be 0|1, got %r" % (self.sp_rev,))
+        if self.sp_rev == 1 and self.sp_mode != 'adaptive':
+            problems.append(
+                "OEPLB_SP_REV=1 requires SP_MODE=adaptive (got %r)"
+                % (self.sp_mode,))
@@ from_env @@
+            sp_rev=int(os.environ.get('OEPLB_SP_REV', '0')),
```

### 3.3 controller.py

```diff
@@ SpController 实例化 @@
             theta_h_low=cfg.v21_theta_r_low, n_min=cfg.v21_n_min)
+            theta_h_act=cfg.v21_theta_r_act, rev=int(getattr(cfg, "sp_rev", 0)))
@@ [PB-OEPLB-SP] ACTIVE banner：格式串增加 rev=%d theta_out=%.6f 两字段 @@
@@ jsonl 行（"sp_n_min_eff" 旁）@@
+                "sp_exit_streak": self._sp.exit_streak,
```

（jsonl 行统一新增 sp_exit_streak 字段：rev0 臂恒 0。三臂同 schema，便于配对比较；
该字段为日志 schema 扩展，不触及 W/H/S、阈值或协议。）

## 4. 参数确定过程（R3 合规声明）

θ_out 与 m_exit 于本文档提交时即冻结，**先于任何 rev1 运行**；两者分别是校准战役
冻结常量（v21_theta_r_act）与部署系统冻结确认惯例（2），**没有任何参数取自
sp dev W1/W2 的服务时间**，也不存在依据 W1/W2 结果回调的通道（dev 协议 §7 明确
诚实收口条款：机制改善但时间无优势 → 收口，不开参数网格）。
封存 W1/W2 机制日志仅用于 §3 之后的离线重放（规则行为检查），见 PROTOCOL §6 免责声明。

## 5. 稀疏采样对 h_rate 波动与阈值可比性的影响（R3 说明义务）

1. **方差放大（已封存证据，PD-1 §4.3）**：S4 下 h_rate 由 ~1/4 样本估计，
   逐周期 CV 相对稠密臂放大 1.42–1.55×（W1：0.960 F_sparse vs 0.678 F_dense vs
   0.614 A；W2：0.884/0.655/0.581）。注意 F_sparse 的 H=64 与 F_dense 的 H=16
   窗长不同，该对比含窗长混杂，只作为"S4 抬高逐周期方差"的方向性证据。
2. **对滞回带宽度的诚实陈述**：校准对的带宽 θ_out/θ_in=1.129（12.9%）**窄于**
   上述噪声放大倍率（1.42–1.55×），故单周期假越 θ_out 在构造上仍可能发生；
   rev1 的主要防线是**连续 2 周期持续确认**（两次独立假越线的联合概率远低于单次），
   带宽只吸收 θ_in–θ_out 之间的贴线漂移（v1 全部 31 次低余量退出的余量为
   1.02–1.52×θ_in，其中低于 1.129× 的部分被带宽直接吸收）。
3. **阈值可比性**：θ_in/θ_out 定义在同一估计量 h_rate = h_w/n_fresh 上；
   覆盖门 n_min_eff = n_min//S 保证任何阈值判定前 fresh 样本 ≥ 3,080,192（S4），
   即阈值永不在小样本上评估。**θ 不随 S 重标定**——按 S 缩放 θ 会引入新的自由参数，
   违背"参数仅用校准材料确定"；方差问题交给覆盖门 + 滞回 + 持续确认处理。
4. **跨臂可比性**：A 臂（rev0/rev1）TRACK/QUIET 均为 H16，θ 在自身各态间可比；
   跨臂比较 h_rate 水平时须沿用 §4.3 的窗长混杂警示（F_sparse 行聚合 64 窗）。

## 6. 不变项清单（R4；diff 触及面 = 全部改动面）

| 不变项 | 验证方式 |
|---|---|
| SP_TABLE 档位 (1,16,16)/(4,16,64)/(1,8,8) | diff 不含 SP_TABLE；测试断言原值 |
| planner/rebalancer、α、迁移接受规则 | diff 不触及 rebalancer.py/controller 迁移路径 |
| 联合提交协议（phase2 折叠哈希、phase3 confirm、phase4 apply） | 哈希仅按 rev 版本化标签；协议代码不动；测试断言 rev0 哈希逐位等于 v1 原实现 |
| PD-7 seal / PD-8 W_after 中和 / PD-9 门 / seed | diff 不触及 |
| window_v2.py / window_v2p1.py（冻结） | diff 不触及 |
| v1 行为（rev=0） | 离线重放：封存 4 条 A 臂日志逐行喂 rev0 代码，须复现 32/32 已提交转移（等价性回归） |

## 7. 离线验收（作者指令三；先于任何 GPU 执行）

测试文件 $EXP/scripts/test_sp_rev1_logic.py（新文件；原 test_sp_controller_logic.py
32+40 断言保持冻结并须继续全绿）。场景矩阵：

| 场景 | 构造 | 期望 |
|---|---|---|
| A 门槛附近波动 | QUIET 内 h_rate 序列 0.0105/0.0118/0.0103/0.0120/0.0101（θ_in<·<θ_out 与单周期 ≥θ_out 交替） | 全程 QUIET；exit_streak 1→0→1→0；无转移 |
| B 持续余量上升 | h_rate 0.0115 连续 2 周期（<z_lo，覆盖通过） | 第 1 周期 QUIET（streak=1），第 2 周期 → TRACK（T7），seal S 变更 |
| B' 带内长期驻留 | h_rate ∈ [θ_in, θ_out) 连续 10 周期 | 全程 QUIET（滞回带吸收 v1 的 31 次低余量退出形态） |
| C 真实变化 | C1: z≥z_hi；C2: event=confirmed_grow；C3: event=urgent_shrink；C4: swaps_started=1 | 各自单周期 → RECOVER（T2/T4/T5） |
| D 样本不足 | n_fresh < n_min//4 | 单周期 → TRACK（T6，非 RECOVER） |
| E 同时触发 | E1: z≥z_hi ∧ 覆盖失败；E2: guardrail ∧ h_rate streak 满；E3: 覆盖失败 ∧ streak 满 | E1/E2 → RECOVER（P1 优先）；E3 → TRACK（P2 先于 P3 记录，同目的地） |
| F rev0 等价回归 | 封存 4 条 A 臂日志逐行重放 rev0 | 32/32 转移、streak/recover_cycles 轨迹与 sp_state_after 逐行一致；proposal_hash 与原实现逐位一致 |
| G rev1 重放（规则行为检查） | 同一日志喂 rev1 | 32 次退出重分类（§8 免责：不预测吞吐/路由） |
| H 快照对称 | rev1 下 snapshot→propose→restore | exit_streak 复原；HOLD 回滚安全 |

## 8. 旧日志重放免责声明

重放仅检查**规则行为**（给定已记录信号，rev0/rev1 各会作出什么转移决定）。
v1 运行在退出后即离开 S4 采样，"若当时未退出，下一 QUIET 周期的 h_rate/z/event"
是**不可观测的反事实**；因此重放**不声称**能预测修订后的吞吐、服务时间或路由轨迹。
任何此类预测只能来自 PROTOCOL_dev_r1 的新运行。
