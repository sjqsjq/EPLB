# INCIDENT #4 — m_E3_F_b4（F 臂，E3 multidomain_switch）— E3-w17 299-swap NCCL 悬挂

**状态**: 已闭环（RESULT_RETIRE 09:46:52Z，修订版 checker 第 3 次实弹）。method crash ≠ HALT（计划书 §9.9）。
**关联**: INCIDENT #2（m_E3_P_b6）、#3（m_E3_F_b8）——本起为指纹第 3 次精确复现；#5（m_E3_F_b3）在途，指纹第 4 次复现（见文末）。

## 1. 时间线（UTC）

| 时刻 | 事件 |
|---|---|
| 08:46:52Z | BOOT_OK m_E3_F_b4（server pid 1060452；F 臂 sp_mode=fixed S/H/W=1/16/16 rev=0）；bench 客户端 pid 1066374（timeout 3600s），E3 流 N=8192 conc=256 |
| 08:48:38Z | window #17：**6 个 rank（DP1/2/3/4/6/7）各 issued 299 swap(s)**（98566144 tok global, dp=8）；**DP0/DP5 缺席 issue 日志**；sp_run_end: sp_fwd_total=282, window_count=17, swaps_started_total=4, transitions=[]（fixed 模式无转移，正常） |
| 08:58:38Z | NCCL watchdog：ranks {0,1,5,6} WorkNCCL(SeqNum=5, OpType=COALESCED, NumelIn=0, NumelOut=0) 600s 超时 |
| 08:59:38Z | watchdog 线程 terminate → c10::DistBackendError → Fatal Python error: Aborted 级联；scheduler 死亡；frontend 半死（排水期仍回 200 OK），GPU 归零 |
| 09:46:52Z | BENCH_TIMEOUT=3600s → SIGTERM bench；checker：bench.json 缺失 + 4 个 fatal marker → **RESULT_RETIRE**（修订版崩溃证据路径第 3 次实弹，分类正确） |
| 09:48Z~ | driver 自愈推进下一格（无需干预，与 #2/#3 一致） |

## 2. 指纹对比（四起事故，跨 P/F 两臂、四个 block）

| 事故 | 格 | 臂 | 流 | 崩溃窗口 | issued swaps | sp_fwd_total | DP 缺席 | watchdog ranks | SeqNum |
|---|---|---|---|---|---|---|---|---|---|
| #2 | m_E3_P_b6 | P (adaptive rev1) | E3 | w17 | 299 | 282 | DP0/DP5 | {0,1,5,6} | 5 |
| #3 | m_E3_F_b8 | F (fixed rev0) | E3 | w17 | 299 | 282 | DP0/DP5 | {0,1,5,6} | 5 |
| #4 | m_E3_F_b4 | F (fixed rev0) | E3 | w17 | 299 | 282 | DP0/DP5 | {0,1,5,6} | 5 |
| #5(在途) | m_E3_F_b3 | F (fixed rev0) | E3 | w17 | 299 | (282 预期) | DP0/DP5 | {5,6,+待核} | 5 |
| 幸存通道 | F_b1, F_b7, P_b8 等 | P/F | E3 | w17 通过 | **300** | 522(至运行尾) | 无 | — | — |

**新证据（#4/#5 强化）**：两次运行的 v2p1 w17 行**逐字节相同**（z=124.79036828239538, s_change_logical=0.4611823558807373, n_fresh_assignments=98566144）——E3 固定流回放下，w17 的**检测器统计是确定性的**；分叉（299+DP0/DP5 分歧 vs 300 全体一致）发生在 **plan→issue 层的非确定性环节**（跨 rank 计划分歧：一个边界 swap 对在部分 rank 被纳入/排除）。299-vs-300 是分歧的症状而非原因。

## 3. 处置（冻结规则，零协议变更）

- RESULT_RETIRE（method crash，§9.9）；失败启动 retire 不重试；矩阵不裁剪；W/H/S 与阈值不动。
- 修订版 checker（sha a44e1e34…，崩溃证据门控）第 3 次实弹，分类正确。
- **M2 诊断（≤4 boots，战役后）**：现已有 4 起同指纹样本 + 幸存对照（300-swaps 通道），可离线比对 w17 计划分歧点；无需新 boot 即可先做日志取证。

## 4. 家族影响（冻结统计规则）

- **P-vs-F\***：block 4（F 侧 E3）丢失 → 上限 **n≤5**（已失 6/8/4；若 #5 b3 退役 → **n≤4**）。
- **P-vs-I / P-vs-E\***：仍 n≤7（b6 P 侧丢失）；P 臂 E3 剩余通道若再挂将继续收窄。
- **E3-F 通道完成率**：b1 V, b7 V, b8 R, b4 R, b3 R(在途) → 2V/3R；剩余 F×E3：b2, b5, b6。
- 完成率台账按 (cond,arm) 记录退役格，统一分析时如实报告（含退役）。

## 5. 预测登记（本起写入时）

E3 剩余 swap 臂通道（F×3：b2/b5/b6；P×剩余；A2×剩余）以当前 E3-F 危险率（3/5）估计，预期再有 ~2-4 起退役；**不改协议、不绕行、不重试**——如实记录并把 w17 临界状态留给 M2。

## 6. 事故 #5 速记（在途，闭环时另立文档）

m_E3_F_b3：10:53:20Z BOOT_OK → 10:55:06Z w17 issued 299（DP1/2/3/4/6/7，DP0/DP5 缺席）→ 11:05:06Z watchdog（SeqNum=5 COALESCED，ranks 含 5/6）→ abort 级联（4× Fatal Python error）→ 排水中，SIGTERM 预计 11:53:20Z。
