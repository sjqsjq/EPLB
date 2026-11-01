# INCIDENT #5 — m_E3_F_b3（F 臂，E3 multidomain_switch）— E3-w17 299-swap NCCL 悬挂（指纹第 4 次精确复现）

**状态**: 已闭环（RESULT_RETIRE 11:53:20Z，修订版 checker 第 4 次实弹，分类正确）。method crash ≠ HALT（§9.9）。
**主文档**: 完整时间线模板、指纹对比表、M2 取证线索见 INCIDENT_E3F_b4_hang.md（#4）；本文档只记 #5 特有事实与增量结论。

## 1. 时间线（UTC）

| 时刻 | 事件 |
|---|---|
| 10:53:20Z | BOOT_OK m_E3_F_b3（F 臂 sp_mode=fixed 1/16/16 rev=0），E3 流 N=8192 conc=256，bench timeout 3600s |
| 10:55:06Z | window #17：DP1/2/3/4/6/7 issued **299 swap(s)**（98566144 tok global, dp=8）；**DP0/DP5 缺席 issue 日志**；sp_run_end: sp_fwd_total=**282**, window_count=17, swaps_started_total=**4**, mode=fixed, transitions=[] |
| 11:05:06Z | watchdog：**ranks {0,1,5,6}**（与 #2/#3/#4 完全相同 rank 集）SeqNum=5 OpType=COALESCED NumelIn=0 600s 超时 |
| ~11:06Z | abort 级联（4× Fatal Python error: Aborted），scheduler 死亡，GPU 归零，frontend 半死排水 |
| 11:53:20Z | BENCH_TIMEOUT=3600s → SIGTERM → checker：bench.json 缺失 + fatal markers → **RESULT_RETIRE** |
| 11:54:28Z | driver 自愈：下一格 m_E4_E_b6 BOOT_OK → 11:58:49Z VALID（无需干预） |

## 2. 增量取证结论

1. **指纹四连**：w17 / 299 swaps / fwd=282 / swaps_started=4 / DP0-DP5 issue 缺席 / watchdog ranks {0,1,5,6} / SeqNum=5——跨 P/F 两臂、四个 block（b6/b8/b4/b3）逐项相同。**ranks {0,1,5,6} 的重复出现**提示分歧方固定为 DP0/DP5（及其 TP 对偶 rank），即计划分歧发生在**特定 rank 对**上而非随机 rank 子集。
2. **确定性临界态再确认**：v2p1 w17 统计与 #4 逐字节相同（z=124.79036828239538, s_change_logical=0.4611823558807373, n_fresh=98566144）——E3 流第 17 窗口的检测/规划输入完全确定；悬挂与否取决于 plan→issue 层非确定性（到达时序/线程调度），一个边界 swap 对在 DP0/DP5 与其余 6 rank 间纳入不一致 → P2P 批次不匹配 → COALESCED 永久悬挂。
3. **M2 输入**：4 崩溃样本 + 幸存对照（300-swaps：F_b1/F_b7/P_b8 等）；离线可先比对幸存与崩溃运行的 w17 计划行（无需新 boot），再决定是否用 ≤4 boots 复现取证。

## 3. 处置与家族影响（冻结规则，零协议变更）

- RESULT_RETIRE，不重试、不裁剪、阈值/W/H/S 不动；完成率台账按 (cond,arm) 记录。
- **P-vs-F\***：block 3（F 侧 E3）丢失 → 上限 **n≤4**（已失 6/8/4/3）。P-vs-I/P-vs-E\* 仍 n≤7。
- **E3-F 通道终局**: b1 V, b7 V, b3 R, b4 R, b8 R → **2V/3R**；剩余 F×E3：b2/b5/b6（危险率按观测 ~3/5-1/2）。
- **预测登记更新**：E3 剩余 swap 臂通道（F×3 + P×~4 + A2×~6 ≈ 13）按观测危险率 ~1/3 估计，预期再 ~3-4 起退役；如实记录，不改协议。
