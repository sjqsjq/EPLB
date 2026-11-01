# INCIDENT #7 — m_E3_P_b3（P 臂，E3 multidomain_switch）— E3-w17 悬挂（指纹第 5 次精确复现 + 首次活体 rank 集取证）

**状态**: 已闭环（RESULT_RETIRE 16:46:18Z，修订版崩溃证据路径第 5 次实弹，分类正确）。method crash ≠ HALT（§9.9）。
**主文档链**: #4（INCIDENT_E3F_b4_hang.md，指纹表）、#5（INCIDENT_E3F_b3_hang.md）、#6（INCIDENT_E4F_b7_hang.md，六事故总表+通用签名）。

## 1. 时间线（UTC）

| 时刻 | 事件 |
|---|---|
| 15:46:18Z | BOOT_OK m_E3_P_b3（**P 臂 sp_mode=adaptive rev1**，崩溃时 state=TRACK，S/H/W=1/16/16，transitions=0），E3 流 N=8192 conc=256 |
| 15:48:03Z | window #17：DP1/2/3/4/6/7 issued **299 swap(s)**（98566144 tok global, dp=8）；**DP0/DP5 缺席**；sp_run_end: sp_fwd_total=**282**, window_count=17, swaps_started_total=**4**, mode=adaptive, state_final=TRACK |
| 15:54:33Z | **活体取证（首次）**：watchdog 触发前 GPU 利用率已分裂——**ranks {0,1,5,6} 100% 忙等 / {2,3,4,7} 0%**——实时锁定卡死 rank 集 |
| 15:58:03Z | watchdog 证实：**Rank 0/1/5/6**（与活体预测完全一致）SeqNum=**5** OpType=COALESCED NumelIn=0 600s 超时（12 行） |
| ~15:59Z | abort 级联（4× Fatal Python error），scheduler 死亡，bench 排水 |
| 16:46:18Z | BENCH_TIMEOUT=3600s → SIGTERM → checker：bench.json 缺失 + fatal markers → **RESULT_RETIRE** |
| 16:47:29Z | driver **71 秒自愈**：m_E2_E_b1 BOOT_OK → 16:49:21Z VALID |

## 2. 增量取证结论

1. **E3-w17 指纹五连**：299 / fwd=282 / swaps_started=4 / DP0+DP5 issue 缺席 / watchdog ranks {0,1,5,6} / SeqNum=5——跨 P（#2、#7）/F（#3、#4、#5）两臂、五个 block（b6/b8/b4/b3/b3）逐项相同。
2. **首次活体确认卡死 rank 集**：15:54Z（悬挂后 6 分钟、watchdog 前 4 分钟）GPU 利用率分裂 {0,1,5,6}=100% / {2,3,4,7}=0%，随后 watchdog 在完全相同的 rank 集触发。证明：(a) 卡死方是 COALESCED P2P 批次不匹配中等待缺失 peer 的 4 个 rank（NCCL kernel 忙等→100% util）；(b) 其余 4 rank 的对应 op 已完成或未进入（0%）；(c) 该 rank 集在 E3 家族五起中固定——分歧方恒为 DP0/DP5 侧。
3. **P 臂 adaptive 状态无关性**：崩溃时状态机在 TRACK（0 转移，swap 活动持续使 quiet_cond 不成立），即 P 臂以与 F 臂完全相同的 (1,16,16) 密集日程签发时同样命中 w17 临界态——竞态在 plan→issue 层，与节奏自适应无关（但 F 臂 5/7 过度代表与其恒定密集日程一致）。
4. **事故登记册（7 起）**：F 臂 5 起（#1/#3/#4/#5/#6）+ P 臂 2 起（#2/#7）；E3-w17 家族 5 起 + E4 2 起（w55 #1、w37 #6）。A1/A2/A3/A4/E/I 迄今 0 起。

## 3. 处置与家族影响（冻结规则，零协议变更）

- RESULT_RETIRE（崩溃证据路径），不重试、不裁剪、阈值/W/H/S 不动；完成率台账按 (cond,arm) 记录。
- **P-vs-I / P-vs-E\***：block 3（P 侧 E3）丢失 → 上限 **n≤6**（blocks 6+3 失）。**P-vs-F\***：仍 **n≤3**（block 3 此前已因 F 侧丢失）。
- **P×E3 终局在途**: b1 V, b3 R, b6 R, b7 V, b8 V → 3V/2R；剩余 P×E3：b2/b4/b5。
- **预测登记更新**：MAIN 剩余高危通道 ≈ P×E3 b2/b4/b5、F×E3 b5/b6、A2×E3 及 E4 swap 臂余量；按观测 E3 危险率 ~5/19≈1/4，预期再 ~1-3 起退役；如实记录，不改协议。
- **M2（≤4 boots，战役后）**：7 样本（含首次活体 rank 集证据）；离线优先比对 w17 幸存(300)/崩溃(299)计划行与 DP0/DP5 侧规划分歧点。
