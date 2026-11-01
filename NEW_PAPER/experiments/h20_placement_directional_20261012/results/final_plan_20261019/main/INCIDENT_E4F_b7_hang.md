# INCIDENT #6 — m_E4_F_b7（F 臂，E4 fullduplex）— E4-w37 299-swap NCCL 悬挂（新指纹变体）

**状态**: 已闭环（RESULT_RETIRE 13:30:13Z）。method crash ≠ HALT（§9.9）。
**判决路径**: 与 #1（m_E4_F_b4，同 E4）相同——bench **自然结束**（ok=1024 errors=1024, N=2048, t=2452.76s），完成率路径判 RESULT_RETIRE；**非**修订版崩溃证据路径（#2-#5 用该路径，bench.json 缺失）。两条路径分类均正确。

## 1. 时间线（UTC）

| 时刻 | 事件 |
|---|---|
| 12:49:18Z | BOOT_OK m_E4_F_b7（F 臂 sp_mode=fixed 1/16/16 rev=0），E4 fullduplex N=2048 conc=256 |
| 12:50:13Z | window #37：7 个 rank（DP0/1/2/3/5/6/7）各 issued **299 swap(s)**（45817856 tok global, dp=8）；**DP4 缺席 issue 日志**；sp_run_end: sp_fwd_total=**602**, window_count=37, swaps_started_total=**7**；v2p1 w37: **z=null, s_change_logical=null**（E4 检测统计在崩溃窗口未定义），h_rate=0.0616 |
| 13:00:13Z | watchdog：**ranks {2,3,4,5,6}** SeqNum=**7** OpType=COALESCED NumelIn=0 600s 超时 |
| ~13:01Z | abort 级联（5× Fatal Python error），scheduler 死亡 |
| ~13:30Z | bench 自然结束：ok=1024 / errors=1024（50% 完成率）→ checker 完成率路径 → **RESULT_RETIRE t=2452.76** |
| 13:31:20Z | driver 自愈：m_E3_E_b5 BOOT_OK → 13:34:47Z VALID（67 秒恢复，无需干预） |

## 2. 取证：通用签名 vs 家族特异指纹

**六起事故总表**：

| # | 格 | 臂 | 流 | 崩溃窗口 | issued | sp_fwd | DP 缺席 | watchdog ranks | SeqNum | 判决路径 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | m_E4_F_b4 | F | E4 | w55 | 300 | — | — | — | 14 | 完成率(自然结束) |
| 2 | m_E3_P_b6 | P | E3 | w17 | 299 | 282 | DP0/DP5 | {0,1,5,6} | 5 | 崩溃证据 |
| 3 | m_E3_F_b8 | F | E3 | w17 | 299 | 282 | DP0/DP5 | {0,1,5,6} | 5 | 崩溃证据 |
| 4 | m_E3_F_b4 | F | E3 | w17 | 299 | 282 | DP0/DP5 | {0,1,5,6} | 5 | 崩溃证据 |
| 5 | m_E3_F_b3 | F | E3 | w17 | 299 | 282 | DP0/DP5 | {0,1,5,6} | 5 | 崩溃证据 |
| 6 | m_E4_F_b7 | F | E4 | **w37** | 299 | **602** | **DP4** | **{2,3,4,5,6}** | **7** | 完成率(自然结束) |

**结论更新**：
1. E3-w17 家族指纹（fwd282/DP0+DP5/ranks{0,1,5,6}/SeqNum=5）是**流特异的确定性临界态**；#6 证明通用机制更广：**"299-swap 签发 + 1-2 个 DP 缺席 issue 日志 → P2P COALESCED 批次不匹配 → 永久悬挂"**，窗口号/缺席 rank/SeqNum 随流与运行而变。
2. 299（非 300）出现在 6 起中的 5 起（#1 为 300 悬挂）——op 数非普适规则（与 #3 文档结论一致），但"少数 rank 计划分歧"是所有事件的共同核心。
3. **F 臂过度代表**：6 起中 5 起为 F（固定密集 1/16/16 rev=0），1 起为 P。F 臂每 16 forward 规划一次、签发密度最高；P 臂自适应会进入 QUIET 降低签发频次。A1-A4/A2/E/I 迄今 0 起。这与"竞态概率 × 签发次数"一致，也提示 F 臂（密集日程）承担最高悬挂暴露——完成率台账如实反映。
4. #6 的 v2p1 z=null/s_change=null：E4 双流下 w37 检测统计未定义但规划仍签发 299 swaps——M2 需核查 null-z 窗口的规划输入来源（headroom 路径）。

## 3. 处置与家族影响（冻结规则，零协议变更）

- RESULT_RETIRE（完成率路径），不重试、不裁剪、阈值/W/H/S 不动。
- **P-vs-F\***：block 7（F 侧 E4）丢失 → 上限 **n≤3**（blocks 3/4/6/7/8 失，仅余 1/2/5）。**P-vs-I / P-vs-E\***：仍 n≤7（仅 block 6 失）。E-vs-F、F-vs-I 等次级家族按同规则在统一分析时逐 block 核定。
- **F 臂完成率**：E3 2V/3R（b1,b7 V；b3,b4,b8 R）、E4 5V/2R（b4,b7 R）。剩余 F 格：E3 b2/b5/b6、E4 b3 等——危险率按观测登记，预测再有 ~2-4 起退役（E3/E4 swap 臂剩余通道合计）。
- **M2（≤4 boots，战役后）**：6 样本（2 流、2 臂、2 判决路径）+ 幸存对照；先离线比对 w17/w37 计划行与 DP 缺席模式。
