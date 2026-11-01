# INCIDENT #3 — m_E3_F_b8 (MAIN): F-arm E3 window#17 悬挂 → RESULT_RETIRE；E3-w17 "299 异常指纹"确立

- 状态: CLOSED — verdict 2026-10-19T05:37:26Z RESULT_RETIRE（修正版 checker crash-evidence 路径**第二次实战触发**：4 类 fatal markers + bench.json 缺失/SIGTERM）
- 分级: method crash per plan §9.9 → RESULT_RETIRE。非基础设施失败；**非 HALT**。不重试；矩阵不裁剪；无中途改参。
- 意义: 第三起 swap 提交臂悬挂；与事故 #2 **同一悬挂点**（E3 window#17）但换到 F 臂——跨臂在同一流事件上复发，确立 E3-w17 异常指纹（见 §3）。

## 1. 涉事格

- cond=E3（8192 req，in512/out1，conc=256，BENCH_TIMEOUT=3600s），arm=F（forced "1 16 16"），block=8。
- ARMFLAGS（L139）: `sp_mode=fixed sp_s=1 sp_h=16 sp_w=16 sp_rev=0 force_s=0 no_commit=0 pair_rule=gap`。
  - 观察项：F 臂所有格（m_E3_F_b1/b7、m_E2_F_b2 抽查）`sp_rev=0` 一致——冻结启动脚本行为，rev 语义属 adaptive 模式；跨格一致性=公平性保持。仅记录，无行动。
- sp_run_end.json: mode=fixed，S/H/W=1/16/16（冻结值，无漂移），sp_fwd_total=282，window_count=17，swaps_started_total=4 —— **与事故 #2 (m_E3_P_b6) 完全相同的统计指纹**。

## 2. 时间线（UTC 2026-10-19）

| 时刻 | 事件 | 证据 |
|---|---|---|
| 04:36:28 | server 启动 | server.log 首行 |
| 04:37:06 | ARMFLAGS | L139 |
| 04:37:26 | BOOT_OK | campaign_marks.tsv |
| ~04:36:57 | bench 开始（8192 req conc=256） | bench.log L1 |
| 04:39:05 | **window#17: issued 299 swap(s)** (98566144 tok global, dp=8) → COALESCED SeqNum=5 永不完成；TIMING n_ops=299 仅 DP1/DP2/DP6/DP3 打印，DP0/DP5 无 issue/TIMING 行（与 #2 相同的 rank 间分歧） | L7441-7446 |
| 04:49:06 | watchdog timeout ranks 1/0/5/6（SeqNum=5, COALESCED, NumelIn=0, 600000ms） | L7686-7698 |
| ~04:49+ | DistBackendError terminate ×1、Fatal Aborted ×2、watchdog 行 ×12、Scheduler exception ×3 | server.log |
| ~05:36:57 | BENCH_TIMEOUT=3600 SIGTERM → bench.json 缺失 | driver |
| 05:37:26 | RESULT_RETIRE（crash-evidence 路径） | verdict.txt |

单格总代价 ≈61 min（与 #2 的 62 min 一致）。

## 3. 新取证发现：E3-w17 "299 异常指纹"（移交 M2）

| E3 通过 window#17 的格 | 臂 | w17 发出 swaps | sp_fwd_total(全程) | 结局 |
|---|---|---|---|---|
| m_E3_F_b1 | F | **300** | 522 | VALID |
| m_E3_F_b7 | F | **300** | 522 | VALID |
| m_E3_P_b8 | P | **300** | 522 | VALID |
| m_E3_A2_b8 | A2（不同配对规则） | 185 | 522 | VALID |
| m_E3_P_b6（#2） | P | **299** | 282（崩溃时） | RESULT_RETIRE |
| m_E3_F_b8（#3） | F | **299** | 282（崩溃时） | RESULT_RETIRE |

- 同一密封流（确定性请求序）+ 同一冻结布局动力学下，**所有幸存 E3 通过都在 w17 发出 300 swaps；两起崩溃通过都发出 299** ——跨臂（P/F）一致。规划确定性被打破 → **规划/状态竞态**：偏常 299-op 批次与悬挂强相关（P/F 通过中 ~2/5 出现偏常）。
- DP0/DP5 在两起事故中都缺席 w17 的 issue/TIMING 日志但其 watchdog 触发 → 可能与其余 rank 的批次内容不一致（P2P 配对错位假说）。因果链（偏常规划→跨 rank 批次不一致→COALESCED 死锁，或同一底层分歧的两个症状）= M2 诊断目标（≤4 boots，campaign 后）。
- **限定**：op 数非普适规则——事故 #1 在 E4 w55 以 300-op 悬挂，而 m_E4_P_b4 在 E4 以 299-op 幸存。指纹是"流×事件"特异的；共同主线仍是 swap 提交路径大迁移批次的概率性竞态。

## 4. 累计退役登记（截至 53 verdicts：50 VALID + 3 RESULT_RETIRE ≈5.7%）

| # | 格 | 臂/流 | 悬挂点 | SeqNum | watchdog ranks | bench 收敛 | verdict 路径 |
|---|---|---|---|---|---|---|---|
| 1 | m_E4_F_b4 | F/E4 | w55, 300 swaps | 14 | 4 ranks | 自然结束 ok=1281 err=767 t=1861s | 原路径 |
| 2 | m_E3_P_b6 | P/E3 | w17, 299 swaps | 5 | 5/6/0/1 | SIGTERM ~02:45:42Z | 修正路径首战 |
| 3 | m_E3_F_b8 | F/E3 | w17, 299 swaps | 5 | 1/0/5/6 | SIGTERM ~05:36:57Z | 修正路径第2战 |

臂退役：F 2（E4×1, E3×1）、P 1（E3×1）；I/E/A1/A2/A3/A4 零退役。免疫解释不变：I 无 swap、A4 no_commit、E 官方 EPLB 路径。

## 5. 家族影响（冻结统计规则下）

- **确证家族 P-vs-I / P-vs-E\***：仍仅 block 6 缺（m_E3_P_b6）→ n=7 完整块。#3 是 F 臂，不影响确证家族。
- **次级家族 P-vs-F\***：block 8 F 侧缺（#3）+ block 6 P 侧缺（#2）→ 当前 n≤6 完整块（若无更多退役）。
- **前瞻登记（非设计变更）**：E3 剩余 swap 提交臂通过 P×6、F×5、A2×7；若 w17 偏常危害率维持 ~1/3，预期 E3 再退役 ~3-5 格；E4 剩余 F×7、P×4 携带 w55 型危害。协议应对不变：继续跑、按 (cond,arm) 记完成率、最终报告 limitations 量化。**不做任何中途参数/矩阵变更（冻结规则）。**

## 6. 证据文件

- `$F/main/m_E3_F_b8/`: server.log（行号见上）、sp_run_end.json、bench.log、meta.txt、verdict.txt（bench.json 缺失=SIGTERM 路径）
- `$F/main/INCIDENT_E4F_hang.md`（#1）、`$F/main/INCIDENT_E3P_hang.md`（#2）
- `$F/logs/campaign_marks.tsv`（BOOT_OK 04:37:26Z；VERDICT 05:37:26Z；INCIDENT 行）
