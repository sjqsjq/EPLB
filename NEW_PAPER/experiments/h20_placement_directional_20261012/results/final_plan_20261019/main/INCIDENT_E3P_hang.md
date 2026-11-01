# INCIDENT #2 — m_E3_P_b6 (MAIN seq 22/168): P-arm async-swap migration NCCL COALESCED hang → RESULT_RETIRE

- 状态: CLOSED — converged 02:45:44Z (RESULT_RETIRE via amended crash-evidence path, first live firing of the incident-#1 amendment)
- 分级: **method crash per plan §9.9 → RESULT_RETIRE**。非基础设施失败；**非 HALT 条件**。矩阵不裁剪；不重试（冻结规则）。
- 重要性: 事故 #1（m_E4_F_b4，F 臂）同签名的第二次发生，但这次在 **P 臂——确证比较臂**。事故 #1 文档中"F 臂强制节奏暴露"的定性由本文档取代/泛化：悬挂发生在**任何提交 swap 的臂**的迁移路径上，与臂无关。

## 1. 涉事格

- run_list_main.csv seq 22: cond=E3（multidomain switch，8192 req，in512/out1，conc=256，BENCH_TIMEOUT=3600s），arm=P（adaptive，params=""），block=6。
- ARMFLAGS（server.log L139-140）: `sp_mode=adaptive sp_s=1 sp_h=0 sp_w=16 sp_rev=1 force_s=0 record_all_phase=0 no_commit=0 pair_rule=gap`
- sp_run_end.json: mode=adaptive，S_final=1 H_final=16 W_final=16（rev1 冻结值，无漂移），sp_fwd_total=282，window_count=17，swaps_started_total=4，transitions=[]。

## 2. 时间线（UTC 2026-10-19）

| 时刻 | 事件 | 证据 |
|---|---|---|
| 01:44:39 | server 启动日志开始 | server.log 首行 |
| 01:45:19 | ARMFLAGS（adaptive P 臂确认） | L139-140 |
| 01:45:42 | bench 开始: 8192 req e3_multidomain_switch.jsonl conc=256 | bench.log L1；ps etimes 反推 |
| 01:45:44 | BOOT_OK | campaign_marks.tsv |
| 01:45:54 | window#1: 268 swaps → batch_isend_irecv ≈1.40s（首批慢）成功 | L1095-1110 |
| 01:46:00 | window#2: 49 swaps 成功 | L1516+ |
| 01:46:38 | window#9: **299 swaps 成功**（近上限批次，悬挂前 42s） | L4265-4284 |
| 01:46:43 | window#10: 259 swaps 成功 | L4691+ |
| 01:47:20 | window#17: issued 299 swaps (98566144 tok global, dp=8) → **COALESCED SeqNum=5 永不完成** | L7435-7440（TIMING n_ops=299，仅 DP1/2/3/4/6/7）；watchdog 600036ms 反推 |
| 01:57:20 | watchdog timeout ranks 5/6/0/1（Timeout(ms)=600000, OpType=COALESCED, NumelIn=0, NumelOut=0） | L7703, 7707, 7711, 7715 |
| ~01:57+ | `c10::DistBackendError` terminate → Fatal Python error: Aborted 多 rank 级联 | L7763, 7782, 7841, 7860, 7935 |
| 02:17:23 | frontend 半死（/health 超时路径同 #1）；bench 以每请求 600s 客户端超时排空（5277/8192 已记录） | bench.log 尾部 |
| ~02:45:42 | BENCH_TIMEOUT=3600 SIGTERM → bench.json 缺失 | driver 配置；ps etimes=3044 @02:36:26 |
| 02:45:44Z | 修正版 checker crash-evidence 路径 → RESULT_RETIRE（4 个 fatal markers；02:36Z 离线演练在同一 artifact dir 上产生相同判定） | verdict.txt |

## 3. 根因链（确证 vs 假设分级）

**确证（日志事实）**：

1. 悬挂点与事故 #1 完全同类：PB-OEPLB async swapper 的 `batch_isend_irecv` P2P 迁移集合通信（WorkNCCL OpType=COALESCED，NumelIn=0/NumelOut=0=纯 P2P 批次），SeqNum=5=本 run 第 5 次 batch_isend_irecv 事件，永不完成 → 600s watchdog → abort 级联 → scheduler 全死。
2. 悬挂批次的 rank 间分歧：DP1/2/3/4/6/7 打印了 n_ops=299 的 TIMING begin()（isend 侧 9-15ms 返回）；**DP0 与 DP5 既无 window#17 issue 行也无 TIMING 行**，但其 watchdog 捕获 SeqNum=5 超时（fired ranks = 0,1,5,6）。→ 移交 M2 诊断。
3. bench 客户端（sha 锁定，不可改）以每请求 600s 超时排空悬挂请求；3600s SIGTERM 使 bench.json 不写出。

**假设（分级）**：

- H1（强）：**概率性竞态**，非确定性故障。同一 run 内悬挂前 4 批大迁移全部成功（268/49/299/259 ops，含悬挂前 42s 的一次 299-op）；P 臂此前 4 格 VALID（m_E1_P_b2/b6、m_E2_P_b5、m_E4_P_b4）。
- H2（中）：**非臂特异**。#1=F（强制），#2=P（自适应）；共同机制=经 batch_isend_irecv 提交 swap。迄今免疫：I（从不 swap）、A4（no_commit：只规划不迁移，E3 上 2× VALID）、E（官方 EPLB 迁移路径，迄今 6 格 VALID 0 悬挂）。
- H3（中）：**非流特异**。F 在 E4 悬挂但通过 E3（m_E3_F_b7 VALID）；P 通过 E1/E2/E4 但在 E3 悬挂；A2（提交 swap 的 pair-rule 变体）在 E3 VALID（m_E3_A2_b8）。
- H4（弱，待 M2）：暴露/节奏因子——两次悬挂都发生在窗口快速翻转+近上限批次的早期阶段（#1: 62s 内 55 窗口；#2: 93s 内 17 窗口）。复现与根因=M2 诊断预算（≤4 boots，campaign 后）。

## 4. 处置（全部依冻结规则，无需咨询）

1. §9.9: method crash = 结果 → RESULT_RETIRE；与基础设施失败分开记录；非 HALT。
2. 不重试（冻结 retire 规则）；不裁剪矩阵；不中途改参（W/H/S、max_total_ops=300 属冻结臂机制，"计时开始后不得改"）。
3. 修正版 checker（事故 #1 时冻结，sha a44e1e34…）的 SIGTERM/crash-evidence 路径在本格**首次实战触发**；02:36Z 离线演练已在同一 artifact dir 上验证输出 RESULT_RETIRE。
4. **确证分析影响**：g_b(P,B)=(1/4)Σ_j log(T_{B,j,b}/T_{P,j,b})，j=E1-E4 四工作流，需要 block b 全部 4 个 P 格。m_E3_P_b6 退役 → block 6 对 P 家族不完整 → P-vs-I 与 P-vs-E\* 的完整块数 n=7（计划 §8.1: n=8 是预算档非功效保证）。按完成率结果如实报告。
5. 完成率台账按 (cond,arm) 记账；两起悬挂进入最终报告 limitations + swap-race 发现节。

## 5. 与事故 #1 对照

| | #1 m_E4_F_b4 | #2 m_E3_P_b6 |
|---|---|---|
| 臂 | F（forced 1/16/16） | **P（adaptive，确证臂）** |
| 流 / N | E4 fullduplex / 2048 | E3 multidomain / 8192 |
| 悬挂批次 | window#55，300 swaps | window#17，299 swaps |
| SeqNum | 14 | 5 |
| watchdog ranks | 4 ranks | ranks 5/6/0/1 |
| run 内先前成功大批次 | 有（calib F ×10 VALID 含 300-op） | 有（268/299/259 ops 于 86s 内） |
| bench 收敛 | 自然结束 ok=1281 err=767 t=1861.07s | SIGTERM ~02:45:42Z → crash-evidence 路径 |
| verdict | RESULT_RETIRE（原路径） | RESULT_RETIRE（修正路径首战） |

## 6. 证据文件

- `$F/main/m_E3_P_b6/`: server.log（行号见上）、sp_run_end.json、v2p1_rank0.jsonl、bench.log、meta.txt、verdict.txt
- `$F/main/INCIDENT_E4F_hang.md`（事故 #1；其尾部 addendum 指向本文档）
- `$F/logs/campaign_marks.tsv`（BOOT_OK 01:45:44Z；INCIDENT 行；VERDICT 行）
- `$F/scripts/integrity_check_fp.py`（修正版 sha a44e1e34…；旧版保存于 C_configs/integrity_check_fp_pre_amendment_20261019.py）
