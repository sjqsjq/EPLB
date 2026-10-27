# prover_book_switch_20261027 归档 README

跨域切换 stage2 域效应对照战役(prover→book)。本目录为 GitHub 精选档; 全档(含逐boot server.log/perreq 与流文件 30MB)在 /data/minghua/sjq/PAPER_EXPERIMENTS/prover_book_switch_20261027/(清单 ARCHIVE_SHA256SUMS.txt); 工作现场 /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/prover_book_switch_20261027/。

定位: 作者 2026-10-27 中止 pc 战役(stage2=code)后指定的补充对照: stage2 换成 book(paper freq6 book↔prover 同族域对), 只跑 2 reps, 跑完作者+agent 联合分析。**n=2, df=1, 全部数字为描述性 NON-INFERENTIAL, 无推断统计**。

## 结论速览 (12/12 boots, 零异常)

- 流: prover[0:2048]@≈2048tok → book[0:1000]@≈4438tok, n=3048, sha 2da34672…, 全行 O=1 conc=32; book 用发布集全量 1000 行(作者要求 1024, 已披露)。
- 六臂: I / ES(R16 静态计数+冗余) / ED(周期100 EPLB) / MT(pinned_fair) / DF(datafore greedy) / P0(冻结 AERO-base rev1 RG=0)。轮换 r1: ES ED MT DF P0 I; r2: ED MT DF P0 I ES。
- prover 块均值(req/s): DF 11.23 ≈ MT 11.16 > P0 10.79 > ES 10.49 > ED 9.74 > I 9.42 (校准收益兑现, DF/MT vs I +19%)。
- book 块均值(req/s): ED 4.56 > P0 4.50 > MT 4.43 > I 4.41 > ES 4.38 > DF 4.36 (全臂 4.23-4.72, 极差约11%; prover 块 19% 差距消失)。
- P0 轮间分裂(本战役最大现象): r1 book 4.29 最低 + b_late 3.97(early→late −10%), 27 次 swap 滴灌; r2 book 4.72 最高 + b_late 4.55 无跌幅, 仅 6 次 swap。同冻结参数同种子, 差异来自切换点与观察窗相位对齐; 两轮失衡度均完全恢复(0.0118/0.0104 vs prover 基线 0.0103)。

## 归因(联合分析结论, 详见 analysis/REPORT_pb.md §7)

1. 静态收益不足的根源是校准不迁移: prover-frozen 布局对 code 计数仅降 1.6% 失衡, 对 book 仅降 5.5% — DF/MT/ES 的 stage2 收益从校准时刻就不存在; 域适配布局天花板极高(hr→0.0001, r_raw→1.00; cdx0 replay 每 forward +16~27.5%)。
2. P0 的 stage2 短板在开销端不在失衡端: 失衡两轮全恢复, 吞吐跟随 swap 扰动量(27次→最低, 6次→最高); pc 战役同构(失衡降75%截断, 吞吐 −12% vs I)。
3. O=1 系统性压制收益端; ED 批量周期重平衡两轮 book 块最稳最高(4.59/4.52), 批量适应扰动成本低于连续滴灌。

## 机制对照

- P0: pb_r1 85窗 边界w27 z峰45.3 swaps27 grow_blocked×40; pb_r2 56窗 边界w30 z峰17.1 swaps6 grow_blocked×9; 双轮 detection/recording/planning/migration/state_transition 全 true, state=TRACK。
- ED: 两轮各 169 重平衡行 ≈ 21 事件 × 8 DP rank; 静态四臂(I/ES/MT/DF×2轮)各仅 1 行 server_args dump, 周期重平衡=0。
- 离线失衡网格(同一份计数×各布局, headroom 与 P0 公式同口径): analysis/imbalance_offline.{py,json,md}, 三重验证(cdx0 归档 r_raw 逐位一致 / DF 重算自洽 hr=0.00004 / prover 切片 cosine 0.9892)。

## 目录

- PROTOCOL_pb.md 预注册协议(0-8节) / LEDGER_pb.tsv append-only 治理日志(含 PC_COUNT_CORRECTION、窥视行、QUEUE_COMPLETE、CRON_PATROL_DELETED)
- analysis/: REPORT_pb.md(权威报告) + descriptive_pb.{py,json,md}(12+14 boot 表/P0轨迹/ED/静态扫描) + imbalance_offline.{py,json,md}(失衡网格, 可重跑)
- scripts/(4件 sha 锁定) / code_lock/ / placements/(3份静态布局, sha 登记) / B_data/registry.json(流谱系)
- boots_curated/: 12 boots 小件(bench json, MECH_CHECK, bench.log, boot_meta, sp_run_end, v2p1_rank0); server.log/perreq/流文件大件在 /data 全档
