# REPORT_pb — prover_book_switch_20261027（12/12 完成，n=2 描述性，NON-INFERENTIAL）

状态：QUEUE_PB_COMPLETE 2026-10-27T14:12:33Z；零 ABORT、零 rc≠0、INFRA 预算未动。
授权：作者 2026-10-27 消息（stage2 换 book、只跑两次、跑完作者+agent 联合分析）；联合分析已完成（在线汇报+离线网格），本报告为约定收尾件（PROTOCOL_pb §8）。

## 1. 设计摘要（细节见 PROTOCOL_pb.md，冻结件 sha 均登记）

- 流：pb_prover_book_switch.jsonl，n=3048，sha256 2da34672…；block1 prover[0:2048]（in≈2048tok，与 pc 战役 block1 字节同源）→ block2 book[0:1000]（BookCorpus in≈4438tok，发布集全量 1000 行；作者要求 1024，发布集只有 1000，已披露未静默调整）；全行 max_tokens=1、temperature=0；conc=32。
- 六臂与 pc 战役完全一致（launcher 字节同 sha cfbdaee7…）：I / ES / ED / MT / DF / P0（冻结 AERO-base rev1，RG=0，v2.1 knobs，反调参条款生效）。
- 轮换：r1 ES ED MT DF P0 I；r2 ED MT DF P0 I ES。守卫链全过（流 sha、代码锁×4、launcher/bench sha、placement 锁、boot_meta、ready、gpu-clear、mech_check）。

## 2. 主结果（req/s；stage 切分 idx2048；book early[2048:2560) / late[2560:3048)）

| boot | total | prover | book | b_early | b_late | Δbook vs I(同轮) |
|---|---|---|---|---|---|---|
| I_r1 | 7.00 | 9.70 | 4.37 | 4.24 | 4.28 | — |
| ES_r1 | 7.30 | 10.56 | 4.38 | 4.36 | 4.18 | +0.4% |
| ED_r1 | 7.20 | 9.90 | 4.59 | 4.58 | 4.38 | +5.1% |
| MT_r1 | 7.50 | 11.12 | 4.39 | 4.31 | 4.26 | +0.5% |
| DF_r1 | 7.60 | 11.27 | 4.50 | 4.46 | 4.31 | +3.0% |
| P0_r1 | 7.30 | 10.97 | 4.29 | 4.41 | 3.97 | −1.8% |
| I_r2 | 6.80 | 9.13 | 4.44 | 4.43 | 4.23 | — |
| ES_r2 | 7.20 | 10.42 | 4.37 | 4.30 | 4.22 | −1.6% |
| ED_r2 | 7.10 | 9.58 | 4.52 | 4.53 | 4.30 | +1.8% |
| MT_r2 | 7.50 | 11.19 | 4.47 | 4.46 | 4.26 | +0.6% |
| DF_r2 | 7.30 | 11.19 | 4.23 | 3.98 | 4.30 | −4.8% |
| P0_r2 | 7.60 | 10.61 | 4.72 | 4.68 | 4.55 | +6.3% |

两轮均值：prover 块 DF 11.23 ≈ MT 11.16 > P0 10.79 > ES 10.49 > ED 9.74 > I 9.42（校准收益兑现，DF/MT vs I +19%）；book 块 ED 4.56 > P0 4.50 > MT 4.43 > I 4.41 > ES 4.38 > DF 4.36（全臂挤在 4.23-4.72，极差约 11%；prover 块 19% 差距在 book 块消失）。total 由 prover 块主导（请求数 2048 vs 1000 且 prover req/s 高一倍余）。

## 3. P0 边界与 swap 动态（失衡度确实降下来了；吞吐跟着 swap 扰动量走）

度量定义（源码核验）：h_rate = headroom/新分配数，物理槽空间失衡度；z = 窗口分布对 EMA 参考（β=0.5）的漂移触发信号，非失衡度；h_max/h_min 字段为哨兵垃圾值未使用。

| boot | 窗口数 | 边界窗 | h_rate pre→spike→plateau | z spike→plateau | swaps | grow_blocked |
|---|---|---|---|---|---|---|
| pb_P0_r1 | 85 | w27 | 0.0106→0.0490→0.0118 | 45.3→0.76 | 27 | ×40 |
| pb_P0_r2 | 56 | w30 | 0.0103→0.0904→0.0104 | 17.1→0.80 | 6 | ×9 |
| pc_P0_r1 | 39 | w30 | 0.0103→0.0753→0.0175 | 15.5→3.59 | 11 | ×11 |
| pc_P0_r2 | 39 | w30 | 0.0103→0.0742→0.0175 | 12.6→3.96 | 11 | ×11 |

- pb 两轮失衡度均完全恢复到 prover 基线（0.0118 / 0.0104 vs 基线 0.0103）；pc 平台 0.0175 为降 75% 后被 run 尾截断（边界后仅 10 个观察窗）。
- 关键分裂：同冻结参数同种子，r1 滴灌 27 次 swap ↔ book 4.29 全臂最低 + b_late 3.97（early→late −10%）；r2 仅 6 次 swap ↔ book 4.72 全臂最高 + b_late 4.55 无跌幅。失衡恢复程度两轮几乎相同，吞吐差异跟随 swap 扰动量，与窗口相位对齐（切换点落在观察窗何处）相关，扰动量本身有大轮间方差。

## 4. 离线失衡网格（同一份计数 × 各布局；headroom_rate 与 P0 公式同口径可比）

计数来源：X 战役 record_boot2 e3 录制 .pt（logical_count 1000×94×128，steps 0..511，每步 8192 行；块边界实证 step 132/260/388，+4 滞后）；code = steps[392:512]（与 pc stage2 同一 retok L512 五源拼接群体；[404:512] 敏感性切片结论不变）；book 发布 traces（prefill 过滤 layer_hists，book_4438tok_O10 同源）。布局重放：deepseek.py flat rebalance（importlib 文件路径加载，torch-only 自包含；Qwen3 num_groups=None→auto→deepseek flat 已从源码确认）。

| 布局 | code 计数 hr | code r_raw | book 计数 hr |
|---|---|---|---|
| Identity | 0.0647 | 1.518 | 0.0911 |
| prover-frozen（DF 重算） | 0.0637（−1.6%） | 1.510 | 0.0861（−5.5%） |
| MT pinned_fair | 0.0666 | 1.533 | 0.0907 |
| e3-fair ILP（MoETuner 风格） | 0.0530 | 1.424 | 0.0823 |
| 域适配 greedy（code/book 各自计数） | 0.0001 | 1.0008 | 0.00002 |
| P0 切换前（live） | ≈0.074 | — | ≈0.049-0.090 |
| P0 切换后（live） | 0.0175（截断） | — | 0.0104-0.0118 |

验证三件套：cdx0 归档 r_raw 与重算逐位一致；DF 重算在其自身校准计数上 hr=0.00004（复现忠实）；prover 切片 vs datafore cosine 均值 0.9892。
cdx0 精确层证据（2026-10-16 归档，层 24/71）：code 流量 base-of-code r_raw 1.34/1.19 优于 base-of-prover 1.71/1.74（prover base 比 identity 1.55/1.49 还差）；replay 口径（探索性计时，非在线算法）：code 走 code-base 比 prover-base 每 forward 快 16%（L24 2534.81 vs 2945.35us）/27.5%（L71 2372.67 vs 3024.30us）；book 走 prover-base 罚 +24%（L24）。

## 5. pc 战役对照（stage2 域效应隔离；pc 计数更正后 14 boots）

pc（prover→code，同臂同 stage1）code 块均值：I 33.76、DF 33.04、MT 33.04、ES 32.15、ED 31.60、P0 29.66。P0 vs I：pc −12%，pb +2%（均值 4.50 vs 4.41）。同臂配对下 stage2 域差异：code 块上 P0 失衡只降 75%（截断）且付 11 次 swap + 记录开销；book 块上失衡全恢复且 r2 低扰动形态最优。

## 6. 机制证据

- ED：pb 两轮各 169 重平衡日志行 ≈ 21 事件 × 8 DP rank（r1 12:32:52-12:40:56；r2 13:17:39-13:26:25）；pc 两轮各 121 行 ≈ 15×8。
- 静态四臂（I/ES/MT/DF ×2 轮）：各仅 1 条 rebalance 行 = server_args dump，周期重平衡 = 0 ✓。
- P0：双轮 detection/recording/planning/migration/state_transition 全 true，shw_constant=false，sp_run_end state=TRACK；r1 events {init 1, hold 30, confirmed_grow 12, grow_blocked_headroom 40, midband_hold 2}；r2 {init 1, hold 31, confirmed_grow 15, grow_blocked_headroom 9}。

## 7. 归因结论（作者核心问题：开销还是静态收益不够——两个命题各对一半）

1. 静态收益不足为真，根源是校准不迁移而非 code/book 不值得适配：prover-frozen 布局对 code 计数仅降 1.6% 失衡、对 book 仅降 5.5%——DF/MT/ES 的 stage2 静态收益从校准时刻就不存在，故 stage2 ≈ identity（pb ±3%、pc ±2%）。域适配静态布局可把失衡降到约 0（hr 0.0001/0.00002，r_raw→1.00），cdx0 replay 显示每 forward 收益 16-27.5%：天花板极高，frozen 校准够不着。
2. P0 的 stage2 短板在开销端而非失衡端：两轮 book 失衡均完全恢复，但吞吐跟随 swap 扰动量（27 次→最低+late −10%；6 次→最高）；pc 更极端（失衡降 75%、吞吐仍 −12% vs I：11 次 swap + 记录开销 + O=1 收益端钝感）。
3. O=1 系统性压制收益端：headroom 降 4-6 倍换不来 O=1 吞吐；论文 freq6 +8.5% 分水岭依赖 O=10 的 prefill 敏感性（cdx0 replay 证实失衡→每 forward 时间敏感）。
4. ED（批量周期重平衡）两轮 book 块最稳最高（4.59/4.52），提示批量适应的扰动成本低于连续滴灌 swap。
5. 方向信号（供讨论非结论）：P0 域切换后若将 swap 批量化/限次（r2 的 6 次形态），book 块可从 −1.8% 翻至 +6.3%；扰动量是可控变量，失衡恢复能力已验证。

## 8. 更正与披露

- PC_COUNT_CORRECTION：LEDGER_pc 实有 14 条 DONE 行（r1×6+r2×6+MT_r3+DF_r3，产物齐全），此前停机行文本写 11/24 与账不符；已按 append-only 补更正行，pc 对比按 14 boots 计；pc_P0_r3 为停机时零数据 boot（无 bench json/perreq）排除。
- book 行数为发布集全量 1000（作者要求 1024，已披露）。
- h_rate 全程不解释为还可加速多少；无 payback solver；论文分析模型（cdx0 replay）不表述为在线算法；P0 knobs 冻结无调参。
- ES 为 deepep normal + R16（144 槽）与 128 槽网格不可通约，网格中未列 ES 布局（已在 imbalance_offline.md 注明）。

## 9. 局限

- n=2，df=1，无推断统计（作者选定 2 reps 求速度）；所有 delta 为描述性。
- book 块仅 1000 请求（约 7-8 分钟），P0 相位对齐方差大（r1/r2 分裂即其表现），单流单种子。
- O=1 压缩臂间差距（xd 战役 limitation 同源）；静态臂校准偏向 prover 域（oracle 校准偏向）。
- 离线网格基于聚合计数（record_boot2 步切片 / 发布 traces prefill 过滤），非逐请求在线重放。

## 10. 产物索引

- 本目录：descriptive_pb.{py,json,md}（12+14 boot 表、P0 轨迹、ED/静态扫描）；imbalance_offline.{py,json,md}（网格+验证，可重跑）；REPORT_pb.md（本件）。
- 战役目录：PROTOCOL_pb.md、LEDGER_pb.tsv（append-only 全程）、B_data/registry.json（流谱系+sha）、scripts/（4 件 sha 锁定）、boots/ 12 boot 现场（bench json、perreq jsonl、server.log、sp_run_end、v2p1、boot_meta、MECH_CHECK）。
- 全档：/data/minghua/sjq/PAPER_EXPERIMENTS/prover_book_switch_20261027/（ARCHIVE_SHA256SUMS.txt）；GitHub 精选档：PAPER_EXPERIMENTS/prover_book_switch_20261027/。
