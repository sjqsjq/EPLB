# REPORT_pc — prover_code_switch_20261027（作者中止于 14/24 DONE，partial 战役记录）

状态：2026-10-27T12:15Z 作者指令停止（QUEUE_STOPPED 行）；P0_r3 为停机时刻零数据 boot（START 12:09:37Z，无 bench json/perreq）已排除；有效 boots = 14（LEDGER_pc DONE 行计数，见 PC_COUNT_CORRECTION 更正行：此前停机行文本写 11/24 与账不符，账本 14 条 DONE 为地面真值，产物齐全：r1×6 + r2×6 + MT_r3 + DF_r3）。
后续：作者 pivoted 到 prover_book_switch_20261027（stage2 换 book、2 reps、联合分析）；pc 的描述性对比并入 pb 战役 analysis/descriptive_pb.md §3-§4 与 REPORT_pb.md §5（同臂同 stage1，隔离 stage2 域效应）。本件为 PROTOCOL_pc §8 约定的 partial 记录件。

## 1. 设计（冻结，见 PROTOCOL_pc.md）

流 pc_prover_code_switch.jsonl n=4096：block1 prover[0:2048]（in≈2048tok）→ block2 code[2048:4096]（retok L512 五源拼接，与 e3 code 块同群体）；全行 max_tokens=1（作者 mid-turn 更正，非论文 O=10）、temperature=0、conc=32。六臂 I/ES/ED/MT/DF/P0 与 pb 战役 launcher 字节同（sha cfbdaee7…）。计划 4 轮 24 boots，实际执行至 r3 中途停止。

## 2. 描述性结果（req/s；code 块均值按已完成 boots）

| boot | total | prover | code | c_early | c_late |
|---|---|---|---|---|---|
| I_r1 | 15.30 | 9.78 | 33.27 | 29.15 | 34.24 |
| I_r2 | 15.50 | 9.88 | 34.25 | 29.82 | 35.27 |
| ES_r1 | 15.80 | 10.28 | 32.51 | 27.82 | 33.69 |
| ES_r2 | 16.00 | 10.55 | 31.78 | 27.57 | 32.86 |
| ED_r1 | 14.60 | 9.36 | 31.92 | 28.76 | 32.50 |
| ED_r2 | 14.70 | 9.49 | 31.27 | 28.18 | 31.77 |
| MT_r1 | 16.80 | 11.15 | 32.27 | 28.29 | 33.19 |
| MT_r2 | 17.10 | 11.23 | 34.35 | 30.21 | 35.24 |
| MT_r3 | 16.90 | 11.23 | 32.51 | 29.28 | 33.07 |
| DF_r1 | 16.90 | 11.19 | 33.21 | 28.71 | 34.33 |
| DF_r2 | 17.10 | 11.25 | 34.05 | 29.60 | 35.13 |
| DF_r3 | 16.50 | 10.93 | 31.86 | 27.77 | 33.04 |
| P0_r1 | 16.00 | 10.91 | 28.76 | 28.21 | 28.51 |
| P0_r2 | 16.40 | 11.07 | 30.56 | 25.26 | 32.19 |

code 块均值：I 33.76、DF 33.04、MT 33.04、ES 32.15、ED 31.60、P0 29.66（P0 vs I −12%，全臂最低）。prover 块与 pb 战役一致形态（DF/MT ≈ 11.2 > P0 ≈ 11.0 > ES ≈ 10.4 > I ≈ 9.8 > ED ≈ 9.4）。臂数不均衡（I/ES/ED/P0 各 2，MT/DF 各 3）——轮转中止所致，描述性口径下按臂取已完成 boots 均值并全披露。

## 3. 核心发现（触发作者 pivot 与后续联合分析）

- prover-frozen 静态臂在 code 块无失配罚（DF 33.04 ≈ I 33.76，差 −2%）：O=1 下失配成本不可见。
- P0 边界 w30 检出（z 15.5/12.6），失衡 h_rate 0.0753→0.0175（降 75%，run 尾截断，边界后仅 10 窗），11 次 swap，grow_blocked×11——付了迁移与记录开销，收益端（O=1 吞吐）不兑现 → code 块 −12% vs I。
- 离线网格（pb 战役 analysis/imbalance_offline.md）：prover-frozen 布局对 code 计数仅降 1.6% 失衡（0.0647→0.0637），code 适配 greedy 可到 0.0001——静态收益不足的根源是校准不迁移，非 code 域不值得适配。

## 4. 披露与局限

- partial 战役：非预注册完整执行，r3/r4 未完成；不做任何推断统计。
- PC_COUNT_CORRECTION（append-only 更正行）为本件计数依据。
- h_rate 不解释为还可加速多少；无 payback solver；P0 knobs 冻结。
- 详细账本：LEDGER_pc.tsv（CAMPAIGN_OPEN→QUEUE_STOPPED→更正行全程）。

## 5. 产物

PROTOCOL_pc.md、LEDGER_pc.tsv、B_data/（流+registry）、scripts/（sha 锁定）、boots/（14 有效 boot 现场 + P0_r3 零数据残档）、analysis/REPORT_pc.md（本件）；描述性对比表并入 pb 战役 analysis/descriptive_pb.{json,md}。
