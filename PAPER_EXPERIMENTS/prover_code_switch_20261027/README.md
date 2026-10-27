# prover_code_switch_20261027 归档 README (partial)

跨域切换 stage2=code 战役, 作者于 14/24 DONE 时中止(2026-10-27T12:15Z)并 pivot 到 prover_book_switch_20261027。本目录为 GitHub 精选档; 全档在 /data/minghua/sjq/PAPER_EXPERIMENTS/prover_code_switch_20261027/; 工作现场 /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/prover_code_switch_20261027/。

定位: partial 战役记录(PROTOCOL_pc §8 约定件); 描述性对比并入 pb 战役 analysis/descriptive_pb.md §3-§4 与 REPORT_pb.md §5。**臂数不均衡(I/ES/ED/P0 各2, MT/DF 各3), 无任何推断统计**。

## 结论速览 (14 有效 boots)

- 流: prover[0:2048]@≈2048tok → code[2048:4096](retok L512 五源拼接, 与 e3 code 块同群体), n=4096, 全行 O=1 conc=32。
- code 块均值(req/s): I 33.76 > DF 33.04 ≈ MT 33.04 > ES 32.15 > ED 31.60 > P0 29.66 (P0 vs I −12%, 全臂最低)。
- 核心发现: prover-frozen 静态臂在 code 块无失配罚(O=1 下失配成本不可见); P0 边界 w30 检出(z 15.5/12.6), 失衡 0.0753→0.0175(降75%, run尾截断), 11 次 swap + 记录开销, 收益端不兑现。
- P0_r3 = 停机时刻零数据 boot(无 bench json/perreq), 排除; boots_curated 保留其残档供审计。

## 更正披露

PC_COUNT_CORRECTION(append-only): LEDGER_pc 实有 14 条 DONE 行(r1×6+r2×6+MT_r3+DF_r3, 产物齐全), 此前 QUEUE_STOPPED 行文本写 11/24 与账不符; 账本 DONE 行为地面真值。

## 目录

- PROTOCOL_pc.md / LEDGER_pc.tsv(CAMPAIGN_OPEN→QUEUE_STOPPED→更正行) / analysis/REPORT_pc.md(partial 权威报告)
- scripts/(4件) / code_lock/ / placements/ / B_data/registry.json
- boots_curated/: 15 boot 目录(14 有效 + P0_r3 零数据残档)小件; 大件在 /data 全档
