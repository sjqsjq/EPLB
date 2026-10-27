# cross_domain_e2e_20261027 -- 跨域切换多 baseline 端到端 prefill-heavy 补充探针 (2026-10-27 定位修正: 非论文 §5.3.1 主比较) + 时间线案例 (§5.5, 引用须带定位注记)

生成: 2026-10-27T09:21:38Z (report_gen_xd.py, 数字程序化取自 analysis/summary.json + timeline_5_5.json)
协议: PROTOCOL.md (pre-reg sha febf6616...) + AMENDMENT-1 (post sha a0f4ad5e...); 六臂同期同窗测量, 未与任何历史战役数值拼接。
流: e3_multidomain_switch.jsonl sha d55fe2f7... (n=8192 = sharegpt/prover/book/code x 2048 相位切换, in512/out1 O1, conc 256)。

## 1. 执行摘要
- 24/24 正式 boots 全绿 (bench_rc=0, mech_rc=0, ok=8192, errors=0), 队列 06:40Z-08:49:02Z, 零 INFRA 重试。
- prep 3/3: rec_boot v1 READY_FAIL(recorder 不兼容, 已 INCIDENT 记录), v2 录制成功, ES_val 通过。
- INCIDENT4 (良性): 看门狗 grep 自匹配完成令牌提前触发 --final, 守卫正确拒绝 (rc=1), 无产物污染。

## 2. 主结果 (primary = full-service wall clock; co-report tps; n=4 轮)
| 臂 | wall mean±sd (s) | 逐轮 wall (s) | tps mean±sd | 逐轮 tps |
|---|---|---|---|---|
| I Identity | 190.60 ± 2.76 | 192.6, 192.4, 190.8, 186.6 | 42.98 ± 0.64 | 42.5, 42.6, 42.9, 43.9 |
| ES static EPLB (ES) | 192.13 ± 4.26 | 197.6, 192.0, 191.8, 187.2 | 42.67 ± 0.94 | 41.5, 42.7, 42.7, 43.8 |
| ED dynamic EPLB (ED) | 208.40 ± 4.24 | 206.2, 212.8, 203.6, 211.0 | 39.30 ± 0.79 | 39.7, 38.5, 40.2, 38.8 |
| MT MoETuner-style (MT) | 188.24 ± 5.16 | 185.3, 196.0, 185.8, 185.9 | 43.55 ± 1.17 | 44.2, 41.8, 44.1, 44.1 |
| DF DataFore-style (DF) | 188.20 ± 3.82 | 185.2, 191.0, 192.0, 184.6 | 43.55 ± 0.87 | 44.2, 42.9, 42.7, 44.4 |
| P0 AERO-base (P0) | 199.98 ± 6.40 | 200.9, 191.7, 207.2, 200.1 | 40.98 ± 1.31 | 40.8, 42.7, 39.5, 40.9 |

排序 (wall 快->慢): DF 188.20 ~ MT 188.24 < I 190.60 < ES 192.14 < P0 199.98 < ED 208.40。

## 3. 推断统计 (P0 为参照, 轮配对 t df=3 双侧, Holm x5)
| 对比 | mean diff (s, P0-arm) | 95% CI | t | p raw | p Holm | 结论 |
|---|---|---|---|---|---|---|
| P0_vs_I | +9.38 | [-2.58, +21.34] | 2.497 | 0.0879 | 0.3518 | n.s. |
| P0_vs_ES | +7.84 | [-4.18, +19.87] | 2.076 | 0.1295 | 0.3778 | n.s. |
| P0_vs_ED | -8.43 | [-24.95, +8.10] | -1.623 | 0.2031 | 0.3778 | n.s. |
| P0_vs_MT | +11.74 | [-6.01, +29.49] | 2.105 | 0.1259 | 0.3778 | n.s. |
| P0_vs_DF | +11.78 | [+0.01, +23.55] | 3.186 | 0.0499 | 0.2494 | n.s. |

**Holm 校正后六臂无显著差异。** 唯一 raw p<0.05 为 P0_vs_DF (+11.78s, p=0.0499, Holm 0.2494 n.s., 边界值如实报告)。
P0 慢于 I/MT/DF/ES (raw p 0.05-0.13), 快于 ED 8.43s (p=0.2031)。与历史同流结论一致 (final_plan_20261019: E1/E3 上四臂互不显著)。

## 4. §5.5 机制时间线 (域边界 idx 2048/4096/6144 = 窗口 8/16/24)
### P0 (AERO-base) 边界反应
- 4/4 boots: swap 发起窗 = 边界后第 1 窗 (9-10, 17-18, 25+), 反应延迟 ~1 窗 (~6s); P0_r1-r3 swaps=13, TRACK 常驻。
- P0_r4: 4 次 TRACK<->QUIET 状态迁移 (w14/15/21x2), w9 z=133.1 (边界1 检测尖峰), w14 z=59.2, w21 z=56.4。
- 相位吞吐 (req/s, 4轮均值): sharegpt 40.0 / prover 40.3 / book 39.3 / code 33.3 -- prover 段全臂最高 (适应收益), code 段全臂最低 (w25-32 持续迁移成本)。
### ED (dynamic EPLB) -- 事实更正
- **更正**: 旧口径 16 次/boot 为日志行数; 实为 **2 个独立事件 x 8 DP rank** (period 256 步 x 全程 512 步), 4/4 boots 一致。
- ev1 落在边界2 (idx4096) ±0.8s 内; ev2 落在流末 (边界3 后 +49~51s) -- 边界1/3 未被任何重平衡覆盖。
- 代价可见: prover 段 ttft p95 = 13.26s (全臂最高, 边界2 重平衡停顿), wall 最慢 208.40s。
### 静态臂 (ES/MT/DF) 与 I
- 4/4 boots runtime rebalance 行 = 0, 布局 t=0 冻结 (ES: init_by_eplb+R16; MT: init_by_mapping; DF: rebalance_experts greedy at init; 证据在 MECH_CHECK baseline_facts.injection_evidence)。
- 相位 ttft p95 平稳 (5.6-8.0s), 无边界停顿; MT sharegpt 段 ttft p95 5.61s 全臂最低。

## 5. 画像与求解成本 (单列, 作者第 2 点)
| 项 | 值 |
|---|---|
| 混合域校准录制 boot | 190.86s bench (ok=8192/0err, 42.9 tps) + 启动 ~15min |
| recorder .pt | 48,130,049 B, sha fa049655..., 94 层 x 128 专家, 每层 33,558,288 = 4,194,260 prompt tok x 8 topk (精确核对) |
| MT ILP 求解 (Gurobi 13.0.3 restricted, ILP1 G=8 K=16) | 94 层共 2822.5s; 每层 solve_s 30.0-30.2 (全部触顶 time_limit=30); mean/max imbalance = 1.0000 |
| ES/DF 求解 | 共用同一 logical_count json (sha ddc3943d..., 字节相同); 无离线求解, 布局构建在 boot init 内 (init_by_eplb / greedy rebalance_experts) |

## 6. Limitations (必带四点 + 更正)
1. **O1 prefill-only**: tps 列 = req/s (8192/wall), 无 decode 段, 布局失衡无逐 token 复利放大, 臂间差距结构性压缩 (~±5-7%); 历史同流旧报告原话: E1/E3 上四臂互不显著。
2. **混合域摊平倾斜**: 混合画像 layer0 imbalance 2.379 vs prover 单域 4.093; 每域相位仅 ~48s, 在线适应回报小。
3. **静态臂 oracle 校准**: 画像录自评测流本身 (作者设计第 2 点), 偏向静态臂, 如实披露。
4. **ES/ED deepep normal 混杂**: 官方 recorder/eplb 路径在 auto 下 NotImplementedError, ES/ED 被迫 normal (其余四臂 auto); ED 慢含 2 次迁移停顿 + normal 路径双重因素, 不与 auto 臂作机制归因; ES 若低于历史 E 带同因。
5. **ED 重平衡次数更正**: 16 行 = 2 事件 x 8 rank (见 §4), PROTOCOL §4 旧核对口径按行数计, 以本更正为准。

## 7. 历史交叉验证 (作者质询 数据效果看起来有点不对 的核对)
- 同一条 e3 流旧战役 (final_plan_20261019, 6 策略 x 8 block) tps 区间: I 41.9-43.5 / P 39.7-45.9 / E 40.1-42.1 / A2 36.9-44.2 / A4 41.5-43.1 / F 41.2-45.5; 本次 24 boots 全部落入对应历史带 (详见 notes/E3_historical_crosscheck.md sha 5a3662a9...)。
- 与 recovery_admission_e2e (76-101 tps) 量级差 = 该战役流为 prover_256tok_out1_x4 (输入减半 -> req/s 翻倍) 且单域倾斜 4.093。

## 8. 治理与产物
- LEDGER_xd.tsv append-only 全程记账; 守卫链 (stream/代码/launcher/placement 四重 sha) 每 boot 复核, 0 次 BOOT_ABORT。
- 分析: analyze_xd.py (sha 7f62f265...) --final 24/24; timeline_xd.py (sha 9e91ffe4...)。
- 产物: analysis/{summary.json, per_boot_raw.tsv, timeline_5_5.json, timeline_5_5.md, REPORT_xd.md}; boots/ 24 目录。


---

## Correction Addendum (2026-10-27 appended; 依据: LEDGER_xd.tsv INVESTIGATION VERDICT 行 + prover_code_switch_20261027 PROTOCOL §8 约定)

本战役定位修正为 **prefill-heavy 补充探针**, **不是**论文 §5.3.1 的 crossdomain_freq6 跨域主比较。三点与论文 spec 的偏差:
1. 流: 实跑 e3_multidomain_switch(4 大块 sharegpt/prover/book/code × 2048, in512/out1, conc256), 非 crossdomain_freq6(6seg book↔prover, book_4438tok_O10 + prover_2048tok_out1, O=10, conc=32, N=600);
2. workload: O=1 + conc256(tps=req/s 差距被结构性压缩, prefill 主导);
3. 静态臂校准源: mixed-full-stream oracle, 非 prover-frozen —— 该点移除了跨域失配失败模式本身。

原文全部数字保留, 作为该探针自身的测量依然有效(24/24 boots ok、机制记录、ED 事件事实更正、§5.5 时间线); 但不得作为 §5.3.1 主比较结果引用。§5.3.1 主比较的后续承接战役: prover_code_switch_20261027(partial, stage2=code)与 prover_book_switch_20261027(stage2=book, freq6 book↔prover 同族域对, 12/12, n=2 描述性)——见 EXPERIMENT_MAP.md 对应行。§5.5 域边界时间线案例引用时须带本定位注记。
