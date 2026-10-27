# E3 流历史交叉核对备忘 (2026-10-27, 作者问询"数据效果看起来有点不对"后 撰写)

## 作者决定
- 选项 (a): 按冻结协议跑完 24 boots, 队列不动 ("先跑完再说吧")。
- 本备忘供最终报告 limitations / 讨论节引用。

## 数据集事实 (冻结, 每 boot 守卫链复核 sha)
- final_plan_20261019/B_data/streams/e3_multidomain_switch.jsonl
- sha256 d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879
- n=8192 = 4 域 x 2048 相位切换: sharegpt [0,2048) -> prover [2048,4096) -> book [4096,6144) -> code [6144,8192)
- 输入恰 512 tok (retok p5/p50/p95 = 512/512/512), max_tokens=1 (O1), temp=0, conc 256
- O1 语义: tps 列 = 输出 tok/s = req/s (8192/190.86 = 42.9 精确吻合), prefill 主导无 decode 段

## 历史同流对照 (final_plan_20261019 战役, 6 策略 x 8 block, 同一 e3 流)
- I: 41.9-43.5 | P: 39.7-45.9 (2 退役) | E: 40.1-42.1 | A2: 36.9-44.2 | A4: 41.5-43.1 | F: 41.2-45.5
- 本次前 15 boots 全部落入上述历史区间 (I 42.5, P0 40.8/42.7, ES 41.5, ED 38.5-39.7, MT 41.8-44.2, DF 42.9-44.2)
- 旧 REPORT_final.md 原话: "E1/E3 上四臂互不显著（放置与恒等在 prefill 主导流上差异小）"
- 旧报告: "PB 族在 E2（prover 小 prefill）上一致优于 I 与 E（放置真收益）" — 判别性流是单域小 prefill, 非 E3

## 与 recovery_admission_e2e (76-101 tps) 的量级差解释
- 该战役流 = prover_256tok_out1_x4.jsonl (256 tok 输入, prefill 工作量减半 -> req/s 约翻倍)
- 且 prover 单域倾斜 layer0 imbalance 4.093 vs E3 混合画像 2.379

## 报告 limitations 必带四点
1. O1 prefill-only: tps=req/s, 布局失衡无 decode 复利放大, 臂间差距结构性压缩 (~±5-7%)
2. 混合域摊平倾斜 (2.379 vs 4.093), 每域相位仅 ~48s, 在线适应回报小
3. 静态臂画像录自评测流本身 (oracle 校准, 作者设计第 2 点, 偏向静态臂, 如实披露)
4. ES/ED 被迫 deepep normal (auto 下官方路径 NotImplementedError), 系统性拖累 EPLB 臂; ED 垫底含 16 次/boot 迁移成本; ED 略低于历史 E 带时归因此混杂, 不作机制结论
