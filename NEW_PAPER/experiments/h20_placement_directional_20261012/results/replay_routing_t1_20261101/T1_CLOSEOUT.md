# T1 CLOSEOUT — replay_routing_t1_20261101 (closed 2026-11-01T16:13:09Z)

## 0. 执行账目
- 7/7 boots (1 smoke + 6 formal) mech all_ok=True, bench errors=0, 零 rc=2, 零停线. E3 流 8192 reqs x 2 reps/条件, conc 256, timeout 900.
- 守卫全过: capture sha fa049655…, E3 流 sha d55fe2f7…, 部署 rebalancer==G2 冻结字节 c250026b…, launcher/runner/bench_client sha 锁.

## 1. R口径 (离线 capture 驱动固定路由重放)
- capture logical_count 聚合 ≡ P_e3.npz[P] 逐元素相等 (断言通过). 归档 G2 计划重放: fin_max_raw = ORIG 1.531831 / GAP32 1.314006 / GAP128 1.178292 / GAP300 1.096219 / MD300 1.161569 / BI300 1.095720.
- 生产阈值 1.0962402945: GAP300 (1.096219) 与 BI300 (1.095720) 达到该阈值; 1.02 行仅为该阈值下的算法诊断口径.
- 布局不变量事实: 全支撑计数下 padding_tokens=761120, occupied_rank_slots=752, rank_pair_edges_x8=6016, expert_edges=12032 在全部条件间恒定 — 置换式重规划不改变连接/padding 度量, 判别量是 rank 负载比.

## 2. M口径 (在线迁移 = controller 运行时下发, 非请求重跑/非启动加载)
- 全部 PB 臂 issued 行=窗口数x8 且跨 rank 去重一致 (G3 口径复核通过): GAP32 62窗/1981ops, GAP128 36/4059, GAP300 28/4327, MD300 56/2892, BI300 34/5196; ORIG issued=0 (无 PB, 符合定义).
- 迁移墙钟 (TIMING begin() 相位, rank0 计 / 8-rank 求和): GAP32 14.8s/122.4s, GAP128 2.1s/20.8s, GAP300 7.5s/71.3s, MD300 6.4s/62.4s, BI300 2.8s/26.4s. 无 finish 相位 TIMING 行, 口径如实标注为 begin 相位.
- v2p1 rank0 窗口时间线 64 窗全部 committed 一致; sp_swaps_started 与 issued 窗口数吻合 (GAP32 61/62, GAP300 27/28 为 sp 记账窗口, 差 1 = init 窗).
- 语义事实: --pb-oeplb-max-total-ops 为每窗口预算 (观测 GAP32 均值≈32 ops/窗), issued_ops_dedup 为跨窗累计; 冻结代码原语义, 未改动.
- ready_s (工程开销单列, 不入比较): ORIG 76, GAP32 60, GAP128 64, GAP300 56, MD300 76, BI300 60.

## 3. X口径 (GPU 真实执行短重复 = execution, 不称 replay)
- rep 均值 tps: ORIG 43.1 / GAP32 40.2 / GAP128 43.6 / GAP300 41.9 / MD300 38.2 / BI300 43.2.
- 诚实结论: 在 E3 prefill 重流 (avg_output_tokens=1) 上, 失衡下降未转化为吞吐增益; GAP32 (-6.8%) 与 MD300 (-11.5%) 呈现可测迁移开销; GAP128/GAP300/BI300 相对 ORIG 为中性 (±3% 内, 2 reps 样本不足做显著性声明).
- 不声明任何 GPU 验证收益; R口径失衡下降与 X口径吞吐在本流上解耦, 为如实测量结果.

## 4. 收益模型 (条件式纸面模型, 非在线算法, 无 payback solver)
- moved_mb_est (37.74 MB/op x plan_len, estimate-only): GAP32 1207.7 / GAP128 4830.7 / GAP300 9472.7 / MD300 9019.9 / BI300 9284.0 MB.
- h_rate/h_w 为 SP 内部统计量, 不作 headroom 解释.

## 5. 产物
- analysis/OFFLINE_t1.json (在线字段已回填), analysis/EXPORT_t1_full_table.tsv, analysis/EXPORT_t1_offline_table.tsv, analysis/layouts/layout_*.json (6), boots/* (7, 本地全保留), LEDGER_t1.tsv.
