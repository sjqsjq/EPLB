# PROTOCOL_t1 - 真实路由重放条件比较 (replay_routing_t1_20261101)
FROZEN before any GPU run. Governing spec = author messages 2026-10-31 (三项任务) and 2026-11-01 (D1/D3/D4 decisions, T1修正, 7.5h硬上限, 不看成绩改规则). Append-only discipline; no mid-run rule changes.

## 0. 授权与决策记录
- D1 = 路径A: 全臂 deepep normal + CUDA graph off (统一公共执行路径).
- D4 = 使用现有 prefill 重 capture, 不新增采集.
- T1修正 (author 2026-11-01): 确认 capture 能驱动固定路由重放; 不能把重新跑请求或启动加载当作重放与在线迁移成本; 优先复用模型加载, 通过短重复完成布局重放.
- 预算: T1 GPU <= 1.3h; 本轮 T1+T2 <= 5.6h < 7.5h 硬上限; T3 后置.

## 1. 口径分离 (三口径, 互不冒充)
- R口径 (replay, 离线): capture 计数驱动的确定性重放. 复用 G2 冻结 planner (rebalancer sha c250026b..., deployed 与 G2 冻结字节一致, 本轮实证) 的 5 个已归档 prod1096 计划 (C2 快照 == 本 capture 聚合, 逐元素断言). 产物: 布局, 连接计数, padding, 条件收益模型. 零请求重跑.
- X口径 (execution, GPU 实测): 每条件固定短重复执行段 = E3 冻结流 x 2 reps (同一流, 同一 conc, 重复两次). 请求重跑只计为执行实测, 永不标注为 replay.
- M口径 (migration, GPU 实测): 在线迁移 = controller 运行时窗口下发的 swap ops 与 v2p1 时间线 (swaps_started, issued per window). 启动/模型加载 = 工程开销单列 (ready_s), 永不计入 M/X/R 任何比较量.
- capture 结构事实 (本轮 torch.load 实证): dict{rank:0, logical_count (1000,94,128) int32, average_utilization_rate_over_window:None}, 总和 3154479072, 与 C2 P_e3.npz 逐元素同源. capture 为逐 forward 专家计数, 不含 token 级指派 => token 粒度路由重放不被现有 capture 支持, 如实声明, 不以请求重跑冒充.

## 2. capture 与采集工况
- 文件: cross_domain_e2e_20261027/prep/record_boot2/expert_distribution_recorder_1793078907.6631618.pt, sha256 fa04965525ecb44ccd89e67be525e815a5ae5474ccbb44c6c7c9f9c13dc6df4a (code_lock 锁定).
- 采集时 ServerArgs (server.log 实证): init_expert_location=trivial, deepep_mode=normal => 原布局 = trivial, 采集已在路径A上.
- 工况: E3 流 8192 reqs, conc 256, 190.86s, avg_output_tokens=1 (prefill 重边界, author D4 接受). record_boot = 失败尝试 (core dump), 排除.

## 3. 六条件 (唯一 delta = 规划规则/预算/BI; 其余全同)
| 条件 | 规划 env/flag delta | 说明 |
|---|---|---|
| ORIG | 无 PB (BASE only) | 原布局 trivial, 零规划零迁移 |
| GAP32 | PB(max-total-ops 32), OEPLB_PAIR_RULE=gap | |
| GAP128 | PB(max-total-ops 128), OEPLB_PAIR_RULE=gap | |
| GAP300 | PB(max-total-ops 300), OEPLB_PAIR_RULE=gap | |
| MD300 | PB(max-total-ops 300), OEPLB_PAIR_RULE=maxdelta | |
| BI300 | PB(max-total-ops 300), OEPLB_PAIR_RULE=gap, OEPLB_PLANNER_BEST_IMPROVEMENT=1 | config.py:97 实证 env 即生效 |
- PB flag 块 = G3 launch_arm_g3.sh 字节同款, 仅 max-total-ops 按条件取值: --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02 --pb-oeplb-min-prefill-tokens 256 --pb-oeplb-sync-window 16 --pb-oeplb-max-total-swap-layers 94 --pb-oeplb-max-swaps-per-layer 64 --pb-oeplb-min-swap-ops 8 --pb-oeplb-max-total-ops N --pb-oeplb-decay-factor 0.0 --pb-oeplb-window-floor 8 --pb-oeplb-window-shift-confirm 2.
- 阈值口径: threshold-ratio 1.02 + auto dead_zone => 有效阈值 1.0962402945 (生产口径, G3 同 lineage; 与 G2 离线 prod1096 单元一致).
- env: pb_env_common (V21 冻结常数, G3 字节同款) + OEPLB_SP_MODE=fixed SP_S=1 SP_W=16 SP_SEED=20261018 SP_REV=0 + OEPLB_RECORD_ALL_PHASE=1.
- BASE = G3 字节同款, 唯一 delta: --deepep-mode auto -> normal (路径A). graph off (--disable-cuda-graph) 保持.

## 4. 离线 R口径指标 (公式冻结)
- final_counts = apply_plan(capture聚合, G2归档plan) (driver_g2 冻结函数).
- ratio raw/pad: 同 G2 ratio_stats.
- padding = Sum_l Sum_r (w_e - n_er), w_e = 128*ceil(n_er/128) (G2 T1 protocol 同款, rank_loads_np pad=True).
- 连接计数: 主指标 rank_pair_edges = 8 * Sum_l |{r: rank r 承载 count>0 专家}|; 次指标 occupied_rank_slots = Sum_l |{...}|; 参考 expert_edges = Sum_l #{e: count>0} (布局不变量).
- 条件收益模型 = 纸面条件式: 收益(失衡下降) 以 迁移成本 为条件变量; 迁移成本 = plan ops x 37.74 MB (冻结估计常数, estimate-only) + M口径实测墙钟. 声明: 非在线算法, 无 payback solver, h_rate/h_w 不作 headroom 解释.

## 5. GPU 矩阵 (冻结顺序)
- SMOKE 1 boot: GAP300 臂 + prover_512_smoke.jsonl (零正式数据), 验证 PB x deepep normal 组合 (历史仅在 auto 见证; ES/ED 证明 normal 权重迁移可用). 失败 => 报具体问题, STOP, 不静默改道.
- 正式 6 boots 顺序: ORIG, GAP32, GAP128, GAP300, MD300, BI300. 每 boot: ready 后 E3 流 (8192, sha d55fe2f7..., conc 256, timeout 900) x 2 reps, bench 标签 LABEL_rep1 / LABEL_rep2. ready_s 单列工程开销.
- 迁移会计: server.log issued 行 (跨 rank 8 副本, 去重口径 = 每窗口计一次, G3 已验证 16/16) + v2p1_rank0.jsonl 窗口时间线 + swaps_started.

## 6. 守卫与 STOP
- boot 级守卫: E3 流 sha; pb 代码 7 文件 sha (async_swapper/controller/sp_state/window_v2/rebalancer/config/window_v2p1, 值见 code_lock/shas.env); launcher sha; bench_client_v2 sha 06668243....
- rc=2 => GUARD_ABORT 停队; rc=1 => 记录继续 (INFRA 零数据重启 <= 2); 无自动重试; 失败报具体问题, 不静默缩臂, 不放宽范围.

## 7. 产物与纪律
- boots/LABEL/{server.log, bench*, boot_meta*, MECH_CHECK.json, v2p1_rank0.jsonl}; analysis/{OFFLINE_t1.json, EXPORT_t1_offline_table.tsv, layouts/}; LEDGER_t1.tsv append-only; 命名 t1_*; 全部关键路径程序化拼接, 不手抄.
