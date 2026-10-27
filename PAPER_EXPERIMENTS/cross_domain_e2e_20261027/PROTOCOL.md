# PROTOCOL - cross_domain_e2e_20261027 (跨域切换多 baseline 端到端)

状态: 预注册 (作者 2026-10-27 确认后生效)。本文件先于任何 GPU boot 落盘。
修改规则: 正文不改动; 事后补充只能以 AMENDMENT 段落追加并在 LEDGER 登记。

## 0. 授权与治理
- 作者指令 2026-10-27: 跨域切换多 baseline 端到端, 先不再修改算法;
  六臂同期测量, 不用旧 AERO 数值拼接新基线; 静态方法共用预先冻结的
  混合域校准画像, 随后布局不再更新, 画像与求解成本单列; 先从已有日志
  核对实际控制步数与重平衡机会。AERO 使用 P0 (RG 已有版本对照, 不加臂)。
- 预算: prep <= 3 boot; 正式 24 boot (6 臂 x 4 轮); 零数据 INFRA 重启 <= 2。
- LEDGER_xd.tsv 只追加; 每 boot 记 START/DONE 行; 异常记 INCIDENT 行。
- 结果去向: 论文 5.3.1 跨域主比较; 域边界时间线用于 5.5 案例。

## 1. 已有日志核对结论 (作者第 3 点, 先于本战役完成)
- 动态 EPLB (period 256) 在 E3 上实际触发: 旧 m_E3_E_b3 / m_E3_E_b5
  server.log 各 16 次 rebalance start, banner "will rebalance per 256
  iterations", 两独立 boot 互证。
- 控制步数参考 (旧 E3 A 臂): m_E3_P 事故 boot sp_fwd_total=282 @17 窗;
  m_E3_A2_b8 sp_fwd_total=522 @32 窗 (512 true steps, 27 swaps)。
- E3 bench 成本: 旧记录 187-204s (约 40-44 tps), 为 prover 流约 2.2 倍。
- 风险登记: INCIDENT_E3P_hang.md (final_plan_20261019) - 旧代码域边界
  单窗 299 swaps 触发 NCCL COALESCED 挂死, 臂无关迁移路径风险;
  现行冻结 async_swapper 2e4652a0 已有 55+ prover boots 零挂死,
  但 E3 边界批量更大, 预案见第 8 节。

## 2. 代码锁表 (与 recovery_admission_e2e_20261027 相同冻结基线)
| 组件 | sha256 |
|---|---|
| pb_oeplb/async_swapper.py | 2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f |
| pb_oeplb/controller.py | 4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697 |
| pb_oeplb/window_v2.py | 810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74 |
| pb_oeplb/sp_state.py (patched; P0 以 RG=0 运行时与原始 ba572f93 字节等价) | 525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466 |
| bench_client_v2.py (final_plan_20261019/scripts) | 066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c |
| launch_rec_e3.sh (本战役 prep 录制 launcher) | b2fc8a2984b76f060351b501e3fa661f645039182687fdde208157baa89f6c42 |
| run_rec_boot.sh (本战役 prep 录制 driver) | 2a675521fdc41e0ec083944b0710fce1d9f435d67591220688df1f5703bb1870 |
| launch_arm_xd.sh (六臂 launcher) | 待 prep 后冻结 (AMENDMENT-1) |
| run_xd_boot.sh (正式 runner) | 待 prep 后冻结 (AMENDMENT-1) |
| 静态臂 placement x3 (E3 混合域画像) | 待 prep 后冻结 (AMENDMENT-1) |

冻结机制: placement sha 写入 code_lock/placement_shas.env (一次写入,
不再修改); runner 启动时 source 并逐 boot 校验, 不匹配即 BOOT_ABORT。

## 3. 请求流 (优先复用已有冻结 E3 四域流, manifest 已核对)
- 路径: final_plan_20261019/B_data/streams/e3_multidomain_switch.jsonl
- sha256: d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879
- n=8192, uniq_text_hash=8192, retok_len p5/p50/p95 = 512/512/512 (in512/out1)
- 域顺序与边界 (registry.json 核对): sharegpt [0,2048) -> prover [2048,4096)
  -> book [4096,6144) -> code [6144,8192), 每域 2048。
- 规模: 8192 请求, O1, 并发 256, bench timeout 900s。

## 4. 六臂定义 (基座统一: sealed rev1 BASE + --disable-cuda-graph + mem 0.8
     + --disable-radix-cache; 与上战役一致, 仅 placement 换 E3 版)
| 臂 | 机制 | a2a | 冗余 | placement |
|---|---|---|---|---|
| I | Identity, 默认初始布局 | auto | 0 | 无 |
| ES | 静态 EPLB: E3 混合域 logical_count -> init_by_eplb/rebalance_experts 官方路径, 此后布局不更新 | normal (官方 recorder 在 auto 下 NotImplementedError, 沿用上战役口径并披露) | R16 | datafore_e3_placement.json |
| ED | 动态 EPLB: 官方 --enable-eplb, auto/R16/period256 (E3 上实际 16 次/boot 已核) | normal | R16 | 无 |
| MT | MoETuner 风格: E3 画像 P_e3.npz -> 逐层 Gurobi ILP1 聚类 (G=8 x K=16) -> init_by_mapping, 此后布局不更新 | auto | 0 | placement_e3_fair.json |
| DF | DataFore 风格: E3 logical_count -> greedy least-loaded 静态重映射 (rebalance_experts 执行), 此后布局不更新 | auto | 0 | datafore_e3_remap_placement.json |
| P0 | AERO-base: 冻结自适应控制器, OEPLB_SP_RG=0 (sp_state patched 以 RG=0 运行时字节等价原始), 在线重平衡 | auto | 0 | 无 |

混合域校准画像来源 (作者第 2 点): 调查确认无既有冻结混合域画像
(MT/ES/DF 旧画像全部为 prover 单域; E3 旧 boots 未开 recorder)。
按报告的补齐方案: 1 个 prep 录制 boot (launch_rec_e3.sh, per_token
recorder, identity 布局, E3 全流) -> recorder .pt 离线转换:
  a) ES/DF: .pt 聚合 logical_count[94,128] -> datafore_e3_placement.json;
     DF 版再经 greedy least-loaded 重映射 -> datafore_e3_remap_placement.json;
  b) MT: pt2npz.py -> P_e3.npz -> solve_ilp.py (Gurobi ILP1) ->
     placement_e3_fair.json + meta (逐层 solve_s = 求解成本单列);
  c) 录制 boot 墙钟与 .pt 大小 = 画像成本单列。
画像覆盖全部四域 = 静态臂的强校准 (对静态臂有利), profile 与 eval 流
同源属性在论文中如实披露 (与上战役口径一致)。

## 5. Prep boots (预算 <=3)
| # | 名称 | 目的 | 通过判据 |
|---|---|---|---|
| 1 | rec_e3 (profile recording) | E3 全流 per_token 录制, 产出混合域画像素材 | bench rc=0 且 ok=8192; recorder .pt >=1 且 sha 登记 |
| 2 | ES_val (注入验证) | 验证 E3 画像经 init_by_eplb 注入成功 | server.log 出现 init_expert_location from init_by_eplb 证据行; bench rc=0 |
| 3 | (机动) | 仅当 1/2 失败且原因为基础设施时启用 | - |
离线转换 (无 GPU): pt -> logical_count -> ES/DF placement; pt -> P_e3.npz
-> Gurobi ILP1 -> MT placement; 三个 placement + meta 的 sha 写入
code_lock/placement_shas.env 并以 AMENDMENT-1 追加到本文件。
ILP 求解不收敛或超时 -> 报告具体层与原因, 不静默换求解器。

## 6. 正式执行锁定顺序 (24 boot, 每臂 4 次, 轮内换序轮换)
- 轮换规则: 第 r 轮顺序 = 基础序 (I,ES,ED,MT,DF,P0) 左旋 r 位;
  r=1..4。锁定后写入 ORDER_TABLE.json, 队列脚本由其生成, 不事后更改。
- 每 boot: runner 守卫链 (流 sha -> 4 代码 sha -> launcher/bench sha ->
  3 placement sha) -> 起服 -> ready 等待 (225x4s) -> bench (8192, conc256,
  timeout 900) -> mech_check -> boot_meta.json.end -> LEDGER DONE 行。
- rc=2 (守卫失败) -> GUARD_ABORT 停队列人工介入; rc=1 (mech/ready 异常)
  -> LEDGER 记录后继续, 操作员在 INFRA 预算内分诊, 不静默丢弃。

## 7. 指标与统计
- 主指标: 完整服务墙钟 (bench total_time_s), 输出吞吐 (tps)。
- 机制记录 (5.5 素材): 各臂在域边界 (请求 2048/4096/6144) 附近的状态/
  规划/迁移时间线。P0: v2p1_rank0.jsonl 逐窗行 + sp_run_end.json;
  ED: server.log rebalance start 时刻序列 (预期 16/boot); ES/MT/DF:
  布局冻结点 (注入证据行); I: 空基线。边界前后窗口负载迁移量单独成表。
- 画像/求解成本单列: 录制 boot 墙钟 + .pt 大小 + ILP 逐层 solve_s 合计。
- 统计: P0 对其余 5 臂, 按轮配对 (df=3), 双侧配对 t, Holm 校正 5 比较;
  每臂报告 4 轮均值/标准差与逐轮值。
- 六臂同期测量声明: 本战役所有数值来自 2026-10-27 起的同窗口 boot,
  不与 recovery_admission_e2e_20261027 (prover 流) 数值拼接比较。

## 8. 应急预案
- 域边界挂死 (INCIDENT_E3P_hang 签名: NCCL COALESCED SeqNum 停滞 +
  watchdog 600s): bench timeout -> RESULT_RETIRE 路径; 记 INCIDENT 行;
  零数据时在 INFRA 预算 (<=2) 内重启该 boot 一次; 同签名第二次挂死 ->
  停队列报告作者, 不静默弃臂, 不扩大修改范围。
- 就绪失败: READY_FAIL 路径 (--no-bench mech), LEDGER 记录, 人工分诊。
- ES/ED deepep normal 口径: 与上战役一致, 论文披露其对 EPLB 臂的
  系统性影响方向。
- 队列中断 (连接断开): 所有长任务 setsid detach + 日志文件 + 轮询恢复。

## 9. 归档与 git
- /data/minghua/sjq/PAPER_EXPERIMENTS/cross_domain_e2e_20261027/ 全量
  (含 server.log/perreq) + ARCHIVE_SHA256SUMS.txt manifest。
- GitHub 精选目录 /workspace/EPLB/PAPER_EXPERIMENTS/cross_domain_e2e_20261027/
  (PROTOCOL/LEDGER/code_lock/ORDER_TABLE/analysis/scripts/placements +
  boots_curated 小文件), plumbing commit 推送, 返回链接与 commit。
- EXPERIMENT_MAP.md 更新: 5.3.1 跨域主比较行 + 5.5 时间线案例行。
- 代码零修改声明: 本战役不修改任何冻结代码; 新增文件仅限本战役目录
  与归档目录; sp_coupling_dev 脏状态不触碰。

## AMENDMENT-1 (appended by finalize_prep_watcher at profile freeze; resolves sec.2 placeholders)
- Prep #1 (rec_e3 v1, per_token+auto): READY_FAIL zero-data. Root cause:
  _DetailSinglePassGatherer crashes on single-node deepep
  (num_tokens_per_rdma_rank=None, expert_distribution.py:406). INCIDENT logged.
- Prep #2 (rec_e3 v2b, proven launch_profile.sh recipe stat+normal+graph128
  +manual HTTP arm): SUCCESS. Arm 200 at 05:25:00Z; bench ok=8192 err=0,
  190.86s, 42.9 tps; stop+dump rc=0/0; recorder .pt rank0 shape (1000,94,128)
  sum=3154479072 sha=fa04965525ecb44ccd89e67be525e815a5ae5474ccbb44c6c7c9f9c13dc6df4a
  (48130049 B). Mid-boot incident: no curl in env; first bench pass ran
  unarmed => zero profile contamination (radix off, artifacts renamed
  INVALID_partial); re-armed via python urllib, fresh full bench in same
  server session. Profile covers the FULL E3 stream (all four domains).
- Pipeline: pt2npz.py -> P_e3.npz (per-layer sum 33558288 = 4194260 prompt
  tokens x 8 topk, exact; layer0 imbalance 2.379 vs prover-only 4.093).
  ES/DF shared json logical_count 94x128, byte-identical files (author key
  point 2: shared pre-frozen mixed-domain calibration profile; DF greedy
  least-loaded executes inside sglang rebalance_experts at init without
  redundancy; ES runs eplb algorithm with R16).
- MT solve cost (single-listed): Gurobi ILP1 G=8 K=16 time_limit=30
  mip_gap=0.005; mean_imbalance=1.0000 max_imbalance=1.0000
  total_solve_s=2822.5; profile capture cost = rec boot bench
  190.86s + pt 48.1MB; per-layer solve_s in placement_e3_fair.json.meta.json.
- Frozen shas (code_lock/placement_shas.env, read-only 444):
  PL_ES=ddc3943d13cd8ffe67be3014d239260e170215b1a340f5fa84efca312b6cf306
  PL_DF=ddc3943d13cd8ffe67be3014d239260e170215b1a340f5fa84efca312b6cf306
  PL_MT=ddb38bfd03e191fe6d45acded55980c8ceb233ddd10ff94a167e4bf214ce3f66
  PL_MT_META=351adaac208cdc65e7d67386ad793c4845441b25ad6b7f6157e1334e7f0a6d9c
- Prep #3 = ES_val (injection validation). Formal 24-boot queue auto-launches
  only on ES_val PASS (runner rc=0 AND init_by_eplb injection evidence in
  server.log). Otherwise STOP + triage.
