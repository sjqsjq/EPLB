# recovery_admission_e2e_20261027 预注册协议 (PROTOCOL)

状态: 已在任何正式 GPU 花费之前锁定。本文件与 LEDGER_rge2e.tsv (append-only) 共同构成治理记录。
修订规则: 仅允许追加带时间戳的 Amendment 小节, 不允许改写已锁定内容。

## 0. 授权与治理

- 授权: 作者任务指令 2026-10-26 —— 完成一项机制修改 (恢复准入策略) 加一组多 baseline 端到端比较, 保留现有版本和旧结果。
- 预算登记 (作者中途核减 少重复几次吧 之后): 正式 = 4 轮 x 7 臂 = 28 boots; 准备 <= 3 boots; 零数据 INFRA 重启 <= 2; 不基于速度追加轮次。
- 旧版本保留: 原 sp_state.py 备份于 code_lock/sp_state.py.orig_ba572f93; AERO-base 路径 (OEPLB_SP_RG=0) 与原版逐字节等价 (离线已证)。phase-2 (abl_prover_sd_20261026) 及全部历史结果目录不动。
- 禁止拼接: 七臂全部在同一时期用同一冻结流重测, 不得混入任何历史均值 (含已发表数值 62.4/61.8/68.4/73.7/75.3/70.5, 仅作背景引用)。
- 所有 boot (含准备) 落在 boots/ 下, 每次 boot 由 run_rge2e_boot.sh 执行守卫链并在 LEDGER 记行。

## 1. 机制修改: AERO-RG (恢复准入门控)

唯一改动 (sp_state.py, QUIET 态 rev1 分支): 孤立 confirmed_grow (无 swap_pending 且 swaps_started<=0) 不再作为变化证据 change_ev, 即不再单独触发 QUIET→RECOVER。语义: 稳定窗口内的自然增长不是负载变化。

不变式 (全部离线验证, 见 §3):
- 不删除底层事件; window_v2.py 与 W_EA 更新逻辑完全不动。
- 忽略孤立 grow 后继续评估其余出口条件 (elif 链 P1a z>=z_hi / P1b guardrail / P1c change_ev / P2 coverage / P3 hysteresis 自然推进, 无提前 return)。
- 同窗口真实告警、压力/收缩事件、迁移、护栏事件保持原优先级; coverage 失败出口与持续 margin 上升出口不变。
- S/H/W 档位、阈值、确认次数、planner、预算、统计维护、迁移实现全部不动。
- 独立开关 OEPLB_SP_RG (env): 0 = AERO-base (逐字节等价原版), 1 = AERO-RG; rg=1 仅在 rev=1 且 mode=adaptive 下合法, 否则构造时 raise。
- 跨秩一致性协议保留: proposal_hash 在 rg=1 时统一加 sp1r1rg 标签 (所有状态), 混版本集群哈希必然不一致 → 迁移降级为 HOLD (安全侧)。
- 记录策略版本与转移原因: 孤立 grow 被忽略时追加信息性 reason ([AERO-RG] lone confirmed_grow ignored ...); 发生转移时 apply 记录前置 [AERO-RG] policy=rg rev=1。

## 2. 代码锁 (每 boot 由 runner 守卫, 不匹配即 rc=2 中止)

| 组件 | SHA-256 |
|---|---|
| sp_state.py 原版 (备份) | ba572f93e2d2c2558cfe2dd8cf0a750ed790e7f544f383cd6c5ebf3768b16329 |
| sp_state.py AERO-RG 版 (src==site-packages 部署) | 525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466 |
| async_swapper.py (冻结) | 2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f |
| controller.py (冻结) | 4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697 |
| window_v2.py (冻结, src==部署) | 810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74 |
| launch_arm_rge2e.sh (本战役启动器) | c2c272594f1851df2ff829f493f8717bc2072fdf2db30fe2b710975899916e18 |
| run_rge2e_boot.sh (boot runner) | 5fd6d44a7f68025541bb20b22485fef31fbc0f915bb433da210c77bc6a9787a9 |
| mech_check_rge2e.py (事实检查, 克隆自 mech_check_p2.py + 5 处补丁) | 54177b648d6ffe7e06b159b052424779ac03ae0304a7e22720021100f8d402e7 |
| bench_client_v2.py (冻结) | 066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c |
| 流 prover_256tok_out1_x4.jsonl (8192 行 = 2048 唯一 x4) | b312ba50c39abe1a81e85bd16f27f2b7913ae7fffb57550efb5b702fb34aabf0 |
| placement datafore_prover (ES 用, logical_count 总和 48,563,408 = prover 全量) | c1787ad19f16ef5d840d5a7c7e364ecf23fb3caafba682e56c0f21991a298f29 |
| placement datafore_remap (DF 用, 总和 2,908,736 = head1024 公平分裂规模) | 07f913570a9c7e0518e7ef39718b38e9f93405678a0a0eb2a7c3adab81e5d85e |
| placement pinned_fair (MT 用, solver=moetuner_ilp1_greedy) | 299cea283e93efe69d3aec8ab71f210908d8b262d45f602079ed3daebe2a2a60 |
| placement pinned_fair meta (逐层 solve_s, Gurobi 合计约 1015.5s) | 125ac2454474e718f838d6222e300bc2437ec5e7dfe2b7820bc1998b24676bbf |
| MT profile 源 pinned10x_head1024.jsonl | 10660f2ef383e65fbc1dc0a4785362e9c4648edb4353f969ad3578187f76b4b1 |
| queue_prep1 / queue_prep23 / queue_formal | 700331e17777cb3da813bcc3501fd310632c11372b8fb313509e40a18f199c7c / 21bb64c7f350a099590ed269a2d50ec7fe2e175945a8c3b3ef392eaead1dcc63 / 5ba40efe58f478804eeca0e82db80508787727b31baff9937d9433fdc67c2082 |
| 旧战役 launcher (参考, 冻结只读) | c2f6b9dd6e57c64d48379b8bf7b6c012df0bd9c48c9bd254f5ca16299d5f8ede |

冻结 v2p1 旋钮 (P 臂, launch_arm_rge2e.sh 内逐字保留): KAPPA=3875.1768798828125, N_MIN=12320768, M_SHRINK=2/M_GROW=3, DWELL 2/2, U_MAX=4, W_UNCERTAIN_TARGET=16, URGENT_EPISODE_CAP=3, BETA=0.5, EA_W=16, PROBE_MODE=off, WINDOW_CONTROLLER=21, seed=20261018, rev=1, mode=adaptive, quiet_streak=3, recover_min_cycles=2。

## 3. 离线验证结果 (先于任何 GPU 验证 boot 完成)

- tests_rgv/test_branches.py: BRANCH_TESTS_PASS n_checks=30 全部 True (T1-T17)。关键项: T1 孤立 grow 原版进 RECOVER 而 RG 保持 QUIET 且带注释; T2/T3 grow 伴随 pending/started 时两版都进 RECOVER; T4/T5 P1a 与 guardrail 优先级不变; T8 忽略后 P2 coverage 出口仍能触发; T9 P3 两步确认与 band 重置不变; T10/T11 TRACK 与 RECOVER 提案逐字段一致且哈希按设计分叉; T13 双 rg1 实例 500 步一致 (跨秩一致性); T14 3x500 步扫描 orig 与 new0 (rg 关) 提案/哈希/转移/驻留逐字节一致; T15 非法组合构造即 raise; T16 转移记录标签; T17 snapshot/restore。
- tests_rgv/replay_p_b1.py: 用 apsd2_P_b1 的 18 行 v2p1 记录严格回放。REPLAY_ORIG_EXACT OK (7 转移逐条 state/transition/reasons 逐字节一致); REPLAY_NEW0_IDENTICAL OK (补丁版 rg=0 与原版逐字节一致, 即旧版本行为完整保留); REPLAY_RG1_COUNTERFACTUAL: 两次孤立 grow 的 QUIET→RECOVER (w6, w11) 消失, 4 条 [AERO-RG] 注释, w6 起驻留 QUIET。rg1 回放是策略层反事实 (基于 base-run 记录信号), 全系统 RG 行为由 GPU 验证 boot 度量。
- 结果文件: tests_rgv/branch_results.json, tests_rgv/replay_results.json。

## 4. 七臂定义与运行路径

统一基座: 密封 rev1 BASE, 仅一处统一改动 —— --cuda-graph-max-bs 128 换成 --disable-cuda-graph, 七臂一致。理由: ES/ED 依赖 deepep normal (官方 EPLB recorder 仅实现 normal/low_latency, auto 会 NotImplementedError), normal 与 CUDA Graph 不兼容; 按作者规则 若图必须联合禁用则统一处理。mem-fraction-static 0.8 为密封值并保持统一 (与 §5.3.1 基线口径 0.78 的偏差在此登记, 公平性由七臂同口径保证)。统一负载: O=1, conc=256, 冻结流 8192 请求。

| 臂 | 实现 | DeepEP | CUDA Graph | 冗余 | 机制 flags | placement 素材 |
|---|---|---|---|---|---|---|
| I (Identity) | 默认连续布局 | auto | 禁用 (统一) | 0 | 无 | 无 |
| ES (静态 EPLB) | SGLang init-expert-location (logical_count → init_by_eplb/rebalance_experts 官方路径) | normal | 禁用 (normal 强制) | R16 | --ep-num-redundant-experts 16 --init-expert-location datafore_prover_placement.json | prover 全量记录 (48,563,408 tokens) |
| ED (动态 EPLB) | 官方 --enable-eplb, 冻结合理配置 E*=e5 (calib/FINAL_SELECTION.json) | normal | 禁用 (normal 强制) | R16 | --enable-eplb --eplb-algorithm auto --ep-num-redundant-experts 16 --eplb-rebalance-num-iterations 256 | 无 (在线自采集) |
| MT (MoETuner 风格) | 逐层 Gurobi ILP1 聚类 (G=8 x K=16) → physical_to_logical_map → init_by_mapping | auto | 禁用 (统一) | 0 | --init-expert-location placement_pinned_fair.json | head1024 公平分裂 profile |
| DF (DataFore 风格) | ISCA 2026 Alg.2 贪心最小负载 ≡ SGLang rebalance_experts on logical_count | auto | 禁用 (统一) | 0 | --init-expert-location datafore_remap_placement.json | head1024 公平分裂记录 |
| P0 (AERO-base) | 现有版本, OEPLB_SP_RG=0 | auto | 禁用 (统一) | 密封 PB | 冻结 PB flags + v2p1 env | 无 profile |
| P1 (AERO-RG) | 新版, OEPLB_SP_RG=1 | auto | 禁用 (统一) | 密封 PB | 同 P0 | 无 profile |

ED 配置依据: e6 事故表明 period=1000 在约 256-271 forward 的流上物理不可触发 (MECH_RETIRE); E*=e5 (auto/16/256) 为校准冻结的 按预测精确触发 配置, 非零触发设定; 每 boot 实际 rebalance 次数由 server.log 中 [EPLBManager] rebalance start 行数记录 (mech_check baseline_facts)。§5.3.1 曾用 period100+buffer32, 本战役改用校准冻结 E*, 偏差在此登记。

profile 与评测流的内容关系 (如实声明): 全部静态素材派生自 prover_256tok_out1.jsonl 记录 (MT/DF 用前 1024 行公平分裂, ES 用全量 8192 请求记录), 评测流 = 2048 唯一请求 x4。profile 内容为评测流内容的子集 (同域设计): 静态臂在评测流的前缀子集上画像再全流评测, 属于对其有利的口径, 报告如实注明; P0/P1 不使用任何 profile (在线自适应)。

profile/求解成本单独入账 (不计入任何臂的 wall clock): MT Gurobi ILP1 逐层求解合计约 1015.5s (meta 逐层 solve_s); DF 贪心求解 (秒级, 记录在 meta); ES 的 init_by_eplb 重平衡为官方在线路径, 其成本保留在 ES 的 wall clock 内 (启动期), 不剥离。

## 5. 准备 boots (<=3, 均不计入正式统计)

- prep1 = rgv_P1 (作者顺序: 新版验证最先): P1 在冻结流上完整跑一次。通过判据 = 行为正确而非速度: 8192 请求成功 (ok=8192 err=0), 驻留合理, 转移 reason 中出现 [AERO-RG] 记录且孤立 grow 不再触发 QUIET→RECOVER, 8 秩 planhash 一致, mech_check 完整性通过。
- prep2 = rgv_ED: 官方 EPLB 兼容性 + banner + 实际 rebalance 次数 > 0 (否则按 §8 报告, 不带病进正式)。
- prep3 = rgv_ES: normal+R16+placement 注入兼容性, injection_evidence 落盘。
- profile 准备 boot = 0: 素材全部复用已登记 sha 的现存产物 (作者允许 能复用现有素材就复用)。

## 6. 正式执行 (28 boots, 顺序预先锁定)

种子 20261027; 基序 [ES I P1 DF ED P0 MT]; 轮 1-3 = 基序循环移位, 轮 4 = 预定反转排列; 首位序列 [ES I P1 MT] —— 无任何臂在全部轮次首位, AERO 臂不总是最先 (作者规则)。锁定文件 order/ORDER_TABLE.{json,md}; queue_formal.sh 由 ORDER_TABLE.json 程序化生成并断言 28 行。

R1: ES I P1 DF ED P0 MT
R2: I P1 DF ED P0 MT ES
R3: P1 DF ED P0 MT ES I
R4: MT P0 ED DF P1 I ES

不按中期结果重排; INFRA 零数据重启 <=2 且逐条记 LEDGER; guard rc=2 停队列待人工分诊; rc=1 boot 保留现场待分诊, 不静默跳过。

## 7. 指标与统计

- 主指标: 全服务 wall clock = bench total_time_s (含全部在线成本: 记录/规划/迁移/EPLB rebalance/注入, 不剥离任何段)。
- 同报: 实际输出吞吐 tps = ok / total_time_s。
- 对比结构: P1−P0 = 机制效应 (恢复准入); P1 vs 五个外部基线 = 全系统效应。
- 配对: 按轮配对 (4 对/对比); 报告全部 28 个原始值、均值差、95% 区间 (paired t, df=3); 检验方法: 双侧配对 t 检验 (命名登记); P1 vs 六臂的 p 值统一 Holm 校正。
- 机制记录 (每 P boot): 真实驻留 (residency_committed_h / residency_final), 采样量 (recorded_steps), 转移原因全列表, 首次迁移位置 (窗口号), 迁移批次 (decode_record_batches/swaps), 专家对计数; rg 特有: rg_annotation_rows, rg_policy_tagged_transitions。
- 轮 1 预选为机制时间线案例 (窗口级轨迹图/表)。

## 8. 应急预案 (作者规则: 报告具体问题, 不静默删臂或扩大修改范围)

- 缺失 profile: 当前无缺失 (三份素材 + profile 源均在 placements/ 且 sha 已验); 若正式期发现损坏 → 停, 报告, 不重采不替换。
- 基线不兼容 (ED/ES 起不来或零触发): prep 期即报告具体报错与日志行, 等待作者决定; 不带病进正式, 不静默改配置。
- 机制错误 (rgv_P1 验证失败): 立即停止全部 GPU 工作 → 修复 → 重做离线验证 (30 项 + 回放) → 重跑 prep1 → LEDGER 与本协议 Amendment 记录; 正式 boots 只在 prep1 通过后开始。
- INFRA 故障 (零数据): 重启 <=2, 逐条记 LEDGER; 超限即停并报告。

## 9. 归档与 git

- 归档根: PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/ (协议、配置与代码锁、顺序表、逐 boot 数据、报告、分析脚本); 本目录为工作现场, 归档时全量收敛并生成中文 README 与 SHA-256 manifest; 大件 (逐 boot 原始日志) 落 /data/minghua 并在归档留指针。
- EXPERIMENT_MAP.md 总索引更新: §5.3.1 多 baseline 端到端比较; §5.4 恢复准入新旧对照。
- GitHub: 归档数据提交并 push (plumbing commit 叠加 origin/main 10362460e1abbee3c83d979a49a10e574d21fd0e, GIT_INDEX_FILE 临时索引, 只增新内容, 绝不触碰 sp_coupling_dev 脏区); 返回链接与 commit。
- 代码修改 (sp_state.py) 单独提交并提供 diff (code_lock/AERO_RG.patch.diff, 79 行)。
