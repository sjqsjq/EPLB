# PROGRESS — 采样与规划窗口联动 (sp_coupling_v1)

方案原文: results/adw_v2p1/next_version/PLAN_sampling_planning_coupling_20261018.md (作者 2026-10-18, 全权执行、不询问、不 push)

## 阶段账本
- [x] P0a 方案文档落盘 + 提交 (main)
- [x] P0b 代码映射文档 MAPPING_sp_coupling.md (提交见 git log; H/W 可安全拆分, 前置采样门可行, PD-1..15 预数据决定)
- [x] P0c 开发分支 sp_coupling_dev 实现 (完成 2026-10-18; 见下"实现记录")
- [x] P0d 离线单元测试 (完成; 见下"测试记录"; 全部通过, off-mode 行为回归无缩水)
- [x] P0e 短服务 smoke (完成 2026-10-18 11:33Z: 3/3 VALID, 0 INFRA; 证据见下)
- [ ] P1  12 次完整服务开发启动 (3臂×2条件×2block, 跑前固定seed臂序, block2 对换)
- [ ] P2  分析报告 (5 问) + 提交

## 预算估算 (基于校准战役实测, 同一节点/同一 harness)
- W1 单启动: boot 64–92s + bench ~352–357s + teardown ≈ 7.5–8 min
- W2 单启动: boot 64–72s + bench ~706–712s + teardown ≈ 13.5 min
- 12 启动 = 6×W1 (~48min) + 6×W2 (~81min) ≈ 2h10m GPU; smoke +8min; 总 GPU ≈ 2.5h
- 结论: 与既有资源包一致, 无需扩容, 按方案 §7 阶段1 执行

## 治理 (方案原文约束, 持续有效)
- 独立开发分支; 保留全部封存结果; 不 push; 不询问作者
- W/H/S 候选值允许在接触本次性能数据前因实现约束修订并书面说明; 计时开始后不得改值
- 跨 rank 时间只用 MAX/per-rank 永不求和; 全部成本留在服务时间; 不挑选有利重复; 失败启动 retire 不重试
- 基础设施失败与结果性失败分开记录; HALT 条件(完整性失败/机制错误) → 停止+完整报告等作者
- 阈值/噪声标度调整只用开发/校准材料一次冻结; 不从性能评价选阈值
- probe/ABA 准入在本版关闭 (mode off), 15 次校准结果作为独立研究保留不改

## 实现记录 (P0c, 2026-10-18)
- src/sp_state.py (新, 15780B): SP_TABLE 冻结 {TRACK:(1,16,16), QUIET:(4,16,64), RECOVER:(1,8,8)}; SpController (propose 纯函数 / apply 仅提交时变更); PD-9 门 block=sp_fwd//S, pos=blake2b(seed:block)%S; PD-2/3/4 进出规则; proposal_hash (sha256→2^58) 折入 phase-2 decision hash; snapshot/restore 支持 HOLD 丢弃 (PD-6); residency 账本
- src/config.py (+135/-1): sp_mode/sp_s/sp_w/sp_seed/sp_quiet_streak/sp_recover_min_cycles + OEPLB_SP_* env + validate_sp_coupling() PD-12 冲突拒绝 (always_record/sample_interval!=1/probe!=off/wctrl!=21/adaptive/decay!=0 全类验证通过)
- src/controller.py (+1064/-5, 23 处补丁, 全部为审计过的替换): 前置门 on_forward_pass_start (model_runner :2660 后 guarded hook, +7/-0); _sp_boundary_dispatch H/W 拆分 (W∧¬H → RuntimeError 机制守卫); H-only 边界 = detSUM+confirmMIN+confirmMAX (3 collectives, r3 fatal 真实可达), H+W = plannerSUM+3 (4, 原 :852→:967 顺序保持); PD-1 ctrl n_min=n_min_full//S 每 phase-2 重设 (对称/确定性); PD-8 rec W_after 在 SP 下永不应用, ctrl.W 钉 w_init=16; PD-7 转换提交时重启 H/W 计数 + planner 封存清零 (sp_seal_S_change, cover_steps 入日志); PD-10 planner SUM fatal-safe (_v2p1_fatal_flag 门控外层 re-raise, off-mode 逐位一致); PD-13 三账本分离 (opportunities/recorded/valid_rows, valid_rows GPU 累积每 H 边界读一次); §8 jsonl sp_* 17 字段 + PROF sp 段 + 日志行 sp 段 + sp_run_end.json (每次提交转换即增量写, SIGKILL 安全, +atexit)
- 部署: deploy_sp_coupling.py (预检 v2.1 部署态 sha 62293ea9/4ba8de2d + 10 文件 SAME + 无活进程/端口 → 备份 backup_pre_sp_deploy/ → 拷贝 3 文件 → model_runner 精确锚点替换 (diff 限定 +7/-0) → py_compile → import smoke IMPORT_OK → sha 锁 deployed_shas_sp_coupling.txt, mr 751dd61c→6580c364)

## 测试记录 (P0d, 2026-10-18)
- scripts/test_sp_state.py: 纯状态机 55 断言 ALL_SP_STATE_TESTS_PASS (T1-T15: 冻结表/QUIET 3 连击进入与 4 类退出/coverage 门 n_min_eff/RECOVER 2 净周期无超时/PD-5 计数复位/swap 破坏 quiet/HOLD 幂等/hash 分歧检测/PD-9 门 25% 精确+无相位锁+每 block ≤1/固定臂 50 周期不变/off 惰性/非法配置拒绝/residency)
- scripts/test_sp_controller_logic.py: 真实 PBOEPLBController (SRC 影子导入) + stub executor + 伪 _decide (复制 collective 顺序) + gloo。world=1: 32 断言 (A1-A22 全生命周期 TRACK→QUIET(z 衰减 48→0.058, 7 周期)→RECOVER(2 净周期)→TRACK→QUIET 再进入; collective 计数逐 forward 验证 H-only=3/joint=4; PD-1 valid_rows×world==n_fresh; PD-7 封存 cover_steps=16; PD-8 _effective_sync_window 恒 16 而 off-mode 生长; B: 接受 swap 使 QUIET 推迟到 cycle6; C: 全 idle 周期 coverage 退出; OFF1-6: 无 sp 字段/legacy 计数/W_after 应用)。world=2: +A22 跨 rank 提交时间线全等, 40 断言 ALL_SP_CONTROLLER_TESTS_PASS
- PD-10 实战验证: world=2 首轮 rank0 断言死亡 → rank1 det-SUM 立即 FATAL 传播 (设计行为, 非静默继续)
- 回归 (PD-14): probe_admission ALL TESTS PASSED (75); window_v2_logic ALL_TESTS_PASS (32); window_v2p1_logic 36 行为 PASS + 3 结构陈旧 FAIL {T0c-controller, T0c-config (新增翻转: 授权 config 补丁的结构性必然, 非行为回归), T4c} — 测试文件未动 (git 证明); multiprocess_faults w4 ALL_FAULT_TESTS_PASS (11 windows); fatal_boundary w2 ALL_FATAL_BOUNDARY_TESTS_PASS
- T0c 删除行审计 (vs 部署 v2.1, 共 8 行, 全部为授权替换): if→elif is_idle (P4); planner SUM 重缩进入 try (P7); error_sentinel/decision_hash window_count→_wid_sp ×3 (P10b/d/e, off 下等值); W_after 应用加 if not _sp_active (P11)

## P0e 修订记录 (计时前, 书面说明)
- controller.py 追加 1 处补丁 (计时前修订, 不影响 W/H/S 值/阈值/门控): sp_run_end.json 原仅在"提交转换"时增量写; 但 teardown 是 pkill -9 (atexit 不执行) 且固定臂/无转换运行永远不落盘 → 改为每个已提交 H 边界刷新 (rank0 本地写, 与 jsonl 同频同级, 已在代码注释标注)。SRC+DEP 同步重部署, sha 锁 v2 (deployed_shas_sp_coupling_v2.txt, sha256sum -c 格式, controller add3208801a6); v1 锁保留为部署历史。重部署后 test_sp_controller_logic world=1 复跑 32 断言全过。
- integrity_check_sp.py (锁定于首个 smoke 启动前): 三臂均要求 ADW21 banner≥8 + SP banner≥8 + 冻结 knob 权威 + meta SP env + S/H/W ∈ 冻结表 + 固定臂零转换 + PD-9 门算术 (S1: opps==sp_fwd; S4: |4·opps−sp_fwd|≤4) + residency==h_cycles + 行数基准 est_steps/H_ref (90%); MECH/RESULT/INTEGRITY 分类与旧 checker 同构; 不比较臂, 不看端点值
- run_sp_coupling_dev.sh (克隆校准驱动骨架): SMOKE(3 启动, 300 请求子集, 不占 DEV 预算)/DEV(12 计时启动上限, INFRA cap 3 单格重启一次, RESULT_RETIRE 不重试继续, MECH/INTEGRITY → HALT exit 8); 臂间唯一差异 = OEPLB_SP_* 块; run_list CSV 由冻结 sp_arm_order.json 生成并 sha 锁定 (smoke 0700617d…, dev d5ba4739…); smoke_w1_300.jsonl = W1 前 300 行 (sha 073f79c2…)

## Smoke 证据 (P0e, 300 请求 ×3 臂, 全部 VALID)
- F_sparse (S4/H64/W64): sp_fwd=74, opps=19 (|4·19−74|=2≤4 门算术✓), recorded=15, prof_record_calls=1410, 1 H 周期 64 步, transitions=[] (固定臂零转换✓), lockstep_ok
- F_dense (S1/H16/W16): sp_fwd=90, opps=90 (S1 门=每 forward✓), recorded=76, record_calls=7144, 5 H 周期, swaps=2, p2l_version=2
- A (adaptive): 与 F_dense 完全同轨迹 (sp_fwd/opps/recorded/record_calls/jsonl z 值逐位相同, swaps=2, 驻留 TRACK 5 周期, transitions=[]) — 未转换时自适应臂与 F_dense 不可区分 (机制零额外开销的直接证据); 未进 QUIET 的原因 = 2 次已接受 swap 打断 quiet_streak (PD-2 正确行为)
- 记录缩减机制验证: F_sparse record_calls 1410 vs F_dense 7144 = 5.07× (= recorded 76/15, 超过 4× 因稀疏采样下服务动态略变)
- 全 rank 一致性: SP banner 8/8 rank ×3 臂; lockstep_ok=true 全部周期 (steps/ts MIN==MAX); n_fresh=Σrank rows (rank0 rows×8≠n_fresh 属预期: 各 rank batch token 数不同, 门与覆盖只用全局 n_fresh); idle forward 推进 sp_fwd 但不推进步计数 (ts 跨 rank 相等证明 idle 序列全局一致)
- 输出正确性: bench ok=300 errors=0 ×3; meta.txt SP env 校验过; sp_run_end.json 每 H 边界刷新 (pkill -9 后完整)

## P1 事件账本
- 11:36Z DEV 启动; 11:43Z seq1 (W1 b1 F_sparse) VALID t=353.73s (门算术精确 |4·243−970|=2, 15 H 周期, 固定臂零转换, swaps=2 在线规划保持, 门开销 15.1ms CPU)
- 11:50Z seq2 (W1 b1 A) MECH_RETIRE → 驱动 HALT exit 8 → 诊断: checker 判据缺陷 (S1 恒等式误套自适应臂; 运行时 72 行 jsonl 全部机制不变量通过) → 修复 checker (A 臂改账本推导界 [916,950], 固定臂保留精确恒等式; 另修复自引入的 res 变量遮蔽) → 同工件重判 VALID t=352.17s → 完整报告 REPORT_sp_mech_halt_seq2.md → HALT 解除, seq3 续跑
- A 臂 W1 科学发现 (P2 素材): QUIET 边界抖动 6 次 (进入 h_rate 0.0080-0.0086, 1-2 周期后退出 0.0109-0.0126, θ=0.0101 无滞回, z 全程 ≪ z_lo); residency TRACK 47/QUIET 6/RECOVER 19 (共 72 H 周期), 11 次封存, 机会仅减 7.1%
- 11:59Z 驱动重启 (pid 63752) 从 seq3 续跑; 12:07Z seq3 (W1 b1 F_dense) VALID t=352.88s → **W1 b1 三臂齐**: F_sparse 353.73 / A 352.17 / F_dense 352.88 (极差 <1.6s ~0.4%)
- 12:14Z seq4 (W1 b2 F_dense) VALID t=353.34s; 12:21Z seq5 (W1 b2 A) VALID t=353.90s — **A 臂 W1 机制账本跨 block 逐字节复现** (驻留 47/6/19, 16 转换, 11 封存, opps 938, rec_calls 86,856, swaps 5, p2l_version 5 全同) → 冻结 seed 下状态机确定性; 抖动为系统性行为非偶发噪声
- 12:28Z seq6 (W1 b2 F_sparse) VALID t=357.21s → **W1 条件收口 6/6 VALID** (均值: F_sparse 355.47 / A 353.03 / F_dense 353.11)
- 12:42Z seq7 (W2 b1 F_sparse) VALID t=704.56s — 首个 W2 (8000 请求切换负载)
- 12:55Z seq8 (W2 b1 A) VALID t=713.20s — **RECOVER 主导驻留** (T72/Q13/R79), 最长 RECOVER 驻留 37 个 H8 周期 (h119→156, ≈106s 估算墙钟) 完整覆盖代码→中文切换段, 零超时退出 (PD-4 ✓); QUIET 抖动依旧 (11 次进入全部 1 周期即 h_rate 退出); swaps=11 (F_dense 6, F_sparse 5)
- 13:08Z seq9 (W2 b1 F_dense) VALID t=708.18s → **W2 b1 三臂齐**: F_sparse 704.56 < F_dense 708.18 < A 713.20 — 切换负载下排序与直觉相反: 省 4× 记录的 F_sparse 最快, 自适应最慢
- 13:21Z seq10 (W2 b2 F_dense) VALID t=708.53s (与 b1 差 0.35s, 复现性极好); 13:34Z seq11 (W2 b2 A) VALID t=716.14s (b1 复现: fwd 2002 同, 驻留 T66/Q12/R93, 34 转换/23 封存/swaps 15) — **A 臂 W2 两 block 均垫底 (~+1%)**
- 13:48Z seq12 (W2 b2 F_sparse) VALID t=709.37s → **PHASE_END DEV: valid_total=11(+1 重判)=12/12, timed 12/12, INFRA 0/3, RESULT retire 0**
- 块内排序汇总: W1 b1 A<F_d<F_s, b2 F_d<A<F_s (A≈F_dense, 差≤0.7s); W2 b1 F_s<F_d<A, b2 F_d<F_s<A (A 恒垫底 +6.8~8.6s vs F_s)

## P2 分析报告 (完成, 2026-10-18)
- 报告: REPORT_sp_coupling_dev_P2.md — 方案 §9 五问全部回答 + PD-1 §4.3 噪声核查 + 抖动发现 + 限制声明 + 治理审计
- 五问要点: Q1 记录削减真实 (F_sparse 4.03-4.14×, 门算术精确, 固定臂零转换零封存); Q2 CPU 成本可忽略 (门+控制面 ≤0.44s ≤0.06% 墙钟) 但 A 臂 W2 行为成本可见 (~+1% 服务时间, 与加密规划→迁移扰动方向一致, 归因为方向性推断); Q3 首规划事件 @64 步 ≈23s, 首 swap @32 步 ≈11s (三臂一致), RECOVER 收敛典型 32 步, 最长 37 周期覆盖切换段零超时; Q4 W1 三臂 ≤0.7% A≈F_dense, W2 A 两 block 垫底 ~1%; Q5 **建议: 机会不足** — 记录成本天花板 <0.1% 墙钟低于噪声, θ 位于 h_rate 自然中位数且无滞回 → QUIET 驻留不住 (32/32 h_rate 退出), 不对当前形态做正式评价
- PD-1 §4.3: S4 使 W1 h_rate/z 的 CV 放大 1.42-1.55× (0.960 vs 0.678; 0.611 vs 0.393) — 噪声放大真实存在但为放大器非根因; 根因 = θ_h_low(0.0101) 落在自然分布中位 (F_dense p50=0.0092/mean=0.0104; F_sparse p50=0.0103) + 冻结设计无滞回带
- 抖动定量: 4 个 A 运行 QUIET 进入 6/6/11/12 次, 驻留全部 1-2 周期, 32/32 退出均为 h_rate (0.0103-0.0129), 零 z/覆盖/护栏退出; 合计 67 次 sp_seal_S_change; A 臂采样机会仅省 7-8%
- 正向机制证据: RECOVER 加密检测 (H8/W8) 在 W2 两 block 均完整捕获切换段 (最长 37 周期, 无超时) — 可作为"切换响应特性"独立研究
- 治理: 冻结值计时后零修改; 12/12 工件全报告无挑选 (含对 A 不利结果); 阈值仅记录未调

## 发布后只读复核与更正 (作者要求, 2026-10-18, 未改代码未跑 GPU)
- EXPORT_evidence_readonly_20261018.md: 三项证据导出 + 真实 forward 驻留重算
- 更正 1: "record 成本天花板 <0.1% 墙钟" 撤回 — 原式只覆盖门+控制面; record/scatter CPU 有直接实测 (PROF 行): 稠密 ~4% / 稀疏 ~1% / A ~3.7% 墙钟每 rank; GPU 核执行无测量; 服务时间可见份额 ≲0.5% (经验上界, 兑换率 ≤17% 且符号不一致)。Q5 建议不变, 论据更正
- 更正 2: "首规划事件 confirmed_grow@64 步" 标注错误 — confirmed_grow 为 v21 窗口控制事件 (SP 下惰性); 真首规划 = p2l v1 @h2/32 步, 与首 swap 同边界行; swap 从未先于规划
- 更正 3: 驻留补真实 forward 口径 (S1 精确/S4 有界, 账本校验精确): W1 A 75.4%/7.7-11.3%/15.0%, W2b1 58.0%/7.6-11.5%/31.6%, W2b2 53.2%/7.8-11.4%/37.2%
- 补充核实: 32 次 QUIET 退出逐条复核 (sp_h_id 键控, 弃用会在 QUIET 期重复的 window_id 键): 31/32 贴线噪声 (h_rate 1.02-1.52×θ, z<2), 1/32 真实漂移 (W2b2 h73 h_rate 4.3×θ, z 24.7→64.5); 覆盖 32/32 过 (6-10× 余量); 退出→RECOVER 为源码显式行为 (sp_state.py L252-253)
- REPORT_sp_coupling_dev_P2.md 追加 §10 更正节 (原文未替换)

## 当前状态
PHASE: P2 完成 + 发布后只读更正完成 — sp_coupling_v1 开发战役收口 (P0a→P0e→P1 12/12 VALID→P2 报告→更正 §10)
NEXT: 等待作者。决策点在报告 §3-Q5 (论据经 §10 更正): 机会不足 (建议) / 有限修订 (滞回带, 需新冻结) / 正式评价 (数据不支持)。无未决 HALT, 无运行中进程, GPU 空闲。

## REV1 有限策略修订（2026-10-18，作者指令；保留封存结果，非正式战役）

- **一、口径修正（完成）**：REPORT §11 + EXPORT Amendment A1（append-only，原文保留）。
  撤回"服务可见记录成本 ≲0.5%""兑换率 ≤17%"作为上界——F_sparse/F_dense 规划频率
  （H64/W64 vs H16/W16）与首次有效布局更新时间（128 步 vs 32 步）不同，臂间净时间差
  不能识别记录的关键路径成本。保留：per-rank record CPU 账本（~4% 稠密 / ~1% 稀疏）、
  完整服务时间、"不曾建立稳定净收益"事实。32 次退出改按实际触发条件报告
  （E1 组 31 次低余量触发 1.02–1.52×θ；E2 组 1 次已确认漂移 z≥z_lo），
  撤回"贴线噪声"笼统认定。
- **二、最小规则修订（完成）**：REV1_quiet_exit_revision_20261018.md（diff+状态转移表
  T1–T10+优先级 P1>P2>P3+参数出处+稀疏采样影响说明+不变项清单）。
  代码已实施于 sp_coupling_dev 分支（未 push）：sp_state.py（rev 门控 QUIET 退出分支、
  SP_CHANGE_EVENTS、SP_QUIET_EXIT_CONFIRM=2、exit_streak 计数器、哈希版本 sp1r1、
  snapshot/restore 扩展）、config.py（OEPLB_SP_REV，rev=1 要求 adaptive）、
  controller.py（theta_h_act/rev 传参、banner rev/theta_out 字段、jsonl sp_exit_streak 字段）。
  参数：θ_out=v21_theta_r_act=0.011395643112507271（校准冻结对）、m_exit=2（冻结确认惯例）；
  零新数值参数，先于任何 rev1 运行冻结，不从 W1/W2 服务时间反选。
  R4 不变项：SP_TABLE 档位、planner、α、迁移接受、联合提交协议、PD-7/8/9、seed、window_v2.py。
- **三、离线验收（完成，全绿）**：scripts/test_sp_rev1_logic.py（新文件）——
  场景 A 门槛波动/B 持续上升/B' 带内驻留/C 真实变化×7/D 样本不足/E 同时触发×4/H 快照对称
  + init 校验；F rev0 等价回归：4 条封存 A 臂日志（72/72/164/171 行）100% 复现已提交
  转移/streak/prop 字段，proposal_hash 与 v1 原构造逐位一致；G rev1 逐边界裁决：32 次
  封存退出 → 0 次立即转移（9 STAY_band + 23 STAY_streak1，含 W2b2 h73 真实漂移），
  分类表 REV1_replay_classification_20261018.md（含反事实免责）；G2 自由运行冒烟干净。
  共 1996 断言。冻结套件 test_sp_controller_logic.py 重跑：world1=32、world2=40 全绿
  （rev0 集成级无扰动）。
- **四、≤12 启动开发验证协议（完成）**：PROTOCOL_dev_r1_validation_20261018.md——
  A_v1/A_rev1/F_dense × W1/W2 × 2 配对 block；臂序预轮转冻结于 sp_arm_order_rev1.json
  （seed 20261018，同 P1 惯例，生成于任何 rev1 计时之前）；主指标=完整服务墙钟（全成本保留）；
  机制指标=真实 forward 驻留/record 次数/转换（含 P 规则标签）/有效布局更新事件（p2l+swaps）；
  n=2 只报告方向；预声明 Q1 贴线往返减少？Q2 有效 QUIET 驻留延长？Q3 W2 变化响应受损？；
  诚实收口条款：机制改善但时间无优势 → 收口，不扩预算、不开参数网格、不回调参数。
- **执行侧（预备完成，未启动计时）**：integrity_check_sp.py 扩展 A_v1/A_rev1 臂身份
  （mode-based fixed 判定修正；封存 A 工件复检 VERDICT:VALID t=352.17 不变）；
  run_sp_rev1_dev.sh 克隆自封存驱动（SMOKE_R1 2 启动不消耗预算 + DEV_R1 12 启动，
  CSV sha 锁定，新部署锁 deployed_shas_sp_coupling_rev1.txt）；DRY_RUN 双阶段 14 行全通过。
- 状态：交付物齐备（diff+转移表+协议+离线验收），等待部署 rev1 → SMOKE_R1 门 → DEV_R1。

### REV1 执行账本（2026-10-18）

- **SMOKE_R1 门（通过，2/2 VALID，0 INFRA，不消耗 12 启动预算）**：
  seq1 A_v1 boot 84s、bench ok=300 errors=0、t=32.82s、VERDICT=VALID；
  seq2 A_rev1 boot 72s、bench ok=300 errors=0、t=31.86s、VERDICT=VALID。
  rev1 专属验证：banner rev=1/rev=0 + theta_out=0.011396 两臂正确；
  meta OEPLB_SP_REV=1/0；jsonl sp_exit_streak 字段全行存在（300 请求≈39 步
  未离开 TRACK，值恒 0，符合预期）；检查器新臂身份（A_v1/A_rev1）通过。
- **DEV_R1 启动**：16:05:07，驱动 pid 146372，run_list sha 锁定
  af48ec31…（12 行）；seq1 = W1 b1 A_v1 引导中。预算 TIMED 12、INFRA 3；
  MECH/INTEGRITY 失败即 HALT（exit 8）等待作者。预计总时长 ~2–2.5h
  （boot ~85s 缓存态 + W1 bench ~353s / W2 ~710s + teardown）。

### REV1 完整性三分（2026-10-18 17:30，DEV_R1 seq1-8 VALID 期间完成）

**观察(a) 跨战役信号偏移 + 观察(b) A_rev1 b1≠b2 —— 统一根因：boot-draw 非确定性（环境性，封存战役中同样存在，非 rev1 代码缺陷）**

1. 首个 H 周期的 n_fresh_assignments 取离散值 {68534272, 74694656, 80855040, 92405760, ...}，相邻差恰为一个引擎步的记录量 6,160,384。取值由 boot/warmup 时序决定（首个 H 窗口内落入的 warmup 步数与首批 batch 组成）。
2. 封存战役内部同样存在：F_dense b1 h1=74,694,656 vs b2 h1=68,534,272；W2 A b1 h1=80,927,232 vs b2=68,534,272（对应封存 W2 A b1=164行/33转换 vs b2=171/34 的轨迹差）。
3. 新战役：A_v1 b1==b2（同 draw 74,694,656 → 字节一致）；F_dense b1==b2（同 draw 68,534,272）；A_rev1 b1 draw=74,694,656，**b2 draw=80,855,040（+1 步）**。
4. A_rev1 b2 分叉机制完全定位：b2 的 h6/h7/h8 h_rate = 0.009905/0.010024/0.009948 连续 3 周期 < θ_in=0.010096 → quiet_streak≥3 → h8 进入 QUIET；b1 的 h7=0.010447 > θ_in → streak 重置 → 至 h26 才进入（与 A_v1 同）。进入后 b2 于 h10 经 P3（0.012044≥θ_out×2）退出，其后信号再未满足 streak-3 → 62行/2转换 vs b1 71行/10转换。
5. 跨战役"臂间数值交换"（封存 F_dense h2 z == 新 A_v1 h2 z 等）是 draw 匹配的假象：z/h_rate 序列 = f(boot draw, 决策)，同 draw 同信号。
6. 排除代码原因：git diff f11bd01..HEAD 仅 rev1 三文件；rev0 重放 proposal_hash 位一致；F_dense 决策账本跨战役完全一致（62行/3交换/988记录/3 p2l；W2 125/5/1998/5）。
7. 处置：**不触发 HALT**（无完整性失败/机制错误；全部 VERDICT=VALID；根因为环境且封存数据中预先存在）。推论：(i) 封存 P1 数值不能作为新战役的数值基线——协议内 A_v1 臂即为此设；(ii) A_rev1 的 n=2 两 block 实为两个不同信号实现，臂间方向性对比强度削弱，如实报告；(iii) v1 臂同样 boot-敏感（封存 W2 b1≠b2 即为证），非 rev1 引入。

### DEV_R1 执行账本 · 收口（2026-10-18 18:08:21 PHASE_END）

- 12/12 VALID，全部 attempt=1；timed_used=12/12，infra_used=0/3；无 RESULT_RETIRE / MECH_FAIL / INTEGRITY_FAIL / HALT。
- 后四次：seq9 W2b1 A_rev1 712.16（17:28:40）；seq10 W2b2 A_rev1 708.42（17:41:52）；seq11 W2b2 F_dense 708.14（17:55:00）；seq12 W2b2 A_v1 717.19（18:08:21）。
- 完整结果、机制表、协议三问（Q1 是/Q2 3-of-4 延长/Q3 未损害）、boot-draw 完整性三分、honest-closure 判定：见 RESULTS_dev_r1_20261018.md。
- Honest closure：机制改善真实且按设计；时间无稳定优势 → 按协议 §5 如实收口，不扩大预算、不开参数网格、不反选阈值。
