
## 2026-10-18 (UTC) — 机制实现阶段完成（§14.2 步骤2 前半）
- **5 个臂标志实现并部署**（全部 additive、默认 OFF）：OEPLB_SP_H（F 臂 fixed (S,H,W)）、OEPLB_SP_FORCE_S=1（A3）、OEPLB_RECORD_ALL_PHASE=1（A1）、OEPLB_PAIR_RULE=maxdelta（A2）、OEPLB_NO_COMMIT=1（A4，含 nocommit_planned_ops_total 计数器）。
- **黄金等价测试 30/30 PASS**（C_configs/test_final_plan_flags.log）：旧(HEAD)-vs-新默认关闭，120步 gate/propose/snapshot/S,H,W trace 逐位一致，覆盖 TRACK/QUIET/RECOVER × rev0/rev1/fixed(1,16,16)；rebalancer 默认 gap 规则 plan 逐 op 一致（15 ops，含 overshoot 层与 diffuse 层）；maxdelta 在构造 overshoot 层选最大转移对（slot0/300）而 gap 选 gap/2 对（slot1/100），diffuse 层两者一致。
- **测试发现并修复 2 个实现缺陷**：(1) sp_h=-1 未拒绝（改为显式 ValueError）；(2) 新校验最初插在 validate_sp_coupling 的 mode=='off' early-return 之后会被绕过（移至最前，off 模式下 PAIR_RULE/RECORD_ALL_PHASE 校验生效）。
- **部署**：SRC→DEP 4 文件 cp + py_compile；SRC_DEP_IDENTICAL 确认；window_v2/v2p1 未触碰。
- **新封存锁** C_configs/deployed_shas_final_plan.txt（14 路径 + 头注，sha256sum -c 通过；4 个被补文件新哈希，其余 10 个与 rev1 锁一致）。rev1 锁原样保留。
- **bench_client_v2.py** 写入 scripts/ 并 sha 锁（C_configs/bench_client_v2.sha256）：v1 严格超集（请求机制逐字节不变），新增 per-request jsonl（idx/id/t_send/ttft/t_last/elapsed/comp/prompt_tokens/error）、p95s、total_prompt_tokens、OEPLB_BENCH_OUT。
- **ARM_FLAGS.md**（任务C核心）：5 标志语义表、I/E*/P/F/A1-A4 每臂精确 env、F 臂解释性决定预注册（仅固定 SP 调度，W_EA 自适应保持活动，依据 window_v2 冻结+F_dense 先例+rev1 迟滞仅属 adaptive）、F0 smoke 每臂机制验收判据表。
- 下一步：任务B 数据构建（E1-E4+校准流 manifest，基于 B_data/index 内容哈希去重与冻结划分）→ 任务 A/D/E/F 文档 → 启动器 → F0 smoke。

## 2026-10-18 (UTC) — 任务B 数据构建完成（§14.2 步骤1→2，全部 CPU 侧数据就绪）
- **6 条流构建并预注册**（B_data/streams/ + registry.json，构建规则 R1-R7 见 scripts/build_streams.py docstring，选择只用来源/长度/域/内容哈希，零性能信息，零 RNG，重跑逐位一致）：
  - e1_code_stable N=3221 @2048tok O1（**N 从 4096 依 §5.1 容量下调，冻结前登记**：code 全域唯一 333 docs/1.108M tok）
  - e2_prover_stable N=4082 @512 O1（**N 从 8192 依 §5.1 下调登记**：prover 4200 docs/1.41M tok, p50=285）
  - e3_multidomain_switch 8192=4×2048 块（sharegpt→prover→book→code，文件级切换）@512 O1（目标满额）
  - e4_fullduplex 2048=4×512 @512 out128 temp0 ignore_eos=true（=E3 块前缀，跨实验复用已登记）
  - calib_screen / calib_confirm 各 1280：S1 code稳定256 O1 + S2 prover稳定256 O1 + S3 四域切换4×128 O1 + S4 长输出 sharegpt128+book128 O128（混合阶段/分段、稳定+变化、短+长输出，§4.2 要求；两阶段用不同内容）
- **拆分**：bucket=int(h,16)%100，0-11 screen / 12-24 confirm / 25-99 test，**doc 级**不相交；实测三向文本重叠=0/0/0。
- **质量验证**：每条流内容级唯一（uniq==n）；重分词回环长度精确（e1 2048/2048/2048，其余 512/512/512，零漂移）；池级 token-ids sha 去重。
- 构造流按 §5.1 标注为"真实内容构造的受控请求流"（拼接语料 token 窗口，stride 已登记：code/prover s256=87.5%/50% 重叠披露，sharegpt/book s512 无重叠）；非自然生产分布，不合成路由。
- 产物：doc_manifest.json（13966 docs 溯源）、splits.json、registry.json（含每流 sha256=计时前冻结）、tokcache.pkl（派生缓存不入库）。
- 下一步：任务 A/C/D/E/F 文档 + 每臂启动器 + run_final_campaign.sh 驱动 → F0 smoke。

## 2026-10-18T20:50Z 驱动完成 + F0 smoke 启动（第4条台账）
- `scripts/run_final_campaign.sh`（rev1 模式驱动）：phases F0SMOKE(8 boots, conc64, N=384)/CALIB1(22, conc256, N=1280)；每次 boot 前全锁校验（deps v3 + bench client + CSV sha + stream sha + NROWS）；TIMED_CAP/INFRA_CAP=3/RESULT_RETIRE 不重试/MECH_RETIRE+INTEGRITY_FAIL→HALT exit8/预算越界→HALT；resume 只认 marks TSV 中 VALID；meta.txt + /proc/<pid>/environ 快照(meta_env.txt)。
- 预注册 smoke 例外（启动前写入驱动头部）：F0 细胞不产生任何评价数据，bench 级 RESULT_RETIRE 按 INFRA 类处理（INFRA_CAP 内同细胞重启一次）；MECH/INTEGRITY 永不重启。
- run-list 冻结并 sha 锁嵌入驱动：F0 8 行（I,E(e1),P,F(2,16,32),A1-A4）；CALIB1 22 行 = e0-e6×2 + F 4候选×2，种子轮转 random.Random(20261019*1000+attempt) 首个满足"同候选不相邻"的洗牌（attempt=1），顺序在任何计时前生成。smoke_slice=calib_screen[0:128]+[1024:1280]（384 条：code O1 + longout O128）。
- `C_configs/EPLB_MECHANISM_NOTES.md`：sglang 0.5.6.post2（实测）EPLB 旗标语义/默认值/候选→CLI 映射/日志标记，事前规则记录不搜索。
- 驱动 bug 修复（基础设施类，零数据，marks 已注记）：首次启动 20:46Z 两败即 HALT——(a) csv.writer 默认 \r\n 污染 artifact_dir（目录名带 CR），(b) ART 未导出致 launch_arm.sh `${ART:?}` 即死。修复：CSV lineterminator=\n 重生成+新 sha，launcher 前缀 ART= 环境。无任何评价数据产生。
- 20:48Z 重启 F0SMOKE：f0_I 引导正常（模型加载✓ DeepGEMM warmup✓ CUDA graph 捕获中，8卡~90GB，无 Traceback）。预计 8 cells ≈ 2-3h。

## 2026-10-18T21:32Z F0 smoke 两次 HALT-修复循环（全部 INFRA 类，零计时数据）
- 20:50Z f0_I **VALID**（384/384 ok=384 errors=0，I 臂惰性标记通过，bench t≈35s）——smoke 首细胞通过。
- 20:51Z f0_E_e1 argparse 秒败×2 → HALT。根因1：launch_arm.sh E/F 分支位置参数错位（$1=ARM 自身，'auto' 喂给 --ep-num-redundant-experts）。修复：ALGO/R/PERIOD=${2..4}、S/H/W=${2..4}；新增 LAUNCH_DRY 干跑模式；17 个臂/参数组合 argv+env 全量干跑验证通过（E 旗标、F fixed env、A1-A4 单旗标、P rev1 冻结集）。
- 修复期间次生事故（已注记 marks）：补丁脚本经 heredoc 传输失败未生效 + 无守卫干跑循环按旧 launcher 真实启动了 I 臂服务器；pkill -9 清理（GPU 复 4MiB），零评价数据。后续补丁一律走 /tmp 脚本文件 + 干跑加 timeout 守卫。
- 21:24Z f0_E_e1 二次秒败×2 → HALT。根因2（机制性，事前事实）：本 sglang 版本 --enable-eplb 自动 recorder='stat'，其 gatherer 仅实现 deepep normal/low_latency，auto → NotImplementedError（expert_distribution.py:314）。本机封存先例 d38_L512_O1_comparison（2026-09-10 同模型同栈 EPLB 基线）即 --deepep-mode normal。
- **预注册偏差**（EPLB_MECHANISM_NOTES.md §6，任何 E 计时前）：E 臂基座 deepep-mode auto→normal（机制实现约束+先例沿用，非调参）；I/P/F/A 保持 rev1 封存基座 auto。公平性 caveat（E decode 无 low-latency buffer）进冻结文件与最终报告 limitations；P-vs-I 主对比不受影响。
- 21:29Z 第三次启动 F0SMOKE：f0_I resume-skip（TSV VALID），f0_E_e1 以 normal 引导中。
- E 臂两次失败均未消耗 TIMED 预算（无 BOOT_OK）；按治理属基础设施/机制配置失败，与结果性失败分开记录。

## 2026-10-18T22:02Z F0 smoke 通过（8/8 VALID）→ CALIB1 启动（第6条台账）
- **SMOKE_PASS 8/8**：I（惰性✓）、E_e1（横幅 period=64✓ rebalance 发生✓ deepep normal）、P（adaptive 35窗 7次提交交换）、F(2,16,32)（fixed S/H/W 精确匹配）、A1（decode_record_batches=514>0，轨迹与 P 全同：35窗/7交换/sp_fwd=570）、A2（pair_rule=maxdelta 回显✓）、A3（force_s=1 回显✓ adaptive✓）、A4（nocommit_planned_ops_total=2079>0 且 swaps_started_total=0，轨迹同 P）。
- 机制读数佐证确定性：P/A1/A4 同种子同 sp_fwd_total=570、同 35 窗；F 臂 sp_fwd 同为 570（同流量）。
- 尾部 summary 解析器 bug（mark() 单 tab 字段 vs 解析按多字段）曾误报 SMOKE_GATE_FAIL 0/8；verdict 本体出自 integrity_check_fp.py，不受影响；解析器已修，summary.json 重建，SMOKE_PASS 落盘。零数据影响。
- smoke 节奏实测：~2.3 min/cell（boot ~2min + bench 384reqs conc64 ~40s）。
- 22:02Z 启动 CALIB1（22 cells：e0-e6×2 + F 4候选×2，冻结种子轮转，calib_screen 1280 reqs conc 256，BENCH_TIMEOUT 2400）。预计 ~2-3h。

## 2026-10-18T22:31Z CALIB1 进展 + e6 事件修正（第7条台账）
- CALIB1 22:02Z 启动后 7 细胞连续 VALID（e2_r1/e4_r2/e2_r2/e1_r2/f_2_16_32_r2/f_1_16_16_r1/e3_r1，~2.2min/cell：boot~75s+bench~60-80s）。
- 22:20Z c1_e6_r1（E period1000）MECH_RETIRE → HALT：rebalance 0 次。诊断（INCIDENT_e6.md）：每 rank 各记 1 行，e1 32 行=4 事件、e4 128 行=16 事件 ⇒ 校准流总 forward pass≈256-271 < 1000 ⇒ 机制-工作负载必然，非构建错误；bench 本体干净（1280/1280，t=77.1s）。
- 预注册修正：e6_r1 退役（消耗 1 计时 boot）、e6_r2 剔除（确定性同果）、CSV 重生成 20 行保持冻结顺序、TIMED_CAP 20、e6 不进 stage-1 排序；"默认 period 1000 在 ~270-pass 基准上从不触发"作为事实进冻结文件与报告。e5(period256) 保留（4 个 E 细胞一致 passes≥256，预期触发 1 次；若 0 触发按同流程处理）。
- 22:31Z 重启 CALIB1，7 个 VALID resume-skip 生效。预算：21 boots ≤ 22 screen。

## 2026-10-18T23:05Z CALIB1 完成 20/20 VALID + stage-1 选择冻结 + CALIB2 启动（第8条台账）
- CALIB1 修正后 20 细胞全 VALID（e6_r1 退役另计；e5 两细胞如预测恰触发 rebalance）。节奏 ~2.3min/cell。boot 总消耗 21 ≤ 22 screen 预算。
- **Stage-1 排序（mean wall s，预注册规则=均值升序，仅在均值精确相等时以更小 R/更简单者破平——本次无精确平局）**：
  F 族：f_1_16_16 55.38 < f_4_16_64 56.58 < f_2_16_32 57.02 < f_1_8_8 59.83
  E 族：e0 79.46 < e5 80.33 < e1 80.91 < e3 82.40 < e2 84.79 < e4 94.40（e6 剔除）
  观察（记录用，不作选择依据之外解读）：PB-fixed 族整体快于 sglang-EPLB 族 ~25-40%；E 族内高频 rebalance（e4 period16）最慢，R=0（e0）最快。
- **topE = e0(auto,R0,P64), e5(auto,R16,P256)；topF = f_1_16_16, f_4_16_64** → STAGE1_SELECTION.json（含 tie 审计、轮转规则、sha）。
- CALIB2：8 细胞（topE×2 + topF×2）on calib_confirm（与 screen/test 零重叠），种子轮转 attempt=10，CSV sha 95f738…；驱动新增 CALIB2 phase（TIMED_CAP=8 ≤ 8 confirm 预算）。23:04Z 启动。
- E*/F* 终选规则（预注册）：仅按 stage-2 mean wall；stage-1 名次不参与终选。

## 9. 2026-10-18 23:44Z — 冻结完成，MAIN（168格主矩阵）点火
- CALIB2 验证收官：8/8 VALID（23:22Z）。终选（stage-2 mean 规则，先于 stage-2 冻结）：**E\*=e5 "auto 16 256"（78.63s）、F\*=f_1_16_16 "1 16 16"（56.935s）**；F 次选差 0.045s 近似并列非精确并列，规则直接生效，已记录。
- P1 流构建（复用封存 R1–R4，无 RNG）：p1_longio.jsonl，N=2048 = 4域×512 窗口（TEST split L2048 s256），round-robin 交错，out128/t0/ignore_eos；code 池与封存 e1 前 512 行逐 prompt 比特一致（断言通过）；sha=3c09e027…；retok 2048/2048/2048。registry_p1.json 含 N 档位与 C_hi 容量依据。
- **C_hi=256 冻结**（仅用容量数与 token 预算，先于任何性能终点）：最差臂 E-R32 KV=190,307 tok，每请求 2176 ⇒ 稳态在飞≈87；L=64 全容纳 vs H=256 ≈2.9× 超额订阅（允许排队）。可行性 smoke（E-R32 突发准入）预注册于 P1 计时前，计 M2 预算。
- 冻结文档 `C_configs/FREEZE_final_plan.md`（11 节）：校准表、e6 发现、E 臂 deepep-normal 偏差+公平性警示、流/矩阵/轮转/统计口径（g_b 按 plan §8.2 原文：j=E1–E4 四工作流等权，块级区间，P-vs-I 与 P-vs-E\* Holm）、§9 失败规则、P1 全设计、预算台账（校准 29/30 封闭，smoke 8/8 封闭）、执行顺序、冻结声明。**写入并提交（6eb430f）于 MAIN 首次 boot 之前。**
- 主矩阵 CSV：run_list_main.csv 168 行 sha=c2e06df4…；轮转=种子约束随机构造（每步在臂≠前且条件≠前的剩余格中均匀抽，死局重试），seed=202610190002 attempt=2；结构校验：21 个 (cond,arm) 格各 8 块、全局无同臂/同条件相邻。P1 CSV：32 行 sha=b324f8ea…，seed=405220380001 attempt=1，方法/压力双约束。注：纯拒绝洗牌双约束通过率≈e^-43，构造法等价且确定性，规则先于计时冻结。
- 驱动补丁：MAIN/P1 相位（CSV/流 sha 双锁入 boot 前检查；MAIN 按 cond 映射 DS/NREQ/timeout：E1 3221/2400s、E2 4082/2400s、E3 8192/3600s、E4 2048/3600s，conc 256；P1 按 cond 映射 conc：PL 64/PH 256，timeout 3600s）；相位结束标记修复（MAIN_DONE/P1_DONE/CALIB2_DONE 各归其位）；bash -n + 两相 DRY_RUN 全通过（168+32 行，映射逐项核对）。
- **MAIN 点火 23:44:28Z**（driver pid 498271）。首格 m_E4_P_b4：BOOT_OK 23:45:38Z（70s），bench 运行中（prefill 正常、8 卡 94GB、closed-loop 排队可见）。预计 168 格 18–30h。
- 纪律：GPU 独占（无并发任务）；恢复只认 TSV "VERDICT <label> VALID"；不做中期选参；本地提交不 push。

## 10. 2026-10-19 00:52Z — 事件：m_E4_F_b4 服务器中途崩溃（方法崩溃类），检查器按 §9.9 对齐
- MAIN 前 8 格全 VALID（E1-P 318.1s、E2-F 84.2s、E2-P 84.6s、E3-A4 190.7s、E4-I 89.1s、E4-P 90.4s、E4-A1 91.6/91.3s）。
- seq=9 m_E4_F_b4（F×E4）：bench 62s 时 window#55 发出 300-op 上限 swap 批，batch_isend_irecv（NCCL COALESCED SeqNum=14）永久悬挂 → 600s watchdog → rank0/3/5 Aborted → gloo 级联 → 调度器死亡（前端半死：/health 超时、/get_model_info 200）。机制标记在位一致（fixed 1/16/16，55 窗，13 次迁移启动）⇒ 非 MECH 类。
- **分类（plan §9.9 / FREEZE §7）**：方法崩溃 = RESULT_RETIRE 完成率结果，不重试不删除。关键对照排除“大批次必死锁”：calib F 10 次 VALID 同样打出 300-op 批；同流 P 臂 299-op 批 VALID；差异变量 = F 固定节奏在持续 decode 下的迁移暴露量（62s 内 55 窗）。竞态假设留 M2 诊断（战役后，≤4 boot）。
- **工具对齐修正（非规则变更）**：原检查器把“bench.json 缺失”一律判 INTEGRITY_FAIL→误 HALT；修正为证据门控（server.log 含致命崩溃标记 + bench.json 缺失/不可解析 → RESULT_RETIRE；否则原语义不变）。旧版封存（sha f127d6e6…→C_configs/…pre_amendment…py），新版 sha a44e1e34…；meta.txt 每格记录 integ_sha 自动区分前后。已判 8 格不重判。离线双路径验证通过。
- 矩阵不裁剪：剩余 7 个 E4_F 格照常尝试；若持续崩溃，各格 boot+≤3600s 超时后 RESULT_RETIRE，E4×F 完成率即报告发现；确证比较 P-vs-I/P-vs-E* 不受影响。ETA 影响：每个崩溃格 ≈63min，总 ETA 上调至 ≈21h。
- 详见 `main/INCIDENT_E4F_hang.md`（时间线/根因链/处置/证据清单）。

## #11 — 2026-10-19 ~02:47Z — INCIDENT #2（m_E3_P_b6，P 臂）闭环：swap 迁移竞态泛化，非臂特异

- **事件**: MAIN seq 22/168，m_E3_P_b6（E3 8192req conc=256，P adaptive，block 6）。01:45:44Z BOOT_OK；01:47:20Z window#17 发出 299 swaps → async swapper `batch_isend_irecv` NCCL COALESCED SeqNum=5 永久悬挂；01:57:20Z watchdog（ranks 5/6/0/1）→ DistBackendError → abort 级联，server 死；bench 客户端排空至 ~02:45:42Z SIGTERM（bench.json 缺失）→ **修正版 checker crash-evidence 路径首次实战触发 → RESULT_RETIRE**（4 个 fatal markers；02:36Z 离线演练同判）。verdict 台账: 2026-10-19T02:45:44Z RESULT_RETIRE（SIGTERM ~02:45:42Z 后 2s，checker exit=1 符合预期）。
- **定性升级**: 与事故 #1（m_E4_F_b4，F 臂）同签名。泛化结论（分级见 INCIDENT_E3P_hang.md §3）：**概率性竞态位于 swap 提交路径（batch_isend_irecv 大批次迁移），臂无关**——F、P 均中招；I（无 swap）、A4（no_commit）、E（官方 EPLB 路径）迄今免疫；非流特异（F 过 E3、P 挂 E3、A2 过 E3）。同 run 悬挂前 4 批大迁移成功（268/49/299/259 ops）→ 非确定性。rank 间分歧细节（DP0/DP5 无 issue/TIMING 行但 watchdog fired）移交 M2（≤4 boots，campaign 后）。
- **处置（冻结规则，无咨询）**: §9.9 method crash = 结果 → RESULT_RETIRE；不重试、不裁剪矩阵、不中途改参；完成率按 (cond,arm) 记账。**确证影响**: block 6 对 P 家族不完整 → P-vs-I / P-vs-E\* 完整块 n=7（§8.1 n=8 为预算档非功效保证）。
- **台账**: MAIN 23/168 verdicts = 21 VALID + 2 RESULT_RETIRE（两退役均 method-crash 类，均 swap 提交臂；事故后下一格 m_E4_E_b1 VALID 02:51:08Z，driver 无需干预自愈推进）；0 HALT / 0 FATAL / 0 BOOT_FAIL / 0 基础设施失败。driver pid 498271 健康，已推进下一格。
- **完成率快照（MAIN，截至 22 verdicts）**: I 4V/0R · E 4V/0R · P 4V/1R(E3) · F 3V/1R(E4) · A1 3V/0R · A2 1V/0R · A4 2V/0R
- **文档/提交**: main/INCIDENT_E3P_hang.md（新）；INCIDENT_E4F_hang.md 尾部 addendum；campaign_marks.tsv INCIDENT+NOTE 行；本条目；git LOCAL ONLY 提交 （HEAD，LOCAL ONLY）。
- **风险预告**: 剩余 P/F/A2/A3 格每格携带同类小概率悬挂风险（迄今 2/22 ≈ 9%，均在大迁移批次）。协议内应对=继续跑、如实记完成率；若同臂连续多格悬挂导致确证家族完整块进一步流失，将在最终报告 limitations 中量化，不做中途设计变更。

## #12 — 2026-10-19 ~06:05Z — INCIDENT #3（m_E3_F_b8，F 臂）闭环：E3-w17 "299 异常指纹"确立，M2 关键线索

- **事件**: m_E3_F_b8（E3，F forced 1/16/16，block 8）04:39:05Z window#17 发出 299 swaps → batch_isend_irecv COALESCED SeqNum=5 悬挂 → 04:49:06Z watchdog ranks 1/0/5/6 → abort 级联；bench SIGTERM ~05:36:57Z → 修正版 checker crash-evidence 路径第二次实战触发 → **RESULT_RETIRE 05:37:26Z**（§9.9 method crash，非 HALT，不重试，不裁剪）。单格代价 61min。
- **关键取证（新）**: 同一密封 E3 流下，**所有幸存 w17 通过发出 300 swaps（F-b1/F-b7/P-b8，sp_fwd_total=522），两起崩溃通过发出 299（P-b6/F-b8，崩溃时 fwd=282）**，跨臂一致；A2（185@w17）幸存。→ 规划确定性被打破：偏常 299-op 规划与悬挂强相关（~2/5 P/F 通过出现偏常）；DP0/DP5 两起事故均缺席 w17 日志但 watchdog 触发 → 跨 rank 批次不一致假说。限定：op 数非普适规则（#1 在 E4 以 300-op 悬挂、m_E4_P_b4 以 299-op 幸存）——指纹为流×事件特异。全部移交 M2（≤4 boots，campaign 后）。
- **观察项**: F 臂 sp_rev=0 跨全部 F 格一致（冻结启动行为，rev 语义属 adaptive；公平性保持，仅记录）。
- **累计退役**: 3/53 ≈5.7%（F 2、P 1；I/E/A1-A4 零）。家族影响：确证 P-vs-I/P-vs-E* 仍 n=7（仅 block 6 缺）；次级 P-vs-F* n≤6（block 6 P 侧 + block 8 F 侧缺）。前瞻登记：E3 剩余 swap 臂通过 P×6/F×5/A2×7，若危害率维持 ~1/3 预期再退役 ~3-5 格——协议应对不变（继续跑、记完成率、limitations 量化），无中途变更。
- **台账**: MAIN 54/168 = 51 VALID + 3 RESULT_RETIRE（m_E1_P_b1 VALID 06:07:06Z 于本条目写入时到达；前一提交信息中 "53/168 (50V+3R)" 为本条目前 1 格的旧计数）；0 HALT/0 FATAL/0 BOOT_FAIL。完成率: A1 4V/0R · A2 5V/0R · A3 1V/0R · A4 4V/0R · E 10V/0R · F 5V/2R · I 8V/0R · P 14V/1R
- **文档/提交**: main/INCIDENT_E3F_b8_hang.md（新，含累计登记表）；campaign_marks.tsv INCIDENT 行；本条目；git LOCAL ONLY（HEAD）。
- **误读澄清（记录防混淆）**: 05:40:25Z verdict 实为 m_E2_A2_b6 VALID（E2 流，t=92.0，artifact 完整）；监测中一度误读为 m_E3_A2_b6 并核查无此格——台账/artifact 无异常。

## #13 — 2026-10-19 ~11:50Z — INCIDENT #4（m_E3_F_b4）闭环 + #5（m_E3_F_b3）在途：E3-w17 指纹第 3/4 次精确复现

- **台账**: MAIN 102/168 = 98 VALID + 4 RESULT_RETIRE（新增退役 m_E3_F_b4，09:46:52Z，修订版 checker 第 3 次实弹，分类正确）；0 HALT / 0 FATAL / 0 BOOT_FAIL / 0 基础设施失败。driver pid 498271 健康（12h+），#4 后自愈推进，期间连落 14 个判决无干预。
- **INCIDENT #4（m_E3_F_b4，F 臂 E3）**: 08:48:38Z w17 六 rank（DP1/2/3/4/6/7）issued 299 swaps、DP0/DP5 缺席 → 08:58:38Z watchdog ranks{0,1,5,6} SeqNum=5 COALESCED NumelIn=0 → 08:59:38Z abort 级联 → BENCH_TIMEOUT SIGTERM 09:46:52Z → RESULT_RETIRE。与 #2/#3 指纹逐项相同（w17/299/fwd282/swaps_started=4/DP0-DP5 缺席/SeqNum=5/ranks{0,1,5,6}）。文档 main/INCIDENT_E3F_b4_hang.md。
- **新证据**: #4 与 #5 的 v2p1 w17 行逐字节相同（z=124.79036828239538, s_change_logical=0.4611823558807373, n_fresh=98566144）→ E3 固定流下 w17 检测统计是确定性的；悬挂分叉发生在 plan→issue 层的非确定性环节（跨 rank 计划分歧，299-vs-300 为症状）。M2（≤4 boots，战役后）现有 4 崩溃样本 + 幸存对照可离线取证。
- **INCIDENT #5（m_E3_F_b3，F 臂 E3）在途**: 10:53:20Z BOOT_OK → 10:55:06Z w17 299-swaps（DP0/DP5 缺席）→ 11:05:06Z watchdog → abort 级联（4× Fatal Python error）→ bench 排水中，SIGTERM 预计 ~11:53:20Z，RESULT_RETIRE 预计 ~11:55Z（checker 第 4 次实弹）。闭环时补 INCIDENT 行 + 文档。
- **家族影响（冻结规则）**: P-vs-F* 失 block4（F 侧）→ n≤5；#5 若退役再失 block3 → n≤4。P-vs-I/P-vs-E* 仍 n≤7。E3-F 通道 2V/3R（在途），剩余 F×E3：b2/b5/b6。预测登记：E3 剩余 swap 臂通道按当前危险率预期再 ~2-4 起退役；不改协议、不重试、不裁剪。
- **监测缺口说明（基础设施，非结果性）**: 09:45Z 定时唤醒未触发（会话非空闲/未重入），09:10Z→11:44Z 无人工拍点；driver 全程自愈，所有判决/产物完整落盘，无数据丢失；#4 闭环文书延至本条补写。教训：作者在线拍（本条即"查询"触发）可兜底，唤醒失灵不影响战役完整性。

## #14 — 2026-10-19 ~12:03Z — INCIDENT #5（m_E3_F_b3）闭环：E3-w17 指纹四连，ranks{0,1,5,6} 固定，P-vs-F* n≤4

- **台账**: MAIN 105/168 = 100 VALID + 5 RESULT_RETIRE（新增 m_E3_F_b3，11:53:20Z，checker 第 4 次实弹分类正确）；0 HALT / 0 FATAL / 0 BOOT_FAIL。driver 498271 健康（12h17m），#5 后 68 秒即自愈开下一格（m_E4_E_b6 VALID 11:58:49Z、m_E2_P_b8 VALID 12:01:47Z）。
- **INCIDENT #5 闭环**: 指纹与 #2/#3/#4 逐项相同（w17/299/fwd282/swaps_started=4/DP0-DP5 缺席/SeqNum=5/ranks{0,1,5,6}）。增量结论：崩溃 rank 集四连相同 → 分歧方固定为 DP0/DP5 侧；v2p1 w17 统计与 #4 逐字节相同 → 确定性临界规划态 + plan→issue 层非确定性分叉。文档 main/INCIDENT_E3F_b3_hang.md；M2 可先离线比对幸存(300)/崩溃(299) w17 计划行。
- **家族影响**: P-vs-F* → **n≤4**（blocks 6/8/4/3 失）；P-vs-I/P-vs-E* 仍 n≤7；E3-F 终局 2V/3R，剩余 F×E3 b2/b5/b6。预测：E3 剩余 swap 臂 ~13 通道 × ~1/3 危险率 ≈ 再 3-4 起退役；不重试不裁剪不改协议。
- **ETA**: 剩余 63 格；干净节奏 ~16:30Z，含预期挂起 ~19:30-20:30Z。

## #15 — 2026-10-19 ~14:35Z — INCIDENT #6（m_E4_F_b7）闭环：E4-w37 新指纹变体，通用签名确立；P-vs-F* n≤3

- **台账**: MAIN 128/168 = 122 VALID + 6 RESULT_RETIRE（新增 m_E4_F_b7，13:30:13Z，完成率路径 ok=1024/errors=1024，与 #1 同路径，分类正确）；0 HALT / 0 FATAL / 0 BOOT_FAIL。driver 498271 健康（14h46m），#6 后 67 秒自愈（m_E3_E_b5 VALID 13:34:47Z）。
- **INCIDENT #6 闭环**: E4 fullduplex F 臂，12:50:13Z w37 七 rank issued 299、DP4 缺席 → 13:00:13Z watchdog ranks{2,3,4,5,6} SeqNum=7 → abort 级联 → bench 自然结束 50% 完成率 → RESULT_RETIRE。**通用签名确立**："299-swap 签发 + 1-2 DP 缺席 → P2P COALESCED 批次不匹配悬挂"；E3-w17 家族指纹为流特异子型。F 臂 5/6 过度代表（密集签发日程暴露最高）；#6 v2p1 z=null 窗口的规划输入待 M2 核查。文档 main/INCIDENT_E4F_b7_hang.md（含六事故总表）。
- **家族影响**: **P-vs-F\* → n≤3**（blocks 3/4/6/7/8 失，仅余 1/2/5）——确证家族 P-vs-F* 样本量受悬挂事故实质压缩，最终报告如实呈现（不重跑、不裁剪、不改协议）。P-vs-I/P-vs-E* 仍 n≤7。F 臂完成率 E3 2V/3R、E4 5V/2R。
- **监测缺口说明（基础设施，非结果性）**: 12:30Z 定时唤醒第二次未触发，12:03Z→14:31Z 无人工拍点；driver 全程自愈（期间 23 个判决落盘、#6 完整发生并闭环），无数据丢失；#6 文书由作者"全部完成了吗"拍（14:31Z）补写。两次唤醒失灵模式相同（会话非空闲/未重入），已改由作者在线拍+短间隔重排兜底。
- **进度与 ETA**: 128/168（76%），剩 40 格；干净节奏 ~17:50Z，含预期挂起 ~19:00-20:30Z；其后 P1（32 格）+ 统一分析。

## #16 — 2026-10-19 ~16:56Z — INCIDENT #7（m_E3_P_b3）闭环：E3-w17 指纹五连 + 首次活体 rank 集取证；P-vs-I/P-vs-E* n≤6

- **台账**: MAIN 146/168 = 139 VALID + 7 RESULT_RETIRE（新增 m_E3_P_b3，16:46:18Z，崩溃证据路径第 5 次实弹分类正确）；0 HALT / 0 FATAL / 0 BOOT_FAIL。driver 498271 健康（17h10m），#7 后 71 秒自愈（m_E2_E_b1、m_E4_P_b3 连落 VALID）。
- **INCIDENT #7 闭环**: P 臂 adaptive（TRACK，0 转移）E3 w17 悬挂 15:48:03Z；指纹与 #2-#5 逐项相同（299/fwd282/swaps_started=4/DP0+DP5 缺席/SeqNum=5/ranks{0,1,5,6}）。**首次活体取证**：15:54Z watchdog 前 GPU util 分裂 {0,1,5,6}=100%/{2,3,4,7}=0%，随后 watchdog 在相同 rank 集触发——卡死方=等待缺失 peer 的 4 rank（NCCL 忙等），分歧方恒为 DP0/DP5 侧。P 臂 TRACK 时与 F 臂同日程同命中 → 竞态在 plan→issue 层，与节奏自适应无关。文档 main/INCIDENT_E3P_b3_hang.md。
- **家族影响**: P-vs-I/P-vs-E* → **n≤6**（blocks 6+3 失）；P-vs-F* 仍 n≤3。P×E3 3V/2R，剩 b2/b4/b5。事故登记册 7 起：F 臂 5 + P 臂 2；E3-w17 家族 5 + E4 2；A1/A2/A3/A4/E/I 0 起。预测：MAIN 尾部再 ~1-3 起退役；不重试不裁剪不改协议。
- **ETA**: 剩 22 格；干净节奏 ~18:40Z，含预期挂起 ~19:30-20:30Z；其后 P1（smoke+32 格）与统一分析。

## #17 — 2026-10-19 ~17:25Z — 首次基础设施事件（m_E2_P_b6 BOOT_FAIL→INFRA_REBOOT n=1→VALID）：冻结预案自愈，infra 预算 1/3

- **台账**: MAIN 152/168 = 145 VALID + 7 RESULT_RETIRE；剩 16 格。driver 498271 健康（17h38m）。
- **事件（基础设施类，与结果性失败分开记录）**: 17:04:41Z m_E2_P_b6 **BOOT_FAIL**（战役首次）→ 冻结 driver 预案自动执行：kill_server_quiet → **INFRA_REBOOT n=1** → 17:06:14Z BOOT_OK → 17:07:46Z **VALID**（ok=4082/4082, err=0, t=88.1s, tps=46.3——E2 prover P 臂干净数据）。
- **合规性核实**: INFRA_REBOOT 逻辑在 run_final_campaign.sh L168-177，属 **FREEZE 基线提交 6eb430f（首次 MAIN boot 前预注册）** 的既定行为：BOOT_FAIL 且 infra_used<INFRA_CAP(=3) → 单次重启重试；第二次 boot 失败 → INFRA_MISSING + **halt**（HALT 条件）。本次为预算内自愈，**无计时影响**（boot 失败发生在 bench 开始前，VALID 数据来自干净重跑），未改任何冻结物。
- **infra 预算状态**: **1/3 已用**；若再发生 2 次 BOOT_FAIL 且重启成功仍可继续，第 4 次 boot 失败（或同一格二次失败）→ HALT → 停、写完整报告、等作者。
- **描述性备注（不用于改参）**: m_E2_P_b6 t=88.1s/46.3tps 为 P 臂 E2 又一强格（对照 I 臂 E2 b2 98.8s/41.3tps；P b2 85.4s/47.8tps）。
- **ETA**: 剩 16 格，干净节奏 ~18:55Z；其后 P1 smoke → P1 32 格 → 统一分析。

## #18 (2026-10-19T18:5xZ) MAIN 阶段收官 — MAIN_DONE 18:30:45Z

**168/168 格全部落账：161 VALID + 7 RESULT_RETIRE。** summary_main.json 已由 driver 写出，MAIN_DONE 标记文件在 main/。台账 VERDICT 行数与 168 精确对账。driver 进程正常退出，GPU 全空（8×4MiB）。

- **7 起 RESULT_RETIRE 全部为方法崩溃（§9.9 结果性失败，非 HALT）**：
  - E3-w17 家族 ×5（#2 m_E3_P_b6, #3 m_E3_F_b8, #4 m_E3_F_b4, #5 m_E3_F_b3, #7 m_E3_P_b3）：299 swaps + DP0/DP5 issue 缺席 + COALESCED NumelIn=0 → NCCL 永久挂起 → 600s watchdog。v2p1 w17 统计逐字节相同（z=124.79036828239538）→ 确定性临界 plan 状态；plan→issue 层非确定性竞态。
  - E4 ×2（#1 m_E4_F_b4 w55, #6 m_E4_F_b7 w37）：同一通用签名（299-swap issue + 1-2 DP 缺席）。#6 为首次完成率路径判决（ok=1024/err=1024）。
  - 通用签名："299-swap issue + 1-2 DP 缺席于 issue 日志 → P2P batch_isend_irecv COALESCED 批不匹配 → 永久挂起"。F 臂 5/7 过代表（恒定 dense 1/16/16 节奏）；P 臂自适应 TRACK 同样中招 → 与节奏无关。A1/A2/A3/A4/E/I：零事故。
- **家族影响（冻结规则）**：P-vs-I / P-vs-E* → n≤6；P-vs-F* → n≤3（blocks 3,4,6,7,8 失；仅 1,2,5 存）。E3-F 终局 3V/3R。
- **基础设施事件 ×1**：m_E2_P_b6 BOOT_FAIL 17:04:41Z → INFRA_REBOOT n=1 → VALID（INFRA 预算 1/3；无计时影响，失败在 bench 前）。
- **计时数据完整性**：161 VALID 格全部有 bench.json + checker 通过；无 W/H/S 中途改动；无阈值再选择；无有利重复挑选。
- 完成率表（含 7 退役格）与 (cond,arm) 明细留待统一分析（§6 冻结规则）。
- **下一步**：P1 阶段 — 先执行预注册可行性 smoke（FREEZE_final_plan.md L153-155：boot 最差臂 E-R32，C_hi=256 突发 in2048/out1 切片，验证准入正常后 kill；M2 诊断预算 boot 1/4），smoke 通过 → 启动 32 格计时矩阵；smoke 失败 → HALT + 完整报告 + 等待作者（不静默改设计）。

## #19 (2026-10-19T19:0xZ) P1 可行性 smoke 通过 + P1 计时矩阵启动

**预注册 smoke（FREEZE_final_plan.md L153-155，M2 诊断 boot 1/4，非测量启动）— SMOKE_PASS：**
- E-R32（最差臂）boot 72s（18:58:57→19:00:09Z），KV 容量 **190,315 tokens**（冻结预估 190,307，符合）。
- 512×in2048/out1 切片（p1_longio.jsonl 前 512 行，max_tokens→"1"；计数/波次为执行器细节，预注册未固定，已记录于 meta.txt）@ C_hi=256 突发：**ok=512 / errors=0 / wall=52.0s**，total_prompt_tokens=1,048,576（=512×2048 精确）。
- 准入正常：排队不拒绝、无 OOM、零崩溃标记、服务器存活至 bench 结束、kill 后 GPU 清空。
- **结论：C_hi=256 在最差臂可行 → 按预注册处置进入 P1 计时矩阵。** 产物在 p1_smoke/；台账 SMOKE_BOOT_OK / SMOKE_VERDICT / NOTE ×3（含 kv_tokens 笔误更正：06→190315，完整 kv_line 在 verdict/meta，无数据影响）。
- 执行器：scripts/p1_smoke.sh（提交 b9d1a3b），kill/gpu-clear/env-clear/health-poll 逐字复用冻结 driver 函数。

**P1 计时矩阵启动 19:05:54Z**（driver pid 1582324，phase P1）：
- check_locks 通过（dep/bench/CSV sha b324f8ea…/DS sha 3c09e027…/32 行全验证）；rows=32 timed_cap=32 infra_cap=3（INFRA 预算已用 1/3，来自 MAIN 阶段 m_E2_P_b6 事件——driver 重启后 infra_used 从 0 重新计数，本阶段可用 3；MAIN 的 1 次已记台账）。
- 首格 p1_ER16_L_b1（E "auto 16 256"，PL conc=64）boot 进行中。
- 设计（冻结）：4 臂（P vs E-R0/R16/R32）× 2 压力（PL conc=64 / PH conc=256）× 4 配对块 = 32 格；流 p1_longio.jsonl（in2048/out128，NREQ=2048，BENCH_TIMEOUT=3600）。
- 监测要点：ER32 臂 KV 压力（smoke 已证 256 可行）、H 相排队属设计内（统一 max-running，报告 queue/in-flight）、任何 OOM/崩溃按 §9 分类（结果性 vs 基础设施分开记录）。
- 预计 32 格 ~3-5h；P1_DONE 后进入统一分析（§6 冻结规则）。

## #20 (2026-10-20T04:1xZ) P1 阶段收官 — P1_DONE 01:26:15Z，32/32 全 VALID

**P1 计时矩阵 32/32 格全部 VALID，零退役、零事故、零基础设施事件**（对比 MAIN 的 7 起方法崩溃：P1 流 in2048/out128 上 E 臂周期重平衡与 P 臂自适应均未触发 299-swap 挂起签名）。summary_p1.json 已由 driver 写出，P1_DONE 标记在 p1/。

- 全程 19:05:54Z → 01:26:15Z（6h20m，32 格，均值 ~11.9min/格）。
- 描述性观察（正式统计以统一分析为准）：P-L ~380-404s vs E-L ~959-1059s；P-H ~272-277s vs E-H ~391-424s；P 臂 lat_p95 全面占优；E-R32（最差 KV 臂）在 C_hi=256 下 8/8 格正常，smoke 预证成立。
- **监测缺口（基础设施类，非结果性）**：22:04Z 武装的 ScheduleWakeup 未触发（本会话第 5 次 misfire；前 4 次 09:45/12:30/16:26/17:49Z）。P1_DONE（01:26Z）后统一分析未能自动接续，直至作者 04:0xZ 询问才恢复。driver 自愈运行全程无数据损失；缺口期间 19 格照常完成并落账。
- 下一步：统一分析（§6 冻结规则）→ 最终 REPORT。

## #21 统一分析完成（2026-10-20 ~05:00Z）— 战役闭环成立
- 表格/统计：build_tables.py（168+32 行 sealed 装配，T=total_time_s 与 verdict t 断言全过）+ stats_unified.py（冻结 §6/§8.2）。修复 sp_behavior 抽取缺 decode_record_batches/prof_record_calls 的 bug 后重跑，A1 操纵核验恢复为 exercised（1020–1054 vs P=0）。
- 确证族（Holm α=0.05，complete blocks 1,2,4,5,7,8，n=6）：P-vs-E G=+36.53% [+34.81,+38.27] p=3.77e-08 显著；P-vs-I G=+2.21% [−0.28,+4.75] p=0.0718 不显著。
- 次要族：P-vs-F*（n=3）−0.52% ns → adaptive 增量收益未确立（§15 第一行）；A2@E3 +8.21% [+2.87,+13.82] p=0.0102（max-Δ 劣于 gap 引导，服务层证据）；A4/A1 ns 且操纵成立；A3@E1 标"未行使"（行为计数与 P 完全相同 → 噪声底）。
- P1 描述：PL P 优势 +155…+162%、PH +46…+48%；R 对 E 臂墙钟无清晰效应，KV 容量随 R 单调降（227k→208k→190k）；32/32 VALID 0 错。
- 完成度：MAIN 94.43%（7 退役计 0 保守）、P1 100%。
- 图表：fig_f6_main_matrix / fig_f6_confirmatory / fig_f7_ablations / fig_f8_p1（逐启动点、退役标注不隐藏、CVD 校验调色板 OKLab ΔE + Vienot 模拟、标记形状二级编码）；f6/f7 forest CSV。
- REPORT_final.md 落盘：Table1/Table2、局限性（E 臂 deepep-normal 执行方式警示随 P-vs-E 出现、swap-race 族、小 n、P-vs-I ns）、§15 措辞适用、治理合规、闭环判定。
- **闭环成立**：168+32 全部执行 + 统一分析通过冻结规则（§16 口径）。loop 停止；X1/X2、R64、M2 剩余/M3/M4 留待作者。
