# PROTOCOL — prover同域调度消融（abl_prover_sd_20261026）

状态：**预注册完成（2026-10-26），未启动GPU**。执行前须作者明确授权（长期指令）。
性质：非正式准备包。第一阶段 = 机制行使检查（2次独立启动），**不产生任何性能优劣结论**。
治理：不做 git commit；本目录 LEDGER_abl.tsv 只追加；冻结文件（launch_arm.sh、FREEZE_record_v2p1.md、supplement_matrix_locked.csv、bench_client_v2.py）一律只读引用，本包不修改任何一行。

---

## §0 版本与参数锁（code_fixed_v2 + 冻结rev1）

部署代码（2026-10-26 live 复核，与 phase-4 v2 冻结逐字节一致）：
| 组件 | sha256 | 说明 |
|---|---|---|
| pb_oeplb/async_swapper.py | `2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f` | v2 race-fix（FIX-3 + DIAG-2） |
| pb_oeplb/controller.py | `4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697` | v2 冻结 |
| scripts/launch_arm.sh | `c2f6b9dd6e57c64d48379b8bf7b6c012df0bd9c48c9bd254f5ca16299d5f8ede` | live（final_plan_20261019/scripts/）与证据包 final_plan_scripts 副本同 sha |
| scripts/bench_client_v2.py | `066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c` | 与 C_configs/bench_client_v2.sha256 锁一致 |

每 boot 前四重 sha 守卫（stream + v2双sha + launcher + client），任一不符 ⇒ BOOT_ABORT rc=2，不启动服务。

rev1 冻结参数：**全部经由 launch_arm.sh 注入，本包零改动**：
- pb_env_common：EA_ALPHA=0.0；OEPLB_WINDOW_CONTROLLER=21；v2p1 冻结 knobs（FREEZE_record_v2p1.md，位于 adw_v2p1/r3_logging/，do-not-edit）：KAPPA=3875.1768798828125、Z_LO=22.643858727592757、Z_HI=314.14010272527486、Z_URGENT=942.4203081758246、THETA_R_LOW=0.010095552687949321、THETA_R_ACT=0.011395643112507271、N_MIN=12320768、M_SHRINK=2、M_GROW=3、DWELL_SHRINK=2、DWELL_GROW=2、U_MAX=4、W_UNCERTAIN_TARGET=16、URGENT_EPISODE_CAP=3、BETA=0.5；OEPLB_PROBE_MODE=off；EA_W=16；OEPLB_V21_LOG=$ART/v2p1_rank0.jsonl。
- P 臂（完整自适应，rev1）：OEPLB_SP_MODE=adaptive、SP_S=1、SP_W=16、SP_SEED=20261018、SP_QUIET_STREAK=3、SP_RECOVER_MIN_CYCLES=2、**SP_REV=1**。
- F 臂（固定 1/16/16）：SP_MODE=fixed、SP_S=1、SP_H=16、SP_W=16、**SP_REV=0**。

SP_REV 口径（解决准备期悬置问题，代码级依据）：config.py L243 守卫 `OEPLB_SP_REV=1 requires SP_MODE=adaptive` —— rev1 只修订自适应状态机（REV1_quiet_exit_revision_20261018.md）；固定臂无状态机（sp_coupling MAPPING PD-11），SP_REV=0 是唯一合法值且由启动断言强制（PD-12 拒绝清单）。因此"冻结rev1参数"对 F 臂的含义 = 共享同一 v2p1 冻结 knobs、同一采样门代码路径（PD-11 公平性条款）、同一 H/W 计数结构；不存在 F 臂设 SP_REV=1 的合法形态。

服务器参数：launch_arm.sh BASE+PB 原文不动（tp8/dp8/ep8、dp-attention、deepep auto、deep_gemm、fp8、mem-fraction 0.8、cuda-graph-max-bs 128、disable-radix-cache、watchdog 600；PB：threshold-ratio 1.02、min-prefill-tokens 256、sync-window 16、max-total-swap-layers 94、max-swaps-per-layer 64、min-swap-ops 8、max-total-ops 300、decay-factor 0.0、window-floor 8、window-shift-confirm 2）。环境文件 /workspace/logs/env_235b.sh（12 exports，live 复核存在）。MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8。

## §1 数据集冻结（对应指示第1点）

**选定主数据集：prover_256tok_out1.jsonl 全量 2048 条**（机器可读记录见 DATASET_FREEZE.json）
- 路径：/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
- sha256：`b4034e3b67ac2925134f123da706117721b7fbfbee47cbf8658f752b6d1d257a`（1,742,640 bytes，2048 行）
- 顺序：文件原生顺序冻结（id tok256_0 … tok256_2047），不洗牌、不截断、不重排
- 请求数 N=2048；**O=1**（实测全部行 max_tokens=1）；temperature=0、ignore_eos=false（实测全量一致）
- 并发 256（bench_client_v2 信号量）
- **唯一 prompt 数：2048**（全字符串去重实测）
- **重复方式：无**——不循环、不重采样、每条唯一；N=2048 即 2048 个不同 prompt 各发一次
- 与 §5.3.1 关系：论文同域 bench 用本文件前 256 行（N=256）；本消融用全量 2048 = 同域同分布 8× 加长
- **pinned10x 含义**：identity 布局下 per-layer 专家负载 ~10× 钉死不均衡（EXPERIMENT_TRACE_INDEX.md L104"pinned, 10×per-layer不均衡, 256tok"；moetuner_baseline_20260916/EXPERIMENT_LOG.md L30"pinned 10× per-layer imbalance"）
- **pinned10x 构造来源**：
  - 数据集预存于 OEPLBdata 旧目录（sjq README.md："single_domain/ 来自OEPLBdata旧目录"），源自 DeepSeek-Prover-V1 256tok 长度分桶；
  - 派生划分实证（2026-10-26 本机验证）：pinned10x_head1024.jsonl = `head -n 1024`（sha `10660f2e…`），pinned10x_tail1024.jsonl = `tail -n 1024`（sha `c760e36b…`），head+tail 级联 sha256 == 全文件 sha256 ⇒ 保序确定性划分（划分命令原文见 REPRODUCE_BASELINES.md §2.1 L98-99）；
  - 10× 性质的量化佐证：MoETuner profile P_pinned_fair.npz（P[94,128]，layer0 imbalance 4.093，ILP 可解至 mean_imbalance 1.000）+ baseline_comparison_20260914/rt_prover/ 路由轨迹（32 个 rank×chunk npz）；
  - **生成脚本：MISSING**——本机（/workspace/EPLB/OEPLB 全树与 sjq datasets）仅检索到使用方引用（REPRODUCE_BASELINES.md、bench_0914.py、driver_0914.sh、theory_a100/*、baselines/a100/*），无构造脚本归档；如实登记，不重建、不推测具体挑选算法。

候选调查（为何不选，详见 REUSE_SURVEY.md §B）：
| 文件 | 行数 | sha256前12 | 排除理由 |
|---|---|---|---|
| e2_prover_stable.jsonl | 4082 | fa788238fa2d | **L512** prover 池（build_streams.py L30/L228："full prover L512 test pool, O1"），非 §5.3.1 的 256tok 同域桶 |
| pinned10x_head1024/tail1024 | 各1024 | 10660f2e/c760e36b | 主数据集真子集；直接用全量 2048 更长且免划分语义 |
| L512_O1_realprover_head2048/tail6144 | 2048/6144 | 6267b7e4/bf396a59 | L512 桶，域长不匹配 |
| prover_2048tok_out1.jsonl | 2048 | c628fc00522b | 2048tok 长 prompt 桶，非 256tok 域 |

## §2 臂定义与公平性（对应指示第2点）

| 臂 | label | launch_arm.sh 调用 | 语义 |
|---|---|---|---|
| 完整自适应 | `apsd_P_b1` | `launch_arm.sh P`（PARAMS 空） | TRACK(1,16,16)↔QUIET(4,16,64)↔RECOVER(1,8,8) rev1 状态机 |
| 固定 1/16/16 | `apsd_F_1_16_16_b1` | `launch_arm.sh F 1 16 16` | S=1,H=16,W=16 恒定，无状态机，SP_REV=0 |

两臂公平性由同一冻结工具链保证（本包不新写任何调度/环境逻辑）：
- **相同初始布局**：两臂均不注入任何离线 placement（BASE/PB 无 placement 参数；ep_dispatch_algorithm=static 默认，s_E3_P_b1 ServerArgs dump 实证）⇒ 均从框架默认 identity 布局出发；
- **相同启动**：同一 launch_arm.sh（sha 锁）、同一 env_235b.sh、同一 MODEL/端口；boot 间 env_clear（清 OEPLB_*/EA_*）+ kill_server_quiet（<1000MiB 门，与 phase-4 冻结脚本同款 pkill 模式，不自匹配）；
- **相同预热**：服务端不加 --skip-server-warmup（sglang 默认 warmup，两臂一致）；bench 客户端 warmup 仅 /health×conc（bench_client_v2 原文注释：zero model-forward / OEPLB-state impact）；
- **完整请求流计时**：bench_client_v2 的 t_start 在第一条请求发出之前、wall 在全部 2048 条完成之后；**无任何剔除**——在线学习、记录、状态转换、规划、迁移全部落在计时窗内；服务端机制时间线由 v2p1_rank0.jsonl（每 H 边界一行）独立记录，mono/wall 时间戳入 bench json；
- 执行顺序预声明：`apsd_P_b1` → `apsd_F_1_16_16_b1`（按作者列举顺序）；两 boot 完全独立（GPU 状态清零后重启），顺序无残留效应；continue-on-fail：任一 boot 失败仍尝试另一 boot，失败如实登记；
- bench_timeout=1800s（参照：phase-4 E2/N=4082 用 2400、实测 bench 89.76s；本流 N=2048 更短）。

## §3 第一阶段机制检查（对应指示第3、4点）

**目的**：仅核查机制是否实际行使。**不判断性能优势**——本阶段不产生、不引用任何 tps/时延对比结论；bench 数值仅作为完整性记录（MECH_CHECK.json 的 bench_record_no_judgment 字段）。

**复用调查结论**（详表见 REUSE_SURVEY.md）：**不存在**同时满足〔code_fixed_v2 双sha + 冻结rev1 + prover 256tok 同域流 + 完整流计时 + conc256〕的既有日志 ⇒ 需新做 2 次独立启动。最近邻正证据 m_E2_P_b1（v1 代码）仅作机制先例引用，不可替代 v2 检查。

**核查项与提取规范**（mech_check.py；字段语义已用 p1_P_H_b1 与 sp_r1_W1_b1_A_rev1 实测校准）：
| 核查项 | 数据源字段 |
|---|---|
| 真实 forward 数 | v2p1 `true_steps_total`（累计，单调性实测通过）max + 每边界 `steps_actual` |
| 有效检测周期 | H 边界行数、`committed` 行数、`event` 值分布、`alert_k`/`z`/`h_rate`/`stable_s` 轨迹、`desync`/`lockstep_ok`/`h_min==h_max` 完整性 |
| S/H/W 轨迹 | (`sp_s`,`sp_h`,`sp_w`) 逐行游程编码 + `sp_transition` 行清单（window_id、before、after、`sp_reasons` 原文）+ `sp_residency` 终值 |
| 实际记录量 | Σ`sp_recorded_steps`、Σ`sp_sample_opportunities`、Σ`sp_valid_rows`、单周期最大 `sp_valid_rows`（QUIET 期应见 N_MIN//4=3080192/周期，E2 实测先例） |
| 规划次数 | `sp_plan_opportunity`=True 行数、`s_change_logical` 行数、`det_map_version` 终值与递增次数 |
| 迁移次数 | Σ`sp_swaps_started`、`sp_swap_pending` 行数、server.log `[PB-OEPLB] window#N: issued` 窗数、PLANHASH 8-rank 一致窗数 |

**事先尺度预期（诚实登记，非门槛）**：以 m_E2_P_b1 为类比（L512×4082=2.09M prompt tokens → 256 forwards → 16 个 H16 窗；steps≈总token/8192），本流 2048×~256tok ≈ 0.52M tokens ⇒ 预计 **~64 forwards ≈ ~4 个 H16 窗**（±50%）。E2 先例显示：首次规划+迁移最早可在第 2 窗发生（sp_swaps_started=1、det_map_version→2），TRACK→QUIET 转换发生在第 9 窗（144 steps，需 quiet_streak≥3 个有效周期）。因此预期本检查中**检测/记录/规划/迁移可行使，而自适应调窗（QUIET/RECOVER 转换）大概率无行使空间**——这属于第4点预先授权的合法结果。

**反调参条款（预注册，对后续所有阶段有约束力）**：
1. 两次 boot 仅用于机制检查；本轮不产生任何性能优势陈述；
2. 若固定长度（N=2048）内自适应臂未发生调窗/状态转换，**结果原样保留**，登记为"未触发"，与触发情形同等有效；
3. **禁止**为促成触发而调整任何阈值/knob（v2p1 与 SP 参数已冻结，本包无修改路径）；
4. **禁止**反复延长流或重复重跑直到触发；单一固定长度 N=2048，一次执行；
5. 未来若需更长流/更多窗，必须以**新预注册**（新协议+新 sha 锁+作者批准）形式进行，不得作为本检查的"重试"；
6. mech_check.py 的 rc 只反映基础设施完整性（bench 行数、sha 锁、无 watchdog/desync），**机制事实（窗数、转换、swap 数）永不影响 rc**——杜绝"必须触发才算通过"的隐性压力。

## §4 第二阶段预声明（对应指示第5点）

- **正式重复数**：待第一阶段两次 MECH_CHECK.json 落盘并登记后，依据机制事实（forward 数、窗数、是否触发、记录量）另行确定并预注册；本轮不确定、不预告具体数值；
- **固定参照档（现在声明，先于任何本轮数据）**：正式消融的固定参照臂 = **F (S,H,W) = (4,16,64)**——与 sp_coupling PD-11 的 F_sparse 及 rev1 状态表 QUIET 三元组一致，代表"固定稀疏采样"档；该档**不根据第一阶段的时间/触发结果选择或调整**；
- 读法登记：作者原文"固定4/16/64"按单一参照臂 (S=4,H=16,W=64) 理解（与 PD-11/QUIET 三元组一致）；若作者本意为三档 S∈{4,16,64}（H=W=16），可在第二阶段启动前书面改判——两种读法都属事先声明，均不依据数据选择；
- 正式阶段（若启动）将另行预注册：重复数、臂集合（P 自适应、F 1/16/16、F 4/16/64、及作者指定臂）、是否纳入 identity 参照、统计校正（Holm）；本包不预先执行任何正式统计。

## §5 交付物与执行步骤（授权后）

```
abl_prover_sd_20261026/
├── PROTOCOL.md            # 本文（预注册）
├── DATASET_FREEZE.json    # 数据集冻结机器可读记录
├── REUSE_SURVEY.md        # 既有日志复用调查（含最近邻明细）
├── run_abl_boot.sh        # 单 boot 执行器（四重sha守卫→boot_meta→launch→ready→bench→mech_check）
├── queue_abl.sh           # 两 boot 序列（顶部 AUTHORIZATION REQUIRED；P→F）
├── mech_check.py          # 完整性断言 + 机制事实提取（无 swap/调窗门槛）
├── LEDGER_abl.tsv         # 本包账本（只追加）
└── boots/<label>/         # 运行产物（执行后生成）：server.log、bench.log、<label>.json、
                           # <label>_perreq.jsonl、v2p1_rank0.jsonl、boot_meta.json(.end)、MECH_CHECK.json
```
授权后执行：`bash queue_abl.sh`（串行 2 boot；单 boot ≈ 服务器装载 10–15 min + bench 数分钟 + 收尾；总量半天内）。执行完成后：LEDGER 追加 + 向作者提交机制检查报告（触发/未触发均如实登记），再进入 §4 第二阶段决策。

## §6 缺失项登记（本轮新增/沿用）

- prover_256tok_out1.jsonl 生成脚本：**MISSING**（§1；文档链与量化佐证已归档）；
- 实验时刻 CPU/主机内存快照：**MISSING**（沿用证据包 E-index 登记；boot 的 server.log 含启动时 uname/主机信息，不新增快照要求）；
- m_E2_P_b1 类比预测基于 v1 代码与 L512 流，仅作尺度参考，**不作为本检查的任何门槛或预期承诺**。
