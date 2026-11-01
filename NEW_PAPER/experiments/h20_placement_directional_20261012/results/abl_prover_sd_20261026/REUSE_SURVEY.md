# REUSE_SURVEY — 既有同条件日志复用调查（2026-10-26，只读）

判定标准（全部满足才可复用为第一阶段机制检查）：
〔C1〕code_fixed_v2（async_swapper `2e4652a0…` + controller `4964cfc8…`）；
〔C2〕冻结 rev1 参数（v2p1 knobs + P臂 SP_REV=1 / F臂 1 16 16 SP_REV=0）；
〔C3〕prover 同域 256tok 流（prover_256tok_out1.jsonl，sha `b4034e3b…`）；
〔C4〕完整请求流计时（bench_client_v2，无剔除）+ conc=256。

## A. 调查结论

**无任何既有日志同时满足 C1–C4 ⇒ 第一阶段需新做 2 次独立启动（待授权）。**

| # | 候选 | 路径 | C1 v2代码 | C2 rev1 | C3 prover256tok | C4 全流/256 | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | phase-4 v2 补充 64 boots（s_E3_*/s_E4_*） | final_followup_20261020/phase4_core/boots/ | ✓ | ✓（P臂rev1；F臂即 1 16 16） | ✗ E3/E4 多域流（supplement_matrix_locked.csv 65行实测无 prover/E2 行） | ✓ | 不可复用（域不符）；仅作 v2 机制参考 |
| 2 | main m_E2_*（40 boots：I/E/P/F/A2×8） | final_plan_20261019/main/m_E2_* | ✗ v1 代码（async_swapper `85a2dac8…`，PHASE4_CODE_VERSION_ADDENDUM） | ✓ rev1（P臂；F臂params同 1 16 16 惯例） | ✗ e2_prover_stable = **L512** prover 池 4082条（build_streams.py L30/L228），非 256tok 桶 | ✓（conc256、全流） | 不可复用（代码+域长不符）；**最近邻正证据**，见 §C |
| 3 | sp_coupling_v1 dev/dev_r1（sp_dev_*、sp_r1_*_A_rev1/A_v1/F_dense/F_sparse） | results/sp_coupling_v1/dev{,_r1}/ | ✗（deployed_shas_sp_coupling_rev1.txt 实测：async_swapper `93fa4ccd…`、controller `c5ddda4f…`） | rev1 语义的发源战役（SP_REV 门即此役引入） | ✗ W1/W2 开发流 | 部分 | 不可复用；rev1 状态机可行性先例（§C.2） |
| 4 | author_diag adhoc_P_v2_E3 / adhoc_I_v2_E3 | final_followup_20261020/author_diag_20261021/boots/ | ✓（boot_meta code_async_swapper=`2e4652a0…` 实测） | ✓ | ✗ E3 多域 | ✓ | 不可复用（域不符）；证明 v2 代码曾正常行使 P 臂 |
| 5 | p1_P_H/L_b1-4（prep boots） | final_plan_20261019/p1/ | ✗（round-2 修复前时代） | ✓ | ✗ p1_longio 流 | ✓ | 不可复用；本包用其 jsonl 做**字段语义校准**（只读） |
| 6 | f0_*（smoke） | final_plan_20261019/smoke_f0/ | ✗ | ✓（含 f0_F_2_16_32） | ✗ | 短流 | 不可复用 |

## B. 数据集候选调查（对应 PROTOCOL §1 表格的补充事实）

- prover_256tok_out1.jsonl：2048 行、2048 唯一 prompt（全串去重实测）、全部 max_tokens=1 / temperature=0 / ignore_eos=false、id tok256_0..2047；
- pinned10x_head1024 + pinned10x_tail1024 级联 sha256 == 全文件 sha256（`b4034e3b…`，2026-10-26 实测）⇒ 保序确定性划分；划分命令原文：REPRODUCE_BASELINES.md §2.1 L98-99（head -n 1024 / tail -n 1024）；
- e2_prover_stable.jsonl：4082 行、sha `fa788238…`、build_streams.py L228 `rows("prover","test",512,256,slice(None),O1,idpfx="e2_prover")` ⇒ L512 分桶；
- 生成脚本检索：`prover_256tok_out1` / `tok256_` 在 /workspace/EPLB/OEPLB（py/sh/md 全树）与 sjq datasets 内仅命中使用方（REPRODUCE_BASELINES.md、moetuner/bench_0914.py、moetuner/driver_0914.sh、theory_a100/step*.sh、baselines/a100/*.sh）⇒ 构造脚本 MISSING（登记于 PROTOCOL §1/§6）。

## C. 最近邻正证据（只读引用，不替代新 boots）

### C.1 m_E2_P_b1（v1 代码、L512 prover stable、4082 条、conc256、2026-10-19T17:53Z）
- meta.txt 实测：ds_sha `fa788238…`、nreq=4082、conc=256、bench_timeout=2400、launcher_sha `c2f6b9dd…`（与本包冻结 launch_arm.sh 同 sha）、benchv2_sha `0666824…`（同锁）；
- bench：ok=4082、errors=0、total_time 89.76s、tps 45.5（此处仅登记，不作性能比较）；
- v2p1_rank0.jsonl 全 16 行实读：256 forwards；w1 init（h_rate=0.0904，10×不均衡域的高余量开局）；**w2 首次规划+迁移**（sp_swaps_started=1、det_map_version 0→1，z=0.294）；w3 再迁移（→2）；w4/w7 confirmed_grow；stable_s 0→2 递进；**w9 TRACK→QUIET 转换**（sp_transition=True、S=4、W=64、sp_reasons="quiet_streak>=3: z<z_lo & h_rate<theta_h_low & coverage & no pending/accepted swaps…"）；QUIET 期记录量降为 4 steps/周期、valid_rows=3,080,192=N_MIN//4（PD-1 n_min_eff 实证）、plan_opportunity 多为 False、事件 grow_blocked_headroom；server.log issued 窗计数=16；全部行 committed=True、无 desync；
- 尺度换算：2.09M prompt tokens → 256 steps ⇒ ~8164 tokens/step；本消融流 0.52M tokens ⇒ **~64 steps ≈ 4 个 H16 窗**（PROTOCOL §3 预期来源）。

### C.2 sp_r1_W1_b1_A_rev1（rev1 开发战役、旧 async_swapper `93fa4ccd…`）
- 71 行、10 次状态转换实读：TRACK→QUIET（w26，quiet_streak≥3）、QUIET→TRACK（w28，h_rate≥theta_act 双周期迟滞，rev1 R1/R3）、QUIET→RECOVER（w31，confirmed_grow）、RECOVER→TRACK（w45，2 个干净 H8 周期）等 ⇒ rev1 状态机三类转换均有真机行使先例；
- 价值：证明"完整自适应"臂的机制在真实服务器上可全链路行使；限制：代码非 v2、流非 prover256tok，故只作先例引用。

## D. 对第一阶段的影响

- 两次新 boot 不可省略（C1×C3 组合格局无既有覆盖）；
- 预期管理已写入 PROTOCOL §3：~4 窗尺度下，检测/记录/规划/迁移预期可行使（C.1 w2 先例），调窗转换大概率无空间（C.1 需 ≥9 窗）——未触发按反调参条款 2 原样保留；
- 本调查全部为只读操作，未修改任何冻结文件。
