# ARM_FLAGS — final_plan_20261019 机制实现与臂配置（任务C核心）

日期: 2026-10-18 (UTC) · 分支: sp_coupling_dev · 锁: `deployed_shas_final_plan.txt`（14路径，rev1锁原样保留）
等价性证据: `test_final_plan_flags.log`（30/30 PASS；旧(HEAD)-vs-新 默认关闭 120步 gate/propose/snapshot/S,H,W trace 逐位一致，覆盖 TRACK/QUIET/RECOVER × rev0/rev1/fixed）

## 1. 五个新增标志（全部 additive、默认 OFF = 封存 rev1 行为逐位不变）

| env | config 字段 | 代码位置 | 语义 | 使用臂 |
|---|---|---|---|---|
| `OEPLB_SP_H` | `sp_h` (0=H=W) | sp_state.py fixed 块 | fixed 模式 (S,H,W) 解耦 H；校验 H≥1, W≥H, W%H==0, 负数拒绝 | F |
| `OEPLB_SP_FORCE_S` | `sp_force_s` | sp_state.py S property | adaptive 下强制 S=1（状态机/H/W 规则不动）；仅 adaptive，否则 config 拒绝 | A3 (P_denseSample) |
| `OEPLB_RECORD_ALL_PHASE` | `record_all_phase` | controller.py 记录门 L443 | 记录门 `is_prefill → (is_prefill or all_phase)`；decode forward 也进采样记录（仍受 min_record_tokens=32 与 S 门控）；要求 SP mode 开启 | A1 (P_allphase) |
| `OEPLB_PAIR_RULE` | `pair_rule` ('gap'\|'maxdelta') | rebalancer.py 配对分支 | maxdelta=恒取 (最热槽,最冷槽) 最大转移对；合法候选集/接受判据(new_ratio<ratio-0.0005)/tried集/预算全部不变；off 模式也校验（rebalancer 直读 env） | A2 (P_maxdelta) |
| `OEPLB_NO_COMMIT` | `no_commit` | controller.py begin(plan) 处 | plan 照常构建/模拟/记日志，仅跳过 `async_executor.begin(plan)`；swaps_started 保持 0（物理真值）；新计数器 `nocommit_planned_ops_total` 进 sp_run_end；SP QUIET 进入等状态机行为不变 | A4 (P_noCommit) |

不变式：window_v2.py / window_v2p1.py 未触碰（冻结）；θ 不重新拟合；rev1 策略零修改（新分支只在非默认 env 下可达）。

## 2. 每臂精确配置

所有臂共用：模型 Qwen3-235B-A22B-FP8, tp8/dp8/ep8, deepep auto, deep_gemm, fp8, mem-fraction 0.8, cuda-graph-max-bs 128, disable-radix-cache, watchdog 600；bench 客户端 `bench_client_v2.py`（sha 锁 `bench_client_v2.sha256`）；同一请求文件/顺序/采样/并发。

- **I (Identity)**: 基座 launcher，无 `--enable-pb-oeplb`，无任何 EPLB 旗标（自然映射；model_runner PB 钩子因 enable=False 全部旁路，惰性验证见 F0 smoke）。
- **E\* (SGLang-EPLB)**: 基座 + `--enable-eplb --eplb-algorithm <algo> --ep-num-redundant-experts <R> --eplb-rebalance-num-iterations <period>`（校准候选 e0-e6；algo/period 旗标语义在 F0 smoke 用 --help+日志动作计数确认后才冻结）；无 PB 旗标。
- **P (PB-OEPLB-final)**: rev1 冻结 env 原样（run_sp_rev1_dev.sh env_common：EA_ALPHA=0.0, WINDOW_CONTROLLER=21, V21 全参数 KAPPA=3875.1768798828125 / Z_LO=22.643858727592757 / Z_HI=314.14010272527486 / Z_URGENT=942.4203081758246 / THETA_R_LOW=0.010095552687949321 / THETA_R_ACT=0.011395643112507271 / N_MIN=12320768 / M_SHRINK=2 / M_GROW=3 / DWELL 2,2 / U_MAX=4 / W_UNCERTAIN_TARGET=16 / URGENT_EPISODE_CAP=3 / BETA=0.5, EA_W=16, PROBE=off, SP: adaptive S=1 W=16 SEED=20261018 QUIET_STREAK=3 RECOVER_MIN_CYCLES=2 REV=1），五个新标志全默认 OFF。
- **F (PB-fixed\*)**: P 的全部统计/规划/接受/预算路径 + `OEPLB_SP_MODE=fixed OEPLB_SP_S=<S> OEPLB_SP_H=<H> OEPLB_SP_W=<W> OEPLB_SP_REV=0`（候选 (1,16,16),(2,16,32),(4,16,64),(1,8,8)）。**解释性决定（预注册）**：F 只固定 SP 调度 (S,H,W)；window_v2p1 的 W_EA 自适应保持与 P 相同地活动——依据 (a) window_v2.py 冻结不改的治理约束，(b) 已封存 F_dense 先例即在 W_EA 活动下运行，(c) rev1 迟滞属 adaptive 状态机、对 fixed 不适用（SP_REV=0）。
- **A1 = P + `OEPLB_RECORD_ALL_PHASE=1`**（E4 专用）
- **A2 = P + `OEPLB_PAIR_RULE=maxdelta`**（E2/E3 专用）
- **A3 = P + `OEPLB_SP_FORCE_S=1`**（E1 专用，P_denseSample）
- **A4 = P + `OEPLB_NO_COMMIT=1`**（E3 专用，P_noCommit）

消融嵌入位置遵循总方案 §8：A1→E4, A2→E2+E3, A3→E1, A4→E3。

## 3. F0 smoke 机制验收判据（每臂启动正确性）

| 臂 | 判据（服务器日志/JSON） |
|---|---|
| I | 无任何 `[PB-OEPLB]` 日志；无 EPLB rebalance 日志 |
| E | EPLB 实际发生 rebalance（动作计数 > 0，algo/R/period 与配置一致） |
| P | sp_run_end 正常、swaps_started ≥ 0、v2p1_rank0.jsonl 存在 |
| F | 日志 S,H,W == 配置值且不随时间变化 |
| A1 | decode 相位记录数 > 0（record 日志出现非 prefill 批次） |
| A2 | 配对差异可观测（ overshoot 层选最大转移对；对照 P 的 plan 日志） |
| A4 | `nocommit_planned_ops_total > 0` 且 `swaps_started_total == 0` |

## 4. 修改文件清单（SRC→DEP 同步部署，py_compile 通过）
sp_state.py（3+1处）· config.py（字段/env/校验重排至 off-early-return 之前）· rebalancer.py（import os + 配对分支）· controller.py（SpController 参数、标志初始化、记录门、no-commit 分支、run_end 计数器）。补丁脚本与测试存档于 `scripts/patch_final_plan.py`, `scripts/patch_fix2.py`, `scripts/test_final_plan_flags.py`。
