# LOCK record — ADW v2.1 r3 calibration logging campaign (pre-run version lock)

锁定时间：2026-10-17（首次 logging run 启动之前）。
本文件按 calibration_plan_v2p1.md（r3，预注册于 commit 51b83e6）与用户
本轮指令第 1 条要求，在任何 v2.1 logging 数据产生前锁定全部版本信息。

## 1. 代码版本

- git 仓库：/workspace/EPLB，分支 main，HEAD = **51b83e6**
  （"ADW v2.1 r3 (pre-freeze): communicator-fatal boundary + calibration
  plan r3 pre-registration + overhead decomposition"）。仅本地，不推送。
- 工作区状态：跟踪文件干净；仅存在与本战役无关的既有 untracked 文件
  （cdx0_v1/capture_trigger.json、ea_v1 logs、benchmarks/results/*.json）。

## 2. 实际加载文件的完整 sha256（部署于 site-packages）

部署目录 `$PB = /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb`：

| 文件 | sha256（部署后实测 == 源码 $SRC=/workspace/EPLB/OEPLB/src） |
|---|---|
| controller.py | `24ccf3048544ed06cbc4f5342c2c418264688c8c9996cb5deb31ab3347ed7c86` |
| config.py | `66b0760d3e7169bd17ebe3f71818826b9eaf19e9f1e059266279e44e1bb09b16` |
| window_v2p1.py | `d20d340f9d44ae997bbb8f2c311bed40e84673173eba90b9da1658374353fdb8` |
| v2p1_signals.py | `98a13608ac56f0f132daa8d189c9aebfee4e62b6633f009f28c3ca2b66e2c436` |
| window_v2.py（冻结 v2，**未触碰**） | `810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74` |

- 部署前 `$PB` 为冻结 v2 状态（controller `2a4dedb2…`，config `180fa41e…`），
  已备份至 `$PB/pre_v21_backup_20261017/`；冻结 v2 另有
  `$SRC/locked_v2_frozen_20261016/`（MANIFEST 完整）与 git 历史双保险。
- `$PB/__pycache__` 中 controller/config/window_v2p1/v2p1_signals 的陈旧
  .pyc 已删除；部署后 `py_compile` 全部通过（COMPILE_OK）。
- 导入验证（logging 惰性 knob 环境下）：controller/config/window_v2p1/
  v2p1_signals 导入 OK；`WindowControllerV2P1(8, knobs)` 与 `(64, knobs)`
  实例化 OK（CTRL21_INIT_OK / W64_OK）。

## 3. 数据集

- `$EXP/inputs/fullservice_v2/workload/calibration.jsonl`（W0）
- sha256 = `7d53aa0b07dbc9f11f47af44ee4f4106861f5ace6d4eeca1ba41cdf5b01b17f9`
- 行数 = **8037**（与 v2 校准/Phase A/B 同一文件，未改动）。

## 4. 启动链文件 sha256

| 文件 | sha256 |
|---|---|
| `$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh` | `05b5c32f90f6374cad000f0d3a32a62da9a23b82cefdb03e4add114699d67967` |
| `/workspace/EPLB/OEPLB/scripts/run_grid_bench.py`（bench 客户端） | `2d56c9778a5dea3956fffe44b3d5e50b7b7fbd4227c025007ad1e9950d23ddb0` |
| `/workspace/logs/env_235b.sh`（NVSHMEM/NCCL 环境） | `26242c03be2bc1f5b62e3b87662acad6720a0ed5e8b674bd3bedf9ab2eaa2eb1` |

驱动脚本：`$EXP/scripts/run_v2p1_logging.sh`（本战役新建；流程逐字沿用
run_adw_v2_calib_fit.sh 模板：kill_server → export env → setsid nohup
launcher → 等待 "ready to roll"（600×4s，遇 Traceback/OOM/RuntimeError
提前失败）→ `timeout 3600 python3 run_grid_bench.py <tag> <dataset> 256`
→ 收集产物 → kill_server + gpu_clear_wait）。

## 5. 校准模式（4 次 logging run）启动参数

每次 run（预注册顺序 **W=8 → 16 → 32 → 64**，每次一遍 W0=8037 请求）：

```
OEPLB_WINDOW_CONTROLLER=21
OEPLB_V21_KAPPA=1.0
OEPLB_V21_Z_LO=1e12  OEPLB_V21_Z_HI=2e12  OEPLB_V21_Z_URGENT=3e12   (见 §7 偏差说明)
OEPLB_V21_THETA_R_LOW=1e-12  OEPLB_V21_THETA_R_ACT=2.0
OEPLB_V21_N_MIN=0
OEPLB_V21_M_SHRINK=2  OEPLB_V21_M_GROW=3
OEPLB_V21_DWELL_SHRINK=2  OEPLB_V21_DWELL_GROW=2
OEPLB_V21_U_MAX=4  OEPLB_V21_W_UNCERTAIN_TARGET=16
OEPLB_V21_URGENT_EPISODE_CAP=3  OEPLB_V21_BETA=0.5      (规格默认值)
OEPLB_V21_LOG=$OUT/W<W>/v2p1_rank0.jsonl
OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
EA_W=<W>  EA_ALPHA=0.0
unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI   (D 关闭、planner 不变)
```

launcher 服务参数（launch235b_eacampaign.sh，不变）：`--tp 8 --dp 8
--ep-size 8 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode
auto --moe-runner-backend deep_gemm --dtype bfloat16 --quantization fp8
--mem-fraction-static 0.8 --cuda-graph-max-bs 128 --port 30000
--trust-remote-code --disable-radix-cache --watchdog-timeout 600
--enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02
--pb-oeplb-min-prefill-tokens 256 --pb-oeplb-sync-window ${EA_W}
--pb-oeplb-max-total-swap-layers 94 --pb-oeplb-max-swaps-per-layer 64
--pb-oeplb-min-swap-ops 8 --pb-oeplb-max-total-ops 300
--pb-oeplb-decay-factor ${EA_ALPHA} --pb-oeplb-window-floor 8
--pb-oeplb-window-shift-confirm 2`（α=0；W_init=EA_W 即该次固定 W）。

bench：`run_grid_bench.py v21calib_W<W> calibration.jsonl 256`
（文件序、Semaphore(256)、/health 预热、流式 TTFT/TPOT——与 v2 完全相同的
发放语义；预热不触碰 OEPLB 状态）。

## 6. 日志协议版本 ↔ 合格窗过滤器一致性

- 部署后 controller.py 中 `protocol_rev` 写入值为 `"r3"`（grep 实测 2 处：
  正常窗记录 + fatal_collective 记录），无 `det_ok` 字段（r3 已移除）。
- 合格窗过滤器（计划 §2.1，离线分析将逐字执行）：`committed==true` ∧
  `protocol_rev=="r3"` ∧ `event ∉ {init, uncertain_carry, uncertain_hold,
  forced_shrink_uncertain, confirm_hold, confirm_fail_freeze,
  w_desync_freeze, fatal_collective}` ∧ `z != null`。
- **一致**：控制器写 r3 ↔ 过滤器要求 r3；event 枚举与
  window_v2p1.py/window_v2.py 常量一一对应。

## 7. 与计划文本的一处可执行性偏差（启动前记录，非事后）

计划 §1 惰性 knob 写作 `Z_LO=Z_HI=Z_URGENT=1e12`。window_v2.py 构造器
校验要求 `0 <= z_lo < z_hi <= z_urgent`（**严格** z_lo<z_hi），三值相等
会被拒绝（ValueError，实测）。改用 `Z_LO=1e12, Z_HI=2e12, Z_URGENT=3e12`：
惰性意图完全保留——κ=1.0 下 z 的可达上界 ≈ s_change·n_layer ≤ 2×1.05e6
≈ 2.1e6 ≪ 1e12，alert/midband/紧急路径依旧不可达。此偏差在任何 logging
数据产生前记录；不改变 §2–§4 任何规则与门槛。

## 8. 行为验证模式（第 5 次 run，占位）

冻结 knob 将由 FREEZE_record_v2p1.md 确定后**追加记录于此**（EA_W=16、
W_init=16、α=0、D 关闭、其余服务参数与 §5 相同；OEPLB_V21_* 取冻结值、
N_MIN=12320768、W_UNCERTAIN_TARGET=16）。追加时间必须晚于参数冻结、
早于第 5 次启动。

### 8.1 追加记录（冻结后、第 5 次启动前）

- 追加时间：2026-10-17T07:26:45+00:00（晚于 FREEZE_record_v2p1.md 冻结时间 2026-10-17T07:25:26+00:00，
  早于第 5 次启动）。
- 冻结 knob（程序化读取 fit_results_v2p1.json 填入，无手抄；已断言与
  FREEZE_record_v2p1.md §1 表逐字一致）：
  - OEPLB_V21_KAPPA=3875.1768798828125
  - OEPLB_V21_Z_LO=22.643858727592757
  - OEPLB_V21_Z_HI=314.14010272527486
  - OEPLB_V21_Z_URGENT=942.4203081758246
  - OEPLB_V21_THETA_R_LOW=0.010095552687949321
  - OEPLB_V21_THETA_R_ACT=0.011395643112507271
  - OEPLB_V21_N_MIN=12320768（沿用不拟合）
  - 规格默认（不变）：M_SHRINK=2、M_GROW=3、DWELL_SHRINK=2、DWELL_GROW=2、
    U_MAX=4、W_UNCERTAIN_TARGET=16、URGENT_EPISODE_CAP=3、BETA=0.5。
- 行为验证模式其余参数：EA_W=16（W_init=16）、EA_ALPHA=0.0、D 关闭、
  EA_ADAPTIVE_WINDOW/EA_ADAPTIVE_DECAY/EA_PLANNER_BI 未设——服务配置与
  §5 校准模式完全相同；唯一差异 = OEPLB_V21_* 六 knob 从惰性值换为上述
  冻结值（W 由固定变为受控自适应，n_min 由 0 变为 12320768）。
- 行为 run 产物：r3_logging/behavior/{v2p1_behavior_rank0.jsonl, server.log,
  bench.log, v21behavior.json, run.log}；驱动脚本 sha256 见 §10 追加行。
- 失败政策沿用 §9 与计划 §1.2/§5.2：基础设施失败允许一次预注册重启；
  第二次致命 → 停止报告；行为验证失败 → 保存冻结参数与失败记录，停止
  并说明原因，不在同一轮静默重拟合。

## 9. 运行前环境状态

- 8×H20 全部空闲（memory.used = 4 MiB ×8）；无存活 sglang 服务进程
  （ps 中 1790 个 `[sglang::…] <defunct>` 为历史僵尸进程，0 内存 0 GPU，
  自 2026-09-15 累积，不影响启动；既往 v2 校准在其存在下正常完成）。
- 运行失败政策（计划 §1.2）已写入驱动脚本：任槽位失败允许**一次**
  预注册重启（记录在案）；同一战役第二次致命失败 → 停止并报告，
  不追加启动。

## 10. 驱动脚本 sha256（追加于启动前）

- run_v2p1_logging.sh = f3cad76361b7e1348728a2e6991c700a7b238c9dd6c7e7acf258eb701087ebb0
- run_v2p1_behavior.sh = 876c71b9beb8d349c696ef1bb6c69b5711f17e405e4113d608db5bf3d052afb5
