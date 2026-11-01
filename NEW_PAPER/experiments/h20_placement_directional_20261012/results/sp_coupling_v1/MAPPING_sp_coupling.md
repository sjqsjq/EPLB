# MAPPING — 采样/规划联动方案 ↔ 执行端真实冻结代码（§9 第一交付）

日期 2026-10-18。起点核验：部署端 12 个 .py 与 SRC 镜像 sha256 全等（controller 62293ea9…、config 4ba8de2d…、window_v2 810c54da… 冻结、window_v2p1 d20d340f…、v2p1_signals 98a13608…、probe_admission 0678e169…）；校准 16 项锁 sha256sum -c 16/16 通过；仓库 HEAD a7087a2（分支 sp_coupling_dev）。作者侧可读 HEAD 1036246 仅作参照，实施以本机部署端为准。window_v2.py 冻结不改（gen_freeze_v2p1.py:51-52 + T0b 断言）。

## A. 结论先行

1. **H/W 当前硬耦合**在同一窗口计数（controller.py:423-426 单一 `_steps_since_last_check < _effective_sync_window`；边界同时做 planner reduce :852-855 与 detection :962-967；:1601 检测输出反写 W）。**拆分结构上可行**：全 rank 锁步计数（:22-31 docstring，idle rank 同步推进 :421-422）、检测有独立累计器 `_det_load` 与独立集合（:1489-1490）、ctrl.step 是纯函数。所有状态下 H|W（16|16、16|64、8|8），W 边界必为 H 边界子集 → 每 rank 集合调度按边界类型固定同序：H-only=[detSUM,confirmMIN,confirmMAX]；H+W=[plannerSUM,detSUM,confirmMIN,confirmMAX]（保持现有 :852→:962 顺序）。
2. **采样门可前置于当前 forward 的 MoE 记录**：model_runner.py:2660 `forward_pass_id += 1`（pre-forward）与 :2666 `_forward_raw` 之间插入 `on_forward_pass_start`；record 单门（controller.py:341）已同时罩住 planner scatter（:366-368）与 detection scatter（:374-376），满足 §4.1"两个记录都被 S 门控"。现行为（:400-408 在 forward 结束后用上一批模式决定下一批）不满足 §2，SP 模式下废弃、改前置门；SP_MODE=off 时保留原路径逐位不变。
3. **不需要新的大型系统**：状态机为新纯 Python 模块 sp_state.py（probe_admission 的 propose/advance 模式），检测信号全部复用现有 z/h_rate/n_fresh/alert_k/stable_s（v2p1_signals.py + window_v2.py 冻结逻辑），共同提交协议复用现有 2-collective confirm（哈希输入扩展 sp 提案）。

## B. 需改动文件与不变量

| 文件 | 改动 | 新增不变量（测试覆盖） |
|---|---|---|
| SRC/controller.py（→部署镜像） | ① `on_forward_pass_start(fwd_id, forward_batch)` 前置采样门（SP 模式）；② `_steps_since_h/_steps_since_w` 双计数（SP 模式）；③ 边界分派：H 边界跑 `_window_control_v2p1`，W 边界追加 planner reduce/plan；④ phase2 计算 sp 提案并入 decision hash，phase4 仅在 commit 应用状态转换；⑤ 状态转换时计数重启/planner 封口；⑥ SP 模式下 W 边界 plannerSUM 改 fatal-safe（r3 式 raise，:829/:1094 吞错仅保留在 off 模式）；⑦ jsonl/PROF 增加 §8 字段 | 采样门前置于本 forward 记录；S 门同罩两 scatter；空 rank 参加全部集合并推进计数；H/W 跨状态计数正确；rec["W_after"] 在 SP 模式永不应用；转换仅 commit 生效、HOLD 回滚；封口记录覆盖步数与原因 |
| SRC/sp_state.py（新） | TRACK/QUIET/RECOVER 纯状态机：propose(signals)→transition、apply-on-commit、snapshot/restore、决策哈希贡献 | 五条确定性序列：稳定→QUIET；变化→RECOVER（首告警即恢复采样）；低样本→不进/退出；无迁移机会≠规划无动作；异常→共同 HOLD |
| SRC/config.py | SP 旋钮（env-only 模式 :209-251）：OEPLB_SP_MODE off\|fixed\|adaptive、SP_S、SP_W、SP_SEED（冻结）、SP_QUIET_STREAK=3、SP_RECOVER_MIN_CYCLES=2；adaptive 状态表 (S,H,W)：TRACK(1,16,16) QUIET(4,16,64) RECOVER(1,8,8) | 冲突拒绝：SP_MODE≠off 时要求 always_record=False、sample_interval=1、probe_mode=off、window_controller=21、adaptive_* 全关，违者启动即 RuntimeError |
| $SGL/model_executor/model_runner.py | :2660/:2662 之间 2 行受保护插入（`_ctrl is not None` 守卫），精确字符串替换补丁 + py_compile + 前后 sha 入新锁 | 非 pb 运行零影响；控制器为 None 时跳过 |
| SRC/test_sp_state.py（新）+ EXP/scripts/test_sp_controller_logic.py（新） | 离线单测（无 GPU）+ torch 逻辑测试（沿用 v2p1 T 套件模式） | 上表全部不变量 |
| EXP/scripts/run_sp_coupling_dev.sh + run_list_sp_dev.csv + integrity_check_sp.py（新，克隆校准骨架） | 12 启动驱动；完整性检查改以 steps/H 计 jsonl 行数基线；knob marks 换 SP env | 冻结 run list sha 锁；每启动前全锁核验 |
| 部署脚本 deploy_sp_coupling.py（新，仿 deploy_p11_and_lock.py） | SRC→site-packages 复制 + model_runner 补丁 + deployed_shas_sp_coupling.txt 锁生成 | src==deployed 断言；off 模式回归：SP_MODE=off 下 test_window_v2p1_logic 通过集与实现前完全一致（37 PASS + 既有 2 项陈旧 FAIL 不扩大） |

不改动：window_v2.py（冻结）、v2p1_signals.py（复用，z 归一化 :87-96 与 h_rate :82-84 已存在）、rebalancer.py、async_swapper.py、topk.py、run_grid_bench.py、launcher、数据集。probe_admission.py 不改（SP 战役 probe_mode=off，配置层强制）。

## C. 状态机 ↔ 现有信号映射（全部用已冻结阈值，不引入新数值）

| 方案条件 | 代码信号 | 出处 |
|---|---|---|
| 逻辑分布变化证据稳定带 | z < z_lo（stationary band，stable_s 累计） | window_v2.py:227-258 |
| 每层归一化负载机会代理低余量带 | h_rate < theta_h_low（RATE 单位，冻结值 0.0100955…） | v2p1_signals.py:82-84; controller.py:1314 |
| 覆盖达到有效样本门 | n_fresh ≥ n_min_eff（见 PD-1） | window_v2.py:141-166 |
| 无待完成迁移/规划未接受交换 | executor 无 pending 且本周期 plan_started==0；周期无规划机会时单列 opportunity=False 不计入"无动作" | controller.py:1011-1013,1091 |
| 明显逻辑分布变化（QUIET 退出） | z ≥ z_hi（首告警即转 RECOVER 恢复采样；规划动作仍需既有 alert_k≥m 确认） | window_v2.py:174,184-226 |
| 机会代理升高 | h_rate ≥ theta_h_low | 同上 |
| 一致性/状态护栏要求退出 | uncertain / confirm_hold / desync / version tripwire 事件 | window_v2p1.py:55-56; controller.py:1502-1506,1617-1634 |

## D. 预数据设计决定（PD 清单；计时开始前允许因实现约束修订并书面说明，计时后不得改）

- **PD-1** 覆盖门按采样比例预声明：n_min_eff = n_min // S（S=4 时 3080192）。不把样本数×S 伪装覆盖；该值属 §4.3"噪声标度"候选，正式评价前用开发数据核验 S4 背景波动，如需调整仅用开发/校准材料一次冻结；若核验显示 QUIET 因覆盖频繁退出，如实报告，不降门强驻留。
- **PD-2** QUIET 进入 = 连续 3 个有效 H 周期同时满足 C 表前三行 + 无 pending/无接受交换（有效=覆盖门过、非 uncertain、非 hold）。
- **PD-3** QUIET 退出（任一，下一共同边界转 RECOVER）：z≥z_hi 首告警、h_rate≥theta_h_low、n_fresh<n_min_eff、护栏事件。低 h_rate 永不作为阻止恢复观测的条件。
- **PD-4** RECOVER→TRACK：进入后 ≥2 个完整 H8 周期且该两周期均满足稳定带+低余量+覆盖；高余量时保持 RECOVER，无定时强制收敛。
- **PD-5** 任何转换后 QUIET 连击计数清零；TRACK 需重新积累 3 个有效 H16 周期。
- **PD-6** 转换走既有共同提交：sp 提案并入 phase2 decision hash（h 不一致→HOLD→restore，状态不变）；仅 commit 后生效。
- **PD-7** 转换边界：H/W 计数自边界重启；S 变化时 planner 累计器封口清零（原因 sp_seal_S_change 入日志），检测 ref EWMA 更新规则不变（不与封口混淆）；记录实际覆盖步数。
- **PD-8** SP 模式（fixed 与 adaptive）rec["W_after"] 永不应用；window ctrl 内部 W 钉在 w_init=16，仅取其 z/alert_k/stable_s/shrink 确认信号按原确认机制门控规划。
- **PD-9** S4 选样：sp_fwd 为控制器侧全 rank 锁步 forward 编号（on_forward_pass_start 自增，idle 亦推进）；block=sp_fwd//4，pos=blake2b(SP_SEED‖block)%4，sp_fwd%4==pos 才为记录机会；机会内仍按真实后端语义（非 capture、prefill、≥min_record_tokens、有效行掩码）记录实际有效行。
- **PD-10** SP 模式 W 边界 plannerSUM 失败→fatal raise（防跨边界 NCCL 错位）；off 模式行为逐位不变。
- **PD-11** 固定臂=SP_MODE=fixed：H=W=SP_W 恒定、S=SP_S 恒定、W 自适应禁用、无状态机；与 A 臂共用同一采样门代码路径（§6 公平性：F_sparse 获得同等稀疏记录能力）。F_dense=(W16,S1)，F_sparse=(W64,S4)。
- **PD-12** 配置冲突拒绝清单见 B 表 config 行；启动即断言，不静默覆盖。
- **PD-13** 日志三分离：sample_opportunities（S 门选中数）/recorded_steps（实际记录 forward 数）/valid_rows（有效行和），永不互相折算。
- **PD-14** 陈旧测试 T0c/T4c（r2 时代结构断言，冻结校准代码上已 FAIL、15/15 VALID 跑完）：不改不删，实现前后 off 模式通过集必须一致（37 PASS 不缩小）；重新冻结与否留作者决定，入最终报告。
- **PD-15** 既有 OEPLB_SAMPLE_INTERVAL（rank 本地 prefill 计数，有相位分叉隐患）在 SP 模式强制=1 并由前置门取代。

## E. §8 日志落点

- rank0 jsonl（controller.py:1706-1770 扩展）：sp_state_before/after、sp_s/h/w、sp_sample_opportunities、sp_recorded_steps、sp_valid_rows、sp_plan_opportunity、sp_swaps_started、sp_seal_reason、sp_cover_steps、转换事件原因。
- 每 H 边界既有字段沿用：共同 forward 编号(true_steps_total)、单调起止(ts_min/max)、z/h_rate/n_fresh、alert_k/stable_s、映射版本(p2l_version/fp8)、协议确认(h_min==h_max, lockstep_ok)。
- PROF 块（:1932-1949 扩展）：sp_record_calls/sp_record_skips/sp_gate_us；既有 record/allreduce/planbuild/finalize 分列不变。跨 rank 只 MAX/per-rank。
- 每 run meta.txt：SP env 全量 + 代码/数据 sha + 状态驻留汇总（run 末 dump sp_run_end.json：状态时间线、转换表、封口记录、最终布局指纹）。

## F. 预算确认（§7 阶段1 要求先给估算）

单次启动实测参照（校准战役同节点同 harness）：W1 ≈ 7.5–8 min（boot 64–92s + bench 352–357s），W2 ≈ 13.5 min（boot 64–72s + bench 706–712s）。12 启动 = 6×W1 + 6×W2 ≈ 2h10m GPU；smoke（3 臂 × 300 请求子集）≈ +15min；总 GPU ≈ 2.5h。与既有资源一致，不扩容，直接进入实现。

## G. 执行序

P0c 实现（分支 sp_coupling_dev）→ P0d 离线测试（sp_state 单测 + controller 逻辑测试 + 既有套件回归：test_probe_admission 75 PASS、test_window_v2_logic 32、v2p1 37 PASS+2 陈旧、multiprocess faults world=4、fatal boundary world=2）→ P0e 部署+锁+3 臂 smoke（调用计数/状态/输出正确性）→ P1 12 启动（臂序已预冻结 c9aa77f：W1/W2 block1 F_sparse→A→F_dense，block2 反转）→ P2 报告（5 问）。
