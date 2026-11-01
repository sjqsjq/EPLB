# ADW v2.1（version 21）有限修版 — 最小 diff 说明

日期：2026-10-16。范围：**仅**处理三项已核实审计发现（F1/F2/F3，见
`../adw_v2/` 2026-10-16 三点核查记录）。

**框架规则（按用户指令，适用于本目录全部文档）**：
1. 这是正确性/一致性修版，**不预告任何性能提升**；
2. **不**把旧 v2 的正式比较结果因果归因于这三项问题；此前 36 次启动的
   全部结果原样成立，作为冻结 v2 的测量；
3. 冻结 v2 与交付物 G 完整保留，不覆盖。

## 1. 保留证据（冻结版本未动）

| 文件 | 冻结 sha256（前16） | 状态 |
|---|---|---|
| `window_v2.py` | `810c54daf1c4b9ec` | **未修改**（测试 T0b 逐次校验） |
| `locked_v2_frozen_20261016/controller.py` | `2a4dedb22d4bcaba` | 冻结快照 + `MANIFEST.sha256` |
| `locked_v2_frozen_20261016/config.py` | `180fa41eae6e9394` | 冻结快照 |
| `locked_v2_frozen_20261016/window_v2.py` | `810c54daf1c4b9ec` | 冻结快照 |

对现网 v2（`OEPLB_WINDOW_CONTROLLER=2`）路径：所有新增代码均以
`version == 21` 门控，v2 行为逐字节不变；冻结单元测试
`test_window_v2_logic.py` 重跑 **ALL_TESTS_PASS**；6 个 phaseB V2 正式日志
（350 窗）用冻结 knob 离线重放 **0 失配**（T13b）。

## 2. 变更清单

**新文件（不动旧逻辑）**
- `OEPLB/src/v2p1_signals.py`（sha256 `98a13608ac56f0f1…`）：F1/F2 信号函数
  （logical_load 逻辑身份聚合、row_distribution、change_score、headroom
  公式与冻结 v2 逐式相同、headroom_rate、z_from_signals 归一化同 v2）。
- `OEPLB/src/window_v2p1.py`（sha256 `2ba43052dda6c975…`）：
  `WindowControllerV2P1(WindowControllerV2)` 子类，仅加：θ 字段改为
  **rate 单位**的取值校验（0<lo<hi≤2，占位 0 拒启）、snapshot/restore
  （9 状态字段全量回滚）、decision_hash / error_sentinel / check_unanimous
  （F3 确认原语）。状态机 step() 本体不改。
- `$EXP/scripts/test_window_v2p1_logic.py`：离线验收 T0–T14（30 项全过）。

**修改文件（纯增量：controller.py +275/-0，config.py +35/-0）**

| 插入点 | 位置锚 | 内容 | 对应项 |
|---|---|---|---|
| A | `__init__` v2 分支后 | `elif version==21: _init_window_control_v2p1`；其他版本号拒启（1/2/21 白名单） | 门控 |
| B | `record_next_layer` scatter_add 后 | version 21 时对**独立**检测累加器 `_det_load` 做第二次 scatter_add（共享行不改；planner 累加器 `self.load` 不受 v2.1 影响） | F1 |
| C | forward 边界 carry/decay 块后 | `_det_load` 独立窗边界（UNCERTAIN carry 仅作用于检测累加器，最多并 1 窗，继承逻辑） | F1 |
| D | `_decide_and_begin_swap` v2 分派后 | version 21 调 `_window_control_v2p1(global_load, global_tokens)`（与 v2 同位置：token/阈值门之前） | 门控 |
| E | `_warmup_all_rank_pairs` 末尾 | 预热重排后：`_p2l_version += 1`、清 `_det_load`、`_det_map_version` 锁齐 | F1 |
| F | warmstart 批量重排 `load.zero_()` 后 | 同 E（清检测累加器 + 版本锁齐；低 n_fresh 窗由继承 UNCERTAIN 处理） | F1 |
| G | refine 批量重排 `load.zero_()` 后 | 同 E | F1 |
| H | `_try_finish_pending_swap` 计数随专家迁移循环后 | 对 `_det_load` 镜像同一 pair-swap 重映射 + 版本锁齐 → 窗末用**当前** p2l 聚合即精确（含提交前计数） | F1 |
| I | `_dump_detailed_load` 前 | 新方法 `_init_window_control_v2p1` / `_v2p1_confirm_allreduce` / `_window_control_v2p1` | F1+F2+F3 |

`config.py`：新增 `v21_*` 字段 17 个 + `OEPLB_V21_*` 环境变量解析
（θ 命名 `v21_theta_r_low/act`，单位为 **rate = h_w/n_fresh**）。
v2 字段与解析不动。

完整 diff：`controller_v21.diff`（340 行）、`config_v21.diff`（51 行）。

## 3. `_window_control_v2p1` 控制流（唯一 W 权威，version 21 下）

1. **快照** `ctrl.snapshot()`（9 状态字段）；
2. **映射版本绊线**：`_det_map_version != _p2l_version` → 本窗按错误处理
   （F1：任何未登记 p2l 变更路径都会触发 HOLD 而不是用错映射）；
3. `_det_load` clone + all_reduce（独立集合通信，各 rank 同位置）；
4. F1 信号：`logical = logical_load(det, 当前 p2l)` → p、s_change 对
   **逻辑身份 ref EWMA**（ref 永不需要重映射——逻辑身份在交换下稳定）；
   z 归一化公式与 v2 相同（κ 需按新信号空间重校，见校准方案）；
   F2：`h_rate = h_w / n_fresh ∈ [0,1]` 传入 step() 的 h_w 参数位；
5. `rec = ctrl.step(z, h_rate, n_fresh)` —— 仅为**提案**；
6. **确认**：所有 rank（含错误路径，用确定性 error_sentinel）把决策哈希
   提交到每窗一次 2 元 int64 SUM all_reduce；`sum == world×h` 当且仅当
   全 rank 同值 → 集合通信日程**构造上不可能错位**（修掉 v2 残余缺口：
   异常路径跳过每 16 窗哈希计数）；
7. **提交当且仅当** 提案存在且哈希一致：应用 W、carry、ref EWMA
   （uncertain/desync 事件冻结 ref，继承 v2 语义）、streak 清零；
   否则全 rank 一致回滚（restore 快照，W 保持）、streak+1；
   streak≥2 → `on_desync()` 冻结在确定性 w_init（继承安全模式）；
8. rank0 JSONL 记录（含 h_rate、proposed_event、committed、
   confirm_fail_streak、p2l_version、desync），`c_window_ms` 计入 det
   all_reduce 与确认 all_reduce（在线开销全部入账，不隐藏）。

任何分支不抛出，服务继续。

## 4. 不变式（与 v2 phase-1 相同）

- `decay_factor==0.0` 强制（**α=0**）；`adaptive_window/adaptive_decay/
  calibrate_adaptive_sensitivity` 必须为 False（**D 关闭**，单权威规则 M3）；
- planner 路径零改动：`self.load`、比例门、交换接受、执行器均为原冻结
  代码；v2.1 只替换**检测与窗口控制**；
- `OEPLB_V21_KAPPA≤0` 或 θ rate 占位 0 → 启动即拒（不允许未冻结值上线）；
- 心跳/集合通信成本计入 c_window_ms；全 rank 决策一致后才改 W；
- 不声称 no-slower-than-fixed。

## 5. 部署状态

**尚未部署**到 site-packages（当前仅离线验收）。获准跑真机时按新哈希锁
（FREEZE_record_v2p1）部署四文件 + 增量 controller/config，并沿用
step6_common.sh 的 verify_hashes 模式新增 v2.1 哈希校验。
