# ADW v2.1 有限修版 r3 —— 最小 diff、边界落实与验收证据

日期：2026-10-17。状态：**预冻结（r3）**；未部署；logging 未启动（等批准）。
范围：仅处理 r2 评审提出的三项 logging 前事项 + 开销口径补全。
**不预告任何性能提升；不把旧 v2 的正式结果因果归因于本次修复。**

## 0. 保存性（r3 触碰面）

| 文件 | r3 状态 | sha256（前16） |
|---|---|---|
| window_v2.py（冻结 v2） | **从未改动** | 810c54daf1c4b9ec |
| locked_v2_frozen_20261016/ | 未动 | 见 MANIFEST |
| window_v2p1.py | r3 未改（=r2） | d20d340f9d44ae99 |
| v2p1_signals.py | r3 未改（=r2） | 98a13608ac56f0f1 |
| config.py | r3 未改（=r2） | 66b0760d3e7169bd |
| controller.py | **r3 唯一改动文件**：对冻结版仍纯增量（+360/-0，difflib 验证） | 24ccf3048544ed06 |

diff 文件：`controller_v21_r2_to_r3.diff`（195 行，r2→r3 增量）、
`controller_v21_r3.diff`（425 行，对冻结版全量增量）。

## 1. 阻塞项 3：通信器致命边界（代码已落实）

### 1.1 边界定义（与评审要求逐字对应）

- **健康通信器上的本地计算异常**（相 2：tripwire/信号/step/hash）
  → rollback + 错误哨兵 + prop_valid=0 → **共同 HOLD，服务继续**；
  连续 2 次 → 冻结于 W_init（继承 v2 安全模式）。
- **通信器不可恢复故障**（相 1 det SUM / 相 3 confirm MIN/MAX 集合
  通信本身异常）→ **FATAL 日志 + 尽力而为 `fatal_collective` JSONL
  证据行 + 裸 re-raise → 终止本次运行**；可疑通信器上**不再进行
  任何普通确认或提交**。恢复 = 重建通信器 = 作业级重启（NCCL 官方
  致命/异步错误处理语义），由框架故障传播流程接管，控制器不自建
  容错。

### 1.2 r2→r3 代码增量（controller.py，唯一改动文件）

删除（r2 的"带病保日程"路径）：
- `det_ok` 标志及其"det 失败 → 跳过相2 → 仍尝试相3确认"的续行逻辑；
- 相 3 失败 → 追加 err_reason 后继续 HOLD 的处理；
- 日志字段 `det_ok`。

新增：
- 相 1 except：`logger.error(FATAL…)` + `_v2p1_fatal_record(1, e)`
  （best-effort，自身 guarded，绝不掩盖 re-raise）+ `raise`；
- 相 3 except：同上（phase=3）+ `raise`；
- 相 2 整体保留但不再嵌套于 `if det_ok:`（语义不变：本地可恢复域，
  内部无任何集合通信）；
- 方法 docstring：四类相位描述更新 + 显式 "Fault boundary (r3)" 段；
- 日志 `protocol_rev` → "r3"（正常窗记录与 fatal 记录双处）。

传播链（已核实，非假设）：`_window_control_v2p1` ← `_decide_and_begin_swap`
← `on_forward_pass_end` ← `model_runner.forward`（site-packages
model_runner.py:2678，直接调用、**无 try/except 包裹**）← scheduler
主循环 → 进程崩溃 → 服务终止（其余 rank 后续集合通信失败/超时退出）
→ 作业级重启。未新增任何容错系统。

### 1.3 验收证据

**(a) 真实多进程致命边界测试**（`test_v2p1_fatal_boundary.py`，
gloo/world=2/CPU，pg 超时 25s；证据 `fatal_boundary_evidence.json`
+ 双 rank 增量 JSON）——7/7 PASS：

| 判定 | 检查项 |
|---|---|
| PASS | evidence files written by BOTH ranks before termination |
| PASS | w1 healthy: joint COMMIT on both ranks (protocol baseline) |
| PASS | w2 injected rank: det-collective fatal recorded, re-raised, NO ordinary confirm attempted (seq has no cmin/cmax), NO commit |
| PASS | w2 peer rank: its OWN real collective errored/timed out -> phase-1 fatal recorded, re-raised, NO confirm, NO commit |
| PASS | NO rank produced ANY window-3 record (run terminated at the fatal window; no continuation of the normal schedule) |
| PASS | NO rank reached the clean-exit marker after w2 |
| PASS | both processes died with the fatal exception propagated (terminated_with recorded; mp.spawn raised) |

逐窗记录（seq=该窗实际执行的协议集合通信；committed；fatal 类；错误摘要）：

| 窗 | rank | seq | committed | fatal | err |
|---|---|---|---|---|---|
| w1 | rank0 | det_sum,cmin,cmax | True | - | - |
| w1 | rank1 | det_sum,cmin,cmax | True | - | - |
| w2 | rank0 | - | False | phase1_det_collective | RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:54 |
| w2 | rank1 | - | False | phase1_det_collective | RuntimeError: simulated communicator fatal error on det SUM (NCCL asyn |

即：w1 健康 → 共同 COMMIT；w2 注入 rank1 det 集合通信致命异常 →
记录后 re-raise 进程终止；rank0 的**真实**集合通信随即失败
（gloo: Connection closed by peer）→ 同样 FATAL 终止；致命窗两侧
均无 cmin/cmax、无 commit；无 w3；spawn 以异常收场。
**对照**：11 窗 recoverable 类测试（相 2 注入）r3 下复跑全过——
两类边界各有真机证据。

**(b) recoverable 类 11 窗故障注入测试 r3 复跑**：
- cpu4/gloo：`{"all_windows_ok": true, "n_windows": 11, "commits_matched_design": true, "exactly_3_protocol_collectives_per_window_every_window": true, "final_W": 16, "final_eff_W": 16, "frozen_after_double_failure": true}`
- gpu8/nccl（8×H20，真实 [94,128]）：`{"all_windows_ok": true, "n_windows": 11, "commits_matched_design": true, "exactly_3_protocol_collectives_per_window_every_window": true, "final_W": 16, "final_eff_W": 16, "frozen_after_double_failure": true}`
- gpu8 计时（本轮重测，与 r2 轮并列报告、不挑选）：窗协议中位
  **411.3 µs**（8 rank 中位值范围 399.7–413.8；r2 轮为 567–572 µs；
  det SUM 60.3–62.1 µs；confirm MIN+MAX 181.2–185.1 µs；
  每 rank 并行，**不跨卡相加**）。

**(c) 离线套件**：39 PASS / 0 FAIL（新增 T16a–g：det_ok 移除、
相1/相3 except 以 bare raise 收尾、fatal 记录器 guarded、相2 无
re-raise、protocol_rev r3 双记录、docstring 边界段）；350 窗冻结
回归 0 mismatch（T13b，6 个 phaseB V2 日志）。原始输出：
`test_v2p1_raw_output_r3.txt`。

## 2. 开销口径补全（评审要求的三点澄清 + 分解测量）

单卡 H20 微基准 `bench_v2p1_overhead.py` →
`overhead_decomposition_gpu.json`（空闲 GPU、真实形状 [94,128]/
索引 65536；两种计时口径并列）：

**(i) 测试专用 all_gather_object**：harness 证据收集，**不是协议
的一部分**，不在任何计时内（窗协议计时只含 det clone+SUM、本地
step/hash、confirm MIN/MAX、sync）。

**(ii) 第二次 scatter_add（记录路径，之前不在 0.57ms 内——评审判断
正确）**，按控制器**精确增量表达式**（复用已算的 flat/mask）：

| 口径 | 每层 | ×94 层/forward |
|---|---|---|
| GPU 串行（批量 K=50，纯核时间） | 30.57 µs | 2.87 ms |
| GPU 串行（每调用，含发射开销） | 48.50 µs | — |
| CPU 入队（与活体 _prof_record 同口径类） | 26.14 µs | 2.46 ms |

活体交叉核对：正式 v2 战役 server.log `[PB-OEPLB-PROF]
avg_record_us`（**一次** scatter_add、CPU 口径、6 启动）中位数的
中位 **88.94 µs**（各启动中位 [91.95, 84.65, 88.94, 84.38, 91.84, 86.15]）——与 bench CPU 口径量级相容
（活体值含 flat/mask 计算与第一次 scatter，本就应更大）。
对照：v2 规划器 scatter 本体 GPU 每调用 32.67 µs → v2.1 把记录
路径 scatter 成本约**翻倍**。

**(iii) 窗边界信号束**（r2 计时替身漏掉的相2本地部分：
logical_load+row_distribution+change_score+EWMA+z）：
**318.6 µs/窗**，其中 change_score 占 255.5 µs（含 .item() 同步）。
仅每 W 个 forward 在 scheduler 线程暴露一次。

**临界路径暴露划分**：记录路径（每 forward，forward hook 内、
计算流上的新增核 → 每个 forward 都串行暴露）；窗协议+信号束
（每 W 个 forward 一次，scheduler 线程阻塞）。相对量级：
GPU 新增/forward ≈ 2.87 ms + 窗协议摊销，对 ~350 ms/forward 的
活体墙钟（W0：~1044 forward / ~365s）≈ **0.82%**（order-of-magnitude only; ~350ms includes inter-request gaps and decode; GPU-added kernel time is what serializes into the forward）。

**修正后的每 forward 新增总账（GPU 串行口径，单 rank，不跨卡相加）**：

| W | 记录路径 | 窗协议（r2轮/r3轮） | 信号束 | 合计（r2轮/r3轮） |
|---|---|---|---|---|
| 8 | 2.87 ms | 71.2/51.4 µs | 39.8 µs | 2.98/2.96 ms |
| 16 | 2.87 ms | 35.6/25.7 µs | 19.9 µs | 2.93/2.92 ms |
| 32 | 2.87 ms | 17.8/12.8 µs | 10.0 µs | 2.90/2.90 ms |
| 64 | 2.87 ms | 8.9/6.4 µs | 5.0 µs | 2.89/2.88 ms |

结论（口径陈述，非性能声称）：评审的摊销表（71/36/18/9 µs/forward）
对**窗协议**成立（本轮复测给出并列的 51/26/13/6 µs 第二
组值），但 v2.1 **新增服务开销的大头是记录路径第二次 scatter_add
（~2.9 ms/forward，占新增总量 ~97%），与 W 无关**；
"检查成本随窗口增大而摊薄"仅适用于窗协议+信号束那 ~1–3% 部分。

## 3. 阻塞项 1+2：校准计划 revision r3（预注册，logging 前）

全文见 `../calibration_plan_v2p1.md`（修订 r3 段）。要点：

**W64 样本冲突（方案 B 选定，预注册）**：窗口数算术（预期）
W8≈129–132 / W16≈64–66 / W32≈32–33 / **W64=16**；扣 init+护栏后
合格静止窗 ≈104/51/25/**12** → 旧"每 run ≥30"规则对 W64 **和 W32**
都不相容，已移除，代之以按用途的池规则：κ/z 带与 θ_r 用 4-run
合并池（门：静止池 ≥150，预期 ≈192–198；Z_c 池 ≥20，预期 ≈45）。
W64 run 预注册用途 = 机制运行验证 + 入池样本 + 小样本诊断点；
**不能声称** = W64 单独的任何分位数/中位数独立估计、任何需
W64 n≥30 的门。方案 A（3 遍连续 W0 = 24111 请求 ≈18–20 min 服务
墙钟，W64 槽位工作量 ×3）已作为完整预注册文本备查，未选定
（理由：G1 已降级为诊断，无门需要 W64 单 run 样本量；重复遍
样本相关而非独立；与预算冻结纪律相悖）。实数低于预期 >30% →
停止并报告，不降门槛、不追加启动。

**G1（F2 尺度不变性）被测命题修正**：验收（门）= 同一状态成对
缩放测试——PT1（机制级：T5/T6b/T7，(h,n)→(c·h,c·n) 决策轨迹
逐一相同，已 PASS）+ PT2（数据驱动，logging 后离线：对每个合格
静止窗构造到 W64 工作量尺度的缩放对，h_rate 逐窗完全相等 +
冻结决策函数逐窗相同；门=100% 恒等，任何 mismatch → F2 实现
缺陷 → 停止）。跨 W h_rate 中位数比**降级为诊断 G1-D**（布局
历史混杂是合法解释；逐点报 n；预注册解释带 ≤1.5 相容 / 1.5–3
排查 / >3 冻结前停下，只决定是否排查，不回写 knob）。

**G4（F1 逻辑身份不变性）被测命题修正**：验收（门）= 受控离线
测试 T2（swap 后 det 镜像重映射与 oracle 逐元素相等）+ T3/T3b
（同一逻辑工作量跨 swap：逻辑空间 z≈3e-5 vs 物理槽 z=147 伪影
对照）。"swap 窗 z ≤ z_lo"门**撤销**（真实需求变化引发 swap 时
z 本应高）。logging 数据角色 = 按上下文报告：每个 swap 窗的触发
事件/段分类/z 轨迹 + 诊断 D1（R_swap=静止段 swap 窗与非 swap 窗
z 中位比，解释带 ≤2 相容 / >2 冻结前排查）与 D2（swap 后 3 窗内
z 回落比例，按段分层，不作失败判据）。记录质量筛选（合格窗/
committed/非 init/合并样本 ≥5 才报比值）保留。

**运行失败政策（配合 §1 代码边界）**：fatal_collective 终止的
启动无拟合样本，属基础设施失败：该槽位允许重启一次（记录在案）；
预算内第二次致命终止 → 停止并报告。预算工作量口径显式化：5 次
启动 × 1 遍 W0（8037 请求 ≈365–370s 服务墙钟/次 + 启动 8–15 min），
合计服务 ≈31–35 min、总墙钟 ≈1.5–2.5 h；"5 次启动"不自动等于
任何历史战役的工作量。

## 4. 边界与不变项

- α=0、D 关闭、planner 与其他服务配置：r3 未触碰。
- 冻结 v2 及其全部正式结果：未覆盖、未改动（T0 系列每轮验证）。
- 未部署（site-packages 仍是冻结 v2 哈希锁）；部署只在新
  FREEZE_record_v2p1 哈希锁下、logging 获批后进行。
- 本文件不声称性能提升；adaptive 净收益仍待校准完成后按独立
  正式协议验证。
