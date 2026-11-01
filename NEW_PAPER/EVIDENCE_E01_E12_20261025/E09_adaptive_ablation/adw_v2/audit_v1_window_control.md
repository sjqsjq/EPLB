# ADW v2 第一步：窗口控制状态机审计（修改前流程）

status: audit_complete_before_any_v2_code_change
date: 2026-10-16
scope: 只审计窗口控制（W 决定路径）；规划器/执行器/服务路径不在本审计修改范围。
authority: 以五臂战役哈希锁定并验证 MATCH 的**已安装**代码为准：
  /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/{controller.py(1206行),config.py,rebalancer.py}
  == /workspace/EPLB/OEPLB/src/{controller.py,config.py,rebalancer.py}（git 已提交，locked_artifacts 全部 MATCH）
  注意：/workspace/EPLB/benchmark/sglang_src/python/sglang/... 是**过期副本**（controller.py 1140 行，9月版），
  与运行代码不一致，禁止作为审计或修改依据（本次已核实并排除）。

## 1. 时钟与触发模型（修改前）

- 钩子：model_runner.py:2677-2678 每个全局 forward step（含 IDLE）调用一次
  `on_forward_pass_end(forward_batch)`；topk.py:988-990 每层每 prefill forward 调用
  `record_next_layer(topk_ids)`（物理槽 id，scatter_add_ 本地零通信）。
- W 的单位 = 全局 forward step 数（非请求数、非 token 数）。每 rank 本地计数器
  `_steps_since_last_check`，因每 rank 每全局 step 恰好调用一次，天然 lockstep，
  检查时刻不需要跨 rank 共识；检查窗口内做一次 all_reduce(SUM) 聚合 94×128 负载。
- 记录采样：只记 prefill（`is_extend`），且每 rank 每次调用要求
  `topk_ids.shape[0] >= min_record_tokens(32)`；`sample_interval=1`（全采）。
- 每轮服务实测规模：7963 请求 / 并发 256 ≈ 1000-1100 个全局 forward step；
  W=16 → ≈ 52-67 个检查窗口/轮（各臂窗口数随 W 轨迹变化：R1 实测 max window#
  A00=67, A10=56, A01=64, A11=52, P11=64）。

## 2. 决策信号清单（修改前，含污染与样本量问题）

| 信号 | 计算 | 数据来源 | 问题 |
|---|---|---|---|
| `_current_avg_ratio` / `_prev_window_ratio` | 94 层逐层 max_rank/mean_rank 后取平均 | all-reduced **含衰减历史**的 global_load | α=0.5 时 ≥50% 是上一窗历史；控制器自身 swap 也改变它（placement 改善→r 变化，与域变化不可分） |
| delta = \|cur_r − `_adw_last_ratio`\| | 相邻两窗 ratio 差 | 同上 | 固定阈值 0.03/0.003，无样本量修正；W 收缩→每窗新鲜 token 变少→r 噪声变大→delta 更易超阈→继续收缩（正反馈，只有 floor=8 兜底；RESET 后 3 窗冷却是唯一例外） |
| `_last_cos_sim` | 相邻两窗 global_load 向量逐层余弦的均值 | 同上（衰减历史向量） | α=0.5 时相邻窗共享历史→自相似抬高 cos_sim，0.85/0.95 阈值语义随臂不同；小 W 时纯噪声比较 |
| `_last_avg_ratio` | 8 个采样层的 max/mean 均值 | global_load | 只用于阻止 grow（>1.02×1.05=1.071 时） |
| `global_tokens` | global_load.sum() | **衰减累计器总和** | 不是本窗新鲜样本量；min_prefill_tokens=256 检验的是衰减后总量；α=0 时才等于新鲜计数 |
| 新鲜样本量 | **不存在此变量** | — | 每窗有效 token/assignment 数从未单独记录（M1 要求的核心缺口） |

## 3. 窗口变更点全图（修改前，共 2 个子系统 5 个变更点 + RESET + 不可达 tier）

子系统 S1（`_decide_and_begin_swap` 内，gated: cfg.adaptive_window 且 cos_sim 非 None）：
- **D shift-shrink**（controller.py:849-866）：cos_sim<0.85 连续
  `window_shift_confirm_windows`(=2, launcher) 窗 → `W←min(W, max(window_floor(=8,launcher), W//2))`
  （C3 修复保证收缩单调不扩张）。
- **E stable-grow**（:873-891）：cos_sim>0.95 连续 confirm 窗（W<base 时 2 窗；W≥base 时 4 窗），
  且 `_last_avg_ratio`≤1.071 才允许；`W←min(2×sync_window(=32), W×2)`。
- 0.85≤cos_sim≤0.95：两计数器都清零。

子系统 S2（check step 内、`_decide_and_begin_swap` 返回后执行，gated: cfg.adaptive_window；
changepoint 跟踪与 decay→0 gated: cfg.adaptive_decay）：
- **A ratio-jump shrink**（:423-448）：delta>0.03 → `W←max(8, W//2)`；
  若 adaptive_decay 同时 `decay→0`（一步清空历史）；RESET 冷却期（3 窗）内跳过。
- **B converged-grow**（:450-458）：delta<0.003 连续 3 窗 → `W←min(128, W×2)`。
- **C volatile-grow**（:460-468）：0.003≤delta≤0.03 连续 3 窗 → `W←min(128, W×2)`（"以稳制动"）。

RESET（gated: cfg.adaptive_decay，:807-830）：converged_ratio 跟踪 + 跳变检测
（cur > converged+0.03 且 prev < converged+0.015）→ 清零 load 历史、
`_reset_cooldown=3`、**本窗 early-return（放弃本窗规划与 cos_sim 跟踪）**。

tier（calibrate_adaptive_sensitivity，:514-552）：战役 launcher 固定 False →
`_calib_done=True`，**不可达**（warm-start 同样不可达）。审计确认无其他路径修改
window_floor/shift_confirm。（C7 已将其与 adaptive_window 解耦。）

**同一窗口内 S1 与 S2 都会执行，且方向可相反**（如 E-grow 与 A-shrink 同窗触发，
后执行的 S2 覆盖 S1）；两个子系统计数器互不复位（S2 的 jump 只清 converge/volatile，
不清 S1 的 shift/stable）。合成包络：W∈[8,128]（S2 上限 128，S1 上限 32，不一致）。

窗口内完整顺序：force_wait 挂起 swap → all_reduce → ratio 循环（94 层×8 rank .item()）
→ RESET 检测（命中则 early return）→ cos_sim 跟踪 + S1 变更 → token/阈值/预算/bias 门
→ 规划（planner 规则 = heuristic_first_improvement；P11 = best_improvement_rank_pair）
→ min_swap_ops(8)/defer 门 → 异步 P2P begin → 返回 check step → S2 变更 → load×decay。

## 4. 实测行为（五臂战役，全部保留）

- W 时间驻留分数：A10 {8:0.180,16:0.276,32:0.496,64:0.080}；A11 {8:0.140,16:0.362,32:0.263,64:0.235}。
- 事件计数（R1 server log 原始 grep，含 8 rank 重复；DP0 分类见 log_check）：
  A10 shrink 80/grow 56/shift-confirm 16/stable-confirm 0；A11 shrink 48/grow 56/shift 24/stable 8/RESET 56；
  A01 RESET 32（无窗口变更，adw=0）；A00/P11 全 0（固定 W=16）。
  DP0 reset 分类（全战役口径，log_check_w_reset_cost.json）：A01 共22 {A8,B1,C3,D10}；A11 共35 {A10,B8,C7,D10}
  ——多数为段中/噪声级（C/D 类），与真实域边界相关的 A 类仅约 1/3。
- 成本（每 start，~370-390s 服务）：record 7.5-10.2s（与 W 无关，按 forward 采样计）；
  allreduce(含 94 层 ratio 循环) 1.3-1.7s ≈ **23-28ms/检查窗**；planbuild 0.06-0.17s；
  finalize ~0.3s；迁移墙钟 sum 7.0-9.8s / 9-16 次 / max 1.7-2.7s（并列报告，从不扣减）。
- 净效果：干净 W 增量对比 A11−A01 = +3.95s [−0.94,+8.84]（α=0.5+D 相同，只差 adw）；
  A10−A00 臂均值差 +4.79s（α=0 干净 W 增量，未列入冻结对比族）；
  A11−A00 = +10.40s [+5.00,+15.80] 为**两套完整在线策略**之差（adw + 基础α 0→0.5 + D 三因素合成）。

## 5. 缺陷清单（映射到 v2 的 M1/M2/M3）

- F1（M3）双子系统同窗写 W、方向可相反、计数器互不复位、上限不一致（32 vs 128）——无单一权威。
- F2（M1）全部阈值（0.03/0.003/0.85/0.95/1.071）与样本量无关；W↓→样本↓→噪声↑→更易触发 W↓ 的正反馈
  （bias=c/√N 机制存在于代码但战役中 OFF：bias_correct=False, bias_gate=0）。
- F3（M1）检测信号与规划历史共用同一衰减累计器：r 与 cos_sim 都被 α 历史、被自身 swap 结果污染；
  无"新鲜归一化检测统计 vs 规划用历史"的分离。
- F4（M1）每窗新鲜 token/assignment 数从未记录；global_tokens 是衰减总量；窗口证据量差异不可见。
- F5（M2）窗口决策完全不看剩余可回收收益（headroom）：dead_zone(1.096) 只门 swap 不门 W；
  grow 仅由 cos_sim+ratio 阈值驱动。
- F6（M2/M3）决策不含检查成本：c_check≈23-28ms/窗已可测（allreduce prof），但 J(W) 型权衡不存在。
- F7（M3）无最低驻留期/迟滞（除 RESET 后 3 窗冷却）；shrink 与 grow 的确认为数不一致（jump 立即收缩 vs 3 窗才扩）。
- F8（M3）RESET 命中窗被整窗丢弃（early return 不规划）且随后 3 窗禁止收缩——检测到变化点时反而抑制响应。
- F9（M3）起始 W 来自 cfg.sync_window（launcher 16，非代码硬编码）；floor 8 / ceiling 32,128 为魔法数。
- F10（M1）cos_sim 在 α=0（纯新鲜比较）与 α=0.5（历史自相似抬高）下语义不同，同一阈值跨臂不可比。
- F11（一致性不变量，v2 必须保持）：现所有 W 变更输入均来自 all-reduced 逐位一致张量或 lockstep 本地计数；
  任何新统计进入 W 决策前必须 all-reduce 或证明 rank 不变（历史教训：tier 曾因本地计数在 0.5 边界两侧
  选出不同 tier，靠 all_reduce 全局计数修复，:478-512 注释）。
- F12（成本自伤，可无损优化）：每检查窗 94 层×8 rank `.item()` 同步（≈752 次 GPU→CPU sync）占
  allreduce prof 的主体，可向量化而不改语义。

## 6. v2 冻结保留（不修改）

规划器规则（heuristic_first_improvement / P11 的 best_improvement_rank_pair）、rebalancer 接受规则、
预算（max_swaps_per_layer 64, max_total_swap_layers 94, max_total_ops 300, min_swap_ops 8）、
threshold_ratio 1.02、dead_zone auto 1.096（C6 全臂一致）、AsyncSwapExecutor、force_wait 防死锁点、
record 热路径与采样、钩子位置、load 历史随 swap 重映射（controller.py:_try_finish_pending_swap 内）。
窗口控制之外的 D（历史遗忘）在第一阶段主比较中保持两臂相同设置（见用户第五节：先隔离窗口）。
