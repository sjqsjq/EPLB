# ADW v2.1 正式评价 — 一次性完整报告（T1–T5）

协议：PROTOCOL_formal_eval_v2p1（FINAL LOCKED，§15 作者确认）。本报告为
终点后一次性交付；未做任何中期分析。方向约定 Δ = time(前者) − time(后者)，
Δ<0 = 前者更快。主终点 = 完整服务墙钟 total_time_s（含全部真实开销，无任何扣除）。

## T1 全部 48 次启动原始记录（含退役/缺失，按臂失败计数）

| seq | 轮 | 条件 | 臂 | label | 判定 | total_time_s | tps | boot_s | attempt | INFRA-R |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | r1 | W1 | A | fev_r1_W1_A | VALID | 352.48 | 11.3 | 105 | — |  |
| 2 | r1 | W1 | F | fev_r1_W1_F | VALID | 349.69 | 11.4 | 64 | — |  |
| 3 | r1 | W1 | L | fev_r1_W1_L | VALID | 355.18 | 11.3 | 72 | — |  |
| 4 | r1 | W1 | M | fev_r1_W1_M | VALID | 348.50 | 11.5 | 64 | — |  |
| 5 | r1 | W2 | L | fev_r1_W2_L | VALID | 703.70 | 11.4 | 60 | — |  |
| 6 | r1 | W2 | M | fev_r1_W2_M | VALID | 700.58 | 11.4 | 78 | — |  |
| 7 | r1 | W2 | A | fev_r1_W2_A | VALID | 704.61 | 11.4 | 60 | — |  |
| 8 | r1 | W2 | F | fev_r1_W2_F | VALID | 718.01 | 11.1 | 72 | — |  |
| 9 | r2 | W2 | F | fev_r2_W2_F | VALID | 699.12 | 11.4 | 72 | — |  |
| 10 | r2 | W2 | L | fev_r2_W2_L | VALID | 699.28 | 11.4 | 60 | — |  |
| 11 | r2 | W2 | M | fev_r2_W2_M | VALID | 703.43 | 11.4 | 64 | — |  |
| 12 | r2 | W2 | A | fev_r2_W2_A | VALID | 702.02 | 11.4 | 60 | — |  |
| 13 | r2 | W1 | M | fev_r2_W1_M | VALID | 349.97 | 11.4 | 64 | — |  |
| 14 | r2 | W1 | A | fev_r2_W1_A | VALID | 360.90 | 11.1 | 72 | — |  |
| 15 | r2 | W1 | F | fev_r2_W1_F | VALID | 349.40 | 11.4 | 72 | — |  |
| 16 | r2 | W1 | L | fev_r2_W1_L | VALID | 351.16 | 11.4 | 64 | — |  |
| 17 | r3 | W1 | L | fev_r3_W1_L | VALID | 348.28 | 11.5 | 68 | — |  |
| 18 | r3 | W1 | M | fev_r3_W1_M | VALID | 350.15 | 11.4 | 64 | — |  |
| 19 | r3 | W1 | A | fev_r3_W1_A | VALID | 355.93 | 11.2 | 92 | — |  |
| 20 | r3 | W1 | F | fev_r3_W1_F | VALID | 353.78 | 11.3 | 68 | — |  |
| 21 | r3 | W2 | A | fev_r3_W2_A | VALID | 712.83 | 11.2 | 76 | — |  |
| 22 | r3 | W2 | F | fev_r3_W2_F | VALID | 705.42 | 11.3 | 68 | — |  |
| 23 | r3 | W2 | L | fev_r3_W2_L | VALID | 709.52 | 11.3 | 68 | — |  |
| 24 | r3 | W2 | M | fev_r3_W2_M | VALID | 712.96 | 11.2 | 73 | — |  |
| 25 | r4 | W2 | M | fev_r4_W2_M | VALID | 714.32 | 11.2 | 64 | — |  |
| 26 | r4 | W2 | A | fev_r4_W2_A | VALID | 704.91 | 11.3 | 68 | — |  |
| 27 | r4 | W2 | F | fev_r4_W2_F | VALID | 702.53 | 11.4 | 64 | — |  |
| 28 | r4 | W2 | L | fev_r4_W2_L | VALID | 707.30 | 11.3 | 64 | — |  |
| 29 | r4 | W1 | F | fev_r4_W1_F | VALID | 354.85 | 11.3 | 80 | — |  |
| 30 | r4 | W1 | L | fev_r4_W1_L | VALID | 355.22 | 11.3 | 68 | — |  |
| 31 | r4 | W1 | M | fev_r4_W1_M | VALID | 357.76 | 11.2 | 76 | — |  |
| 32 | r4 | W1 | A | fev_r4_W1_A | VALID | 359.96 | 11.1 | 64 | — |  |
| 33 | r5 | W1 | A | fev_r5_W1_A | VALID | 355.39 | 11.3 | 72 | — |  |
| 34 | r5 | W1 | F | fev_r5_W1_F | VALID | 353.46 | 11.3 | 64 | — |  |
| 35 | r5 | W1 | L | fev_r5_W1_L | VALID | 355.43 | 11.3 | 80 | — |  |
| 36 | r5 | W1 | M | fev_r5_W1_M | VALID | 354.66 | 11.3 | 68 | — |  |
| 37 | r5 | W2 | L | fev_r5_W2_L | VALID | 715.20 | 11.2 | 72 | — |  |
| 38 | r5 | W2 | M | fev_r5_W2_M | VALID | 712.69 | 11.2 | 64 | — |  |
| 39 | r5 | W2 | A | fev_r5_W2_A | VALID | 712.38 | 11.2 | 64 | — |  |
| 40 | r5 | W2 | F | fev_r5_W2_F | VALID | 711.41 | 11.2 | 76 | — |  |
| 41 | r6 | W2 | F | fev_r6_W2_F | VALID | 709.90 | 11.3 | 64 | — |  |
| 42 | r6 | W2 | L | fev_r6_W2_L | VALID | 707.16 | 11.3 | 72 | — |  |
| 43 | r6 | W2 | M | fev_r6_W2_M | VALID | 710.34 | 11.3 | 64 | — |  |
| 44 | r6 | W2 | A | fev_r6_W2_A | VALID | 711.18 | 11.2 | 68 | — |  |
| 45 | r6 | W1 | M | fev_r6_W1_M | VALID | 357.20 | 11.2 | 68 | — |  |
| 46 | r6 | W1 | A | fev_r6_W1_A | VALID | 357.31 | 11.2 | 64 | — |  |
| 47 | r6 | W1 | F | fev_r6_W1_F | VALID | 353.76 | 11.3 | 84 | — |  |
| 48 | r6 | W1 | L | fev_r6_W1_L | VALID | 353.94 | 11.3 | 65 | — |  |

**逐臂失败计数（全部报告，不删除）**：
- A 臂：非 VALID 0 次（无）
- F 臂：非 VALID 0 次（无）
- L 臂：非 VALID 0 次（无）
- M 臂：非 VALID 0 次（无）

幸存者偏差声明：以下 T2 各区间**只描述成功运行中的性能**；A/M 臂的失败计数
（上表）是部署代价的一部分，与速度区间并排解读，不得删除失败后仅凭成功样本
宣布部署获益。缺失配对规则：n≥4 → 照常给区间并加旗标；n≤3 → 仅描述性
（"insufficient pairs"）。

## T2 配对差与区间（每条件独立；八个边际 95% CI，不校正，全部报告）

八个区间为**边际 95%**，无整族 95% 覆盖保证；整体结论须同时权衡全部八个
区间、逐臂失败计数与 T5 范围声明，不得事后挑选排除零的一项。

### 条件 W1

F 臂 W1 有效轮均值（%描述的分母）：352.49 s

| 比较 | n对 | 缺失轮 | Δ均值(s) | sd | t 95% CI | 置换p(最小2/2ⁿ) | 置换95%CI | Wilcoxon p | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| A−F (primary) | 6 | — | 4.51（+1.28% of F） | 3.61 | [0.71, 8.30] | 0.0312 | [0.36, 4.51] | 0.0312 | 排除零（t 与置换一致） |
| A−L (secondary) | 6 | — | 3.79（+1.08% of F） | 4.65 | [-1.09, 8.67] | 0.1250 | [-0.46, 4.25] | 0.1562 | compatible with zero（跨零） |
| A−M (secondary) | 6 | — | 3.95（+1.12% of F） | 4.01 | [-0.25, 8.16] | 0.0312 | [0.12, 3.95] | 0.0312 | 方法间不一致，判为不确定 |
| M−F (auxiliary) | 6 | — | 0.55（+0.16% of F） | 2.64 | [-2.22, 3.32] | 0.6562 | [-0.80, 1.35] | 0.6875 | compatible with zero（跨零） |

### 条件 W2

F 臂 W2 有效轮均值（%描述的分母）：707.73 s

| 比较 | n对 | 缺失轮 | Δ均值(s) | sd | t 95% CI | 置换p(最小2/2ⁿ) | 置换95%CI | Wilcoxon p | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| A−F (primary) | 6 | — | 0.26（+0.04% of F） | 7.08 | [-7.17, 7.69] | 0.9375 | [-2.23, 2.49] | 0.4375 | compatible with zero（跨零） |
| A−L (secondary) | 6 | — | 0.96（+0.14% of F） | 2.95 | [-2.14, 4.06] | 0.4375 | [-0.87, 1.83] | 0.4375 | compatible with zero（跨零） |
| A−M (secondary) | 6 | — | -1.07（-0.15% of F） | 4.49 | [-5.78, 3.65] | 0.6562 | [-1.88, 0.81] | 0.6875 | compatible with zero（跨零） |
| M−F (auxiliary) | 6 | — | 1.32（+0.19% of F） | 10.10 | [-9.27, 11.92] | 0.7500 | [-2.90, 4.23] | 0.4375 | compatible with zero（跨零） |

## T3 A 臂行为链（检测/收缩分开定义，同一动作不重复归因）

窗口→行号映射：累计 steps_actual 占比 × 条件总行数（§9）。W2 真实边界 = 行 4000。
检测 = 边界后首个 committed 窗口 z ≥ z_hi(314.14010272527486)；收缩 = 首个 committed
confirmed_shrink/urgent_shrink。两延迟分开报告，不相加。亚阈值 alert 仅作上下文。

### W1
- r1: 窗口 20（W 变化 2 次，轨迹 16→32→64）
  - committed 事件：{'init': 1, 'hold': 17, 'confirmed_grow': 2}；亚阈值 alert(上下文)：0
- r2: 窗口 20（W 变化 2 次，轨迹 16→32→64）
  - committed 事件：{'init': 1, 'hold': 17, 'confirmed_grow': 2}；亚阈值 alert(上下文)：0
- r3: 窗口 20（W 变化 2 次，轨迹 16→32→64）
  - committed 事件：{'init': 1, 'hold': 17, 'confirmed_grow': 2}；亚阈值 alert(上下文)：0
- r4: 窗口 20（W 变化 2 次，轨迹 16→32→64）
  - committed 事件：{'init': 1, 'hold': 17, 'confirmed_grow': 2}；亚阈值 alert(上下文)：0
- r5: 窗口 20（W 变化 2 次，轨迹 16→32→64）
  - committed 事件：{'init': 1, 'hold': 17, 'confirmed_grow': 2}；亚阈值 alert(上下文)：0
- r6: 窗口 20（W 变化 2 次，轨迹 16→32→64）
  - committed 事件：{'init': 1, 'hold': 17, 'confirmed_grow': 2}；亚阈值 alert(上下文)：0

### W2
- r1: 窗口 37（W 变化 4 次，轨迹 16→32→64）
  - 边界窗 idx=19；检测窗 idx=20（z=652.0，延迟 1 窗）；首收缩窗 idx=21（距边界 2 窗）
  - committed 事件：{'init': 1, 'hold': 30, 'confirmed_grow': 3, 'hold_alert': 1, 'confirmed_shrink': 1, 'midband_hold': 1}；亚阈值 alert(上下文)：1
- r2: 窗口 37（W 变化 4 次，轨迹 16→32→64）
  - 边界窗 idx=19；检测窗 idx=20（z=634.7，延迟 1 窗）；首收缩窗 idx=21（距边界 2 窗）
  - committed 事件：{'init': 1, 'hold': 30, 'confirmed_grow': 3, 'hold_alert': 1, 'confirmed_shrink': 1, 'midband_hold': 1}；亚阈值 alert(上下文)：1
- r3: 窗口 37（W 变化 4 次，轨迹 16→32→64）
  - 边界窗 idx=19；检测窗 idx=20（z=634.7，延迟 1 窗）；首收缩窗 idx=21（距边界 2 窗）
  - committed 事件：{'init': 1, 'hold': 30, 'confirmed_grow': 3, 'hold_alert': 1, 'confirmed_shrink': 1, 'midband_hold': 1}；亚阈值 alert(上下文)：1
- r4: 窗口 37（W 变化 4 次，轨迹 16→32→64）
  - 边界窗 idx=19；检测窗 idx=20（z=634.7，延迟 1 窗）；首收缩窗 idx=21（距边界 2 窗）
  - committed 事件：{'init': 1, 'hold': 30, 'confirmed_grow': 3, 'hold_alert': 1, 'confirmed_shrink': 1, 'midband_hold': 1}；亚阈值 alert(上下文)：1
- r5: 窗口 37（W 变化 4 次，轨迹 16→32→64）
  - 边界窗 idx=19；检测窗 idx=20（z=614.3，延迟 1 窗）；首收缩窗 idx=21（距边界 2 窗）
  - committed 事件：{'init': 1, 'hold': 30, 'confirmed_grow': 3, 'hold_alert': 1, 'confirmed_shrink': 1, 'midband_hold': 1}；亚阈值 alert(上下文)：1
- r6: 窗口 37（W 变化 4 次，轨迹 16→32→64）
  - 边界窗 idx=19；检测窗 idx=20（z=634.7，延迟 1 窗）；首收缩窗 idx=21（距边界 2 窗）
  - committed 事件：{'init': 1, 'hold': 30, 'confirmed_grow': 3, 'hold_alert': 1, 'confirmed_shrink': 1, 'midband_hold': 1}；亚阈值 alert(上下文)：1

## T4 M 臂完整性与代价记账 + M−F

- W1 r1: committed 62 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=2.84，累计=236 ms（rank0 并行记账，**不从服务时间扣除**）
- W2 r1: committed 125 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=1.38，累计=261 ms（rank0 并行记账，**不从服务时间扣除**）
- W1 r2: committed 62 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=2.62，累计=236 ms（rank0 并行记账，**不从服务时间扣除**）
- W2 r2: committed 125 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=1.76，累计=308 ms（rank0 并行记账，**不从服务时间扣除**）
- W1 r3: committed 62 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=1.52，累计=172 ms（rank0 并行记账，**不从服务时间扣除**）
- W2 r3: committed 125 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=2.10，累计=459 ms（rank0 并行记账，**不从服务时间扣除**）
- W1 r4: committed 62 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=3.27，累计=301 ms（rank0 并行记账，**不从服务时间扣除**）
- W2 r4: committed 125 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=1.52，累计=335 ms（rank0 并行记账，**不从服务时间扣除**）
- W1 r5: committed 62 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=2.01，累计=252 ms（rank0 并行记账，**不从服务时间扣除**）
- W2 r5: committed 125 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=2.09，累计=455 ms（rank0 并行记账，**不从服务时间扣除**）
- W1 r6: committed 62 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=1.96，累计=259 ms（rank0 并行记账，**不从服务时间扣除**）
- W2 r6: committed 125 窗，W_after 集合=[16]（必须=[16]），控制器 c_window_ms 中位=1.67，累计=483 ms（rank0 并行记账，**不从服务时间扣除**）

M−F 为辅助描述性比较（T2 表内）：同期估计监测净效应，不预设大小或可分辨性；
c_window_ms 是 PROF 式累计时间，不等于暴露在关键路径上的代价，只作记账披露。

## T5 范围声明与限制（原文重申）

- 本报告只回答：A−F（部署净价值）、A−L（相对场景固定参考）、A−M（窗口策略作用）、
  M−F（监测代价辅助）。正、负或不确定都是合法结论；"compatible with zero" 按原样保留。
- 限制（协议 §11，全部有效）：prompt 重复/重合（125/250 unique×32，与校准集共享
  1383 条 unique prompt，L9）；O1 工作流（max_tokens=1，非真实生成分布）；单节点
  8×H20；W0-only 校准外推（W1/W2 未见；未演练路径存在）；A/M 与 F/L 机制不完全
  对称（仪表化差异已披露，M 臂即为其对照）；σ 估计来自 df=3 历史数据；L 臂窗口为
  单启动 argmin 事前选择（选择噪声已披露）；z_lo 边界口径敏感性已存档不回改。
- 公平性：全部臂同启动器、同模型、同并发 256、同数据集 sha；每启动前部署代码
  sha256sum -c 校验；A/M banner 8-rank + knob 唯一性门；F/L 负门（无 ADW banner）。
- 统计：n=6 固定；不校正多重性（边际区间）；置换伴随完整枚举（最小 p=2/2ⁿ，
  n=4→0.125，n=5→0.0625，n=6→0.03125，事前接受其保守性）；方法间不一致→不确定。
- 本轮结束后形成论文结论，不再进入无期限校准（作者指令，§15）。

