# ADW v2.1 三项验收结果（离线，CPU，无服务器）

日期：2026-10-16。测试：`$EXP/scripts/test_window_v2p1_logic.py`
（28 PASS / 0 FAIL，`ALL_V2P1_TESTS_PASS`）；冻结回归：
`test_window_v2_logic.py` → `ALL_TESTS_PASS`（未改动重跑）。
原始输出：`test_v2p1_raw_output.txt`、`test_v2_frozen_regression_output.txt`。

**框架规则**：以下为正确性/一致性验收，**不构成性能提升的预告**；旧 v2
正式结果**不**因果归因于这三项问题。

## 验收 1 — 检测分布统一逻辑身份，映射版本正确（F1）

| 检查 | 结果 | 关键数字 |
|---|---|---|
| T1 logical_load 守恒 + 冗余 p2l 逐专家分箱正确 | PASS | 质量守恒、每 bin 与 oracle 相等 |
| T2 窗中 pair-swap + 计数随专家迁移 + 当前 p2l 聚合 == 从未交换的 oracle | PASS | gathered == oracle（逐元素） |
| T3 同一逻辑负载跨交换：逻辑空间 s_change ≈ 0 | PASS | s_log = 5.96e-08（z ≈ 3e-05） |
| T3b 对照：物理槽空间（v2 方式，ref 不重映射）出现伪变化 | PASS | s_phys = 0.2942，z_phys = 147.09 |
| T4a 错误哨兵按 window_id 确定性 | PASS | — |
| T4b 映射版本绊线位于 det all_reduce 之前 | PASS | 源码结构校验 |
| E/F/G/H 四个映射提交点全部登记版本并镜像重映射/清零 | （代码插入点，见 minimal_diff.md §2） | — |

结论：交换提交（增量 H、warmstart、refine、预热）后检测分布不再有布局
耦合伪差；任何未登记的 p2l 变更路径在运行时触发 HOLD（绊线），而不是
用错误映射聚合。真机 8-rank 运行时验证列入 phase-2 短机制验证。

## 验收 2 — headroom 统一工作量尺度后重新校准（F2，机制部分）

| 检查 | 结果 | 关键数字 |
|---|---|---|
| T5 rate 尺度不变性：(h,n)→(c·h,c·n)，c∈{0.5,1,2,4,8} 轨迹全同 | PASS | 5/5 一致（W_after+event 全序列） |
| T6a 对照演示：冻结 v2 原始计数 θ 把窗大小当失衡（同 rate 0.20，n 差 4 倍 → 决策不同） | PASS | small=shrink_rejected_low_headroom vs big=hold_alert… |
| T6b v2.1 rate θ 在同两尺度下决策全同 | PASS | — |
| T7 h_rate ∈ [0,1]（随机 200 例 + 手算 0.5 例 + 不可分专家 floor 例 + 零保护） | PASS | h=60 → rate=0.5；热专家 → 0；n=0 → 0 |
| T14 占位/非法 rate 阈值拒启；合法与"惰性"（>1，仅 logging 用）接受 | PASS | (0,0)/(0.2,0.1)/(0.5,3.0)/(0.3,0.3) 全部 ValueError |

**阈值数值本身尚未冻结**：κ 与 z 带必须在新信号空间（逻辑身份）联合
重校，θ_r 为跨 W 合并的 rate 分位数 —— 规则见
`calibration_plan_v2p1.md`（预注册，W0-only，先于任何 logging run）。
本验收对象是**机制**（尺度不变、单位正确、占位拒启），不是数值。

## 验收 3 — 窗口及控制状态先共同确认再提交（F3）

| 检查 | 结果 | 关键数字 |
|---|---|---|
| T8 一致性判定：同哈希 → True；单 rank 异哈希 → False；每窗恰一次集合通信 | PASS | mock all_reduce 计数 = 1 |
| T9 非一致 → 9 状态字段全量回滚；streak≥2 → 冻结 w_init；后续窗 w_desync_freeze | PASS | streak=2, W=16, frozen |
| T9b 快照字段覆盖完整转移状态（9 字段） | PASS | — |
| T10 全 rank 同错 → 一致哨兵仍 HOLD（提交需提案非 None）；混合哨兵 → 非一致 HOLD；错误路径也恰好参加一次集合通信 | PASS | 日程不可能错位 |
| T11 回滚后重放：rec 逐字段全等（含 window_seq） | PASS | — |
| T12 决策哈希确定性、对全部提交字段+window_id 敏感、∈[0,2^58)、与错误命名空间分离 | PASS | 5/5 字段敏感 |
| T4c/d/e 确认集合通信每窗恰一次、错误路径与确认之间无 return、提交=提案∧一致 | PASS | 源码结构校验 |

结论：v2 残余缺口（单 rank 异常跳过每 16 窗哈希计数 → 日程错位）被
构造性消除：确认改为**每窗**、**无条件**参加，错误路径以确定性哨兵入列。

## 回归与保留

| 检查 | 结果 |
|---|---|
| T0a 冻结快照与 MANIFEST.sha256 一致 | PASS |
| T0b window_v2.py 未动（810c54da…） | PASS |
| T0c controller.py / config.py 相对冻结版**纯增量**（+275/-0、+35/-0） | PASS |
| T13b 冻结 v2 + 冻结 knob 重放 6 个 phaseB V2 正式日志：**350/350 窗** W_after+event 全对（phaseC 消融日志因 knob 不同按设计排除） | PASS |
| 冻结单元测试重跑 | ALL_TESTS_PASS |

## 遗留（phase-2，待批准）

- 真机短机制验证（8 rank 分布式）：交换窗伪差实测、绊线运行时触发、
  confirm/hold 实弹演练、额外开销入账（每层第二次 scatter_add +
  每窗 2 次 all_reduce）；
- 校准 logging runs（4×W0，W 固定 8/16/32/64）→ knob 冻结 → 1 次行为
  验证 run；共 5 次 W0 启动，**不是**性能战役；
- 独立正式协议另文另批。
