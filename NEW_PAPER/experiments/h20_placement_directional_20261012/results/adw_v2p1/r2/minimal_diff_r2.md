# ADW v2.1 r2 — 最小 diff、各 rank 通信序列与状态一致性证据

日期：2026-10-17。性质：有限修版内的第二次修订（r1 → r2），处理
2026-10-16 只读核查确认的两个结构性问题。**v2.1 尚未冻结、未部署、
未跑过 logging**，r2 属预冻结修订。**不预告性能提升；不把旧 v2 结果
归因于这些问题。**

## 0. r1 → r2 处理的两个问题（用户核查提出）

- **问题①（判定非对称角落）**：r1 确认判定为各 rank 本地谓词
  `sum(h) == world × h_own`。哈希全同时正确；但原则上存在"某 rank 的
  h 恰等于分歧哈希的算术均值"时该 rank 单独判 True 的角落（概率
  ~world·2⁻⁵⁸/窗，构造上非零）→ 可能出现单 rank 提交、其余 HOLD 的
  分歧 verdict。
- **问题②（det 集合通信可被本地错误跳过）**：r1 的映射版本绊线位于
  det all_reduce **之前**；非对称触发时，出错 rank 跳过 det collective
  直达确认 collective，与其余 rank 的 [94,128] det all_reduce 配对
  → NCCL 尺寸不匹配挂起。

## 1. r2 修法（最小 diff）

**window_v2p1.py**（`d20d340f9d44ae99…`，r1 为 2ba43052…）：
`check_unanimous`（SUM 判定）替换为 `confirm_commit(h, prop_valid, reduce_fn)`：
每窗固定顺序两次 2 元 int64 集合通信（先 MIN 后 MAX），归约
`[h, prop_valid]`；判定 **commit ⟺ (h_min==h_max) ∧ (v_min==1)**，
只依赖归约后的比特一致数据 → **verdict 对称性由构造保证**（问题①
的谓词形态被整体移除，不再可表达）。`decision_hash`/`error_sentinel`/
`snapshot`/`restore`/状态机子类结构不变。

**controller.py**（`f5b9a1e8433f448d…`，r1 为 901f31b9…）：
`_window_control_v2p1` 重排为四相结构 + `_v2p1_confirm_reduce`（MIN/MAX）
替换 `_v2p1_confirm_allreduce`（SUM）：

- **相1（集合，无条件）**：det SUM all_reduce。任何本地检查都不先于它
  （问题②关闭：本地状态不可能让任何 rank 跳过）。
- **相2（本地，可恢复范围）**：映射版本绊线 → 信号 → step() 提案 →
  决策哈希；**内部无任何集合通信**。异常 → 回滚 + 确定性哨兵 +
  `prop_valid=0` → 联合 HOLD，服务继续。
- **相3（集合，无条件）**：确认 MIN + MAX。
- **相4（对称提交）**：`commit ∧ rec≠None ∧ prop_valid==1` 才提交
  （防御性合取；协议下 commit ⟹ 后两者成立）；否则一致回滚、
  streak+1、≥2 冻结 w_init（继承安全模式）。

**可恢复异常范围（明确定义，写入方法 docstring）**：
- 可恢复 = 仅相2 本地计算（绊线、信号、step、哈希）→ HOLD 并继续服务；
- 不可恢复 = 集合通信自身失败（相1/相3，NCCL 级故障）→ 控制器 HOLD 并
  大声记日志（`det_ok=false` / `confirm_reduce_failed`），仍按序尝试后续
  collective 以保日程对齐；通信真断时由服务器 NCCL watchdog 终止进程，
  不在控制器内掩盖。

**纯增量性质保持**（相对冻结 v2）：controller.py +319/-0，
config.py +35/-0；window_v2.py 未动
（`810c54daf1c4b9ec…`）。config.py 本轮零改动。
完整 diff：`controller_v21_r2.diff`。

## 2. 各 rank 完整通信序列证据（真实多进程故障注入）

测试：`$EXP/scripts/test_v2p1_multiprocess_faults.py`——真实
torch.distributed 进程组（非 mock），逐窗镜像控制器 r2 四相结构，
按设计日程注入故障；每窗用 all_gather_object 收集**全部 rank** 的
协议通信序列、verdict、committed、完整控制器状态元组（12 字段+历史），
断言逐 rank 全同。证据原件：`fault_evidence_cpu4.json`（gloo，world=4，
det [4,8]）、`fault_evidence_gpu8.json`（**nccl，world=8，NVIDIA H20，
真实形状 det [94,128]=12032 元 int64**）。

注入日程：w3 单 rank 本地异常；w5 单 rank **版本绊线（非对称）**；
w7/w8 单 rank 分歧提案（h_rate+0.37 → 哈希必不同）；w10 全 rank 异常。

### cpu4（gloo，world=4）

| 窗 | 注入 | 各rank协议序列（全同） | 序列一致 | 状态一致 | verdict | committed | event | W/eff |
|---|---|---|---|---|---|---|---|---|
| w1 | — | 1:det_sum:32/1:cmin:2/1:cmax:2 | True | True | True | True（设计 True） | init | 16→eff 16 |
| w2 | — | 2:det_sum:32/2:cmin:2/2:cmax:2 | True | True | True | True（设计 True） | hold | 16→eff 16 |
| w3 | r1:local_error | 3:det_sum:32/3:cmin:2/3:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 16→eff 16 |
| w4 | — | 4:det_sum:32/4:cmin:2/4:cmax:2 | True | True | True | True（设计 True） | hold | 16→eff 16 |
| w5 | r3:version_mismatch | 5:det_sum:32/5:cmin:2/5:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 16→eff 16 |
| w6 | — | 6:det_sum:32/6:cmin:2/6:cmax:2 | True | True | True | True（设计 True） | grow_blocked_headroom | 16→eff 16 |
| w7 | r1:h_rate_delta | 7:det_sum:32/7:cmin:2/7:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 16→eff 16 |
| w8 | r2:h_rate_delta | 8:det_sum:32/8:cmin:2/8:cmax:2 | True | True | False | False（设计 False） | confirm_fail_freeze | 16→eff 16 |
| w9 | — | 9:det_sum:32/9:cmin:2/9:cmax:2 | True | True | True | True（设计 True） | w_desync_freeze | 16→eff 16 |
| w10 | r0:local_error,r1:local_error,r2:local_error,r3:local_error | 10:det_sum:32/10:cmin:2/10:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 16→eff 16 |
| w11 | — | 11:det_sum:32/11:cmin:2/11:cmax:2 | True | True | True | True（设计 True） | w_desync_freeze | 16→eff 16 |

final：`{"all_windows_ok": true, "n_windows": 11, "commits_matched_design": true, "exactly_3_protocol_collectives_per_window_every_window": true, "final_W": 16, "final_eff_W": 16, "frozen_after_double_failure": true}`

### gpu8（nccl，world=8，H20，真实 det 形状）

| 窗 | 注入 | 各rank协议序列（全同） | 序列一致 | 状态一致 | verdict | committed | event | W/eff |
|---|---|---|---|---|---|---|---|---|
| w1 | — | 1:det_sum:12032/1:cmin:2/1:cmax:2 | True | True | True | True（设计 True） | init | 16→eff 16 |
| w2 | — | 2:det_sum:12032/2:cmin:2/2:cmax:2 | True | True | True | True（设计 True） | hold | 16→eff 16 |
| w3 | r1:local_error | 3:det_sum:12032/3:cmin:2/3:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 16→eff 16 |
| w4 | — | 4:det_sum:12032/4:cmin:2/4:cmax:2 | True | True | True | True（设计 True） | hold | 16→eff 16 |
| w5 | r7:version_mismatch | 5:det_sum:12032/5:cmin:2/5:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 16→eff 16 |
| w6 | — | 6:det_sum:12032/6:cmin:2/6:cmax:2 | True | True | True | True（设计 True） | confirmed_grow | 32→eff 32 |
| w7 | r1:h_rate_delta | 7:det_sum:12032/7:cmin:2/7:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 32→eff 32 |
| w8 | r6:h_rate_delta | 8:det_sum:12032/8:cmin:2/8:cmax:2 | True | True | False | False（设计 False） | confirm_fail_freeze | 16→eff 16 |
| w9 | — | 9:det_sum:12032/9:cmin:2/9:cmax:2 | True | True | True | True（设计 True） | w_desync_freeze | 16→eff 16 |
| w10 | r0:local_error,r1:local_error,r2:local_error,r3:local_error,r4:local_error,r5:local_error,r6:local_error,r7:local_error | 10:det_sum:12032/10:cmin:2/10:cmax:2 | True | True | False | False（设计 False） | confirm_hold | 16→eff 16 |
| w11 | — | 11:det_sum:12032/11:cmin:2/11:cmax:2 | True | True | True | True（设计 True） | w_desync_freeze | 16→eff 16 |

final：`{"all_windows_ok": true, "n_windows": 11, "commits_matched_design": true, "exactly_3_protocol_collectives_per_window_every_window": true, "final_W": 16, "final_eff_W": 16, "frozen_after_double_failure": true}`

**关键读法**：
- 每一窗、每一 rank 恰好 3 次协议集合通信（det_sum + cmin + cmax），
  故障窗也不例外 → 日程对齐由实证支持（r1 的问题②场景 w5：非对称
  绊线 rank 仍完成了 det_sum，无挂起，测试在超时内正常结束）；
- verdict/committed/状态元组逐窗全 rank 一致（含 HOLD 窗与冻结窗）；
- committed 序列与设计日程完全一致；w8 双失败 → 全 rank 冻结
  w_init=16（gpu8 上 w6 曾一致提交 confirmed_grow 16→32，冻结后回到 16，
  证明提交与回退都是全 rank 同步的）；
- w10 全 rank 异常：哨兵哈希一致（h_min==h_max）但 v_min=0 → 联合
  HOLD，绝不提交。

## 3. 协议开销入账（仅计量，非服务测量，非性能声明）

gpu8/nccl/H20/world=8、真实 det 形状 [94,128] int64，30 预热 + 50 计时：

| 项 | 中位数 |
|---|---|
| 整窗协议（3 次集合 + 本地 step/哈希/克隆，含同步） | 567–572 µs/rank（p90 600–608，max 2045–2102） |
| det SUM all_reduce 单独 | 60–61 µs |
| 确认 MIN+MAX（2×2元）单独 | 208–211 µs |

运行时该开销全部计入 rank0 日志 `c_window_ms`（含相1+相2+相3）。
相对每窗数十秒的服务窗口，此为上界性质的入账数字；正式开销结论只能
来自带服务的机制验证 run（phase-2，未启动）。

## 4. 离线验收更新（T0–T15 全过）

- T4b：det all_reduce 无条件（绊线已移到 collective 之后、det 使用之前）；
  T4b2：相2 无集合通信；T4c：confirm_commit 恰一次且位于相2/相4 之间；
  T4g：确认 reducer 为 MIN/MAX 各一次；
- T8：共同判定四情形（全同全有效→commit；单哈希分歧→hold；单提案
  无效→hold（尽管哈希全同）；全哨兵→hold）+ 顺序 MIN→MAX + 每窗两次；
- T10：错误窗恰好参加两次确认 collective；一致哨兵仍 HOLD；
- T15（新）：verdict 对称性——分歧世界中不同本地哈希的两个 rank 视角
  得到**相同** HOLD verdict（r1 的不对称谓词形态已不可表达）；
- T13b：350/350 窗冻结 v2 重放回归**不受 r2 影响**（仍全对）；
- 冻结 v2 单元测试重跑 `ALL_TESTS_PASS`。

## 5. G1/G4 校准判据修订（logging 前完成，见 calibration_plan_v2p1.md 修订 r2）

- 新增 §2.1 **合格窗**定义：`committed==true ∧ protocol_rev=="r2" ∧
  event ∉ {init, uncertain_*, confirm_hold, confirm_fail_freeze,
  w_desync_freeze} ∧ z≠null`；全部拟合样本只取合格窗；单 run 合格窗
  <30 → "样本不足"如实报告并停止拟合。
- **G1（修订）**：4 个固定 W 的合格静止窗 `h_rate` 中位数
  max/min ≤ 1.5；同批窗 raw `h_w` 中位数之比**仅报告存档**（预期 ~3×，
  作 F2 前后对照）；样本不足不记 PASS。
- **G4（修订）**：合格 swap 窗（相邻合格窗间 p2l_version 变化）中位
  z ≤ z_lo **且**样本 ≥5（不足记"不确定"；允许并入行为验证 run 的
  swap 窗一次）；辅助报告 swap/静止窗 z p90 之比；失败必须排查后
  才能冻结。
- G2/G3/G5、拟合规则 FORM、W0-only 边界、5 次启动预算、反调参条款
  全部不变。

## 6. 状态与边界

- 本轮改动：window_v2p1.py（判定原语）、controller.py（v21 方法四相
  重排）、离线测试（T4/T8/T10 更新 + T15 新增）、故障注入测试（新）、
  校准方案（G1/G4+§2.1 修订）。config.py、v2p1_signals.py、window_v2.py、
  冻结 v2、交付物 G、全部旧结果：**未动**。
- 未部署 site-packages，未运行任何 logging run / 行为验证 / 服务战役。
- 8-rank 设备验证为协议级最小验证（无模型、无请求、分钟级、GPU 空闲
  时段执行），不构成任何服务或性能结论。
