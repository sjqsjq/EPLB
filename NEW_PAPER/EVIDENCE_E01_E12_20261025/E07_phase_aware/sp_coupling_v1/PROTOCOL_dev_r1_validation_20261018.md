# PROTOCOL：rev1 开发验证（≤12 计时启动；非正式性能战役）
日期 2026-10-18 ｜ 作者指令四 ｜ 前置：REV1_quiet_exit_revision_20261018.md（最小 diff + 状态转移表）已交付，离线验收全绿（test_sp_rev1_logic.py 1996 断言 + 冻结套件 32+40 断言）

## 1. 设计

- **三臂**：`A_v1`（原自适应臂，OEPLB_SP_REV=0，v1 冻结行为）、`A_rev1`（修订臂，OEPLB_SP_REV=1）、`F_dense`（稠密参照，fixed S1/H16/W16）。
  三臂差异**仅**在 SP 块（含 REV 开关）；planner、α、迁移接受规则、S/H/W 档位、
  联合提交协议、全部冻结 v21 阈值逐字不变（R4）。
- **条件**：W1（w1_stable_code_4000，4000 请求）与 W2（w2_switch_code_chinese_8000，8000 请求），数据集 sha 锁定（与 P1 相同的三个 sha 常量）。
- **配对 block**：每条件 2 个 block；臂序**预先轮转**并冻结于
  sp_arm_order_rev1.json（seed=20261018，与 P1 同一生成惯例，块 2 = 块 1 逆序；
  3 臂 × 2 块无法完全位置平衡——沿用 P1 已声明的 caveat），
  生成于任何 rev1 计时数据之前。运行清单 run_list_sp_rev1_dev.csv（12 行，sha 锁定进驱动）。
- **预算**：TIMED_CAP=12（每启动一次服务 + 一次 bench；RESULT_RETIRE 消耗预算、不重试、不补位）。
  INFRA_CAP=3（零数据启动失败：同 cell 重启动一次，不消耗 TIMED 预算；超限 HALT）。
  SMOKE_R1 = 2 次 300 请求冒烟（A_v1、A_rev1 各 1），**不消耗** 12 启动预算（P0e 先例），
  仅验证 rev1 启动、banner rev/theta_out 字段、jsonl sp_exit_streak 字段、检查器新臂身份、bench ok。
- **失败语义（与 P1 相同）**：MECH_RETIRE / INTEGRITY_FAIL → 立即 HALT（exit 8），写完整报告，等待作者；
  HALT 永不因端点值触发；基础设施失败与结果性失败分开记账。

## 2. 指标（预先声明）

**主指标**：完整服务墙钟（bench t，秒）——全部成本保留在服务时间内
（并发 256、O1 数据集、real requests only、无延迟 floor、probe off、不扣除任何实测成本）。

**机制指标**（全部来自既有账本，无新剖析）：
1. 真实 forward 驻留：TRACK/QUIET/RECOVER 占比（S1 态精确、S4 态区间法，与 EXPORT §④ 同法）；
2. 记录次数：record_calls（PROF 行 + jsonl Σsp_recorded_steps×94 交叉核对）与 per-rank record CPU（MAX-rank，跨 rank 永不求和）；
3. 转换：次数、类型（QUIET→TRACK vs QUIET→RECOVER）、P 规则标签（jsonl sp_reasons 前缀 P1a/P1b/P1c/P1d/P2/P3）、seal 次数；
4. 有效布局更新事件：p2l_version 提交序列、Σswaps_started、首 p2l>0 的累计步数。

**统计口径**：每 cell n=2，**只报告方向**（逐 block 配对列出），不做正式优势声明、不做显著性检验、不合并 block。

## 3. 预声明问题（作者指令四原文）

- **Q1 修订是否减少贴线往返**：A_rev1 vs A_v1 的 QUIET 退出次数与"退出→再进入"往返数
  （v1 封存基线：W1 16 转换/11 seals，W2 33–34 转换/22–23 seals，32/32 退出为 h_rate 低余量触发）。
- **Q2 是否延长有效 QUIET 驻留**：QUIET 真实 forward 驻留占比
  （v1 封存基线：W1 7.7–11.3%，W2 7.6–7.8–11.5%）与最长连续 QUIET 段。
- **Q3 是否损害 W2 变化响应**：从首个真实漂移信号（首个 z≥z_lo 或 h_rate≥2×θ_out 的 H 周期）
  到 (a) S1 采样恢复、(b) 漂移后首个被接受 swap 的步数延迟；A_rev1 vs A_v1（封存）与 F_dense 同表列出。
  rev1 语义下该响应可能经 P3→TRACK（而非 v1 的立即 RECOVER）；延迟 ≥1 个 H 周期是否可接受由数据说话，不先验辩护。

## 4. 诚实收口条款（作者指令四原文落实）

- 若机制指标改善（Q1/Q2 成立）但主指标仍无方向一致的时间优势 → **如实收口**：
  报告"机制改善、时间无优势"，rev1 不升级为正式战役，**不自动扩大预算、不再开参数网格、
  不根据本轮服务时间回调 θ_out/m_exit**（参数冻结声明见 REV1 §4）。
- 若 Q3 显示 W2 响应受损（延迟增加且伴随路由/服务时间劣化）→ 如实报告损害，
  rev1 定性为"以变化响应换取驻留"的失败权衡，同样收口。
- 本轮 12 启动的**全部**结果（含不利结果）进 PROGRESS 账本与报告；不挑选有利重复。

## 5. 治理（全部沿用）

LOCAL ONLY 永不 push；W/H/S 值计时开始后不得改；跨 rank 时间只用 MAX/per-rank；
失败启动 retire 不重试；阈值只用校准材料一次冻结（θ_out=theta_r_act、m_exit=2，
先于本轮任何运行冻结）；window_v2.py 不动；保留全部已封存结果（本轮产物写 dev_r1/，
不覆盖 dev/）；驱动每启动前 sha256sum -c 部署锁（新锁文件
deployed_shas_sp_coupling_rev1.txt，由 deploy_sp_coupling.py 重新部署后生成）。

## 6. 旧日志重放免责（重申 REV1 §8）

本报告引用的封存基线（32 次退出、驻留占比、转换/seal 数）来自 v1 已封存运行；
rev1 的离线重放**仅**检查规则行为（给定已记录信号 rev1 会作何决定），
**不预测**修订后的吞吐、服务时间或路由轨迹——那些只能由本协议的新运行给出。
