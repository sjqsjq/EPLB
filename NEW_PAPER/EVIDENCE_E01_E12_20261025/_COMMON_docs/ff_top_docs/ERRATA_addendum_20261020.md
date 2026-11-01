# ERRATA 追加勘误（2026-10-20，followup 战役阶段一 CPU 复核）

**性质：追加式。旧 REPORT_final.md / stats_report.txt / 表格一律不覆盖、不重写；本文件与 `cpu_audit/` 为唯一勘误载体。** 旧报告末尾仅追加一行指针。

## E1 完成度分解（回答"724505 vs 722200 的 2305 来自哪里"）

- 计划请求（MAIN）= 767256（168 格 planned_nreq 之和，含退役格计划量）。
- **冻结主口径 = 94.43%（724505/767256）**：退役挂死格（5 起，NCCL watchdog 终止、客户端被杀、完成数不可测）计 0；**退役崩溃但仍有实测的部分成功格保留实测 ok**（m_E4_F_b4 ok=1281/err=767；m_E4_F_b7 ok=1024/err=1024）——这正是 FREEZE §6/§9.9"RESULT_RETIRE 计入完成率报告、不得删除"的执行方式，先于分析冻结，非事后选择。
- **2305 = 1281 + 1024**，即上述两个部分成功格的实测成功请求。若把 7 个退役格全部计 0（严格口径）= 722200/767256 = **94.13%**。两个口径并列报告，主口径不变。
- 分解表 `cpu_audit/completion_decomposition.csv`：
  - 启动成功率（server ready & bench 完成判定 VALID）= 161/168 = **95.83%**；server 达到 ready（BOOT_OK 台账）= 168/168 = 100%（7 起失败均发生在 bench 期间，非启动失败）。
  - 请求完成率（冻结主口径）= **94.43%**；严格口径 = **94.13%**；仅对有测量启动的口径 = 724505/726296 = **99.75%**（参考口径，不作主口径）。
  - 失败启动中的部分成功请求 = **2305**（2 个崩溃格）；5 个挂死格部分成功 = 0（不可测，计 0）。
  - 逐条件：E1/E2 100%；E3 89.58%（两口径同，5 挂死格计 0）；E4 冻结 97.81% / 严格 95.00%（差=2305）。
  - P1：65536/65536 = 100%，32/32 VALID。
- 结论：旧报告 94.43% 数字与其口径定义自洽、可复算；作者指出的 94.13% 为同数据的严格变体，两者差异完全由 2 个部分成功格解释，无隐藏计数。

## E2 消融校正表（回答"A2@E3 的 p=0.0102 是否已校正"）

- **0.0102 是未校正 raw p**（配对块 t 检验，n=6）。旧报告与图 F7 均标注 "(unadj.)"，未将其作为校正后证据使用，也未据此宣称 superiority。
- 校正后（`cpu_audit/ablation_adjustment.csv`）：5-消融族内 Holm = **0.0512**；含 P-vs-F\* 的 6 检验次级族 Holm = **0.0615**。两者均 >0.05 → **校正后不显著**。
- 冻结族定义（FREEZE §6）：Holm 强制仅用于确证族两检验（P-vs-I、P-vs-E）；次级族（P-vs-F\* + 5 消融）单独申报、不与确证族混合校正；仅当从次级族提出形式 superiority 主张时才需校正——本战役未提出，A2@E3 仅作为机制层描述性证据（gap 引导 vs max-Δ）。
- A3@E1 保留在族内（n=8，raw p=0.0662），**未做任何事后删除**；不为旧战役追显著。

## E3 A3 解释修正

- 旧措辞（REPORT §0/§4 与图表注）："A3 对比 = 每启动执行噪声底" —— **修正为**：A3（denseSample）的**预期采样操纵未行使**（recorded_steps 796–798 与 P@E1 逐位相同、swaps 49≡49、sp_fwd 810≡810），因此该时间对比**不携带关于采样机制的因果信息**：既不能归为采样收益，也不能归为"纯噪声"（后者仍是归因）。正确表述：操纵缺席 → 对比不可解释为采样效应；E1 同臂每启动 ±8% 的变异性仅说明"若存在采样效应，本包功率不足以检测"，属适用性不足声明。
- 后续 followup 阶段六（采样验证包）按作者新协议以"真正行使"为前置门，不依赖本对比。

## E4 confirmed_grow 语义修正

- 旧表述风险：把 `confirmed_grow` 读作"外部工作负载变化被检测到"。**修正为**：`confirmed_grow` 是窗口控制器的**内部稳定扩窗/内部控制事件**——window_v2.py（冻结版）中 m_grow=3 个连续 stationary 窗（z<z_lo 且 h_w<θ_h_low 等内部统计条件）确认后允许 W 向上扩一档；属 v2.1 两窗确认漂移检测器词汇表内的控制律事件（`EV_CONFIRMED_GROW`，window_v2.py:38；sp_state.py:68-80 注释同义）。
- 对外部变化的反应性证据属于 alert/shrink 事件族（hold_alert / urgent_shrink / confirmed_shrink，z≥z_hi 或 z≥z_urgent）；grow 族不构成外部变化检测的证据。机制账本中 transitions（P@E2 1–4、P@E3 0–4）的解释同步按此口径。

## E5 配置导出（`cpu_audit/config_table.csv`，逐臂实际 server_args 解析 + CUDA graph 日志证据）

| 臂 | prefill/decode attn | CUDA graph 热路径 | deepep | R | 算法/重平衡周期 | S/H/W 与初始布局 |
|---|---|---|---|---|---|---|
| I | fa3 / fa3(默认) | **capture 成功**（"Capture cuda graph begin" 各 rank） | auto | 0 | 无 EPLB/PB | 无机制；init=trivial |
| E | fa3 | **禁用**（WARNING: deepep_mode=normal） | normal | 16（主）/0,16,32（P1） | eplb_algorithm=auto，period=256；recorder='stat'(auto-set) | 官方 init 规则；init_expert_location=trivial |
| P | fa3 | capture 成功 | auto | 0 | PB-OEPLB rev1：SP_MODE=adaptive，S 初值 1（在线可变），W 初值 16，ADW21 v2.1 冻结旋钮（κ=3875.1768798828125 等） | 逻辑主映射 identity；init=trivial |
| F | fa3 | capture 成功 | auto | 0 | PB 机制 + SP_MODE=fixed **S/H/W=1/16/16**；W_EA(ADW21) 适应按预注册解释保持 ACTIVE | 同 P |
| A1/A2/A3/A4 | fa3 | capture 成功 | auto | 0 | P + 单消融开关（record_all_phase / pair_rule=maxdelta / force_S=1 / no_commit） | 同 P |

- 所有臂 attention 后端一致（fa3；decode/prefill 未单独覆盖）；差异仅在 deepep 模式（E=normal，预注册偏差，EPLB_MECHANISM_NOTES.md sec.6）与机制开关。E 臂 CUDA graph 禁用 + recorder 常开 = 旧报告 §10.1 执行方式警示的物理来源，本表为其配置级证据。
