# REPORT_final — final_plan_20261019 统一分析报告（冻结 §6/§8.2 规则执行）

- 生成时间：2026-10-20（UTC），统一分析阶段（§14.2 最后一项）。
- 范围：主矩阵 168 次计时启动（161 VALID + 7 RESULT_RETIRE）+ P1 32 次（32 VALID）+ 统一统计/图表。
- 状态：**战役全部冻结步骤完成**。确证族、次要族、描述族、完成度、机制计数、局限性、§15 收尾措辞全部在下文给出。
- 产物位置：`results/final_plan_20261019/analysis/unified/`（本目录）。代码：`build_tables.py`、`stats_unified.py`、`figures.py`；数据：`main_table.{csv,json}`（168 行）、`p1_table.{csv,json}`（32 行）、`unified_stats.json`、`stats_report.txt`、`f6_forest_values.csv`、`f7_forest_values.csv`；图：`fig_f6_main_matrix.png`、`fig_f6_confirmatory.png`、`fig_f7_ablations.png`、`fig_f8_p1.png`。
- 指标口径：T = 客户端单调钟 total_time_s（与 verdict t 一致性断言 |T−t|<0.5s 全部通过）；跨 rank 时间只用 MAX/每 rank，不求和；不扣除任何实测成本；real requests only；probe_mode=off；无 delay floor。

---

## 0. 执行摘要（headline，诚实口径）

1. **确证族（冻结：恰好 2 个检验，Holm α=0.05，complete blocks [1,2,4,5,7,8]，n=6）**
   - **P-vs-E：G = +36.53% [+34.81%, +38.27%]，Holm p = 3.77e-08 → 显著。**（PB 自适应放置相对官方 EPLB，四工作流等权对数时间比。）
   - **P-vs-I：G = +2.21% [−0.28%, +4.75%]，Holm p = 0.0718 → 不显著。**（相对恒等放置的整体优势未确立；单工作流 E2 显著，见 §3。）
2. **P-vs-F\*（次要族，complete blocks [1,2,5]，n=3）：G = −0.52% [−2.92%, +1.94%]，p = 0.457 → 不显著。** 自适应相对校准后固定窗口的附加收益**未确立**（n=3，CI 很宽）。按 §15 第一行措辞：不得写"全部收益来自 adaptive"。
3. **机制层正向结果（次要族，描述性证据）**：gap 引导候选规则（PB_gap）优于纯 max-Δ 贪心 —— A2@E3：A2 更慢 G=+8.21% [+2.87%,+13.82%]（未校正 p=0.0102；5 个消融族内 Holm 后处于边界 0.050），且 A2 换入更多（26–28 vs P 12–14 swaps/boot）却无服务收益；A2@E2 同向（+8.63%，p=0.0588）。
4. **消融操纵核验**：A4@E3（只算不提交）操纵成立（nocommit_planned_ops 8853–8878 > 0 且 swaps_started=0）→ 其对比 +1.87% ns 说明 E3 上"提交换入"相对"只规划"无可测墙钟收益；A1@E4（全阶段记录）操纵成立（decode_record_batches 1020–1054 vs P=0）→ +2.44% ns，decode 记录成本≈0；**A3@E1（denseSample）未被行为行使**（recorded_steps 796–798 与 P@E1 完全相同、swaps 49=49）→ 按 §7 标"未行使"，其对比（+5.71%，p=0.0662）解释为每启动执行噪声底（E1 同臂 T 范围 ±8%），不作机制推断。
5. **P1（描述性区间估计，§10.3 边界：无 ad-hoc SLO、非确证）**：长入长出流（in2048/out128）下 P 臂全面领先官方 EPLB 各冗余档：PL（conc=64）P 优势 G 形 +155.3%…+162.3%；PH（conc=256）+46.1%…+47.6%。冗余 R∈{0,16,32} 对 E 臂墙钟**无清晰效应**（差异在同臂每启动噪声内），而 KV 容量随 R 单调下降（P 227.2k ≈ R0 227.1k → R16 208.6k → R32 190.2k tokens，min over 8 ranks）。32/32 VALID、0 错误、完成度 100%。
6. **完成度**：MAIN 724505/767256 = **94.43%**（7 个退役崩溃格按 0 保守计入；部分完成格保留实测 ok）；P1 65536/65536 = **100%**。退役格全部计入完成度报告、从未删除（§9.9）。
7. **闭环判定**：168+32 格全部执行完毕且统一分析按冻结规则完成 → 按计划 §16/§9，**性能闭环成立**（附带 §10 全部局限性与 §11 措辞约束）。

---

## 1. 设计与冻结回顾

- 主矩阵 168 = E1{I,E,P,F,A3}×8 + E2{I,E,P,F,A2}×8 + E3{I,E,P,F,A2,A4}×8 + E4{I,E,P,F,A1}×8（run_list_main.csv sha256=c2e06df4…，驱动每次 boot 前复核）。
- P1 32 = {P, E-R0, E-R16, E-R32} × {PL conc=64, PH conc=256} × 4 块（run_list_p1.csv sha256=b324f8ea…；轮转 seed=405220380001 预注册）。
- 统计冻结（§8.2）：g_b(P,B) = (1/4)Σ_j log(T_{B,j,b}/T_{P,j,b})；G = exp(mean g_b)−1；complete-block 单元；对数比上 t 区间；Holm 仅用于 2 个确证检验；次要族（P-vs-F\* + 5 消融）单独声明、未校正 p 全部报告；每工作流另报 raw 时间/配对差/time-reduction(1−T_P/T_B)/throughput-gain(T_B/T_P−1)，互不混用。
- 请求流冻结 sha（先于任何主矩阵计时）：e1 dee24c50…，e2 fa788238…，e3 d55fe2f7…，e4 a024c5e6…；P1 流 3c09e027…（registry 见 `C_configs/FREEZE_final_plan.md` §4）。N 实际值：E1=3221、E2=4082、E3=8192、E4=2048（唯一内容容量，预登记调整，重复请求不冒充独立样本）。

## 2. 确证族结果（F6b：fig_f6_confirmatory.png）

| 检验 | complete blocks | n | G=exp(mean)−1 | 95% t-CI | Holm p | 判定 |
|---|---|---|---|---|---|---|
| P-vs-E | 1,2,4,5,7,8（丢 3,6：P@E3 退役） | 6 | **+36.53%** | [+34.81%, +38.27%] | 3.773e-08 | **显著** |
| P-vs-I | 1,2,4,5,7,8（同上） | 6 | **+2.21%** | [−0.28%, +4.75%] | 7.182e-02 | 不显著 |

逐块 g_b（P-vs-E）：b1 +0.3041, b2 +0.2965, b4 +0.3091, b5 +0.3068, b7 +0.3284, b8 +0.3232（块间高度一致；E4 项贡献 +0.94…+1.08，E2 项 +0.13…+0.28，E1/E3 项≈0 或负）。
逐块 g_b（P-vs-I）：b1 +0.0023, b2 +0.0121, b4 +0.0525, b5 −0.0050, b7 +0.0466, b8 +0.0224（符号不一致、幅度小 → 整体不显著与块间异质性一致）。

## 3. 每工作流效应（未校正；四类量分别命名；F6a：fig_f6_main_matrix.png）

| 对比 | 工作流 | n块 | T_P 范围(s) | T_B 范围(s) | G | 95% CI | p | time-reduction | throughput-gain |
|---|---|---|---|---|---|---|---|---|---|
| P-vs-I | E1 | 8 | 315.6–370.7 | 322.9–332.0 | −1.90% | [−6.28,+2.69] | 0.354 | −2.07% | −1.77% |
| P-vs-I | E2 | 8 | 84.6–97.1 | 98.4–104.7 | **+13.50%** | [+8.61,+18.60] | 2.53e-04 | +11.78% | +13.63% |
| P-vs-I | E3 | 6 | 178.5–206.3 | 188.4–195.6 | +1.52% | [−4.42,+7.82] | 0.549 | +1.36% | +1.66% |
| P-vs-I | E4 | 8 | 90.4–101.7 | 89.1–95.0 | −4.51% | [−8.99,+0.18] | 0.057 | −4.88% | −4.38% |
| P-vs-E | E1 | 8 | 315.6–370.7 | 324.1–331.5 | −1.56% | [−5.95,+3.04] | 0.443 | −1.72% | −1.43% |
| P-vs-E | E2 | 8 | 84.6–97.1 | 103.9–112.0 | **+21.17%** | [+15.20,+27.45] | 4.32e-05 | +17.34% | +21.36% |
| P-vs-E | E3 | 6 | 178.5–206.3 | 194.8–204.1 | +4.44% | [−2.37,+11.73] | 0.159 | +4.09% | +4.62% |
| P-vs-E | E4 | 8 | 90.4–101.7 | 248.6–270.9 | **+172.73%** | [+160.35,+185.69] | 2.89e-10 | +63.28% | +173.10% |
| P-vs-F | E1 | 8 | 315.6–370.7 | 321.2–366.8 | +4.40% | [−2.96,+12.33] | 0.206 | +3.88% | +4.74% |
| P-vs-F | E2 | 8 | 84.6–97.1 | 84.1–93.8 | −1.27% | [−5.32,+2.96] | 0.496 | −1.40% | −1.16% |
| P-vs-F | E3 | 4 | 178.5–206.3 | 179.9–198.8 | −1.01% | [−9.77,+8.60] | 0.751 | −1.15% | −0.88% |
| P-vs-F | E4 | 6 | 91.5–101.7 | 90.4–109.3 | +1.09% | [−4.75,+7.28] | 0.659 | +0.95% | +1.23% |

读法：PB 族（P/F）在 E2（prover 小 prefill）上一致优于 I 与 E（放置真收益）；E4（长 decode）上 P≈I≈F 而 E 慢 2.6×（执行方式主导，见 §10.1）；E1/E3 上四臂互不显著（放置与恒等在 prefill 主导流上差异小）。**P 不占优的格子（E1、E4-vs-I）全部如实列出，未隐藏。**

## 4. 次要族（单独声明；F7：fig_f7_ablations.png）

- **P-vs-F\***（complete blocks [1,2,5]，n=3）：g_b = −0.0017/+0.0024/−0.0163 → G = **−0.52% [−2.92%,+1.94%]**，p=0.457，ns。小 n 警示：CI 宽，仅能排除 |效应|>~3% 级别之外的结论；自适应 vs 固定窗口的增量收益未确立。
- 5 个条件×消融对比（各 vs 同条件 P，未校正 p 全报；族内 Holm 仅作敏感性备注）：

| 对比 | n块 | G | 95% CI | p(未校正) | 行为核验 | 解释 |
|---|---|---|---|---|---|---|
| A3 denseSample@E1 | 8 | +5.71% | [−0.49,+12.29] | 0.0662 | **未行使**（recorded_steps 796–798 ≡ P；swaps 49≡49） | 对比=每启动执行噪声底（§7 规则），不作机制推断 |
| A2 maxdelta@E2 | 8 | +8.63% | [−0.40,+18.49] | 0.0588 | 行使（swaps 6–8 vs P 2） | max-Δ 换入更多、服务更慢（边界） |
| A2 maxdelta@E3 | 6 | **+8.21%** | [+2.87,+13.82] | 0.0102 | 行使（swaps 26–28 vs P 12–14） | gap 引导优于 max-Δ 的服务层证据（族内 Holm 边界 0.050，不宣称形式 superiority） |
| A4 noCommit@E3 | 6 | +1.87% | [−3.91,+8.01] | 0.451 | 行使（nocommit_planned 8853–8878>0，swaps=0） | E3 上提交换入相对只规划无可测墙钟收益 |
| A1 allphase@E4 | 8 | +2.44% | [−3.20,+8.41] | 0.347 | 行使（decode_record_batches 1020–1054 vs P=0） | decode 阶段记录成本≈0（噪声内） |

## 5. 机制计数（每启动范围；sp_run_end.json 抽取）

| 臂@条件 | swaps_started | transitions | sp_fwd | recorded_steps | nocommit_planned | decode_record_batches |
|---|---|---|---|---|---|---|
| P@E1 | 49 | 0 | 810 | 796–798 | 0 | 0 |
| P@E2 | 2 | 1–4 | 266 | 135–204 | 0 | 0 |
| P@E3 | 12–14 | 0–4 | 522 | 438–509 | 0 | 0 |
| P@E4 | 12–21 | 0 | 1162–1178 | 134–136 | 0 | 0 |
| F@E1/E2/E3/E4 | 49 / 2 / 13–14 / 12–14 | 0 | 810/266/522/1162 | 796–797/252–254/509–510/132–136 | 0 | 0 |
| A1@E4 | 12–22 | 0 | 1162–1194 | 136–140 | 0 | **1020–1054** |
| A2@E2 / A2@E3 | 6–8 / 26–28 | 0 | 266 / 522 | 252–254 / 508–510 | 0 | 0 |
| A3@E1 | 49 | 0 | 810 | 796–798 | 0 | 0 |
| A4@E3 | **0** | 0 | 522 | 508–509 | **8853–8878** | 0 |

P@E2/E3 的 transitions>0 与 recorded_steps 变动 = 自适应窗口状态机实际行使（ADW21 v2.1 ACTIVE，冻结 κ=3875.17 等旋钮）；F 臂 transitions=0（固定窗口无状态迁移）与定义一致。

## 6. P1 描述性结果（F8：fig_f8_p1.png；§10.3：仅区间估计，无 SLO、非确证）

| 压力 | 臂 | T mean (s) | T range | tps_out | ttft_p99 | lat_p99 | ok/err | KV_min |
|---|---|---|---|---|---|---|---|---|
| PL(64) | P | 402.2 | 380.3–429.2 | 653.1 | 11.13s | 17.01s | 8192/0 | 227258 |
| PL | E-R0 | 1056.7 | 959.2–1144.7 | 249.4 | 6.89s | 36.24s | 8192/0 | 227053 |
| PL | E-R16 | 1048.6 | 959.4–1180.3 | 251.5 | 6.96s | 36.67s | 8192/0 | 208673 |
| PL | E-R32 | 1027.2 | 969.5–1105.6 | 256.0 | 8.39s | 36.52s | 8192/0 | 190219 |
| PH(256) | P | 273.9 | 271.9–276.7 | 957.2 | 30.75s | 39.91s | 8192/0 | 227162 |
| PH | E-R0 | 401.1 | 378.9–446.0 | 656.5 | 28.69s | 57.18s | 8192/0 | 225713 |
| PH | E-R16 | 404.6 | 386.8–429.3 | 649.0 | 27.90s | 56.15s | 8192/0 | 207150 |
| PH | E-R32 | 402.8 | 388.3–424.4 | 651.5 | 28.13s | 56.62s | 8192/0 | 190219 |

配对 P 优势（G 形，t-CI df=3，**区间估计**）：PL vs R0/R16/R32 = +162.32% / +160.19% / +155.28%；PH = +46.13% / +47.62% / +47.00%。
观察（描述性）：(a) R 对 E 臂墙钟无清晰效应（R0↔R32 差异在同臂噪声内，PL 上 R32 名义最快）；(b) KV 容量随 R 单调近线性下降（每 +16 R ≈ −18.5k tokens，min over 8 ranks），P 无冗余 ≈ R0 容量；(c) P 臂 PL 上 ttft_p99 高于 E 臂（11.1 vs 6.9–8.4s）但 e2e lat_p99 大幅更低（17.0 vs 36.2–36.7s）；PH 上 P ttft_p99 30.75s 略高于 E 臂 ~28s（准入前端排队），总墙钟与 lat_p99 仍占优；(d) 可行性 smoke（E-R32、C_hi=256 突发 512 请求）先于计时通过：SMOKE_PASS（512/512 ok、0 err、KV=190315）。

## 7. 完成度报告（§9.9：退役计入、不删除）

| cond | arm | boots | valid | retired | completed/planned | rate |
|---|---|---|---|---|---|---|
| E3 | F | 8 | 5 | 3 | 40960/65536 | 62.50% |
| E3 | P | 8 | 6 | 2 | 49152/65536 | 75.00% |
| E4 | F | 8 | 6 | 2 | 14593/16384 | 89.07% |
| 其余 18 组 | — | 8 | 8 | 0 | 全量 | 100.00% |

- MAIN 聚合请求完成度：**724505/767256 = 94.43%**（NCCL 挂死格客户端被 watchdog 终止、完成数不可测，保守计 0；部分完成格保留实测：m_E4_F_b4 ok=1281/2048 err=767；m_E4_F_b7 ok=1024/2048 err=1024）。
- P1 聚合：**65536/65536 = 100%**（32/32 VALID，0 错误，0 事件）。

## 8. Table 1 — 环境 / 版本 / 开关 / 每臂容量

| 项 | 值 |
|---|---|
| 硬件 | 单机 8×NVIDIA H20（nvidia-smi） |
| 模型 | Qwen3-235B-A22B-FP8（/root/models/ms_cache/…，quantization=fp8，dtype=bfloat16） |
| 栈 | sglang 0.5.6.post2；torch 2.9.1+cu128；CUDA 12.8；attention fa3；sampling flashinfer；moe_runner deep_gemm |
| 并行 | tp=8、dp=8（enable_dp_attention）、ep=8、pp=1；mem_fraction_static=0.8；chunked_prefill=1024；max_prefill_tokens=16384；fcfs；disable_radix_cache=True；cuda_graph_max_bs=128；watchdog 600s |
| MoE 传输 | moe_a2a_backend=deepep；**deepep_mode：I/P/F/A 臂=auto；E 臂=normal（官方 EPLB 路径要求；CUDA graph 因此禁用 — 冻结 §3 公平性警示）** |
| I 臂 | 静态 BASE（无记录、无换入；trivial init） |
| E 臂 | 官方 EPLB：enable_eplb=True，ep_num_redundant_experts=16（主矩阵）/ P1 {0,16,32}，eplb_rebalance_num_iterations=256；参数串 "auto R 256" |
| P 臂 | PB-OEPLB ADW21 v2.1：threshold_ratio=1.02、min_prefill_tokens=256、sync_window=16、max_total_swap_layers=94、max_swaps_per_layer=64、max_total_ops=300、min_swap_ops=8、min_record_tokens=32、decay=0、自适应窗口 V2.1（floor=8、shift_cos=0.85、stable_cos=0.95、confirm=2）、冻结校准常数 κ=3875.17（两阶段校准后一次冻结，计时开始后未改）；相选择记录（仅 prefill） |
| F 臂 | PB 机制 + 固定窗口 S/H/W=1/16/16（参数串 "1 16 16"），无自适应 |
| A 臂 | A1=P+全阶段记录；A2=P+max-Δ 配对规则；A3=P+强制 S=1 密集采样；A4=P+只规划不提交 |
| 并发/超时 | 主矩阵全 256；bench 超时 E1/E2 2400s、E3/E4 3600s（防挂死，非判据）；P1 PL=64 / PH=256（排队准入允许，预注册选 256 理由见冻结 §P1） |
| 版本/ provenance | repo /workspace/EPLB 分支 sp_coupling_dev；驱动 FREEZE 提交 6eb430f；run_list sha 见 §1；smoke_slice sha 8b224dee…、calib_screen sha d4fa14d2…（D_matrix/runlist_locks.json）；分析代码见本目录 |
| 每臂原生 KV | 同 BASE 0.8；实测 min-over-ranks：P≈227.2k、E-R0≈227.1k、E-R16≈208.6k、E-R32≈190.2k（P1 实测；E 臂权重副本代价如实报告，§10.2 口径） |

## 9. Table 2 — 全部失败 / 不支持 / 范围 / 旧阴性结果

| 类别 | 条目 | 处置 |
|---|---|---|
| 结果性失败（7） | m_E3_P_b3、m_E3_P_b6、m_E3_F_b3、m_E3_F_b4、m_E3_F_b8（NCCL watchdog 挂死：大批量 swap race，299-swap COALESCED batch-mismatch 指纹）；m_E4_F_b4、m_E4_F_b7（崩溃+部分完成，err>0） | 全部 RESULT_RETIRE，计入完成度（§7），不删除、不重跑替代（冻结预案：同格二次失败→INFRA_MISSING+HALT 未触发） |
| 基础设施事件（1） | 1 次 boot 级失败，按冻结预案单次重启恢复（INFRA_CAP=3 内，预算 1/3） | 与结果性失败分开记录；不影响任何格判定 |
| 校准期机制退役（1，历史） | CALIB1 MECH_RETIRE（主矩阵前的机制类退役） | 已封存记录，不计入主矩阵 |
| 监测缺口（基础设施类，非结果） | ScheduleWakeup 5 次误发（09:45/12:30/16:26/17:49/22:04Z）；最大空窗 2026-10-20 01:26–04:08Z（P1_DONE 后统一分析未自动启动） | 台账/PROGRESS #20 记录；驱动自跑无数据丢失；不改变任何结果 |
| 不支持/范围 | 单机单模型四工作流；无质量类指标（MMLU 型不在范围）；无多节点；P1 仅描述；E4 P-vs-I 边界负向（−4.51%，p=0.057）与 P@PH ttft_p99 偏高均披露 | 进 §10 局限性 |
| 旧阴性结果 | h20_placement_directional_20261012 历史封存（rev1 阴性/混合）保留不改；θ 未重拟合；未从历史评价墙钟挑数据（§16） | 封存原样 |
| 崩溃取证 | $F 下 core-pt_nccl_watchdg-* / core-sglang::data_pa-* 共 9 个 core 文件保留 | 证据留存，不清理 |

## 10. 局限性与效度威胁

1. **E 臂执行方式偏差（冻结 §3 预注册警示，必须随 P-vs-E 解释出现）**：本栈官方 EPLB 要求 deepep_mode=normal → CUDA graph 禁用、decode 无 low-latency buffer，且 recorder 常开。故 E4 +172.73% 与 P1 的 P 优势中，**执行方式惩罚与放置策略效应不可分离**；prefill 主导流（E1/E3）上 E≈P≈I 提示该惩罚在 O1 流上小，E2 的 +21% 间隙可能含官方路径 recorder/周期 rebalance 开销。P-vs-I 确证比较不受此影响。机制分解超出冻结范围（M3 trace / M4 replay 留待作者）。
2. **swap-race 挂死族**：7 起全部集中于大换入量格（E3 w17 签名 / E4 F 臂）；PB 提交路径在并发大批量下的稳定性限制，watchdog 捕获并退役；I/A 臂零挂死、P 臂仅 E3 两起。论文须作为系统限制披露。
3. **P-vs-I 整体不显著 + E4 边界负向**：自适应在 decode 重、低并发流上开销无对应收益（−4.51%，p=0.057）；不隐藏。
4. **小 n**：确证族 n=6 块（丢 3,6）；P-vs-F\* n=3 → 自适应增量收益的 CI 过宽，只能"未确立"；P1 每臂×压力 4 boot，仅区间估计。
5. **P 臂 PH ttft_p99 偏高**（30.75s vs E ~28s）：准入前端排队；e2e 与墙钟仍占优，披露。
6. **A3 未行使**：denseSample 操纵未改变行为 → 该对比仅噪声底信息。
7. 单模型/单机/4+1 流/并发档位有限：外部效度受限；未做质量指标。
8. 监测误发为会话基础设施缺陷（非实验缺陷），已如实记录。

## 11. §15 收尾措辞适用性

适用**第一行情形**（P 优于 E 确证；P-vs-I 整体不显著；P 不优于 F）：
- 可写：PB 族放置（含自适应实现）相对**官方 EPLB 路径**在本四工作流等权口径下有显著且大的墙钟优势（+36.53%，Holm p=3.8e-08）；在 prover 型小 prefill 流上相对恒等放置有显著收益（E2 +13.50%）；P1 长入长出压力扩展中 P 臂全面领先且容量无冗余代价。
- 可写（机制层）：gap 引导候选规则优于 max-Δ 贪心（A2@E3 +8.21%，未校正 p=0.0102，族内 Holm 边界）；noCommit/allphase 消融显示提交与 decode 记录的墙钟代价≈0（E3/E4 噪声内）。
- **不可写**：全部收益来自 adaptive；adaptive 相对校准固定窗口有性能优势（P-vs-F\* ns，n=3）；P-vs-I 整体优势（ns）；任何 SLO 达标声明（P1 仅描述）。
- 必须随 P-vs-E 出现：§10.1 执行方式警示；随退役格出现：§7 完成度口径。

## 12. 治理合规声明

- 全部提交 LOCAL ONLY，未 push；W/H/S 与 κ 等旋钮计时开始后未改；跨 rank 时间只用 MAX/每 rank；失败启动按冻结预案（INFRA_CAP 内单次重启；同格二次失败→HALT 未触发）；不挑选有利重复、不做中期选参、不裁剪矩阵；阈值/噪声标度仅用开发/校准材料一次冻结；基础设施失败与结果性失败分开记录；退役格计入完成度不删除；real requests only、无 delay floor、probe_mode=off、不扣除实测成本；window_v2.py 冻结未改；旧封存结果保留；§16：未改 rev1 策略、未重拟合 θ、未从已有评价墙钟挑数据。
- 闭环口径：准备完成≠闭环；**闭环仅在 168+32 跑完并通过统一分析后成立 —— 现已成立**（本报告即统一分析产物）。

## 13. 产物索引

`analysis/unified/`：build_tables.py、stats_unified.py、figures.py、main_table.csv/json、p1_table.csv/json、unified_stats.json、stats_report.txt、f6_forest_values.csv、f7_forest_values.csv、fig_f6_main_matrix.png、fig_f6_confirmatory.png、fig_f7_ablations.png、fig_f8_p1.png、REPORT_final.md（本文件）。
上游：main/（168 格封存产物）、p1/（32 格）、p1_smoke/、D_matrix/run_list_*.csv、C_configs/FREEZE_final_plan.md、logs/campaign_marks.tsv、PROGRESS.md（#18–#21）。
留待作者（未自动启动）：X1/X2 扩展、R64 +8 格、M2 剩余诊断（3/4 boot、离线 w17/w37 对照）、M3 trace、M4 replay。

--- 勘误指针（2026-10-20 追加，不覆盖上文）：见 results/final_followup_20261020/ERRATA_addendum_20261020.md（完成度分解 94.43/94.13/2305、消融 raw vs Holm、A3"未行使"措辞、confirmed_grow 语义、逐臂配置表）。
