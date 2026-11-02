# PAPER_CLOSEOUT_DATA.md -- 论文收口数据整理（六战役统一口径汇总）

生成: 2026-11-02 (CPU-only 数据整理阶段; 作者指令: T1/T2 完成后停止新增 GPU 任务, 转入数据整理/作图/论文定稿; 不因结果不显著或未领先而追加轮次)
纪律: 本文档只做已有测量数字的程序化搬运与口径标注, 不新增任何推断, 不改写任何冻结报告。h_rate/h_w 不作 headroom 解释; 无 payback solver; CPU 战役不得声称已验证真实 GPU 收益。

## 0. 战役清单与口径分类（论文引用前必查此表）

| 战役 | 目录 | 口径 | 规模 | 状态 |
|---|---|---|---|---|
| T2 六臂服务主比较 | sixarm_service_t2_20261101 | O32 conc256 闭环服务（非 SLO）, GPU 真实执行 | 36/36 boots 全绿 | 作者指定的主比较, 已完成 |
| T1 三口径最小验证 | replay_routing_t1_20261101 | R=离线capture回放 / X=GPU真实执行短重复x2 / M=在线迁移记账 | 7/7 条件 | 已完成, 不扩采 |
| XD 跨域补充探针 | cross_domain_e2e_20261027 | O1 prefill-only conc256（非 §5.3.1 主比较, 定位修正见其 Addendum） | 24/24 boots | 已完成 |
| PBD 加倍流描述 | prover_book_double_20261027 | O32 服务流 n=5344, n=2 轮描述性（NON-INFERENTIAL） | 12/12 boots | 已完成 |
| G3 相位选择 | pf_all_phase_select_20261030 | GPU boot 级, PF vs ALL 记录相位, n=4 每格描述性 | 16/16 boots all_pass | 已完成 |
| G2 规划器矩阵 | planner_matrix_c1c4_20261031 | CPU 确定性算术（settrace 锚点, 零 GPU, 零计时） | 52/52 cells 计划一致 | 已完成 |

## 1. T2 -- 主比较（作者指令优先级第一）

来源: sixarm_service_t2_20261101/T2_CLOSEOUT.md (sha16 118226274b0e2a6d) + analysis/EXPORT_t2_service_table.tsv (36 行, sha16 0a8a8b51e6ee308a)。
配置: 6 臂 (I/ES/ED/MT/DF/P0) x 3 轮 x 2 场景; Path A (deepep normal, graph off); RG=0; 除 ES/ED 机制必需 redundancy=16 外零冗余。
场景: PROV = prover_1024_out32_ie conc256; PBD = pbd_prover_book_double_out32_ie n=5344 conc256（两场景 conc 一致, 流长/构成不同, 已声明）。
完整性: 36/36 all_ok, mech 全绿, 零 rc=2 停摆; 队列总 wall 6.8h < 7.5h 上限。

### 1.1 PROV 场景（tps mean, 3 轮）
| 臂 | tps | 区间 | lat_mean ms | tpot ms |
|---|---|---|---|---|
| ES (static EPLB) | 28.06 | 27.26-28.63 | ~9066 | ~218 |
| DF (DataFore-style) | 28.06 | - | - | - |
| MT (MoETuner-style) | 28.01 | - | - | - |
| P0 (PB-OEPLB) | 26.67 | - | - | - |
| I (Identity) | 26.54 | - | - | - |
| ED (dynamic EPLB) | 24.84 | 22.96-26.06（方差最大） | - | - |

**如实陈述: PROV 上 P0 未领先。** 三个静态重布置臂 (ES/DF/MT ~28.0 tps) 领先 I (+5.4~5.7%) 与 P0 (+0.5%); ED 最低。n=3 轮, 按作者指令不追加轮次, 不做显著性扩权。

### 1.2 PBD 场景（tps mean, 3 轮）
| 臂 | tps |
|---|---|
| P0 | 6.13 |
| ED | 6.11 |
| ES | 6.06 |
| MT | 5.99 |
| DF | 5.94 |
| I | 5.87 |

lat ~41-43s, tpot 745-803ms。全臂窄带 5.87-6.13（极差 ~4.4%）; P0 名义最高但与 ED/ES 差距在轮间波动量级内。**如实陈述: 窄带, 不构成领先结论。**

### 1.3 缺陷披露（必须随结果引用）
DEFECT_FIX: launch_arm_t2.sh 第 19 行字节拼接时丢失 PL= 前缀 → ES/MT/DF 臂 10 个 boot ~8s 即死（set -u unbound variable）; I/P0/ED 不受影响。原子修复（launcher+LSHA2 守卫同步）, boots_fail1/ 归档, queue_t2_rerun.sh 按 ORDER_TABLE 原序重跑 10/10, 协议与轮转顺序零变更, 零臂丢弃。--pb-oeplb-max-total-ops 观测语义 = 每窗口预算（GAP32 ≈ 32 ops/window）。

## 2. T1 -- 最小真实执行验证（现有 capture, 不扩采）

来源: replay_routing_t1_20261101/T1_CLOSEOUT.md (sha16 2de240717b7d3b81) + analysis/EXPORT_t1_full_table.tsv。
条件: ORIG / GAP32 / GAP128 / GAP300 / MD300 / BI300（7/7 完成, mech_all_ok=True, bench errors=0）。

### 2.1 R 口径（离线 capture 回放, fin_max_raw）
| 条件 | plan_len | fin_max_raw |
|---|---|---|
| ORIG | 0 | 1.531831 |
| GAP32 | 32 | 1.314006 |
| GAP128 | 128 | 1.178292 |
| GAP300 | 251 | 1.096219（= 生产阈值 1.0962402945 的死区地板） |
| MD300 | 239 | 1.161569 |
| BI300 | 246 | 1.095720 |

### 2.2 X 口径（GPU 真实执行, 短重复 x2, 标注 execution 非 replay）
**如实陈述: prefill-heavy E3 流上无吞吐收益。** GAP32 −6.8%, MD300 −11.5%; GAP128/GAP300/BI300 中性 ±3%; n=2 不做显著性声明。失衡下降与吞吐在该流上解耦; padding/connections 布局不变（已核）。ready_s 为工程开销列, 永不进比较。

### 2.3 M 口径（在线迁移记账）
issued_lines = 8 x windows 全条件 G3 核验通过（dedup_ok=True）; 例: GAP32 62 窗 496 行, BI300 34 窗 272 行; migration_wall_rank0_s 2.1-7.5s; v2p1 swaps_started/committed 与窗口记账一致。

## 3. XD -- 跨域切换补充探针（引用必带定位注记）

来源: cross_domain_e2e_20261027/analysis/REPORT_xd.md + summary.json。
定位: **非论文 §5.3.1 crossdomain_freq6 主比较**（Correction Addendum: 流/workload/校准源三点偏差）; 可用作 §5.5 机制时间线案例（须带注记）。
主结果（n=4 轮, wall 秒, 快→慢）: DF 188.20 ~ MT 188.24 < I 190.60 < ES 192.13 < P0 199.98 < ED 208.40。
推断: 以 P0 为参照 Holm 校正后六臂无显著差异; 唯一 raw p<0.05 为 P0_vs_DF (+11.78s, p=0.0499, Holm 0.2494 n.s., 边界值如实报告)。
机制时间线: P0 4/4 boots 域边界后 ~1 窗（~6s）发起 swap; ED 实为 2 事件 x 8 rank（更正旧 16 行口径）, 边界 1/3 未被覆盖, prover 段 ttft p95 13.26s 全臂最高; 静态臂 0 次运行时 rebalance。
Limitations 四点必带: O1 prefill-only 结构性压缩臂间差距; 混合域摊平倾斜（layer0 2.379 vs 单域 4.093）; 静态臂 oracle 校准偏向静态臂; ES/ED deepep normal 路径混杂（不与 auto 臂作机制归因）。

## 4. PBD -- 加倍流描述性（n=2, NON-INFERENTIAL）

来源: prover_book_double_20261027/analysis/descriptive_pbd.md。
流: n=5344（prover[0:3344) + book[3344:5344)）。
分段均值（req/s）: prover 段 DF 11.15 > MT 11.14 > P0 10.99 > ES 10.39 > ED 9.84 > I 9.72; book 段 ED 4.58 > P0 4.45 = DF 4.45 > I 4.41 > MT 4.40 > ES 4.38。
臂排序在 pb 原流（3048）上复现（§3b 对照表）。
P0 边界反应确定性: r1/r2 完全一致（windows=109, boundary_w=48, z_spike=109.0, swaps_total=23, TRACK 常驻）; h_rate 数值仅为控制器内部统计, 不作 headroom 解释。
ED 行记账: 297 行 = 事件 x 8 DP rank。静态臂 rebalance-lines=1（仅 init）。

## 5. G3 -- 记录相位选择（PF vs ALL, 描述性）

来源: pf_all_phase_select_20261030/analysis/EXPORT_g3_boot_table_20261031.tsv + recon_g3_facts.json（guardrails: n=4 每格, DESCRIPTIVE only, moved_mb 为估算 = ops x 37.74MB）。
16/16 all_pass。均值（秒）: ARC PF 23.43 vs ALL 34.71; PROV PF 35.63 vs ALL 41.31。
配对差（ALL−PF, %）: ARC +16.6~+86.9; PROV +12.9~+22.9。
迁移记账: PF 只录 prefill 相位 → issued windows 更少（ARC 3 vs 8; PROV 3-6 vs 9-10）, dedup_check 全 True（lines = 8 x windows）。

## 6. G2 -- 规划器矩阵（CPU 确定性, 零 GPU 声明）

来源: planner_matrix_c1c4_20261031/analysis/CLOSEOUT_g2.md + EXPORT_g2_rule_table_20261031.tsv（52 cells, control-vs-traced 计划逐 op 一致 52/52; 冻结规划器 sha256 c250026b...）。
快照: C1 calibration (init 1.5514) / C2 crossdomain P_e3 (1.5318) / C3 book (2.3106) / C4 datafore prover (2.4984)。
六发现（原文 F1-F6 浓缩, 引用以 CLOSEOUT 为准）:
- F1 口径效应受预算门控: 32/128 预算下两口径计划全同; 300 预算下生产口径产生更小计划（C1 281 vs 299; C2 251 vs 299）且 finmax 落在阈值本身（C2 1.0962 = 死区地板可视化）。
- F2 深失衡快照打满预算: C3/C4 两口径 plan=300, finmax 同值（1.2157/1.2281）-- 约束从阈值切换为预算。
- F3 半 gap 可行性: gap 分支 183 次激活, 100% 存在可行 delta>0 候选, 172/183 最优候选落在 0.25*gap 内。
- F4 max-delta 过冲为零: 严格判据（delta_max > gap/2 翻转序）在全部 52 格 0 次（含 4617 次 maxdelta 分支试验）-- 守卫在真实快照上严格保守, 按测量事实报告, 不做美化。
- F5 搜索效率: maxdelta 环境烧试验不成比例（C3 1312 trials → 229 ops）; gap 规则在 32/128 预算零拒绝。
- F6 BI 对照: best-improvement 计划 268/246/299/300 ops, finmax 1.1630/1.0957/1.3195/1.2111 -- 同骨架, 不同枚举。
Limitations: 快照为 T1 输入的一致性恢复（非新采集）; 确定性算术无轮间方差; **CPU-only: 本文档任何数字不佐证已实现的 GPU 加速。**

## 7. 论文整合规则（定稿时执行）

1. T2 是本轮唯一"统一配置三轮主比较"; 呈现 PROV/PBD 双场景表 + DEFECT_FIX 披露, P0 不领先照实写。
2. T1 以三口径呈现: R 表（含生产阈值地板）+ X 如实"无吞吐收益"+ M 记账核验; ready_s 单列工程开销。
3. XD 只作 §5.5 时间线案例与补充探针, 每处引用带定位注记; 不得回填 §5.3.1。
4. PBD/G3 描述性表格不加推断词; G2 全部数字带 CPU-deterministic 标签。
5. 口径标签强制: O=1 pure-prefill microbench（旧 §5.2/§5.3.1, +17.46%/+20.7%/+8.5% 属此口径）与 O32 conc256 service（T2/PBD）不得混排同一比较表。
6. h_rate/h_w 永不写作 headroom; 无 payback 声明; 1.02 行标注"该阈值下的算法诊断"。
