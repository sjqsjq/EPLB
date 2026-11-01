# FREEZE — final_plan_20261019 主矩阵 / P1 预注册冻结文档

冻结时刻：2026-10-18T23:4xZ（本文档写入并 git 提交于 MAIN 第一次 boot 之前）。
授权：作者指令（2026-10-18）“一个一个实验去做……不用咨询我”“继续跑文档中的实验，持续监测”。
治理文档：`/workspace/EPLB/NEW_PAPER/PB_OEPLB_FINAL_EXPERIMENT_PLAN_20261019.md`（下称 plan）。
本文档冻结后，任何对矩阵、臂序、数据流、n、判据、统计口径的修改都构成方案变更，
必须另立修正案（amendment）并保留本文件原样；禁止中期选参（plan §9.10）。

路径记号：`$F = /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019`

---

## 1. 校准结果冻结（E*/F* 已定，选择过程封闭）

### 1.1 阶段一（screen 流，1280 reqs @ conc256，每候选 2 次）
均值墙钟（秒，升序）：

| F 候选 | mean | | E 候选 | mean |
|---|---|---|---|---|
| f_1_16_16 | 55.375 | | e0 (R0,p256) | 79.46 |
| f_4_16_64 | 56.575 | | e5 (R16,p256) | 80.33 |
| f_2_16_32 | 57.015 | | e1 (R16,p64) | 80.91 |
| f_1_8_8 | 59.83 | | e3 (R32,p256) | 82.395 |
| | | | e2 (R32,p64) | 84.79 |
| | | | e4 (R16,p16) | 94.395 |

晋级：topE = {e0, e5}，topF = {f_1_16_16, f_4_16_64}；截断处无精确并列（预注册 tie-break 未触发）。
选择规则（阶段一前已冻结）：mean wall 升序；仅精确相等才 tie-break（更小 R/更简单者优先）。

### 1.2 阶段二（confirm 流，独立语料 split，每候选 2 次）
| 候选 | run1 | run2 | mean |
|---|---|---|---|
| f_1_16_16 | 57.41 | 56.46 | **56.935** |
| f_4_16_64 | 57.43 | 56.53 | 56.980 |
| e5 | 78.62 | 78.64 | **78.630** |
| e0 | 79.57 | 79.74 | 79.655 |

### 1.3 终选（规则在阶段二运行前已冻结：仅按 stage-2 mean wall）
- **E\* = e5 = EPLB `--eplb-algorithm auto --ep-num-redundant-experts 16 --eplb-rebalance-num-iterations 256`**
- **F\* = f_1_16_16 = PB-fixed `S=1 H=16 W=16`**（OEPLB_SP_MODE=fixed, SP_REV=0）
- F 两候选差 0.045s（≈0.08%）为近似并列但**非精确并列**，mean 规则直接生效，不使用 tie-break；记录在案。
- 数据文件：`$F/calib/STAGE1_SELECTION.json`、`$F/calib/FINAL_SELECTION.json`、
  `$F/calib/stage1/summary_calib1.json`、`$F/calib/stage2/summary_calib2.json`。

## 2. e6 事件与修正（已封存，此处汇总引用）

默认 period=1000 在本基准流（约 256–271 个 forward pass/rank）上**物理上不可能触发**任何
rebalance（生成器 `for _ in range(period)` 永远走不完）。e6_r1 判 MECH_RETIRE（1 次计时 boot 已消耗，
bench 本身干净：1280/1280，t=77.14s），e6_r2 移除，CALIB1 CSV 重冻结为 20 行（sha 2c365171…），
TIMED_CAP=20。详见 `$F/calib/stage1/INCIDENT_e6.md`。
**该发现（默认 period 在短基准上从不生效）必须进入最终报告**，不是可丢弃的失败。
对照：e5（period=256）在 4 个 E 格中均按预测精确触发。

## 3. E 臂预注册偏差（唯一臂级偏差）

事实：`--enable-eplb` 自动置 `expert_distribution_recorder_mode='stat'`；'stat' gatherer 仅在
deepep_mode ∈ {normal, low_latency} 实现，`auto` 抛 NotImplementedError（eplb/expert_distribution.py:314）。
处置（EPLB_MECHANISM_NOTES.md §6，先于任何 E 计时封存）：
- **仅 E 臂**将 BASE 中 `--deepep-mode auto` 替换为 `--deepep-mode normal`；其余全部臂保持 rev1 封存 BASE（auto）。
- 先例：封存实验 d38_L512_O1_comparison（2026-09-10，同模型同栈）以 `--deepep-mode normal` + recorder='stat' 成功运行。
- **公平性警示（必须进 limitations）**：E 臂 decode 阶段无 low-latency buffer，可能低估官方 EPLB 的墙钟表现；
  P-vs-I 确证比较不受影响；P-vs-E 的解释必须附带此警示。

## 4. 请求流冻结（全部先于任何主矩阵计时封存）

`$F/B_data/registry.json`（e1–e4，构建规则 R1–R7 预注册于 build_streams.py docstring）：

| 流 | N | 输入长 | 输出 | sha256 |
|---|---|---|---|---|
| e1_code_stable | 3221 | 2048 | O1 (t=0) | dee24c502f7f301d92ce242c671c7a59278fb4749ecef1da74a40c36cb8fc682 |
| e2_prover_stable | 4082 | 512 | O1 (t=0) | fa788238fa2dcc90b12b8b2e18ab723509dd92937d1a8ff664ffec47627bc78f |
| e3_multidomain_switch | 8192 | 512 | O1 (t=0) | d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879 |
| e4_fullduplex | 2048 | 512 | out128 (t=0, ignore_eos) | a024c5e61d0dbf321dee6245ef3d5f270da7e4374304d910e943cdd0657bc0a6 |

N 调整（plan §5.1 授权，registry 内已登记，先于冻结）：E1 目标 4096→实际 3221（code 唯一内容容量），
E2 目标 8192→实际 4082（prover 唯一内容容量）。禁止把重复请求冒充独立样本。

P1 流（`$F/B_data/registry_p1.json`，构建复用封存 R1–R4 规则，无 RNG）：

| 流 | N | 构成 | 输出 | sha256 |
|---|---|---|---|---|
| p1_longio | 2048 | 4 域 × 512 窗口（TEST split，L2048 s256），round-robin 交错（成分平稳） | out128 (t=0, ignore_eos) | 3c09e0270a60d185c801646c3cebc7157a99086d0c10c8b4e78c5dc615068457 |

完整性：code 域前 512 窗口与封存 e1 流逐行 prompt 比特一致（构建时断言通过）；2048 行内容 hash 全唯一；
retok 长度 p5/p50/p95 = 2048/2048/2048。N=2048 档位选择依据 CPU 盘点（四域池容量均 ≥512），先于构建冻结。

## 5. 主矩阵冻结（168 次计时启动）

- 条件 × 臂 × 块（plan §5.2/§7/§8）：
  - E1 code_stable：{I, E\*, P, F\*, A3} × 8 块 = 40
  - E2 prover_stable：{I, E\*, P, F\*, A2} × 8 块 = 40
  - E3 multidomain_switch：{I, E\*, P, F\*, A2, A4} × 8 块 = 48
  - E4 fullduplex：{I, E\*, P, F\*, A1} × 8 块 = 40
- 臂实现（launch_arm.sh，封存 sha 锁）：I = 静态 BASE；E = E\* 参数 + §3 偏差；P = PB 全策略；
  F = F\* 固定 SHW；A1 = P+全阶段记录；A2 = P+maxdelta 配对规则；A3 = P+强制 S=1；A4 = P+只算不提交。
- 并发：全部 256（plan §5.2）。每条件对全部臂/块使用同一文件、同一顺序、同一采样参数（§9.4）。
- bench 超时（防挂死，不是判据）：E1/E2 2400s，E3/E4 3600s。
- 运行顺序：`$F/D_matrix/run_list_main.csv`，sha256 = c2e06df4b9daa9acfeedda21a84624c5172893babdcec8697aff0bff6aa2dc60，
  168 行，已嵌入驱动锁（每次 boot 前复核）。
- **轮转规则（预注册）**：种子约束随机构造 — rng = random.Random(20261019\*10000 + attempt)，
  每步在“与前一格臂不同且条件不同”的剩余格中均匀抽取；死局则 attempt+1 重构；首个完成解冻结。
  **实际：seed=202610190002，attempt=2**。完整顺序与位置：`$F/D_matrix/rotation_main.json`。
  （注：纯拒绝洗牌在双约束下概率≈e^-43，故采用等价的逐步约束构造；规则先于任何主矩阵计时在此冻结。）
- 块 = 配对重复（每 (条件,臂) 恰 8 块，每块 1 次计时启动）；n=8 为预算档位，不是功效保证（plan §8.7）。
- 禁止中途加格、换序、改数据（plan §8.7/§9.10）。

## 6. 统计口径冻结（plan §8 原文口径）

- 主终点：每工作流完整墙钟 T（客户端单调时钟；闭环并发下客户端排队与服务器延迟分开定义）。
- **确证比较（事前定义，仅两个）**：P-vs-I 与 P-vs-E\*，在四工作流等权组合上的 log 时间比。
  逐 block b：g_b(P,B) = (1/4)·Σ_{j=1..4} log(T_{B,j,b} / T_{P,j,b})，**j = 四工作流 E1–E4**
  （每条件取该 block 的一次运行）；G(P,B) = exp((1/n)·Σ_b g_b) − 1，n=8。
  G 是等权工作流几何吞吐比增益，**不是**按请求数/token 加权的总体生产收益。
  以完整 block 为单位估计区间（log 比/配对差 t 区间为主要估计）；两个主比较 Holm 校正；
  同时完整报告各工作流效应及未调整区间，不包装为独立确认性胜利。
- 8 块 × 5/6 臂**不宣称**恰好完整拉丁方（plan §8.1）；轮转表 + seed + 实际起止/位置全部保存。
- 次级族（单独申报，不与确证族混合校正）：P-vs-F\*；5 个 条件×消融 对比
  （A3@E1、A2@E2、A2@E3、A4@E3、A1@E4，各 vs 同条件 P）。
- 报告义务：每工作流原始时间、配对差、时间缩减率、吞吐变化；吞吐与 T 为相关派生指标，不作两份独立获胜证据。
- 失败格处理见 §7；RESULT_RETIRE 计入完成率报告，不得删除后假装 n 完整。

## 7. 统一协议与失败规则冻结（plan §9）

- 冷控制：每格独立 boot = 均匀初始专家布局 + 空控制历史（保留首布局适应成本，不预热控制态）。
- I/P/F 逻辑主映射 = identity；E 冗余槽按官方 init 规则。radix cache off（BASE 内 --disable-radix-cache）。
- 锁检查：deps v3 / bench client / CSV / 流 sha 在**每次 boot 前**复核；不匹配 = HALT。
- 失败分类：
  - 零测量基础设施失败（boot 不成）：INFRA_CAP=3 次/相位内有界重启；耗尽 = HALT。
  - 方法崩溃/OOM/超时 = RESULT_RETIRE：作为完成率结果保留（正式相位不重试、不删除）。
  - 机制/完整性失败（MECH_RETIRE / INTEGRITY_FAIL）= HALT（exit 8），写完整报告等待作者。
- 判据永远不由终点数值触发；W/H/S 值在计时开始后不得改；跨 rank 时间只用 MAX/per-rank，永不求和。
- 不做中期选参：运行期间只允许健康/配置检查，禁止基于终点调整顺序、数据、n。

## 8. P1 冻结（plan §10；32 次计时启动）

- 问题：官方 EPLB 的冗余度 R 与 KV 压力的权衡 vs P_final（无冗余权重拷贝）。
- 臂（4）：P（PB 全策略，无额外参数）；EPLB auto/period256 × R ∈ {0, 16, 32}
  （算法与 period 采用校准终选 E\* 的 auto/256，仅 R 变化；§3 偏差同样适用于这些 E 臂）。
- 流：p1_longio.jsonl（§4；真实长输入/长输出：in2048/out128，t=0，ignore_eos）。
- 压力（2）：
  - **L：conc=64** — 每请求 KV 预算 2176 tok；64×2176 = 139,264 < 最差臂容量 190,307 ⇒ 稳态全容纳，无排队压力。
  - **H：C_hi = 256（冻结）** — 依据仅为真实 KV 容量与每请求 token 预算（先于任何性能终点）：
    实测容量（现有 boot 日志 "KV Cache is allocated. #tokens:"）：E-R32=190,307（最差臂）、E-R16=208,750、
    E-R0=227,168、P=227,292、I=227,269、F=227,258 ⇒ 最差臂稳态在飞 ≈ 190,307/2176 ≈ 87 请求。
    候选 {256, 512, 1024} 均可经排队准入（§10 允许排队）；选 **256** 因为：对最差臂已形成 ≈2.9× 超额订阅
    （与 L 形成决定性对比）、与主矩阵并发一致（操作统一）、排队墙钟有界。512/1024 不改变压力定性结论且加倍超时风险。
  - H 相允许排队：统一 max-running 约束（服务器调度器默认），报告实际排队/在飞数。
- 每臂原生 KV 池：所有臂同一 BASE `--mem-fraction-static 0.8`，8 卡（§10.2 满足：E-R32 权重拷贝多、KV 池天然更小，即其原生代价，如实报告）。
- 矩阵：4 臂 × 2 压力 × 4 配对块 = 32；运行顺序 `run_list_p1.csv`，
  sha256 = b324f8eac6d0ece81aedf24088e97548059a44fe19a6a1f8847ad6993c3ecbfd；
  轮转：同 §5 构造算法（约束 = 方法不同且压力不同），**seed=405220380001，attempt=1**；
  `rotation_p1.json`。
- **可行性 smoke（先于 P1 计时，非测量启动，计入 M2 诊断预算 ≤4）**：boot 最差臂 E-R32，
  以 C_hi=256 突发 in2048/out1 切片请求，验证准入正常（无 OOM/崩溃/拒绝服务）后 kill；
  结果记入台账。若 smoke 失败 → C_hi 降档为不可行事件处理：HALT 并报告（不静默改设计）。
- 指标（§10.4）：吞吐-并发、TTFT/完成时间 p99、KV 容量 vs R、错误/超时率、排队/在飞计数；
  报告 weight/runtime-reserved/migration-peak/KV-token-capacity；4 重复仅作范围估计。
- R64 可选 +8 次：**不自动启动**（plan 授权外，留待作者）。

## 9. 预算台账（冻结时点）

| 预算项 | 上限 | 已用 | 备注 |
|---|---|---|---|
| 校准 boot | 30（22 screen + ≤8 confirm） | **29**（screen 21 含 e6_r1 retire；confirm 8） | 封闭 |
| F0 smoke boot | 8 | 8 | 封闭，8/8 VALID |
| 主矩阵计时启动 | 168 | 0 | TIMED_CAP=168 嵌入驱动 |
| P1 计时启动 | 32 | 0 | TIMED_CAP=32 嵌入驱动 |
| M2 诊断 | ≤4 | 0 | 预留 1 次给 P1 可行性 smoke |
| M4 replay | ≤24 | 0 | 未动 |
| M3 traces | ≤4 | 0 | 未动 |
| INFRA 重启 | 3/相位 | F0:1（烟测例外）CALIB1:0 CALIB2:0 | 驱动内计 |

## 10. 执行顺序与监控

1. MAIN（168 格，串行，预计 18–30h）：驱动 `scripts/run_final_campaign.sh MAIN`，
   台账 `logs/campaign_marks.tsv`，产物 `main/<label>/`。
2. P1 可行性 smoke（M2，1 boot）→ P1（32 格，预计 8–16h），产物 `p1/<label>/`。
   **GPU 上不允许任何并发任务**（8 卡独占，plan §14.2）。
3. 统一分析（CPU）：确证 g_b/Holm、次级族、完成率、图（plan §13）、台账、limitations、最终 REPORT。
4. HALT 纪律：完整性失败/机制错误/驱动死亡且状态不明 → 停止一切、写完整报告至 PROGRESS.md 与 REPORT、等待作者。
5. 恢复规则：仅 TSV 中 `VERDICT <label> VALID` 可跳过；其余一律重跑本格（重启后 resume 语义不变）。
6. 提交纪律：git 本地提交，永不 push；保留全部旧封存版本；不改 rev1 策略、不重拟合 θ、不从已有评价墙钟挑数据。

## 11. 冻结声明

本文件与下列 sha 一并构成冻结基线，写入时刻早于 MAIN 第一次 boot：

- run_list_main.csv sha256 = c2e06df4b9daa9acfeedda21a84624c5172893babdcec8697aff0bff6aa2dc60（168 行）
- run_list_p1.csv sha256 = b324f8eac6d0ece81aedf24088e97548059a44fe19a6a1f8847ad6993c3ecbfd（32 行）
- 流 sha：见 §4（e1–e4 + p1_longio）
- 驱动：scripts/run_final_campaign.sh（MAIN/P1 相位，CSV/流 sha 双锁）
- 轮转：rotation_main.json（seed 202610190002, attempt 2）、rotation_p1.json（seed 405220380001, attempt 1）
- E\* = e5 "auto 16 256"；F\* = f_1_16_16 "1 16 16"；C_hi = 256；N_P1 = 2048

此后唯一允许的运行时动作：健康检查、台账记录、按 §7 分类的失败处置、既定分析。
不得把“准备完成”称作“论文性能闭环”（plan §16）——闭环仅在 168+32 格跑完并通过统一分析后成立。
