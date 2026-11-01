# PHASE-2 预注册 — prover同域调度消融 正式阶段（写于任何 phase-2 计时数据存在之前）

时间：2026-10-26。本文件一经落盘即冻结；分析规则只能由本文预先声明，不得依据 phase-2 结果回调。

## §0 授权链
- 作者指令"开跑吧"（phase-1，已执行完毕）→ phase-1 报告向作者列明第二阶段三项待决（重复数、F(4,16,64) 参照确认、更长流需新预注册）→ 作者指令"**迅速继续**"（2026-10-26）。
- 解读：授权按 phase-1 PROTOCOL §4 既定框架推进第二阶段（含条款 5 要求的新预注册与相应 GPU 执行）；未收到对 F(4,16,64) 主读法的书面改判 ⇒ 按预声明主读法执行。授权链与解读全文登记于 LEDGER_abl.tsv。

## §1 输入：phase-1 机制事实（仅事实，无性能输入）
64 forwards / 4 个 H16 窗 / stable_s_max=2（距 quiet_streak≥3 差 1 个有效周期）/ 迁移 2 次 / det_map_version→2 / 尺度模型实证 8066 tokens/step。结论：N=2048 无调窗行使空间 ⇒ 依据指示第 5 点与条款 5，做**一次性、预先声明的**流加长。

## §2 流冻结（一次性加长，封顶）
- 文件：streams_p2/prover_256tok_out1_x4.jsonl = 冻结原文件（sha `b4034e3b…`）的 **4× 逐字节级联**（实测 data == orig*4 为 True）
- sha256：`b312ba50c39abe1a81e85bd16f27f2b7913ae7fffb57550efb5b702fb34aabf0`；6,970,560 bytes；8192 行
- 唯一 prompt 数 2048，每个恰好出现 4 次；重复方式 = 全文件重复、相对顺序不变、id 按设计重复（id 不进 server payload；payload 仅 prompt/max_tokens=1/temperature=0/ignore_eos=false）；无洗牌、无子采样、无重新生成
- N=8192，O=1，conc=256（与 phase-4 E3/E4 的 nreq/conc 惯例一致）；bench_timeout=3600
- 尺度预期：~2.06M prompt tokens ⇒ ~256 forwards ≈ **~16 个 H16 窗**（E2 先例 QUIET@w9/144 steps ⇒ 转换有行使空间）
- **加长封顶**：仅此一次 4×；若 phase-2 仍未见状态转换，结果原样保留，禁止再加长/再重跑直到触发（条款 3–4 精神沿用）

## §3 臂与设计（24 boots）
| 臂 | labels | 语义 |
|---|---|---|
| P 完整自适应 rev1 | apsd2_P_b{1..8} | launch_arm.sh P（SP_MODE=adaptive, SP_REV=1，冻结 knobs） |
| F 固定 1/16/16 | apsd2_F_1_16_16_b{1..8} | launch_arm.sh F 1 16 16（SP_REV=0，config 守卫强制） |
| F 固定 4/16/64（预声明参照） | apsd2_F_4_16_64_b{1..8} | launch_arm.sh F 4 16 64（= PD-11 F_sparse / QUIET 三元组） |

- 顺序预声明：每 block b 内 P → F1 → F4，b=1..8；boot 间完全独立（env_clear + kill_server_quiet <1000MiB 门）
- 公平性规则与 phase-1 完全一致：同一 launch_arm.sh（`c2f6b9dd…` 守卫）、同一 bench_client_v2（`0666824…` 守卫）、v2 双 sha 守卫（`2e4652a0…`/`4964cfc8…`）、流 sha 守卫（`b312ba50…`）、identity 初始布局、默认服务端 warmup、客户端 /health-only warmup、完整流计时无剔除

## §4 正式重复数
每臂 **8** 次独立启动（冻结的 phase-4 b1..b8 家族惯例，Holm 兼容）。本数在任何 phase-2 计时前固定。

## §5 分析预注册（先于数据）
- 主指标：每 boot 完整流 tps（bench json：comp_tokens/wall）
- 主对比集合（仅此两项进入 Holm）：**P vs F(1,16,16)**、**P vs F(4,16,64)**；双侧 Welch t 检验（每臂 n=8 boot 级样本）；Holm 校正，α=0.05；描述统计报 median 与 min/max
- F1 vs F4 仅作描述性报告，不入主对比集合；不做事后添加
- 机制终点（每 boot，mech_check_p2 事实字段）：窗数、状态转换清单（window_id/before/after/reasons）、Σswaps_started、det_map_version 终值、Σvalid_rows、stable_s_max；按臂汇总为计数与轨迹，与性能结果分开陈述
- 解释护栏：h_rate 为低余量带信号，**不得**表述为"还可以加速多少"；不把论文分析模型说成已实现在线算法；机制未触发如实登记
- 缺失/失败策略：continue-on-fail；completeness rc≠0 的 boot 登记原因并从分析集排除；不自动重试；某臂可用 boot < 8 时按可用 n 分析并登记缺口，是否补跑由作者决定；**排除永不基于性能数值或机制事实**

## §6 治理
phase-1 工件只读不动；新增文件：本文件、DATASET_FREEZE_PHASE2.json、streams_p2/、run_abl_boot_p2.sh、mech_check_p2.py、queue_abl_p2.sh、aggregate_p2.py、boots_p2/、RESULTS_PHASE2.md（跑完后写）；LEDGER 只追加；无 git commit；不修改任何冻结文件。
