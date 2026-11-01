# author_diag — 跨域矩阵 (freq6 / E3 / E4) × (identity / v2 / gh_pristine) × (paper / campaign 配置)

日期: 2026-10-21 ~ 2026-10-22 (UTC)。作者问题: "为什么新系统不如老系统优秀(跨域)"。
本目录为独立 ad-hoc 诊断账本 (LEDGER_adhoc.tsv)，与正式 64-boot 矩阵、prep、fix 轮完全隔离；
不回填正式结果，不改动任何冻结控制律。所有 boot 均为: 8×H20, Qwen3-235B-A22B-FP8
(/root/models/ms_cache 替代已消失的 /data/models 原路径，已披露)。

## 代码版本
- **v2 (冻结)**: async_swapper 2e4652a0…, controller 4964cfc8…, config 52f7f4d7… (含 FIX-1/2/3 + DIAG-1/2 + v2.1 S/H/W + probe)。
- **gh_pristine**: GitHub origin/main 8 文件原封包 (async 93fa4ccd… = code_lock 同源, controller ada04359…, config 232ecbd5…)。
- 每轮 gh boot 前备份 DEP、trap 恢复 v2 并 sha 校验；LEDGER 中 RESTORE_v2* 行全部 OK。

## 两种启动配置
- **paper flags** (复现 §5.3.1): --disable-cuda-graph, mem 0.78, decay 0.5, adaptive-window + adaptive-decay ON, threshold 1.02, sync-window 16。
- **campaign flags** (本轮正式矩阵 P/I 臂): cuda-graph ON, mem 0.8, decay 0, adaptive-window OFF, v2.1 env (κ=3875.18, z_lo/hi/urgent=22.6/314.1/942.4, θ_r=0.0101/0.0114, N_min=12320768, W_uncertain_target=16 …)。

## 结果 (median, tps; freq6 取 run2 steady)

| workload | 配置 | identity | v2 | gh_pristine | v2 vs I | gh vs I |
|---|---|---|---|---|---|---|
| freq6 (book↔prover, N=600, conc=32, O=10) | paper | 4.45 {4.5,4.4} | 4.60 {4.8,4.1,4.6} | 4.70 {4.9,4.3,4.7} | **+3.4%** | **+5.6%** |
| freq6 | campaign | 4.80 (n=1) | 4.80 (n=1) | — | **0.0%** | — |
| E3 (multidomain switch, 8192 req, conc=256, O=1) | paper | 42.45 {42.6,42.3} | 42.9 {43.1,39.8,42.9} | 41.7 {41.7,41.2,42.6} | +1.1% | −1.8% |
| E3 | campaign | 41.92 (formal n=8) | 43.06 (formal n=8) | 38.5 (n=1) | +2.7% | −8.2% |
| E4 (fullduplex, 2048 req, conc=256, O=128) | paper | 1212.8 (n=1) | 1166.7 {1177.4,1147.3,1166.7} | 1199.7 {1199.7,1148.1,1218.4} | **−3.8%** | **−1.1%** |
| E4 | campaign | 2840.25 (formal n=8) | 2613.95 (formal n=8) | — | **−7.9%** | — |

噪声界 (同 cell 多轮极差): freq6 steady ±0.4 (4.1–4.9); E3 v2 ±3.3 (39.8–43.1), gh ±1.4; E4 ±70 (~6%)。
单轮差 < 噪声界的不得作为结论。gh vs v2 freq6 三轮配对 gh 全胜 (+0.1/+0.2/+0.1)，方向一致但幅度贴噪声下沿。

## 机制记录 (行为计数, //8 ranks)

| boot 类 | windows | swaps | resets | 备注 |
|---|---|---|---|---|
| freq6 gh paper | 14–22 | 2009–2863 | 80–88 | RESET 频繁, ADW 收缩/扩张活跃 |
| freq6 v2 paper | 13–19 | 2117–2571 | 同量级 | v2 legacy 路径完整保留 paper 行为 |
| freq6 v2 campaign | 19 | 2089 | **0** | z 峰值 94.3 << z_hi 314 → midband_hold; grow_blocked_headroom; S/H/W=1/16/16 冻结 |
| E3 paper (v2≡gh) | 11 | 1911 (gh r2:1856) | 7 | v2 与 gh 行为逐窗一致 |
| E4 paper v2 | 3–7 | 780–1707 | 7–8 | ADW grow 32 后行为跨 boot 有变异 |
| E4 paper gh | 3 | 781–796 | 8 | |

## 结论 (对 "为什么新系统不如老系统")

1. **老系统的跨域增益在老配置下依然复现**: gh_pristine + paper flags freq6 median +5.6% (r1 单轮曾 +8.9%，3 轮中位数收敛到 +5.6%；论文 +8.5% 在噪声带内)。
2. **修复没有杀死 legacy 增益**: v2 + paper flags freq6 median +3.4%，方向为正；v2 与 gh 在 freq6 上 3 轮配对差 0.1–0.2 tps，贴噪声。E3 上 v2 与 gh 行为逐窗一致 (11 窗/1911 swap/7 reset)。
3. **增益消失发生在 campaign 启动配置，不在代码**: 同一 v2 代码，freq6 campaign flags = 0.0% (4.8 vs 4.8)。机制证据: v2.1 阈值 (z_hi=314) 远高于 freq6 实际 z 峰值 (94.3) → 全部 midband_hold，S/H/W 冻结 1/16/16；headroom θ 门槛在 0.010–0.029 区间频繁 grow_blocked；RESET 0 次 (paper 配置 80–88 次)；decay=0 无遗忘；recording_ratio 1.3% 采样门饿死自适应信号。
4. **E4 对任何 PB 配置都不利**: paper flags 下 identity 1212.8 > gh 1199.7 > v2 1166.7；campaign 下 −7.9%。E4 (O=128 全双工) 的瓶颈不在专家放置，swap 开销纯损耗。
5. **"新系统不如老系统"的准确表述**: 不是 v2 代码劣化，而是 (a) v2.1 阈值/门控针对 campaign 流校准，对 freq6 这类 paper 工作负载失配 → 跨域增益归零; (b) E4 类负载本来就不该触发 PB; (c) 老配置下老代码的 +5.6% 与 v2 的 +3.4% 差异贴噪声，需要 n>3 才能定 "fixes 是否吃掉了 ~2pp"——不显著≠等价。

## 治理
- e4_v2_paper_r3 于 19:22:20Z 因 sgl-kernel DeepGEMM JIT CUDA 断言崩溃 (rc=124, bench 超时)。**不同故障类** (非 COALESCED 死锁类; fix 轮 2/2 约束不适用)，如实记录后单跑 r4 补齐 (1166.7, faults=0)。
- 全部本地提交，不 push。旧 200 次实验、失败、协议、代码锁、原分析保留，新旧不混池。
- 本矩阵是描述性诊断，不构成新的正式声明；写入论文的声明仍按 ONEPAGE_claims.md 及其 ADDENDUM 执行。
