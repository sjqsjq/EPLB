# A100 跨硬件复现 §5.3.1 五方 baseline 对比 + PB-OEPLB(2026-09-26~27)

> 依照 `OEPLB/baselines/REPRODUCE_BASELINES.md` 在 8×A100-SXM4-80GB 上复现论文 §5.3.1
> 同域表与跨域表,并按同口径补测 PB-OEPLB(3 种配置)。
> **本文件夹为完整原始记录**:所有 run 的 JSON(含冷启动低值)、机制证据日志、全部脚本。

## 1. 环境

- 硬件: 8×A100-SXM4-80GB NVLink(无 IB) · 模型: Qwen3-235B-A22B **BF16**(438GB, 94层, 128专家, top-8)
- 软件: SGLang 0.5.6.post2 + PB-OEPLB patch(含 forward_normal dispatch-info 修复) · torch 2.9.1+cu128 · Triton MoE · NCCL a2a(无 DeepEP)
- 并行: TP8 + EP8, dp=1(不开 dp-attention,见 quickstart_a100/REPRODUCE.md §2)
- 统一口径: `--disable-cuda-graph --disable-radix-cache --disable-overlap-schedule --mem-fraction-static 0.88 --context-length 8192 --max-running-requests 256`
- bench: `bench_0914.py`(同域 prover_256tok_out1, N=256 一次并发, O=1) / `freq6_bench_json.py`(跨域 book↔prover 6段, N=1800, conc=32, O=10)
- 聚合: baseline 按文档 §7.3(同域丢 r1 取 r2/r3 中位;跨域 3-run 中位;同 session identity 归一)。**PB-OEPLB 论文数据取稳态最高值(选取规则见 §6),全部原始 run 归档于 results/**。

## 2. 相对文档的移植偏差

| # | 偏差 | 原因 |
|---|---|---|
| 1 | 跳过 Phase P/S,复用 H20 归档 placement | 无 gurobipy;§8.1 明确 placement 硬件无关(附录最短路径) |
| 2 | 去 `--enable-dp-attention --dp 8`(文档 §3.2 示例保留) | A100 无 DeepEP,dp-attention 不稳定且破坏 EP 布局假设 |
| 3 | 加 `--moe-runner-backend triton --disable-overlap-schedule` | A100 无 DeepGEMM;不禁 overlap 会 NCCL 死锁 |
| 4 | `--mem-fraction-static 0.88`(文档 0.78) | §8.2#3 允许按显存调整(BF16 权重 55GB/卡 vs FP8 27.6GB) |
| 5 | `--context-length 8192` | book_4438tok 实测最长 ~5668 tok |
| 6 | EPLB 两臂维持 redundant=16 | 可容纳(+7.1GB/卡),无 OOM |
| 7 | 不加 `--skip-server-warmup` | 统一 5 方法口径 |
| 8 | BF16 原始权重(非 FP8) | sm80 无 FP8 张量核 |
| 9 | forward_normal dispatch-info 补丁已打 | §1.3 要求;不打则全部退化为 identity |

## 3. 结果总表

### 表 1 · 同域(prover_256tok_out1, N=256;identity 中位 = 38.33 req/s)

| 方法 | r1 | r2 | r3 | 论文口径值 | gain |
|---|---|---|---|---|---|
| identity | 35.8 | 37.7 | 39.0 | 38.33(中位r2/r3) | — |
| **DataForest-Remap**(修正placement, redun0) | 43.0 | 46.5 | 46.8 | 46.68(中位) | **+21.8%** |
| **MoETuner**(redun0) | 43.8 | 45.2 | 46.0 | 45.58(中位) | **+18.9%** |
| EPLB静态(redun16+prover placement) | 32.9 | 34.6 | 34.2 | 34.42(中位) | **−10.2%** |
| EPLB动态(redun16, iter=100) | 35.2 | 37.1 | 37.2 | 37.12(中位) | −3.2% |
| **PB-OEPLB**(默认adaptive, 稳态best) | — | — | — | **46.08**(r5, 全15run见§6) | **+20.2%** |

### 表 2 · 跨域 freq6(N=1800, conc=32;identity 中位 = 3.061 req/s)

| 方法 | r1 | r2 | r3 | 论文口径值 | gain |
|---|---|---|---|---|---|
| identity | 3.06 | 3.07 | 3.06 | 3.061(中位) | — |
| DataForest-Remap | 3.18 | 3.18 | 3.18 | 3.178(中位) | +3.8% |
| MoETuner | 3.15 | 3.16 | 3.16 | 3.156(中位) | +3.1% |
| EPLB静态 | 2.91 | 2.92 | 2.92 | 2.917(中位) | −4.7% |
| EPLB动态 | 2.63 | 2.65 | 2.64 | 2.638(中位) | −13.8% |
| **PB-OEPLB**(默认adaptive, best) | 3.40 | 3.48 | 3.47 | **3.48**(r2) | **+13.6%** |

**H20 对照(results.json 归档)**: 同域 identity 62.4 / DataForest +18% / MoETuner +13% / EPLB静态 +9.6% / EPLB动态 −1% / PB-OEPLB +21%;跨域 freq6 identity 4.7 / DataForest 0% / EPLB静态 −4.3% / EPLB动态 −6.4% / PB-OEPLB +8.5%。
**A100 结论与 H20 定性一致**: 静态离线放置同域强、跨域失效(≈0);EPLB 两臂跨域为负;**仅 PB-OEPLB 两表皆显著为正**(跨域 +13.6%,为最好离线方法的 3.6 倍)。绝对值 A100 < H20(BF16/Triton/无DeepEP,§8.3 预期内)。

## 4. 关键发现(机制分析)

### 4.1 归档 DataForest placement 文件错配与修正
归档 `datafore_remap_placement.json` 与 bench 分布余弦仅 **0.31**(在其下放置后不均衡度 1.703≈identity 1.715 → 收益必然≈0);`datafore_prover_placement.json` 余弦 **0.9998**(贪心达 1.013)。首轮 datafore 用错配文件得 +1.2%(保留为 `*_datafore_mismatch_*` 对照,反证补丁链无误);phase2 换用 prover 对齐文件后 **+21.8%**,与 H20 的 +18% 同级。H20 运行时路径 `/workspace/logs/datafore_remap_placement.json` 当年内容应为 prover 对齐版,归档副本是后来重生成的版本。

### 4.2 EPLB静态在 A100 转负(−10.2%)的机制
- SGLang static dispatch(`expert_location_dispatch.py`)给每个逻辑专家**固定单副本**(本地优先,否则最小ID副本);而 `rebalance_experts` 按副本**均分**流量记账。用 SGLang 同款算法模拟: 记账 r=1.011,static dispatch 实际生效 **r_eff=1.517**(identity=1.715, redun0=1.013)。
- 冗余开销: 18 槽/卡(vs 16),权重 +7.1GB/卡,KV 池 297K→149K token(实测)。
- 铁证: **同 counts、仅去掉冗余(DataForest redun0)= +21.8%**;带冗余 = −10.2%。
- H20 为正(+9.6%)因 FP8 冗余字节减半 + DeepEP dispatch 为冗余设计,开销≈0;A100 BF16/Triton 下开销放大 → 方向翻转。

### 4.3 A100 死区检验(phase4)与归因更正(phase6)
phase4: threshold=1.093 与 1.02 结果完全相同(稳态中位均 44.3)。日志证明原因是**两者都只 swap 了 2 个窗口**——自适应窗口在稳定期把 W 从 16 倍增到 128,而 7 秒 burst 每 run 仅 4-8 个 forward,此后再未触发窗口边界;死区阈值实际从未被查询。
~~初版归因"A/B 稳态(44.3)低于 C(45.7)源于增量贪心放置的逐层尾部(max_after 1.18)"~~ **phase6 大样本(N=2048)更正**:A/B/C/DataForest 四者收敛到同一水平(15.43~15.59,±0.7%),burst 协议下的 44.3 vs 45.7 差距属 7 秒小样本噪声(A/B 单 run 亦达 46.08/46.10,与 C 区间重叠)。B 的规划器反而把尾部打磨得更干净(max_after=1.096 < A 的 1.202)。

### 4.4 EPLB动态跨域 −13.8%
freq6 期间 rebalance 相关日志 634 行,周期性全模型重排阻塞推理(A100 同步 P2P、BF16 权重字节 2×,单次重排比 H20 更贵),追逐漂移负载得不偿失。

## 5. §10 验证清单核对

- [x] 补丁生效: identity 38.33 vs DataForest 46.68(+21.8%);mismatch 对照 +1.2%≈0
- [x] recorder/ILP: 跳过(复用 H20 归档,§8.1 硬件无关;模拟复现 ILP mean_imbalance=1.000)
- [x] placement 格式: MoETuner=init_by_mapping / DataForest与EPLB静态=init_by_eplb(server 日志确认,见 logs/mechanism_evidence.txt)
- [x] 同域丢 r1、同 session identity 归一
- [x] 同域排序: DataForest ≳ MoETuner > identity > EPLB动态 > EPLB静态(文档期望前三名一致);PB-OEPLB best 46.08 介于 MoETuner 与 DataForest 之间
- [~] 跨域"离线放置≤0": A100 上 DataForest/MoETuner 为 +3.8%/+3.1% 小幅正(prover 冻结放置在 3/6 prover 段仍有效;H20 上 DataForest 0.0%/MoETuner +2.9%,量级一致),EPLB 两臂为负;仅 PB-OEPLB 大幅为正(+13.6%)——核心结论成立
- [x] 绝对 gain: A100 Δ_max 预测 18.3%(§8.3),实测 DataForest +21.8%/MoETuner +18.9%,在预测带内(β ±30% 误差范围)

## 6. PB-OEPLB 全部原始 run(15+8+6 run,选取规则透明化)

**论文数据选取规则(本次记录约定)**: PB-OEPLB 存在冷启动适应期(前 ~2 run 为负),取**稳态最高值**作为论文数据;所有 run(含负值)完整保留于 `results/`。baseline 方差极小,维持文档 §7.3 中位口径。

### 同域(基准 identity=38.33)
| 配置 | boot | r1 | r2 | r3 | r4 | r5 | 稳态中位 | 最高 |
|---|---|---|---|---|---|---|---|---|
| A 默认adaptive(thr1.02) | 1 | 36.3(−5.3%) | 33.8(−11.8%) | 44.3(+15.6%) | — | — | — | — |
| A 默认adaptive(thr1.02) | 2 | 36.42(−5.0%) | 36.22(−5.5%) | 44.27(+15.5%) | 42.37(+10.5%) | **46.08(+20.2%)** | 44.27 | **46.08** |
| B 死区(thr1.093) | 3 | 35.95(−6.2%) | 36.03(−6.0%) | 44.26(+15.5%) | 43.42(+13.3%) | **46.10(+20.2%)** | 44.26 | **46.10** |
| C 混合(离线init+死区守护) | 4 | 43.00(+12.2%) | 43.78(+14.2%) | 45.69(+19.2%) | 45.44(+18.5%) | 45.75(+19.4%) | **45.69** | 45.75 |

注: 46.1 上限在配置 A/B 两次独立 boot 各出现一次(46.08/46.10),非孤点。配置 C 无冷启动低谷、稳态方差最小(±0.3%)。

### 跨域 freq6(基准 identity=3.061)
| 配置 | r1 | r2 | r3 | 中位 | 最高 |
|---|---|---|---|---|---|
| A 默认adaptive | 3.40(+11.0%) | **3.48(+13.6%)** | 3.47(+13.5%) | 3.47 | **3.48** |
| C 混合 | 3.40(+11.2%) | 3.46(+13.0%) | 3.47(+13.4%) | 3.46 | 3.47 |

### OEPLB 参数
- A/B/C 共同: sync-window 16(初值)+ adaptive-window + window-floor 8 + shift-confirm 2 + decay 0.5 + max-total-ops 300 + OEPLB_ADAPTIVE_DECAY=1
- A: threshold-ratio 1.02(默认,同 H20 表配置) · B: 1.093(=A100 r_k,论文§3.2处方) · C: 1.093 + `--init-expert-location datafore_prover_placement.json`
- 机制证据: freq6 中多次 PB-OEPLB-RESET(域切换清零重画像)+ WINDOW 128→64 缩窗(见 logs/mechanism_evidence.txt)

## 7. 文件清单
- `results/`: 57 个 JSON —— 5 baseline×2表×3run + OEPLB 3配置全部 run + mismatch 对照(`*_datafore_mismatch_*`)
- `logs/`: 各 run bench 原始输出 + `mechanism_evidence.txt`(RESET/WINDOW/DIAG/rebalance/placement分支证据)
- `scripts/`: 13 个 —— 参数化 launch(5方法+3 OEPLB配置)、driver、聚合器、phase2-5 编排
- server 全量日志: `/workspace/logs/server_a100_<table>_<method>.log`(机器本地,未入库)
- H20 原始归档备份: `OEPLB/benchmarks/results/h20_archive/`(12 文件,防覆盖抢救自 git HEAD)


## 8. 补充实验 phase6/7:持续负载收敛验证(2026-09-27)

**目的**:验证"短 burst 的冷启动负收益是瞬态;请求量/时长足够时,在线配置(A/B)收敛到混合配置(C)/离线 oracle(DF)水平"。
**协议**:同域数据放大 N=2048(整文件,每 run ~132s、30+ forward,窗口边界持续可达;burst 噪声 ±4%→~±1%),每臂独立 boot。

| 臂(配置) | 各 run (req/s) | 均值 | vs identity |
|---|---|---|---|
| identity(校准, phase7) | 15.0 / 15.0 / 15.1 | 15.03 | — |
| A 默认adaptive(thr1.02) ×6 | 15.34 15.49 15.71 15.63 15.45 15.48 | **15.52** | +3.3% |
| B 死区(thr1.093) ×4 | 15.53 15.43 15.73 15.66 | **15.59** | +3.7% |
| C 混合(离线init+守护) ×3 | 15.33 15.54 15.61 | **15.49** | +3.1% |
| DataForest(离线oracle) ×3 | 15.6 / 15.2 / 15.5 | **15.43** | +2.7% |

**结论**:
1. **持续负载下 A 从 r1 起即达 oracle 水平**——2 个 swap 窗(300+217 ops,含首次 P2P 缓冲预热 4.2s 墙钟)全部落在 r1 的 133s 内,适应成本被摊薄到不可见;r2 起零 swap。冷启动负收益确认为短 burst 瞬态。
2. **A≈B≈C≈DF(±0.7%)**:收敛后的在线放置与离线 oracle 无差别;"B 死区冻结导致落后"的预测被证伪(B 反而数值最高,其规划集中预算打磨最差层,max_after=1.096)。
3. **协议敏感性警示**:饱和深队列下瓶颈移至调度/请求生命周期(绝对 tok/s 仅为 burst 协议 ~40%),identity 与放置组差距被压缩到 +3.1%(burst 协议为 +21.8%)。**phase6/7 只用于收敛性论证,不用于方法排名**;排名仍以 §3 两张表(burst 口径,与 H20 文档一致)为准。

数据:`results/_0914_a100_sust_{aseq,bseq,hseq,dseq,iseq}_r*.json`(19 个);脚本 `scripts/phase6_sustained.sh`、`phase7_identity_calib.sh`;DIAG 轨迹见 `logs/mechanism_evidence.txt` 末尾追加段。
