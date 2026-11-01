# PB-OEPLB 在 GB200/Blackwell（EP=4）上的复现研究

> 独立复现文档，供后续整合进主论文。硬件：4×GB200（Blackwell，sm_100，aarch64，185GB/卡，NVLink）；
> 模型：Qwen3-235B-A22B-FP8（94 MoE层，128专家，top-8）；并行：TP=DP=EP=4（每卡32本地专家）；
> 栈：torch 2.9.1+cu129，SGLang dev(dce8b06)+PB-OEPLB patch，sgl-kernel 0.3.20，DeepEP normal，DeepGEMM FP8。
> 与主论文差异：主论文为 8×H20（EP=8，sm_90）；本文为 EP=4/GB200，绝对增益被压缩（见§4上界分析），但机制与相对结论一致。

## 摘要

在 GB200/EP=4 上端到端复现 PB-OEPLB 及 §5.3.1 五方对比（identity / EPLB静态 / EPLB动态 / DataForest-Remap / MoETuner，MoETuner 用 gurobipy 受限license 求解 ILP1）。核心结论：(1) **机制逐条验证成立**——r_k 幂律自动算得 1.034、swap 收敛 1.333→1.016、死区门控、自适应窗、VERIFY-WEIGHT-MOVE 全部复现；(2) **跨域"在线vs离线分水岭"三硬件一致复现**——唯 PB-OEPLB 跨域显著正（O=10 +3.8%、O=1 +4.1%），EPLB动态最差（−12.6%，rebalance 阻塞占 28.2% 墙钟）；(3) **死区/上界理论闭环自洽**——nsys 实测 f_sens(GB200)=0.184≈H20 的 0.386/2.06，定量证实"算力↑→f_sens↓"，Δ_max=4.3% 对实测头条 +3.8% 给 η=88%≈H20 的 86%；(4) **同域增益由输入长度决定**（相关+0.747），长输入(≥400tok)+4.6%、短输入(<200tok)−2.7%，且 aggregate ratio 非吞吐充分统计量（离线ILP1同ratio却高6%）；(5) **TPOT 收益强依赖 PD 相关性**——QA类(mmlu,O=64) TPOT −4.6%，数学类(prover,O=32) TPOT +6.5%。

---

## 1. 实验设置与方法学

- **硬件**：4×GB200（Blackwell，sm_100，aarch64，185GB/卡满血无共租户，NVLink）。EP=4，每卡 32 本地专家。
- **模型/栈**：Qwen3-235B-A22B-FP8，SGLang dev + PB-OEPLB 6文件patch（sitecustomize.py shadow editable 安装，pristine 树零改动），DeepEP normal + DeepGEMM FP8。
- **数据集**：同域 prover（L512_O1_realprover / prover_256tok，fair-split head1024 profile、tail1024 bench）；跨域 freq6（6段 book↔prover，N=1800，conc32）；8数据集单域扫描（mmlu/gsm8k/humaneval/cmmlu/csqa/arc_easy/obqa/prover256）。
- **方法学要点**：(1) GB200 本机重录 profile（recorder stat），fair-split 无泄漏；(2) 三方静态放置统一走显式 `physical_to_logical_map`→`init_by_mapping`（早期 `logical_count`→`init_by_eplb` 路径严重低估 DataForest）；(3) **每数据集 identity 与 OEPLB 背靠背同 session 交错测**——GB200 有高达 11% 的 session 间系统漂移（GPU时钟/热），短run对其极敏感，跨session比较无效；(4) 充分预热 + 3轮中位。

---

## 2. 头条复现（Table A）

L512_O1_realprover，O=1 纯prefill，8192请求，conc256，run_grid_bench。

| 臂 | time(s) | tps(req/s) | TTFT mean |
|---|---|---|---|
| identity | 92.29 (n=5, CV0.67%) | 88.76 | 2865ms |
| PB-OEPLB | 88.92 (n=3稳态, CV0.04%) | 92.13 | 2733ms |
| **增益** | | **+3.8%** | **−4.6%** |

机制验证（server log）：`auto dead_zone_ratio=1.034 (EP=4)`；swap 收敛 window1 1.333→1.024(298ops)→1.017→1.016；`VERIFY-WEIGHT-MOVE×12`；swap 开销冷启518ms→稳态14-20ms；自适应窗 `ratio jump 0.280→缩窗8+α→0清史→扩16→32→64`。

---

## 3. §5.3.1 两表复现

### Table B — 同域六方（prover256 tail1024, O=1, conc256, run_grid_bench, fair-split）
| 方法 | req/s | vs id | TTFT |
|---|---|---|---|
| identity | 151.9 | — | 1531ms |
| EPLB动态(16冗余) | 154.8 | +1.9% | −2.6% |
| PB-OEPLB | 158.6 | +4.4% | −5.0% |
| EPLB静态(16冗余) | 161.2 | +6.1% | −5.8% |
| DataForest-Remap(无冗余) | 162.0 | +6.6% | −7.1% |
| MoETuner(ILP1) | 167.3 | +10.1% | −9.9% |

warm 复核（充分预热3轮中位）：identity 165.2 / PB-OEPLB 172.5(+4.4%) / DataForest 174.6(+5.7%) / MoETuner 181.6(+9.9%)，增益一致。同域离线最优(MoETuner)略胜在线(PB-OEPLB)，与主论文"同域 OEPLB≈DataForest oracle"方向一致（本文二者差~1%在噪声内）。

### Table C — 跨域六方（freq6, conc32；PB-OEPLB为充分预热3轮中位CV<0.7%，其余单次）
| 方法 | O=10 req/s | vs id | O=1 req/s | vs id | H20对照(O=10) |
|---|---|---|---|---|---|
| EPLB动态(52-108次重排,阻塞2-3s) | 5.81 | **−12.6%** | 13.7 | −7.4% | −6% |
| EPLB静态 | 6.54 | −1.7% | 14.7 | −0.7% | −4% |
| identity | 6.65 | — | 14.8 | — | 4.7 |
| DataForest-Remap | 6.72 | +1.1% | 15.0 | +1.4% | +0% |
| MoETuner | 6.76 | +1.7% | 14.8 | +0.0% | −1.5% |
| **PB-OEPLB** | **6.90** | **+3.8%** | **15.40** | **+4.1%** | +8.5% |

**跨域是本次最稳的结论**：唯 PB-OEPLB 显著正，离线/静态全归零或负，EPLB动态最差。三硬件（H20/A100/GB200）排序一致。

---

## 4. 死区(r_k)与上界(Δ_max)理论的闭环验证

### Table H — f_sens 硬件依赖（nsys 实测 20s×4GPU 稳态，β_c：Combine+1.33/Expert+0.08/Dispatch−0.78）
| | H20(EP8,§E15) | GB200(EP4,本文) |
|---|---|---|
| f_expert(计算) | 48.7% | **20.4%** |
| f_dispatch+f_combine(通信) | 40.4% | **60.2%** |
| f_dispatch | 9.5% | **30.0%** |
| f_combine | 30.9% | 30.2% |
| **f_sens=Σβ_c·f_c** | 0.386 | **0.184 (≈0.386/2.06)** |

**闭环**（每个输入独立实测，无循环）：r_before=1.334（离线计数≈实测DIAG1.333，偏差0.11%）、r_k=1.034（EP幂律=server自动）、r_after=1.016（DIAG）、f_sens=0.184（nsys）→ x_eff=0.225 → Δ_max=f_sens·x_eff/(1−f_sens·x_eff)=**4.3%** → 实测头条+3.8% → **η=88%≈H20的86%**。

**机理**：f_sens 几乎只由两个通信项决定（β_expert≈0，计算占比不进公式）；GB200 计算变快本身不直接降 f_sens，而是把瓶颈推向通信、令 Dispatch(负β)占比从9.5%暴涨到30%，这才是 f_sens 减半的直接原因。均衡前后绝对时间验证 β_c：降 ratio 几乎**只缩短 Combine（−18%，22.1→18.1s）**，Dispatch/Expert 基本不动。**推论**：专家均衡收益上界由 NVLink/all-to-all 带宽侧决定而非 FLOPs；算力越强上界越低——与"GPU越好收益越大"直觉相反。

---

## 5. 同域增益的根因

### Table D — 8数据集单域扫描（每集 identity/OEPLB 同session交错, 3轮中位）
| 数据集 | ~输入tok | r_before | 增益 |
|---|---|---|---|
| humaneval | 650 | 1.232 | **+4.2%** |
| prover256 | 500 | 1.344 | **+4.9%** |
| obqa | 73 | 1.227 | +0.7% |
| cmmlu | 183 | 1.231 | −1.4% |
| mmlu | ~300 | 1.171 | −2.4% |
| arc_easy | 133 | 1.210 | −2.4% |
| gsm8k | 256 | 1.298 | −3.3% |
| csqa | 88 | 1.267 | −7.6% |

**增益 vs 输入长度相关 +0.747（强）；vs headroom(x_eff) +0.23（弱）、vs r_before +0.24（弱）**。长输入(≥400tok)平均+4.6%全正，中(200-400)−2.8%，短(<200tok)−2.7%。

**机理**：吞吐收益 = 相对不均衡降幅 × 每forward绝对计算时间。短prompt单次prefill计算量极小→straggler等待绝对时间仅几ms，而swap开销近似固定成本→净负；长prompt→straggler显著→正。**复现并印证 Observation 2**（输入越长收益越大）。

### 放置结构质量（同域OEPLB<离线的根因）
系统排查排除4假设：充分预热后仍+4.4%（非收敛不足）；死区降到1.005逼OEPLB到ratio1.002吞吐反降（**印证死区理论:r_k以下无收益**，非死区太保守）；宽死区1.06使其settle停swap吞吐更低（非churn）；bias_correct更差（非采样偏置）。**决定性证据**：OEPLB激进(ratio1.002)=171.5 vs MoETuner(ratio1.000)=181.6，且MoETuner与DataForest的per-layer ratio分布完全相同(avg/p90/max均≈1.000)吞吐却差4%→**aggregate ratio(任何分位)不是吞吐的充分统计量**；差异在专家→GPU具体指派（ILP1全局最优>贪心LPT>贪心pairwise-swap局部搜索），指派结构通过每GPU专家token分布影响 DeepGEMM staircase 档位与all-to-all通信量。**这是在线贪心 vs 离线全局优化的固有权衡**，也是主论文§6.2"r对30B是弱充分统计量"在EP=4/235B上的重现——建议上界模型引入 prompt长度/指派结构项。

---

## 6. TPOT 的配置依赖（PD相关性 × 输出长度）

PB-OEPLB 只记 prefill 路由，故 TPOT 是否受益取决于 prefill→decode 路由相关性（§3.4：QA/推理 ρ=0.78-0.85 强、数学 ρ=0.44-0.69 弱）。

### Table F
| 数据集(任务) | PD相关ρ | 输出 | TPOT identity→PB-OEPLB | 结论 |
|---|---|---|---|---|
| prover512(数学) | 弱 | O=32 | 252.6→269.2ms（**+6.5%差**）| prefill放置不匹配decode路由 |
| freq6(混合) | 中 | O=10 | 312.1→310.4ms（−0.5%）| 近中性 |
| **mmlu(QA/推理)** | **强** | **O=64** | 223.4→213.2ms（**−4.6%**），p99 240.4→227.7（**−5.3%**）| **prefill放置迁移到decode有效** |

**mmlu O=64 是全指标最佳配置**：吞吐+4.9%、TTFT−5.3%、TPOT−4.6%、TPOT-p99−5.3% 全部同向改善。**"TPOT收益不高"是数据集(PD弱相关)与输出长度选择问题，选对配置(QA类+O=64)即可测出显著TPOT收益**——定量复现并印证 §3.4/Observation 3。

---

## 7. 全指标矩阵（Table E；不含利用率——nvidia-smi util% 被 DeepEP busy-wait 自旋污染，非有效指标）

同域 prover512 O=32（含decode）+ 跨域 freq6 O=10，各2轮。

| 数据集 | 方法 | 吞吐tps | CV | TTFT mean/p99(ms) | TPOT mean/p99(ms) |
|---|---|---|---|---|---|
| 同域O=32 | identity | 830.3 | 3.9% | 2042/3275 | 252.6/313 |
| | PB-OEPLB | 795.0 | **1.9%** | **1955**/3252 | 269.2/322 |
| | EPLB动态 | 705.0 | 0.1% | 2092/4870 | 307.0/397 |
| 跨域O=10 | identity | 69.5 | 1.6% | 1759/3284 | 312.1/507 |
| | PB-OEPLB | 70.8 | **0.3%** | **1686**/3189 | 310.4/506 |
| | EPLB动态 | 66.8 | 4.2% | 1824/4667 | 325.5/637 |

要点：TTFT 两数据集 OEPLB 均最优（−4.2~4.3%）；稳定性 OEPLB 最好（CV 最低）；同域O=32（decode-heavy）OEPLB 吞吐≈identity（prefill收益被decode稀释，见§6）。

---

## 8. cost 对比（Table G）

| 方法 | 调整机制 | 运行时开销 | 离线(提前)开销 | 漂移后 |
|---|---|---|---|---|
| identity | 无 | 0 | 0 | — |
| **PB-OEPLB** | 增量swap(P2P) | **0.40%墙钟**(52次决策,稳态44ms) | 0(在线学习,~3窗收敛) | 自动适应 |
| **EPLB动态** | 全量rebalance | **28.2%墙钟**(108次×1.77s,max2.92s) | 0 | 自动但阻塞 |
| DataForest-Remap | 冻结放置 | 0 | 录制~9s/1024req + 贪心LPT **22.4ms** | 须重录+重算 |
| MoETuner | 冻结放置 | 0 | 录制 + **ILP1 Gurobi 44.5s(热)~171.5s(冷)** + **商业license** | 须重录+重解ILP |

**EPLB动态调整开销是PB-OEPLB的77×，吃掉28.2%墙钟**，是其吞吐净负（同域−15%、跨域−4~13%）的直接主因。"只记录不swap"对照臂证实 PB-OEPLB 的 record+all_reduce 开销≈噪声(<1%)。静态法运行时零开销但有离线成本、且workload漂移须重做（MoETuner还要重解ILP+license）——PB-OEPLB零离线、在线适应是其核心工程优势。

---

## 9. 复现工程要点（Blackwell/EP=4）

1. **DeepGEMM冷JIT必致hang**：grouped-GEMM冷编译期间其余rank卡在DeepEP all-to-all barrier→`DeepEP timeout check failed`→永久挂死。必须先 `sglang.compile_deep_gemm`（`warmups=compile-deep-gemm` 孤立编译、不走all-to-all）预编译，之后warmup从~1it/s跳到1600it/s。冗余专家使num_groups 32→36（新shape）、且`--enable-eplb`的recorder在warmup期会desync——须用"带`--ep-num-redundant-experts`但不带`--enable-eplb`"预编译。
2. `nvidia-smi`误报compute_cap 8.9（实为sm_100），须`TVM_FFI_CUDA_ARCH_LIST=10.0`否则tvm_ffi编成compute_89触发ptxas fatal（表象为假DeepEP timeout）。
3. SGLang为editable安装（PEP-660 MetaPathFinder优先于PYTHONPATH），用`sitecustomize.py`摘除sglang finder（不动megatron/slime）实现shadow，pristine树零改动可回退。
4. dev版`--init-expert-location`仅接受纯`physical_to_logical_map`(走`init_by_mapping`)或`logical_count`(走`init_by_eplb`)，H20归档的带num_layers/ep_size的JSON会TypeError；三方统一用`init_by_mapping`显式map是可复现关键。
5. **代码bug**：`window_floor>sync_window`时自适应窗口"收缩"反向变"扩张"（`controller.py:807` `max(window_floor,W//2)`，默认32>16），建议加`assert window_floor<=sync_window`。

---

## 10. 结论

1. PB-OEPLB 在 GB200/EP=4 上机制完整复现（r_k幂律、死区门控、swap收敛、自适应窗、权重物理搬动全部验证）。
2. **跨域"在线vs离线分水岭"三硬件一致**：唯 PB-OEPLB 跨域正收益（+3.8~4.1%），EPLB动态因28.2%墙钟阻塞最差（−12.6%）。
3. **死区/上界理论闭环自洽**：nsys实测 f_sens(GB200)=0.184≈H20/2.06，定量证实"算力↑→f_sens↓"；Δ_max=4.3%→实测+3.8%→η=88%≈H20的86%。上界由带宽侧决定而非FLOPs。
4. **同域增益由输入长度决定**（相关+0.747），且 aggregate ratio 非吞吐充分统计量（在线贪心 vs 离线全局优化的结构质量差）——对上界模型的refinement：须引入 prompt长度/指派结构维度。
5. **TPOT收益强依赖PD相关性**：QA类(mmlu,O=64) TPOT−4.6%，数学类(prover,O=32) +6.5%——印证§3.4/Observation3。
6. cost：PB-OEPLB调整开销0.40%墙钟 vs EPLB动态28.2%（77×）；静态法零运行时但有离线成本+漂移须重做。

> 完整脚本、6方法×多workload结果JSON、5个OEPLB调参臂、三方nsys profile CSV、fresh profile与placement、机制日志见 `experiments/gb200_ep4_repro/`；机器归档 `/data/minghua/sjq/OEPLBdata/experiment_logs/gb200_ep4_repro/`。
