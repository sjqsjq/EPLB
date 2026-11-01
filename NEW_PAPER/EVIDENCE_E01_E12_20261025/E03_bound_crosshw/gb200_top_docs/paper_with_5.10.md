### 5.10 跨硬件泛化：GB200/Blackwell（EP=4，sm_100，DeepEP/FP8 主线路径）

将§5.3.1五方对比移植到 **4×GB200（Blackwell，sm_100，aarch64，185GB/卡，NVLink）**，走与H20相同的 FP8+DeepGEMM+DeepEP 主线路径，**并行度 EP=4**（每卡32本地专家）。方法学：(1) 本机用原生recorder重录 prover 路由计数、fair-split（head1024 profile / tail1024 bench，同分布不相交无泄漏；新旧profile余弦0.971，证实H20归档profile未严重错配，但本机重录更严谨）；(2) DataForest/EPLB静态/MoETuner 三方**统一走显式 `physical_to_logical_map`→`init_by_mapping` 注入**（早期DataForest误用 `logical_count`→`init_by_eplb` 叠加短burst协议，收益被严重低估，修正后同域从+0.3%→+5.7%）；(3) 全部用 run_grid_bench 持续负载 + **充分预热 + 3轮中位**（替代 N=256 单burst，后者仅~3s、噪声5~10%且冷首run系统性偏低~8%）；(4) MoETuner ILP1 用 `pip install gurobipy` 受限license求解（每层516变量小规模MIP在≤2000上限内，94层44s、mean_imbalance=1.000）。EPLB两臂 redundant16；全部 `--deepep-mode normal --disable-cuda-graph`。

**同域（prover_256tok tail1024, O=1, conc=256, 充分预热+3轮中位）**：

| 方法 | req/s | vs identity | TTFT mean | 收敛ratio | H20对照(§5.3.1) |
|---|---|---|---|---|---|
| identity | 165.2 | — | 1437ms | 1.341 | 62.4 |
| **PB-OEPLB（在线swap）** | 172.5 | **+4.4%** | −5.0% | 1.018 | +20.7% |
| DataForest-Remap（贪心LPT,冻结） | 174.6 | +5.7% | −6.4% | 1.000 | +18.0% |
| MoETuner（ILP1全局最优,冻结） | 181.6 | +9.9% | −7.4% | 1.000 | +13.0% |

**跨域（freq6: 6段book↔prover, N=1800, conc=32）**——PB-OEPLB为充分自适应后3轮中位（CV<0.7%），其余单次：

| 方法 | O=10 req/s | vs id | O=10 TPOT | O=1 req/s | vs id | H20对照(O=10) |
|---|---|---|---|---|---|---|
| EPLB动态（52次重排,每次阻塞2~3s） | 5.81 | **−12.6%** | **+17.8%** | 13.7 | −7.4% | −6% |
| EPLB静态（prover+redundant16） | 6.54 | −1.7% | +2.5% | 14.7 | −0.7% | −4% |
| identity | 6.65 | — | — | 14.8 | — | 4.7 |
| DataForest-Remap（prover冻结） | 6.72 | +1.1% | −0.8% | 15.0 | +1.4% | +0% |
| MoETuner（prover-fit迁移） | 6.76 | +1.7% | −1.9% | 14.8 | +0.0% | −1.5% |
| **PB-OEPLB（RESET+缩窗在线适应）** | **6.90** | **+3.8%** | **−5.3%** | **15.40** | **+4.1%** | +8.5% |

**延迟（identity→PB-OEPLB）**：同域 TTFT 1437→1365ms（−5.0%）；跨域O=10 TTFT −1.0%、**TPOT 337.6→319.7ms（−5.3%）**。同域O=1无decode故TPOT为null（这解释"同域只有TTFT"）；跨域O=10的TPOT−5.3%印证Observation 3。

五点结论。

**其一，"在线vs离线分水岭"在Blackwell/EP=4复现，是本次最稳的结论**：跨域下**唯PB-OEPLB显著为正（O=10 +3.8%、O=1 +4.1%，3轮CV<0.7%）**，离线/静态全归零或负（DataForest +1.1%、MoETuner +1.7%——prover冻结放置对3/6的book段错配），EPLB静态−1.7%，**EPLB动态最差（O=10 −12.6%、TPOT+17.8%）**：52次全量重排、每次阻塞2~3s，域切换下追逐混合流恶化。三硬件（H20/A100/GB200）排序一致。

**其二，O=1 vs O=10 揭示EPLB的decode惩罚**：EPLB动态跨域从O=1的−7.4%恶化到O=10的−12.6%、TPOT+17.8%——输出越长decode占比越大，冗余挤占KV+周期rebalance阻塞的惩罚越重，是论文"EPLB强制normal禁CUDA graph→decode-heavy退化"的EP=4体现。O=1纯prefill下各方压缩到±1.4%（除EPLB动态），PB-OEPLB仍+4.1%。

**其三，同域OEPLB(+4.4%)≈DataForest(+5.7%)、低于MoETuner(+9.9%)——根因是"在线贪心 vs 离线全局优化"的放置结构质量差，而非可调参缺陷**。系统排查排除了4个假设：充分预热后仍+4.4%（非收敛不足）；把死区从r_k=1.034降到1.005、逼OEPLB收敛到ratio 1.002（≈MoETuner的1.000），吞吐反而171.5<172.5（**恰好印证死区理论：r_k以下再均衡无收益**，非死区太保守）；宽死区1.06使其settle停swap，吞吐169.9更低（非churn抖动）；bias_correct 169.5更差（非采样偏置）。**决定性证据**：OEPLB激进(ratio 1.002)=171.5 vs MoETuner(ratio 1.000)=181.6，且MoETuner与DataForest的per-layer ratio分布完全相同(avg/p90/max均≈1.000)吞吐却差4%——**证明aggregate ratio(任何分位)不是吞吐的充分统计量**；差异在专家→GPU的具体指派：ILP1全局最优 > 贪心LPT > 贪心pairwise-swap(从identity局部搜索)，指派结构通过每GPU上专家token分布影响DeepGEMM staircase档位(§3.1)与all-to-all通信量。**这与论文"同域OEPLB≈DataForest oracle"一致**（本节二者差~1%在噪声内）；MoETuner在EP=4(每卡32专家)异常突出，而论文EP=8(每卡16专家)下MoETuner(+13%)<DataForest(+18%)——**MoETuner相对强弱随EP翻转，非OEPLB退化**。同域绝对增益整体被EP=4/GB200上界压缩(论文EP=8同域+18~20%)。

**其四，$f_{sens}$硬件依赖由nsys实测定量证实（回应§3.3"算力↑→$f_{sens}$↓"）**。对identity与PB-OEPLB各采20s×4GPU稳态负载分阶段（$\beta_c$：Combine+1.33/Expert+0.08/Dispatch−0.78）：$f_{expert}$从H20的48.7%降到GB200的**20.4%**、通信占比升到**60.2%**、$f_{dispatch}$从9.5%暴涨到**30.0%**，得 $f_{sens}(GB200)=\mathbf{0.184}\approx0.386/2.06$，与"Blackwell FP8 GEMM约2×于Hopper→计算时间腰斩→MoE步转为通信主导"定量吻合。**机制**：$f_{sens}$几乎只由两个通信项决定（$\beta_{expert}\approx0$）；算力变快本身不直接降$f_{sens}$，而是把瓶颈推向通信、令Dispatch(负$\beta$)占比暴涨，这才是$f_{sens}$减半的直接原因。均衡前后绝对时间验证$\beta_c$：降ratio几乎**只缩短Combine(−18%,22.1→18.1s)**。**闭环**：$r_{before}=1.341、r_k=1.034、f_{sens}=0.184$→同域$\Delta_{max}=4.3\%$，PB-OEPLB头条(L512_O1_realprover)实测+3.8%→$\eta=88\%$≈H20的86%。**推论**：专家均衡收益上界由带宽侧决定而非FLOPs；算力越强上界越低——与"GPU越好收益越大"直觉相反。

**其五，对hinge/死区模型的一个refinement**：其三证明EP=4/每卡32专家下，同r的不同指派吞吐差4~6%→**r在EP=4不是充分统计量**（论文§6.2已承认"r对30B是弱充分统计量"，本发现在EP=4/235B上重现该局限）；建议上界模型引入指派结构项（如每GPU专家token分布的staircase档位）以在低EP/大每卡专家数下保持预测力。

**复现工程要点（Blackwell/EP=4）**：(1) **DeepGEMM冷JIT必致hang**——grouped-GEMM冷编译期间其余rank卡在DeepEP all-to-all barrier→永久挂死；必须先`sglang.compile_deep_gemm`（`warmups=compile-deep-gemm`孤立编译不走all-to-all）预编译，之后warmup从~1it/s跳到1600it/s；冗余专家使num_groups 32→36、且`--enable-eplb`的recorder在warmup期会desync，须用"带`--ep-num-redundant-experts`但不带`--enable-eplb`"预编译。(2) `nvidia-smi`误报compute_cap 8.9(实为sm_100)，须`TVM_FFI_CUDA_ARCH_LIST=10.0`否则tvm_ffi编成compute_89触发ptxas fatal(表象为假DeepEP timeout)。(3) SGLang为editable安装(PEP-660 MetaPathFinder优先于PYTHONPATH)，用`sitecustomize.py`摘除sglang finder实现shadow，pristine树零改动可回退。(4) dev版`--init-expert-location`仅接受纯`physical_to_logical_map`(走`init_by_mapping`)或`logical_count`(走`init_by_eplb`)，H20归档的带num_layers/ep_size的JSON会TypeError；**三方统一用`init_by_mapping`显式map是可复现的关键**。(5) 发现`window_floor>sync_window`时自适应窗口"收缩"反向变"扩张"(`controller.py:807` `max(window_floor,W//2)`,默认32>16)，建议加`assert window_floor<=sync_window`。完整脚本、6方法×3workload结果JSON、5个OEPLB调参臂、三方nsys profile CSV、fresh profile与placement、机制日志见`experiments/gb200_ep4_repro/`，机器归档`/data/minghua/sjq/OEPLBdata/experiment_logs/gb200_ep4_repro/`。

#### 5.10.1 单域增益由输入长度决定（8数据集扫描，对上界模型的refinement）

在GB200/EP=4上对8个单域数据集（prover256/humaneval/gsm8k/mmlu/cmmlu/arc_easy/csqa/obqa，各2048条、O=1纯prefill、conc256）逐一测 identity vs PB-OEPLB。**测量方法论要点**：GB200存在高达11%的session间系统漂移（GPU时钟/热状态，同一identity配置在不同session测得280 vs 312 req/s），短数据集每run仅~7s对此极敏感；故**每数据集identity与OEPLB背靠背同session交错测**（各3轮中位，CV多<3%），消除跨session比较的偏差。

| 数据集 | ~输入tok | r_before | identity | PB-OEPLB | 增益 |
|---|---|---|---|---|---|
| humaneval | 650 | 1.232 | 244.4 | 254.6 | **+4.2%** |
| prover256 | 500 | 1.344 | 170.0 | 178.4 | **+4.9%** |
| gsm8k | 256 | 1.298 | 303.4 | 293.3 | −3.3% |
| mmlu | ~300 | 1.171 | 284.1 | 277.3 | −2.4% |
| cmmlu | 183 | 1.231 | 290.8 | 286.8 | −1.4% |
| arc_easy | 133 | 1.210 | 312.3 | 304.9 | −2.4% |
| csqa | 88 | 1.267 | 328.2 | 303.2 | −7.6% |
| obqa | 73 | 1.227 | 317.8 | 320.1 | +0.7% |

平均−0.9%、3/8为正。**关键规律：增益与输入长度强相关（+0.747），与不均衡headroom几乎无关（x_eff相关+0.23、r_before +0.24）**。长输入（≥400tok）平均+4.6%（全正），中（200-400）−2.8%，短（<200tok）−2.7%。

**机理**：吞吐收益 = 相对不均衡降幅 × **每forward绝对计算时间**。短prompt（25-200tok）单次prefill计算量极小→不均衡的straggler等待绝对时间仅几ms，而PB-OEPLB的swap开销（P2P权重搬动+抖动）是近似**固定成本**→固定开销>微小收益→净负；长prompt（500-650tok）prefill计算量大→straggler显著→纠偏收益盖过开销→正。这**复现并印证Observation 2**（"输入越长OEPLB收益越大：short<medium<long"），且在EP=4/GB200下短输入进一步跌入负区（f_sens低+固定swap开销）。

**对上界模型的refinement**：$\Delta_{max}=f_{sens}x_{eff}/(1-f_{sens}x_{eff})$ 只含相对量$x_{eff}$，缺绝对计算时间项。实测表明应修正为 $\Delta_{max}\propto x_{eff}\times$（每forward计算时间）$\propto x_{eff}\times$ prompt\_length：prover256（长，η=111%）、humaneval（η=137%）符合原模型，但csqa（短，η=−217%）原模型完全失效→**上界模型须引入prompt长度/绝对计算时间维度**，否则对短输入负载会把净负误判为正。这也解释了§5.3.1头条用L512 prover（500tok长prompt）——正落在OEPLB的有利区。

#### 5.10.2 指标补全：静态法离线成本 与 TPOT的配置依赖

**（1）DataForest/MoETuner的离线（提前）开销**。二者运行时无调整开销，但有一次性离线成本，且workload漂移后须重做（正是其跨域失效根因）：DataForest-Remap = 1遍推理录制路由（~9s/1024请求，随集线性）+ 贪心LPT放置计算 **22.4ms**（94层），无额外依赖；MoETuner = 同录制 + **ILP1 Gurobi求解 44.5s(热)~171.5s(冷)/94层 + 需商业Gurobi license**。对照：PB-OEPLB零离线成本（在线学习、~3窗收敛），EPLB动态零离线但运行时每100iter全量重排（本节实测108次×1.77s=191.6s=**28.2%墙钟阻塞**，是PB-OEPLB增量swap 0.40%的77×）。生产环境workload变化时，静态法须周期性重profiling+重算（MoETuner还要重解ILP），PB-OEPLB自动适应。

**（2）TPOT收益强依赖配置（PD相关性×输出长度）**。PB-OEPLB只记prefill路由，故TPOT是否受益取决于prefill→decode路由相关性（§3.4：QA/推理ρ=0.78-0.85强、数学ρ=0.44-0.69弱）：

| 数据集(任务) | PD相关ρ | 输出 | TPOT identity→PB-OEPLB |
|---|---|---|---|
| prover512(数学) | 弱 | O=32 | 252.6→269.2ms（**+6.5%差**）|
| freq6(混合) | 中 | O=10 | 312.1→310.4ms（−0.5%）|
| **mmlu(QA/推理)** | **强** | **O=64** | 223.4→213.2ms（**−4.6%**），p99 240.4→227.7（**−5.3%**）|

数学prover上prefill放置不匹配decode路由→TPOT无益甚至因swap扰动略差；QA/推理(mmlu)上prefill热点≈decode热点→放置同时优化decode→**TPOT改善−4.6%**。这定量复现并印证§3.4/Observation3。**mmlu O=64是本次全指标最佳配置**：吞吐+4.9%、TTFT−5.3%、TPOT−4.6%、TPOT-p99−5.3%，全部同向改善——说明"TPOT收益不高"是数据集(PD弱相关)与输出长度选择问题，选对配置(QA类+O=64)即可测出显著TPOT收益。


### 5.11 四硬件统一上界模型（H20/A100/GB200/H800 综合）

§2.4 的铰链模型经四处硬件截面独立检验后，综合为可操作的统一形式（v2）。设 f_sens 为偏斜敏感时间份额（nsys 按 kernel 占比×敏感系数 β_c 分解：Combine +1.33、Expert +0.08、Dispatch −0.78，§5.10），r_k 为死区右端点；T(r) 扫描的斜率口径与之等价：β=B/T_flat≈f_sens。对任意数据集由离线路由计数零 GPU 算得 r_before，则

$$x_{eff}=\frac{r_{before}-\max(r_{after},r_k)}{r_{before}},\qquad \Delta_{max}=\frac{f_{sens}\cdot x_{eff}}{1-f_{sens}\cdot x_{eff}},\qquad 实得=\Delta_{max}\times\eta - tax$$

**四硬件参数与闭环**：

| 硬件/栈 | f_sens(β) | r_k | Δ_max→实测 | η / 在线税 |
|---|---|---|---|---|
| H20 FP8+DeepEP, EP8 | 0.386 (nsys) | 1.093 (L512扫描) | 22.7%→+17.5~20.7% | η≈86% |
| A100 BF16+Triton/NCCL, EP8 | 0.33~0.36 (T(r)斜率) | 1.053~1.070 (L256) / 1.133 (L494)；57B EP2/4: 1.016/1.026 | +21.1%→DataForest实测+21.8% | η≈100% / OEPLB稳态税≈0.3% |
| GB200 FP8+DeepEP, EP4 | 0.184 (nsys) | 1.034 (幂律自动+闭环验证) | 4.3%→+3.8% | η≈88% / OEPLB税0.40% |
| H800 DeepEP自旋吸收, EP8 | 0.03 | **1.7~2.3** (12点扫描) | 3.9%@r3.2→静态臂3.8% | 税6.5%>Δ_max→全臂≤0 |

**v2 相对 §2.4 的三条修正律**（各有独立实测支撑）：
1. **r_k 幂律的适用域**：r_k−1=c·EP^1.52（c=0.00408）在 **GEMM 主导栈间可迁移**——A100 57B 独立扫描 EP2/EP4=1.016/1.026，H20 定律外推误差≤0.008；GB200/EP4 自动值 1.034≈H20 EP4 的 1.032。在**自旋吸收型通信主导栈**（H800，comm/GEMM=72/13）定律失效（实测 1.7~2.3），须直接扫描判据。
2. **r_k 的 workload 依赖强于硬件依赖**：r_k−1 随 prompt 长度近线性（A100：L256→L494 使 0.06→0.133，×2.2≈长度比 1.93；机制=attention 不敏感份额单边膨胀），chunk 减半 +0.02~0.03；GB200 8 数据集增益 vs 输入长度相关 +0.747（vs r_before 仅 +0.24）。同 workload 下硬件间 r_k 差仅 ±0.5~3.7%。
3. **ratio 非充分统计量（指派结构项）**：同 r≈1.000 下，ILP 全局指派与贪心指派的吞吐可差 ~few%（指派结构经 DeepGEMM staircase 档位与 all-to-all 量影响时间），故上界模型须保留 ~few% 的指派结构误差带——此点四硬件一致成立。**但"ILP 与贪心孰优"的排序在现有数据点上不一致，且该不一致与 EP 完全混淆、不能归因于硬件/kernel**：GB200/EP4 上 MoETuner(ILP)>DataForest(贪心) 约 +4~6%，而 A100/EP8、H20/EP8 上 DataForest≥MoETuner 约 1~2%；**唯一出现反号的 GB200 恰是四硬件中唯一的 EP=4 点**，hardware（Blackwell/DeepGEMM-sm100）与 EP（4 vs 8）两因素在此完全共变，无法解耦。就现有证据，更 parsimonious 的解释是 EP 而非硬件：EP=4 每卡 32 专家，ILP 全局优化相对贪心 LPT 的自由度收益更大；EP=8 每卡 16 专家，贪心已近最优（与本文附§5.10.1 的 EP=4 观察一致）。**须补 GB200/EP8 或 A100/EP4 一个对照点方能分离 EP 与硬件两因素**；在此之前，律 3 只确立"ratio 非充分统计量、存在 ~few% 指派结构误差带"这一四硬件共性，**不对排序反号做硬件/kernel 归因**（避免把 EP 效应误标为硬件效应）。

**决策流程**（把"跑几十次实验"化为"算一次公式"）：① 一次 T(r) 布局扫描（~2h GPU）或 nsys 稳态 20s → f_sens、r_k；② 离线 counts → r_before（零 GPU）；③ 公式给 Δ_max；④ 与控制器税比较（实测：PB-OEPLB 0.3~0.4% 墙钟；EPLB动态 4.7~28.2%）→ Δ_max>tax 才部署。该流程的事后检验：H800 上 11 次端到端结果全部落于公式预测内（零意外，含"税>天花板→全负"）；A100/GB200 闭环 η=86~100%。

**适用域**：铰链线性段验证至 r≤1.8（A100 conc 点 r=4.57 呈 ~5% 上凸，线性外推在该点低估时间）；f_sens 的 nsys 分解现有 H20/GB200 两组（A100 为斜率口径等价推得，H800 待补）；H800 原始扫描表待归档后并入实验索引。

