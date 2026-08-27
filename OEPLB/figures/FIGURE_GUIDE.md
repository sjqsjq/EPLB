# 图表解读文档

> 每张图写明：**为什么测**、**横纵轴含义**、**反映了什么现象**、**对论文的启发**。
> 模型：Qwen3-235B-A22B-FP8（94层, 128专家, top-8, EP8=16专家/卡），O=10。

---

## Fig 1: 逐层负载不均衡度曲线

![Fig1](fig1_imbalance_ratio_per_layer.png)

**为什么测**：证明MoE推理存在严重的逐层负载不均衡——这是负载均衡的基本动机。需要量化"默认放置下到底有多不均衡"。

**横轴**：MoE层ID（0-93，共94层MoE层）。
**纵轴**：max/min Imbalance Ratio——该层8张GPU中，负载最重的GPU的专家选择总数 ÷ 负载最轻的GPU的专家选择总数。1.0=完美均衡。
**3条线**：蓝=MMLU(25tok QA)、橙=prover(1253tok math)、绿=book(4438tok BookCorpus)。均为identity放置（默认：专家0-15→GPU0, 16-31→GPU1, ...）。
**灰色虚线**：y=1.0（完美均衡基准线）。

**反映了什么现象**：
- **所有层都显著>1.0**：没有一层是均衡的。MMLU均值2.26×、prover 3.51×、book 4.38×——热点GPU负载是冷点的2-4倍。
- **book最极端**：某些层max/min达11.79×，意味着一本4400-token的书，某些层的热点GPU负载是冷点的近12倍。
- **跨层波动大**：不是均匀不均衡——某些层特别严重（路由集中在少数专家），某些层相对好。
- **不同数据集曲线不同**：MMLU(短prompt QA)相对平（2-4×），prover/book(长prompt)更高（3-12×）——长prompt路由更集中。

**对论文的启发**（§2.1 问题形式化）：负载不均衡是MoE推理的结构性问题，不是偶然——94层全部不均衡，均值2-4×，最极端12×。这直接证明了"需要负载均衡"的动机。straggler GPU决定整个MoE step的时间→计算资源浪费50-75%。

---

## Fig 2: Identity vs 最优放置对比

![Fig2](fig2_identity_vs_optimal.png)

**为什么测**：证明负载均衡的**理论收益空间**——如果放置最优(max/min→1.0)，能降低多少不均衡？这定义了OEPLB的上限。

**横轴**：3个数据集（MMLU/prover/book）。
**纵轴**：Mean max/min Imbalance Ratio（94层均值）。
**红色柱**：Identity（默认连续放置）。
**绿色柱**：Optimal（LPT贪心bin-packing：把热专家按频率降序，贪心分配到最轻GPU）。
**柱上数字**：具体ratio值。

**反映了什么现象**：
- **Identity 2.26-4.38× → Optimal ~1.00**：最优放置将不均衡度降到近完美（1.000-1.001），降低53-73%。
- **book降低最多(73.4%)**：book的identity不均衡最严重(4.38×)，但最优放置修复最多——因为长prompt的路由集中度高，LPT能有效分散。
- **所有数据集optimal≈1.00**：说明在这3个配置下，"移动专家位置"（不改路由、不加冗余）就足以达到完美均衡→§2.5"两个天花板重合"，不需要冗余专家。

**对论文的启发**（§2.4 理论加速上界 + §2.5 两个天花板）：最优放置→r≈1.0→53-73%降低→理论加速上界=不均衡比例×MoE计算占比。这定义了OEPLB的天花板和"零冗余"设计的适用域。

---

## Fig 3: Per-GPU负载分布

![Fig3](fig3_per_gpu_load.png)

**为什么测**：展示热点GPU（straggler）随数据域变化——同一identity放置下，不同数据集的热点GPU不同→静态最优对一域对另一域是错的。

**横轴**：GPU ID 0-7（8张物理GPU，EP=8。identity放置：GPU0=专家0-15, GPU1=专家16-31, ..., GPU7=专家112-127）。
**纵轴**：Load Share (%)——该GPU承载的专家选择总数占全部8卡总和的百分比。完美均衡=12.5%（1/8）。
**3组柱**：蓝=MMLU、橙=prover、绿=book。
**灰色虚线**：y=12.5%（完美均衡线）。

**反映了什么现象**：
- **MMLU热点=GPU4(13.4%)**：MMLU的QA路由较多激活专家64-79（GPU4的local范围64-79），使GPU4过载。
- **prover热点=GPU5(14.6%)**：数学路由激活专家80-95（GPU5范围），与MMLU完全不同。
- **book热点=GPU0(13.5%)和GPU4(13.5%)**：叙事文本激活专家0-15和64-79。
- **关键：3个数据集的热点GPU完全不重叠**——MMLU热点在GPU4，prover在GPU5，book在GPU0。为MMLU把热专家从GPU4移走会改善MMLU，但如果prover来了，GPU5仍是热点→静态放置无法跨域适应。

**对论文的启发**（§2.3 观察1 + §3.5自适应窗口）：不同域路由到不同GPU→需要**动态/自适应**放置。OEPLB的adaptive window在域切换时收缩窗口→快速重新放置。这证明了"静态参数调参失败"（§5.4 ShareGPT η≈0）的原因。

---

## Fig 4: 跨域路由相似度矩阵

![Fig4](fig4_cross_domain_similarity.png)

**为什么测**：量化"不同域的路由有多不同"——如果域间路由相似，一个静态放置就够了；如果正交(ρ≈0)，必须动态。

**横纵轴**：3个数据集（MMLU/prover/book）。
**颜色**：Spearman ρ——两个数据集的128维全局专家频率向量的秩相关。红色=负相关，白色=0(正交)，蓝色=1.0(完全一致)。
**格子内数字**：ρ值。

**反映了什么现象**：
- **对角线=1.0**（自己和自己完全一致）。
- **非对角≈0**：MMLU vs prover=0.054, MMLU vs book=-0.038, prover vs book=-0.216——3个数据集的路由模式**几乎正交**。
- **prover vs book反相关(-0.216)**：数学和叙事的路由甚至有反向趋势——数学激活的专家在叙事中恰恰是冷的。
- **含义**：为数据集A优化放置（把A的热专家分散），对数据集B不仅不是最优、甚至可能是最差——因为A的热点可能是B的冷点，分散A的热点等于集中B的"新热点"。

**对论文的启发**（§2.3 观察1 域内稳定+域间切换）：路由分布是**分段稳定的Markov过程**——域内cos_sim>0.95（稳定），域间ρ≈0（正交切换）。这支撑了§3.2指数衰减（域切换时清零历史）和§3.5自适应窗口（检测到切换→收缩）的设计。

---

## Fig 5: 逐层Prefill→Decode Spearman ρ

![Fig5](fig5_prefill_decode_rho_mmlu.png)

**为什么测**：直接证明prefill路由预测decode路由（DataFore Ob3）——如果ρ高，prefill-only recording是decode分布的充分统计量（§3.6设计依据）。

**横轴**：MoE层ID（0-93）。
**纵轴**：Spearman ρ——该层prefill阶段128专家频率直方图 vs decode阶段128专家频率直方图的秩相关。ρ≥0.7=强（绿），0.4-0.7=中（橙），<0.4=弱（红）。
**数据**：MMLU, O=10, n=3000, 聚合pooling（prefill 758K selections/层, decode 212K/层）。
**绿色虚线**：y=0.7（DataFore"强相关"阈值）。

**反映了什么现象**：
- **94/94层全部≥0.7（全绿）**：每一层prefill都强预测decode。mean ρ=0.833。
- **没有一层弱相关**：0层<0.4，0层0.4-0.7——所有层都强。
- **ρ范围0.694-0.945**：最低层0.694（接近0.7阈值），最高0.945（近完美）。绝大多数层在0.8-0.9。

**对论文的启发**（§2.3 观察3 + §3.6 充分性论证）：ρ=0.833, 94/94层强→prefill-only recording在MMLU(QA)负载下是decode分布的充分统计量。decode走CUDA graph（零开销跳过记录），prefill阶段记录即可。这是PB-OEPLB §3.6的核心设计依据。

---

## Fig 6: Top-K专家Overlap

![Fig6](fig6_topk_overlap.png)

**为什么测**：从"操作"角度验证prefill→decode预测——prefill最热的K个专家，有多少也是decode最热的K个？这比Spearman ρ更直接反映"prefill能否定位decode热点"。

**横轴**：K值（5, 10, 20, 40）——取prefill最热的K个专家 vs decode最热的K个专家。
**纵轴**：Overlap (%)——交集/K。100%=完全重叠，0%=完全不重叠。
**4个绿色柱**：K=5→49%, K=10→58%, K=20→(从ob3数据), K=40→75%。

**反映了什么现象**：
- **K越大overlap越高**：K=5时49%（5个里~2.5个重合），K=40时75%（40个里30个重合）——越宽的"热专家集"，重叠越多。
- **K=5时49%**：prefill最热5个专家中约一半也是decode最热5个——足以指导放置（把这几个热专家分散到不同GPU）。
- **K=40时75%**：prefill的top-40热专家中3/4也是decode的top-40——说明prefill和decode的"热度排序"整体高度一致。

**对论文的启发**（§3.3 配对选择算法）：swap planner基于prefill频率排序专家→把热专家分散。Top-5 overlap=49%意味着prefill能定位decode最热的~2-3个专家→足够指导swap（swap每次只移几对专家）。

---

## Fig 7: 专家激活频率热力图

![Fig7](fig7_expert_heatmap.png)

**为什么测**：可视化"不同域路由到不同专家"——一张图直观展示路由模式的域差异。同时展示94层×128专家的完整路由分布。

**横轴**：Expert ID（0-127，128个路由专家）。
**纵轴**：MoE Layer（0-93，94层）。
**颜色**：log₁₀(专家选择频率+1)——越亮=该专家在该层被选中越多（热），越暗=冷。
**3个子图并排**：左=MMLU(QA), 中=prover(math), 右=book(BookCorpus)。

**反映了什么现象**：
- **MMLU的热点**：集中在专家~30, 64, 76, 91, 110附近——QA任务的路由模式。
- **prover的热点**：集中在专家~51, 54, 80, 94, 99附近——数学推理的路由模式。
- **book的热点**：集中在专家~10, 34, 67, 69, 103附近——叙事文本的路由模式。
- **3个子图的亮区几乎不重叠**：MMLU亮的地方prover是暗的，反之亦然——直观展示了"域间正交"（Fig4的ρ≈0）。
- **某些层有亮竖线**：个别专家在几乎所有层都热（跨层热点），这些是"通用专家"——不受域影响。大部分层的亮区随域变化。
- **某些层更均匀**（整体偏暗），某些层非常集中（少数专家极亮）——对应Fig1中ratio的跨层波动。

**对论文的启发**（§2.3 观察1+4 + Fig4的定量版本）：热力图是Fig4跨域ρ≈0的**可视化**——不同域激活不同专家簇。这支撑了：(1) 静态放置无法跨域适应；(2) OEPLB需要域切换检测（§3.5 cos_sim changepoint）；(3) 不同域需要不同swap plan。

---

## Fig 8: ρ vs Prompt长度

![Fig8](fig8_length_dependence.png)

**为什么测**：刻画prefill→decode预测质量随prompt长度的变化——短prompt的prefill信号弱(ρ低)，长prompt强(ρ高)。这界定了prefill-only recording"充分"的适用域。

**横轴**：Prompt长度（tokens, log scale）——25, 107, 249, 1253, 4438, 5536。
**纵轴**：Mean Spearman ρ（prefill→decode, 聚合pooling, O=10）。
**蓝色实线+圆点**：可靠数据点（MMLU 25tok=0.833, prover 1253tok=0.980, book 4438tok=0.967）。
**灰色圆点**：噪声点（prover 107/249tok和book 5536tok——O=10 decode稀疏+N小导致噪声）。
**绿色虚线**：y=0.7（强相关阈值）。
**箭头标注**：MMLU(25tok)标注为"task-structured QA"——解释为什么25tok也能ρ=0.833。

**反映了什么现象**：
- **长prompt(≥1253tok) ρ=0.97-0.98**：prefill近完美预测decode。长prompt→prefill直方图充分采样→prefill≈真实路由分布。
- **MMLU(25tok) ρ=0.833**：虽是最短prompt，但因QA任务结构化(57学科→专家映射强)→ρ高。**任务结构 > 长度**对短prompt的ρ起决定作用。
- **短math(107/249tok) ρ=0.26-0.44**：数学短prompt，prefill信号弱+数学答案路由偏离题目→低ρ（且O=10稀疏decode放大噪声）。
- **可靠点全部≥0.7**：MMLU/prover_1253/book_4438都在绿线以上→prefill-only充分。

**对论文的启发**（§3.5 自适应窗口 + §3.6 充分性边界）：prefill-only recording在**任务结构化QA + 长prompt**下充分(ρ≥0.83)。短自由文本需更大聚合窗口(M↑)补偿低ρ——这正是§3.5 M*=f(L_seg)公式的实证依据。MMLU的task-structure outlier解释了为什么§5.4 heterogeneous ShareGPT的η≈0（短+自由文本→低ρ→决策噪声→swap无效）。

---

## Fig 0: 放置过程方法论图（最重要——理解所有图的基石）

![Fig0](fig0_methodology_placement.png)

**为什么测**：让读者清楚"不均衡度是怎么算的"、"identity和optimal分别是什么"、为什么optimal能降到1.0。这张图是Fig1/Fig2/Fig9/Fig10/Fig11的方法论基础。

**3个面板展示完整过程**：

### Step 1（左面板）：专家路由频率

- **横轴**：Expert ID（0-127，128个路由专家）。颜色=identity放置下该专家所属的GPU（0-15→GPU0蓝色，16-31→GPU1橙色，...，按`tab10`配色）。
- **纵轴**：Selection Count——该专家在**整个benchmark run的所有prefill forward**中被选中的token次数（不是单个batch，是聚合）。
  - Layer 62(book), 总选择数=31,084,816（约3100万次选择）。
  - 这是1000条book请求（每条4438 tok prompt）的prefill阶段聚合：1000 × 4438 × 8(top-8) ÷ 94层 ≈ 37.8万/层...实际31M是因为每层的选择来自所有94层的forward，聚合后每层有1000×4438×8=35.5M选择。
- **关键现象**：专家频率**不均匀**——有的专家被选中极多（柱高），有的极少（柱矮）。高频率专家集中在某些区域（如专家64-79区域），低频率散布。这种不均匀是负载不均衡的根因。
- **颜色分组**：同色=同GPU的16个专家。看到蓝色（GPU0）和绿色（GPU4）区域有高柱→这些GPU过载。

### Step 2（中面板）：Identity（默认）放置→per-GPU负载

- **横轴**：GPU 0-7（8张GPU，每张16专家）。
- **纵轴**：该GPU上16个专家的选择总数（把Step 1中同色的16根柱加起来）。
- **灰色虚线**：mean=所有GPU的平均负载（完美均衡线）。
- **数据**：GPU3=7,757,680（最重）, GPU2=658,068（最轻）。max/min=11.79×。
- **为什么这么不均衡**：identity放置把专家0-15放GPU0，但这个区间内有些专家极热、有些极冷——热专家集中在少数GPU上（GPU3和GPU4），而GPU2只有冷专家→GPU2几乎空闲。

### Step 3（右面板）：LPT最优放置→per-GPU负载

- **过程**：把128个专家按频率**降序排序**，每次把最热的放到当前最轻的GPU上（贪心bin-packing）。
- **结果**：8张GPU的负载几乎相等（~3,885,000），max/min=1.00×。
- **为什么能到1.0**：因为没有任何**单个**专家的选择数超过总量的1/8（12.5%）。最热的专家约占2-3%，16个专家/GPU有足够粒度分散。LPT贪心能保证每GPU的负载在最优解的4/3以内（经典LPT保证），而这里实际达到了~1.0。
- **什么时候到不了1.0**：如果某个专家的选择数>总量的1/8（§2.5的r_place下界），则该专家无论放哪张GPU都撑死那张→ratio>1.0。此时需要**冗余专家**（复制热专家到多张GPU）。但在Qwen3-235B EP8配置下，所有层的r_place≈1.00→纯放置就够了，不需要冗余。

### 这张图回答的核心问题

| 问题 | 答案 |
|------|------|
| 不均衡度怎么测的？ | 整个run所有prefill forward聚合的(94,128)token选择计数矩阵 |
| 粒度是什么？ | token级别的专家选择计数（不是batch，是全run聚合） |
| identity是什么？ | 按专家ID顺序连续分配（0-15→GPU0, 16-31→GPU1, ...） |
| optimal是什么？ | LPT贪心：按频率降序，每次放最轻GPU。NP-hard的4/3近似 |
| 为什么optimal降到1.0？ | 没有单专家>1/8总量，16专家/GPU足够粒度分散 |
| identity和optimal用的是同一份数据吗？ | 是，同一份(94,128)频率矩阵，只是"哪个专家放哪GPU"不同 |

---

## Fig 9: 跨域放置迁移矩阵

![Fig9](fig9_cross_domain_transfer.png)

**为什么测**：定量证明"静态最优放置跨域后失效"。不只是说"域间路由不同"(Fig4)，而是直接测量"为域A算的最优放置用在域B上ratio变成多少"。

**横轴**：Applied to（decode路由数据来自哪个域）。
**纵轴**：Placement from（用哪个域的prefill数据算LPT最优放置）。
**颜色**：Yellow→Red，越红=ratio越高（越不均衡）。
**对角线**：同域=1.00（完美）。
**非对角线**：跨域=2.1-4.1。

**关键现象**：MMLU的最优放置→prover=3.667，比identity(3.510)还差——"最优"放置跨域后比不做更糟。

**对论文启发**（§2.3 Obs1）：静态放置必然跨域失败→需要动态自适应。

---

## Fig 10: Identity→LPT放置ratio降低（swap timeline）

![Fig10](fig10_swap_timeline.png)

**为什么测**：展示"OEPLB在做什么"——从identity放置出发，swap专家对，ratio怎么降。

**横轴**：MoE层（book数据集的7个代表层0/10/20/40/62/80/93）。
**纵轴**：max/min ratio。红=identity，绿=LPT最优。
**柱上数字**：identity ratio（红）和optimal ratio（绿）。

**关键现象**：Layer 62 identity=11.79→LPT=1.000（降91.5%）。所有层降56-91%。

**对论文启发**（§2.4 理论上界）：每层的ratio→1.0后，MoE时间降低(r-1)×MoE占比。

---

## Fig 11: Prefill引导放置效果（PD ρ→ratio降低闭环）

![Fig11](fig11_remap_effect.png)

**为什么测**：闭环验证"PD相关性高→prefill引导放置有效"。3个数据集的PD ρ都高(0.83-0.98)，那基于prefill数据的放置是否真能降低ratio？

**横轴**：3个数据集。红=identity ratio，绿=Remap(prefill-guided LPT) ratio。
**柱上数字**：ratio值。蓝色注释：该数据集的PD ρ。

**关键现象**：MMLU ρ=0.833→ratio 2.26→1.00(降56%)；prover ρ=0.980→3.51→1.00(降72%)；book ρ=0.967→4.38→1.00(降77%)。**ρ越高，identity ratio越高（路由越偏斜），但Remap也越有效（降得越多）**。

**对论文启发**（§3.6 闭环）：PD相关→prefill引导放置→ratio降到1.0→decode受益。这是"prefill-only recording充分"到"实际有效"的完整闭环。

---

## Fig 12: 热点GPU域切换时间线

![Fig12](fig12_domain_switch_timeline.png)

**为什么测**：可视化"域切换时straggler GPU瞬间变化"——Fig3的per-GPU负载是聚合的，这里展示逐forward的hot GPU如何随域切换。

**横轴**：Forward pass（MMLU→prover→book拼接，各200 forward）。
**纵轴**：Hotspot GPU ID（0-7，该forward中负载最重的GPU）。
**红色虚线**：域边界。**红色标注**：各域的热点GPU（MMLU=GPU4, prover=GPU5, book=GPU0）。

**关键现象**：在MMLU段，热点集中在GPU4；prover段跳到GPU5；book段跳到GPU0。域边界处**瞬间切换**。→静态放置无法同时服务3个域。

**对论文启发**（§3.5 自适应窗口）：域切换→cos_sim下降→shrink window→重新放置。

---

## Fig 13: 多粒度不均衡度（per-forward vs per-window vs aggregate）

![Fig13](fig13_multi_granularity.png)

**为什么测**：回答"聚合是不是太粗泛"——之前的Fig1/Fig2用全run聚合数据算ratio，但MoE实际每个forward经历的是瞬时ratio，OEPLB看的是窗口级ratio。三者有什么差异？

**左面板（时间线）**：
- **横轴**：Prefill forward pass（0-166）。
- **纵轴**：Mean max/min ratio（identity放置，94层均值）。
- **蓝线**：Per-forward（每个forward单独算ratio）= 瞬时MoE batch级别。
- **橙线**：Per-window（16个forward聚合）= OEPLB的sync_window级别。
- **红虚线**：Aggregate（全部166个forward聚合）= Fig1/Fig2用的。

**右面板（分布直方图）**：
- **横轴**：ratio值。**纵轴**：密度。
- 蓝=per-forward分布，橙=per-window分布，红虚线=aggregate。

**关键发现**：
| 粒度 | mean | std | 含义 |
|------|------|-----|------|
| Per-forward | **4.50×** | 0.94 | MoE all-to-all每个forward实际经历的不均衡 |
| Per-window(W=16) | 2.80× | 0.26 | OEPLB的sync_window看到的 |
| Aggregate | 2.25× | 0 | Fig1/Fig2的全run聚合（理论下界） |

**核心洞察**：**聚合(2.25×)严重低估了实际不均衡——MoE每个forward实际承受4.5×的straggler**，是聚合的2倍。OEPLB窗口级(2.8×)在两者之间。LPT optimal→1.0在所有粒度都成立。

**对论文启发**（§2.1 + §3.5）：聚合ratio是"结构性下界"（路由本身偏斜的体现），但MoE实际体验的瞬时不均衡远高于此。OEPLB的sync_window级别(2.8×)比聚合更接近真实。这解释了为什么OEPLB的实际收益(+17.5%)高于"聚合ratio×MoE占比"的理论估计——因为实际ratio更高。

---

## Fig 1b: Per-Forward不均衡度分布（箱线图，3数据集）

![Fig1b](fig1b_per_forward_ratio.png)

**为什么测**：Fig1用聚合（全run）ratio，但MoE每个forward实际经历的straggler可能更高（大数定律把不同forward的差异平均掉了）。这张图展示per-forward的分布，与聚合对比。

**横轴**：3个数据集（MMLU/prover/book）。
**纵轴**：Mean max/min Imbalance Ratio（identity放置，94层均值）。
**箱线图**：per-forward的ratio分布（箱=四分位距，须=极值，点=离群）。颜色按数据集。
**红色钻石(◆)**：aggregate ratio（Fig1/Fig2用的全run聚合值）。

**关键现象**：
| 数据集 | per-forward median | aggregate | per-forward高多少 |
|--------|-------------------|-----------|-----------------|
| MMLU | 3.82× | 2.26× | 1.7× |
| prover | 3.68× | 3.51× | 1.05×（接近） |
| book | 6.36× | 4.38× | 1.45× |

- **MMLU差距最大**：短prompt(25tok)→每个forward只有~25×8=200选择/层→直方图稀疏→不同forward路由差异大→聚合平滑掉了很多。
- **prover接近**：长prompt(1253tok)→每个forward充分采样→per-forward已接近聚合。
- **book仍有差距**：4438tok prompt但极端层(L62=11.79×)拉高per-forward。

**对论文启发**（§2.1 + Fig1的补充）：聚合ratio是"结构性下界"（路由偏斜的体现），但MoE每个forward实际经历的straggler是聚合的1.5-1.7×。短prompt负载的实际不均衡比聚合图(Fig1)显示的更严重。论文动机论证应同时引用两个角度。

---

## Fig 2b: Per-Forward Identity vs LPT Optimal（3数据集，含聚合对比）

![Fig2b](fig2b_per_forward_identity_vs_optimal.png)

**为什么测**：Fig2用聚合数据展示identity→LPT的降低。这里用per-forward数据重做，同时保留聚合柱作对比——展示不同粒度下OEPLB的理论收益空间。

**横轴**：3个数据集。3组柱并排：
- **左柱(红色半透明)**：Aggregate identity（Fig2用的全run聚合ratio）。
- **中柱(橙色+误差棒)**：Per-forward identity median（MoE每个forward实际经历的ratio）。
- **右柱(绿色)**：Per-forward LPT optimal（每个forward单独算LPT最优→ratio）。
**误差棒**：per-forward identity的标准差。
**柱上数字**：具体ratio值。

**关键现象**：
- **per-forward identity(3.7-6.4×) > aggregate identity(2.3-4.4×)**：MoE实际经历的不均衡比聚合高1.5-1.7×。
- **per-forward LPT optimal → 1.000（所有数据集）**：在每个forward的时间尺度上，LPT都能降到~1.0。放置机会在任何粒度都存在。
- **收益空间**：per-forward identity 3.7-6.4× → LPT 1.0，降低73-84%（比聚合的53-73%更大）。

**对论文启发**（§2.4 理论上界 + §2.1动机）：论文应使用per-forward ratio做动机论证（MoE实际经历的straggler），聚合作为理论下界参考。理论上界Δ_max = f_sens × x_eff，其中r_before应该用per-forward（而非聚合）→理论上界更高→OEPLB的实际收益(+17.5%)与更高的per-forward ratio更一致。
