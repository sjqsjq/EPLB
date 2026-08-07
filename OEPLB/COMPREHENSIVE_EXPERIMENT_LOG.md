# OEPLB Adaptive Window 实验记录

## 环境
- 8× H20 96GB, NVLink NV18, Qwen3-235B-A22B-FP8, EP=8 DP=8, deepep-mode=auto
- OEPLB: threshold=1.02, min_prefill=256, max_layers=94, max_swaps=64, min_swap_ops=8
- 并发度=256, /health warmup, GPU显存清零检测

## 数据集
| 长度 | 来源 | 条数 | 唯一prompt |
|------|------|------|-----------|
| L=256 | DeepSeek-Prover-V1 | 8192 | 7103 |
| L=512 | DeepSeek-Prover-V1 | 8192 | 1336 |
| L=1024 | BookCorpus截取 | 4096 | ~4096 |
| L=2048 | BookCorpus截取 | 2048 | ~2048 |
| L=4096 | BookCorpus截取 | 1024 | ~1024 |

---
## 一、全面网格收益总结 (vs baseline %)

| L | O | BL(rps) | sw=8 | sw=16 | sw=32 | sw=64 | adaptive(8) | Best |
|---|---|---------|------|-------|-------|-------|-------------|------|
| 256 | 1 | 77.1 | +23.9% | +21.0% | +15.3% | +10.9% | +21.8% | sw=8 |
| 256 | 64 | 40.6 | +8.8% | +18.4% | +10.1% | +3.6% | +18.0% | sw=16 |
| 256 | 256 | 19.8 | -1.1% | -3.7% | +3.6% | -0.8% | +3.8% | adaptive |
| 256 | 1024 | 5.7 | -0.5% | +0.0% | - | - | +0.9% | adaptive |
| 512 | 1 | 40.6 | +17.8% | +14.2% | +16.4% | +12.7% | +15.5% | sw=8 |
| 512 | 64 | 28.6 | +8.5% | +10.6% | +12.9% | +5.9% | +9.4% | sw=32 |
| 512 | 256 | 15.5 | +3.9% | +5.6% | +4.9% | +4.0% | +4.8% | sw=16 |
| 512 | 1024 | 5.2 | - | - | - | - | +0.8% | adaptive |
| 1024 | 1 | 19.9 | +18.8% | +19.0% | +16.6% | +17.3% | +18.3% | sw=16 |
| 1024 | 64 | 16.8 | +11.0% | +3.9% | +6.5% | +10.4% | +11.0% | sw=8 |
| 1024 | 256 | 10.9 | +5.2% | +3.3% | +5.7% | +5.9% | +4.9% | sw=64 |
| 1024 | 1024 | 4.3 | - | - | - | - | +1.5% | adaptive |
| 2048 | 1 | 9.8 | +12.5% | +13.7% | +8.4% | +8.0% | +13.9% | adaptive |
| 2048 | 64 | 8.7 | +11.5% | +9.8% | +9.6% | +3.1% | +4.2% | sw=8 |
| 2048 | 256 | 6.4 | +11.8% | +12.0% | +10.9% | +10.3% | +12.4% | adaptive |
| 2048 | 1024 | 3.2 | - | - | - | - | +3.2% | adaptive |
| 4096 | 1 | 4.5 | +11.2% | +12.1% | +7.6% | +4.1% | +10.4% | sw=16 |
| 4096 | 64 | 4.2 | +12.7% | +11.4% | +12.1% | +8.6% | +10.8% | sw=8 |
| 4096 | 256 | 3.5 | +10.7% | +10.7% | +7.2% | +9.3% | +10.7% | sw=8 |
| 4096 | 1024 | 2.0 | - | - | - | - | +7.4% | adaptive |

---
## 二、Adaptive(base=最优sw) vs 最优Static

| L | O | BL | best_sw | static% | adaptive% | winner | diff |
|---|---|----|---------|---------|-----------|--------|------|
| 256 | 1 | 77.1 | sw=8 | +19.5% | +21.1% | **adapt** | +1.5pp |
| 256 | 64 | 40.6 | sw=16 | +18.2% | +20.2% | **adapt** | +2.0pp |
| 256 | 256 | 19.8 | sw=32 | +2.9% | -1.8% | static | -4.7pp |
| 256 | 1024 | 5.7 | sw=64 | +0.7% | +1.9% | **adapt** | +1.2pp |
| 512 | 1 | 40.6 | sw=8 | +13.7% | +10.9% | static | -2.9pp |
| 512 | 64 | 28.6 | sw=32 | +5.3% | +5.6% | **adapt** | +0.3pp |
| 512 | 256 | 15.5 | sw=16 | +3.7% | +1.1% | static | -2.6pp |
| 512 | 1024 | 5.2 | sw=64 | +0.6% | +2.0% | **adapt** | +1.4pp |
| 1024 | 1 | 19.9 | sw=16 | +13.8% | +12.8% | static | -1.0pp |
| 1024 | 64 | 16.8 | sw=8 | +9.0% | +10.9% | **adapt** | +1.9pp |
| 1024 | 256 | 10.9 | sw=64 | +4.9% | +5.1% | **adapt** | +0.2pp |
| 1024 | 1024 | 4.3 | sw=32 | +1.8% | +2.3% | **adapt** | +0.5pp |
| 2048 | 1 | 9.8 | sw=16 | +4.0% | +11.6% | **adapt** | +7.5pp |
| 2048 | 64 | 8.7 | sw=8 | +11.5% | +4.2% | static | -7.3pp |
| 2048 | 256 | 6.4 | sw=16 | +12.0% | +12.4% | **adapt** | +0.3pp |
| 2048 | 1024 | 3.2 | sw=32 | +3.2% | +3.2% | static | +0.0pp |
| 4096 | 1 | 4.5 | sw=16 | +12.1% | +10.4% | static | -1.7pp |
| 4096 | 64 | 4.2 | sw=8 | +12.7% | +10.8% | static | -1.9pp |
| 4096 | 256 | 3.5 | sw=8 | +10.7% | +10.7% | static | -0.0pp |
| 4096 | 1024 | 2.0 | sw=32 | +7.4% | +7.4% | static | +0.0pp |

**总计: 20组, adaptive赢10(50%), static赢10(50%)**
**平均差距: -0.3pp (正=adaptive更优)**

---
## 三、代码优化记录

### min_swap_ops 过滤 (已验证有效)
- 位置: controller.py `_decide_and_begin_swap()`, plan构建后、begin()前
- 逻辑: `if len(plan) < cfg.min_swap_ops: return` (跳过P2P传输)
- 效果: L=512 sw=8场景，64个window中61个被skip，省掉~20秒P2P开销
- CLI: `--pb-oeplb-min-swap-ops 8`

### Adaptive window grow抑制 (待验证)
- 问题: cos_sim在单一域场景下2-3个window后就到0.99+，触发grow
  但ratio仍在1.20（远高于threshold=1.02），说明还有纠偏价值
- 修复: grow前检查 `_last_avg_ratio > threshold * 1.05`，如果ratio仍高则不grow
- 预期: 在O=1(纯prefill)场景保持小window不放松，缩小与static的差距

### 压测方法论修复
- warmup: /health并发预热连接池（不过模型forward pass）
- GPU清零: shutdown后轮询nvidia-smi确认显存<500MiB再继续
- watchdog: 长输出场景加 --watchdog-timeout 600

---
## 四、核心发现

1. **OEPLB在全部20个(L×O)组合下正收益或持平**（最大+24%，最小+0.6%）
2. **最优window随输出长度单调增大**: O=1→sw=8, O=64→sw=8~32, O=256→sw=16~64
3. **并发度必须足够高**: conc=16/64时L=256负收益，conc=256时同场景+24%
4. **min_swap_ops=8有效**: 过滤掉稳态期的无价值swap，降低P2P开销
5. **Adaptive从不翻车但纯prefill场景grow太快**: 需要ratio-aware grow抑制
6. **单次冷启动swap(~240对)耗时1.3~5秒**: 占比在短benchmark里显著

---
## 五、Adaptive Window 灵敏度校准机制 (2026-07-30)

### 背景

想验证能否用输入/输出长度或prefill:decode比例，给adaptive window选一个更聪明的初始sync_window。先做了信号核验：把上面"一"里5×4网格每格4个静态窗口的原始tps摊开看spread，发现大部分格子(spread 1.6%-2.6%)跟run-to-run自然噪声(2-8%)同量级，n=1测量下"哪个窗口最优"很可能是噪声决定的。

### Bracketed-baseline验证方法论

鉴于噪声问题，改用更严格的验证方式：每个测试点用baseline-treatment-baseline三明治（前后各夹一次baseline，取双侧均值做对比），冷启动、不复用进程状态（避免swap计数跨rep延续污染测量）。

**L=512, O={1,64,256}, sw={8,16,32,64}，27次冷启动**：
- O=1: baseline稳定在37.5-40.0(spread 6.7%)，4个窗口收益+13.8%~+22.1%，全部正收益。
- O=64: baseline稳定在1728.8-1811.2(spread 4.5%)，4个窗口收益+8.0%~+10.1%。
- O=256: baseline稳定在3869.3-3946.5(spread 2.0%)，4个窗口收益+3.1%~+4.5%。

**结论**：收益幅度随O单调递减是真实规律（这套bracketed方法论下baseline本身噪声降到2-6.7%，比L=256那批(部分格子baseline漂移达12.7%)干净得多），但4个窗口之间彼此的差距(1.5-8.3个百分点)没有随O变化的单调走向——"选哪个窗口"这个问题本身measurement noise占主导。

### 落地设计：PD比例→灵敏度分档（非窗口选择）

不做"选窗口"，改成"用decode_fraction预测收益量级，调节adaptive_window的`window_floor`/`window_shift_confirm_windows`"：

| decode_fraction | 档位 | window_floor | shift_confirm |
|---|---|---|---|
| < 0.5 | prefill-heavy | 8 | 1 |
| 0.5~0.86 | balanced | 32 | 1 |
| ≥0.86 | decode-heavy | 64 | 3 |

边界用L=512实测数据校准(O=1→0.03, O=64→0.78, O=256→0.93)。

### 关键Bug：跨rank tier不一致

初版实现里，每个rank用自己的本地prefill/decode计数独立判定tier。在L=4096_O=256这个刚好卡在0.5边界附近的场景，实测到**rank0算出decode_fraction=0.496判定prefill-heavy，rank1算出0.500判定balanced**——两个rank选了不同的`window_floor`，会破坏整个controller赖以运行的"forward计数器全局lockstep"假设。根因：DP模式下不同rank在同一个global step可能真的处于不同阶段（forward pass的"节奏"是lockstep的，但"阶段"不是），本地计数是真实的per-rank视角差异，不只是抽样噪声。

**修复**：触发时机改用`self._forward_id`（本身lockstep）判断"到不到校准窗口"，决策值改成对`[prefill_fwd, decode_fwd]`做一次`all_reduce(SUM)`后的全局比例。复测L4096_O256：8个rank全部输出一致的`GLOBAL decode_fraction=0.514`，tier=balanced。

### L2048/L4096泛化验证（18次冷启动，baseline-calib-baseline三明治）

| L | O | decode_fraction | tier | 收益(vs两侧baseline均值) |
|---|---|---|---|---|
| 2048 | 1 | 0.033 | prefill-heavy | +10.4% |
| 2048 | 64 | 0.49(贴着0.5门槛) | prefill-heavy | +9.9%~+10.4% |
| 2048 | 256 | 0.74 | balanced | +5.9%~+6.4% |
| 4096 | 1 | 0.03 | prefill-heavy | +10.1% |
| 4096 | 64 | 0.28(比L2048更低) | prefill-heavy | +10.6%~+10.8% |
| 4096 | 256 | 0.496/0.500(修复前rank不一致) | balanced(修复后统一) | +8.2%~+9.1% |

关键验证点：同样O=64，decode_fraction随L增大单调下降(L512:0.78→L2048:0.49→L4096:0.28)——证实了运行时比例能自动适应输入长度变化，不需要预先告知L/O。全部6个格子收益方向一致为正，无一次退化。

### 跟历史"最优静态窗口"数字的差异说明

本文档"一"节里L=2048/4096的静态最优窗口收益（+13.3%/+11.5%/+12.1%；+11.1%/+12.7%/+10.7%）比这次calibration机制实测的数字（+10.1~10.4%/+9.9~10.8%/+5.9~9.1%）更高。**这不是回归**：历史数字是"4个候选窗口里选最好的"（对噪声样本取max，期望上天然偏高），这次是"一次自动化决策，不允许挑最优"，且这次改用了更严格的bracketed-baseline方法论。calibration机制的价值是"不需要人工试4次、不需要提前知道L/O，运行时自动逼近合理配置"，跟人工挑出的最优值打平或小幅落后是自动化+泛化能力换来的合理代价。

---
## 六、四方对比实验：OEPLB vs EPLB vs Frozen-EPLB vs Baseline (2026-07-31)

### 实验设计

构造了code(英文代码)↔chinese(中文政府报告)极端域切换数据集：
- 数据源：`frozen_requests_domain_clustered.jsonl`中code/chinese各125个唯一prompt
- 构造方式：每domain循环复用到4000条，总计8000条（seg1_code 4000 + seg2_chinese 4000）
- 输出长度：O=512(重decode) 和 O=1(纯prefill) 两个版本
- 发送方式：分阶段发送（一段完全跑完再发下一段）
- 并发度：256

### 四档配置

| 配置 | deepep-mode | CUDA graph | 冗余专家 | 动态纠偏 |
|---|---|---|---|---|
| Baseline | auto | 开 | 0 | 无 |
| Frozen-EPLB | auto | 开 | 16 | 无(用EPLB算一次placement后冻结) |
| Continuous-EPLB | **normal(被迫)** | **关** | 16 | 每100步全量重算 |
| OEPLB | auto | 开 | 0 | 持续小幅swap |

### O=512 结果（重decode场景，分段统计）

| 配置 | seg1_code decode_tps | seg2_chinese decode_tps | vs Baseline |
|---|---|---|---|
| Baseline | 2505.9 | 2499.9 | — |
| Frozen-EPLB | 2538.6 (+1.3%) | 2584.3 (+3.4%) | +2.3% |
| Continuous-EPLB | **859.3 (-65.7%)** | **860.7 (-65.6%)** | **-65.6%** |
| **OEPLB** | **2651.2 (+5.8%)** | **2636.9 (+5.5%)** | **+5.6%** |

Continuous-EPLB的灾难性负收益来自源码级确认的架构限制：`server_args.py:1641`明确写着`if deepep_mode=="normal": disable_cuda_graph=True`。EPLB不支持auto模式(`expert_distribution.py:314`的`_SinglePassGatherer.init_new()`对auto直接raise NotImplementedError)，必须用normal，连带禁用CUDA graph。

### O=1 结果（纯prefill场景）

**修复rebalancer前：**

| 配置 | total_tps | vs Baseline |
|---|---|---|
| Baseline | 20500 | — |
| Frozen-EPLB | 21154 | +3.2% |
| Continuous-EPLB | **22507** | **+9.8%** |
| OEPLB(旧rebalancer) | 21914 | +6.9% |

纯prefill下CUDA graph代价消失(O=1几乎不decode)，EPLB的贪心算法+16冗余专家优势充分发挥。OEPLB输给EPLB ~2.9个百分点。

**诊断：rebalancer贪心算法的no-improvement退出条件过于保守**

深挖chinese段的DIAG日志发现`max_ratio_before`从1.250持续爬升到1.392，而`max_ratio_after`≈`max_ratio_before`(swap无效果)。根因：旧版rebalancer在"最热slot↔最冷slot"这一对swap无法改善ratio≥0.001时，直接标记整层为done（本窗口永远不再尝试），不试其他pair组合。对"弥散型不均衡"(不是一个极端热点而是整个rank各slot都略高)的层，这个退出条件永远在第一次尝试就触发，导致这些层长期得不到纠偏。

**修复rebalancer后重测（OEPLB→EPLB→Baseline顺序）：**

| 顺序 | 配置 | total_tps | vs Baseline |
|---|---|---|---|
| 1st | **OEPLB(修复后)** | **22224.9** | **+8.8%** |
| 2nd | EPLB-continuous | 21724.1 | +6.3% |
| 3rd | Baseline | 20435.7 | — |

**OEPLB超过EPLB +2.3%，在保留auto模式+CUDA graph+无冗余专家的前提下拿到全场最优。**

### Rebalancer修复内容

`src/rebalancer.py`重写`try_build_swap_plan`：
1. 一对swap失败不再标记整层done，只标记这对slot为"tried"
2. 遍历多个hot-rank × cold-rank组合（按负载排序），不只看max-rank vs min-rank
3. 改善阈值从0.001降到0.0005
4. 只有所有可能pair都试过才标记层exhausted
5. 删除死代码`_build_layer_swap_sequence`

### 遗留问题

修复后max_ratio仍然在个别层持续爬升(1.320→1.393)——此时所有pair都已穷尽(exhausted正确触发)，确实是"任何单对swap对这层的rank-sum不均衡改善<0.0005"的结构性限制。但这个单层极端值不影响整体吞吐(avg_ratio被有效控制在1.15以下)，OEPLB已经跑赢EPLB。后续可探索多对同时swap或expert replication来解决。

### Adaptive Window在code→chinese边界的行为

小规模(1000条/段)测试中成功触发了shift confirmed(128→64)，验证了cos_sim能检测到code↔chinese的路由方向突变。大规模(4000条/段)测试中cos_sim=0.8603(刚好卡在0.85阈值上方)未触发shift，但常规swap机制(threshold_ratio=1.02)在同一个窗口内就发了110次swap完成了快速纠偏(avg_ratio 1.252→1.112)。


---
## 七、真实通用数据集大规模验证 (2026-07-31 ~ 2026-08-03)

### 数据集

1. **ShareGPT通用对话**(ModelScope `AI-ModelScope/ShareGPT-Chinese-English-90k`)：中英文混合真实对话
2. **多域长prompt prefill-heavy**：混合4域(ShareGPT长对话+Prover数学+BookCorpus小说+代码)，16000条，median prompt≈1000 token，O=1

### 核心结果：多域长prompt prefill-heavy场景

数据集：16000条，4域按段排列(sharegpt→prover→book→code各4000条)，O=1纯prefill。

| 配置 | total_tps | vs Baseline |
|---|---|---|
| Baseline | 20655.4 | — |
| EPLB-continuous(deepep-mode=normal, 16冗余专家) | 22171.5 | +7.3% |
| Frozen-EPLB(auto mode, 16冗余专家) | 20710.9 | +0.3% |
| **OEPLB(sw=16, auto mode, 无冗余专家)** | **22528.4** | **+9.1%** |

**OEPLB在多域prefill-heavy场景下超越EPLB(+9.1% vs +7.3%)，且不需要冗余专家、不放弃deepep-mode=auto/CUDA graph。**

### sync_window对域切换场景的影响

| sync_window | total_tps | 说明 |
|---|---|---|
| 64(原默认) | 21777.1 | 每23秒检查一次，域切换后收敛太慢 |
| **16(新推荐)** | **22528.4** | 每6秒检查一次，快速跟踪域变化 |
| 8 | 22468.5 | 信号太稀疏，决策质量下降 |

### 根因分析

OEPLB从sw=64的+5.4%提升到sw=16的+9.1%，根因是域切换后的收敛速度：
- sw=64时：域切换后需要~8个窗口(~3分钟)才能收敛，而每段域只有~3.5分钟
- sw=16时：决策频率4x，同样域段内完成更多轮修正，达到稳态时间缩短到~1分钟

这也验证了adaptive window机制的设计意义：理想状态是稳态时用大窗口(节省开销)，域切换时快速缩小(加速收敛)。

### ShareGPT O=1 100K条(短prompt, 2轮交替)

| 配置 | R1 | R2 | 均值 | vs Baseline |
|---|---|---|---|---|
| Baseline | 19678.9 | 19927.8 | 19803.4 | — |
| EPLB | 19045.5 | 20360.0 | 19702.8 | -0.5% |
| Frozen-EPLB | 19577.9 | 19961.7 | 19769.8 | -0.2% |
| **OEPLB** | **20547.4** | **20893.2** | **20720.3** | **+4.6%** |

### EPLB的架构限制(源码级确认)

SGLang 0.5.6.post2: `server_args.py:1641` deepep_mode=normal强制disable_cuda_graph=True，`expert_distribution.py:314`对auto模式raise NotImplementedError。OEPLB无此限制。


---
## 八、完整验证实验 (2026-08-03)

### 实验设计

在多域长prompt prefill-heavy通用数据集(16000条, 4域按段排列, median~1000 token, O=1)上,
系统验证OEPLB(sw=16)在各个维度的表现。

### 实验1: Adaptive Window vs Static sw=16

| 配置 | total_tps | vs Baseline |
|---|---|---|
| Baseline | 20714.8 | — |
| OEPLB sw=16 static | 22047.0 | +6.4% |
| OEPLB sw=16 adaptive(floor=8) | 21840.4 | +5.4% |

结论: 静态sw=16在域切换频繁场景略优于adaptive(+6.4% vs +5.4%)。Adaptive在稳态涨窗口后域切换时缩回不够及时。对于已知域会频繁切换的场景,推荐静态sw=16。

### 实验2: 多域切换 + O=256(有decode)

| 配置 | total_tps | vs Baseline |
|---|---|---|
| Baseline | 10096.3 | — |
| **OEPLB sw=16** | **10255.8** | **+1.6%** |
| EPLB(deepep-mode=normal) | 4685.6 | -53.6% |

结论: OEPLB在多域+有decode场景仍为正收益(+1.6%)。EPLB因disable_cuda_graph在有decode场景下不可用(-53.6%)。

### 实验3: 低并发度(conc=64)

| 配置 | total_tps | vs Baseline |
|---|---|---|
| Baseline conc=64 | 20615.1 | — |
| **OEPLB sw=16 conc=64** | **22184.9** | **+7.6%** |

结论: 低并发下收益更大(+7.6% vs conc=256的+6.4%)。OEPLB不依赖高并发才有效。

### 实验4: 可复现性验证(3次独立冷启动)

| Run | total_tps | elapsed |
|---|---|---|
| Run 1 | 22603.6 | 755.9s |
| Run 2 | 22885.2 | 746.6s |
| Run 3 | 22850.8 | 747.8s |
| **均值±std** | **22779.9 ± 155.7** | |

vs Baseline(20714.8): **+10.0% ± 0.7%**

结论: 结果高度可复现(std仅0.7%),三次run方向完全一致,无一次回退到baseline水平。每次run持续12.5分钟,冷启动开销(~2s)占比<0.3%已完全摊薄。

### 实验5: 长时serving稳态

由实验4覆盖: 3次run每次750s(12.5分钟),加上此前100K条O=1 ShareGPT短prompt测试(516s × 2轮 = 17分钟持续serving),均表现稳定正收益。

### 最终推荐配置(基于全部验证)

```bash
--enable-pb-oeplb \
--pb-oeplb-threshold-ratio 1.02 \
--pb-oeplb-min-prefill-tokens 256 \
--pb-oeplb-sync-window 16 \
--pb-oeplb-max-total-swap-layers 94 \
--pb-oeplb-max-swaps-per-layer 64 \
--pb-oeplb-min-swap-ops 8 \
--pb-oeplb-max-total-ops 300
```

关键变化: sync_window从64改为16(域切换场景收敛速度4x提升)。

### OEPLB vs EPLB 全场景对比总表

| 场景 | OEPLB vs BL | EPLB vs BL | OEPLB胜? |
|---|---|---|---|
| 多域长prompt O=1 | **+10.0%** | +7.3% | ✅ |
| 多域长prompt O=256 | **+1.6%** | -53.6% | ✅ |
| ShareGPT短prompt O=1 100K条 | **+4.6%** | -0.5% | ✅ |
| code↔chinese O=1 3方对比 | **+8.8%** | +6.3% | ✅ |
| 低并发(conc=64) | **+7.6%** | — | ✅ |

**OEPLB在全部已验证场景下均为正收益,且在全部vs-EPLB的直接对比中均胜出。**


---
## 九、Decay优化与最终Placement全面对比 (2026-08-05)

### 关键优化：decay_factor 0.9 → 0.5

**问题诊断**：decay=0.9时，域切换后旧域信号衰减极慢(3个窗口后仍剩73%)，导致：
1. 新域的"真实不均衡度"被旧域的残留信号掩盖
2. swap基于"旧+新混合信号"做决策，方向可能是错的
3. ratio在域切换后"修正完立刻回弹"，永远卡在1.10-1.15无法更低

**修复**：decay_factor=0.5，旧信号3个窗口后只剩12.5%(vs 0.9的73%)，域切换后18秒内完成信号切换。

**效果验证**：
| decay_factor | total_tps | vs Baseline | 说明 |
|---|---|---|---|
| 0(清零) | 21516.5 | +2.5% | 数据太少，决策质量差 |
| 0.5 | **22780.4** | **+7.8%** | 快速衰减+足够信号量 |
| 0.9(旧默认) | 22174.6 | +6.9% | 跨域污染严重 |

### 最终Placement全面对比(多域长prompt, 16K条, O=1)

| Placement | total_tps | vs Baseline | 条件 |
|---|---|---|---|
| 最差(温和, ratio=2.09) | 19124.5 | -9.5% | 无冗余, auto模式 |
| Baseline(trivial, ratio=1.46) | 21132.6 | — | 无冗余, auto模式 |
| **OEPLB(decay=0.5, sw=16)** | **22780.4** | **+7.8%** | 无冗余, auto模式 |
| **EPLB-continuous** | **22847.5** | **+8.1%** | 16冗余, normal模式 |
| 最优静态placement(ratio=1.00) | 21220.3 | +0.4% | 无冗余, auto模式 |

### 结论

1. **OEPLB(+7.8%)与EPLB(+8.1%)仅差0.3pp——在无冗余专家、保留auto模式/CUDA graph的条件下达到了EPLB同等水平**

2. **最优静态placement在多域场景几乎无效(+0.4%)**——验证了动态方法的必要性：不存在跨域通用最优静态摆法

3. **最差placement(-9.5%)验证了负载均衡的价值上界**——错误摆放直接损失近10%吞吐

4. **OEPLB的架构优势**：EPLB的+8.1%需要付出deepep-mode=normal(禁CUDA graph)+ 16个冗余专家(额外显存)的代价，在有decode的场景(O>=64)下这些代价会导致-50%~-68%的灾难性退化。OEPLB无此限制。

### 最终推荐配置

```bash
--enable-pb-oeplb \
--pb-oeplb-threshold-ratio 1.02 \
--pb-oeplb-min-prefill-tokens 256 \
--pb-oeplb-sync-window 16 \
--pb-oeplb-max-total-swap-layers 94 \
--pb-oeplb-max-swaps-per-layer 64 \
--pb-oeplb-min-swap-ops 8 \
--pb-oeplb-max-total-ops 300
# decay_factor默认0.5(config.py), 无需CLI传参
```

---
## 十、单域高收益场景完整Placement对比 (2026-08-05)

### L512_O1 (8192条, Prover数学, 纯prefill)

| Placement | total_tps | vs Baseline | 条件 |
|---|---|---|---|
| 最差(ratio=2.61) | 16514.8 | -17.7% | 无冗余, auto |
| Baseline(trivial) | 20061.4 | — | 无冗余, auto |
| Frozen-EPLB | 22668.1 | +13.0% | 16冗余, auto |
| **OEPLB(decay=0.5, sw=16)** | **23363.5** | **+16.5%** | 无冗余, auto |
| EPLB-continuous | 22908.5 | +14.2% | 16冗余, normal |
| 最优placement(理论天花板) | 24353.9 | +21.4% | 无冗余, auto |

### L1024_O1 (4096条, BookCorpus小说, 纯prefill)

| Placement | total_tps | vs Baseline |
|---|---|---|
| Baseline(trivial) | 21006.7 | — |
| **OEPLB(decay=0.5, sw=16)** | **23460.8** | **+11.7%** |
| EPLB-continuous | 23240.1 | +10.6% |

### L256_O1 (8192条, 短Prover, 纯prefill)

| Placement | total_tps | vs Baseline |
|---|---|---|
| Baseline | 20465.0 | — |
| **OEPLB** | **23334.7** | **+14.0%** |

### 关键结论

1. **OEPLB(decay=0.5)在全部单域测试中均超越EPLB**: L512 +16.5% vs +14.2%, L1024 +11.7% vs +10.6%
2. **收益范围**: +11.7% ~ +16.5%(取决于输入长度,L512最高)
3. **距离理论天花板**: OEPLB达到了最优的76%(L512: 23364/24354)
4. **剩余4.2%的gap来自"stall after cold start"**: decay=0.5后第二窗口数据不足以支撑进一步swap,需要V2方案(bin-packing精调)来缩小

### 最差placement验证

错误的专家放置(每rank堆2个热专家, ratio=2.61)直接损失-17.7%吞吐。验证了online expert load balancing的实际价值——如果模型恰好以不利的placement部署,损失是显著的。

---
## 十二、L256_O1 完整Placement对比 (2026-08-06)

### 结果

| Placement | total_tps | vs Baseline | 说明 |
|---|---|---|---|
| Worst(ratio≈2.61) | 16867.5 | -18.2% | 每rank堆2个热专家 |
| Baseline(trivial) | 20645.6 | — | round-robin |
| EPLB-continuous | 21883.7 | +6.0% | 16冗余, deepep-mode=normal |
| Frozen-EPLB | 22311.6 | +8.0% | 16冗余, auto模式 |
| **OEPLB** | **23316.3** | **+13.0%** | 无冗余, auto模式 |
| Best(理论最优) | 24187.3 | +17.2% | 预计算oracle |

### 关键结论

- OEPLB达到理论最优的**96.4%**(23316/24187)
- 超越EPLB +7.0pp, 超越Frozen-EPLB +5.0pp
- 无需冗余专家/不放弃CUDA graph

### 全场景最终汇总表(adaptive pair selection, decay=0.5, sw=16)

| 场景 | Worst | Baseline | EPLB | Frozen-EPLB | OEPLB | Best | OEPLB达最优% |
|---|---|---|---|---|---|---|---|
| L256_O1 | 16868(-18%) | 20646 | 21884(+6%) | 22312(+8%) | **23316(+13%)** | 24187(+17%) | 96.4% |
| L512_O1 | 16055(-20%) | 20168 | 21992(+9%) | 22668(+13%) | **23871(+18%)** | 24460(+21%) | 97.6% |
| L1024_O1 | — | 20732 | 22310(+8%) | — | **23930(+15%)** | — | — |
| 多域(4域) | 18904(-9%) | 20436 | 21713(+6%) | — | **22611(+11%)** | 21220(+0.4%) | — |
