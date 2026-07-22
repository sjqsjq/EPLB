# OEPLB 8卡 H20 实验报告

## 一、环境

- **硬件**: 8× NVIDIA H20 (96GB/卡), NV18 全互连
- **软件**: SGLang 0.5.6.post2, sgl-kernel 0.3.19, DeepEP v1.2.1 (NVLink patches), PyTorch 2.9.1+cu128
- **模型**: Qwen3-30B-A3B-FP8 (48层, 128专家/层, top-8路由, EP=8→每卡16专家)
- **配置**: TP=8, DP=8, EP=8, deepep-mode=auto, deep_gemm, disable-radix-cache

## 二、发现并修复的 Bug

### Bug 1: p2l 双重索引错误 (rebalancer.py)
**根因**: `topk_ids` 在 SGLang 的 `fused_topk()` 内部已经被 `topk_ids_logical_to_physical()` 转成了 physical slot ID（因为 `ep_dispatch_algorithm="static"` 在 `--enable-pb-oeplb` 时被自动设置），但 rebalancer 的 `_build_layer_swap_sequence` 又用 `lc[p2l[i]]` 做了一次多余的 logical 转换，等于看的是错误的数据。
**影响**: swap 决策基于乱序数据，"看上去在收敛"实际在随机交换。
**修复**: 直接用 `lc[i]`（physical slot 索引），仅在需要 logical expert identity 时查 `p2l[i]`。同时修复多轮模拟中 `lc` 也要跟着 `p2l` 一起 swap。

### Bug 2: all_reduce 原地污染 decay 历史 (controller.py)
**根因**: `torch.distributed.all_reduce(self.load, SUM)` 原地修改了 `self.load`，把 8 个 rank 的本地值求和写回。decay 机制 `self.load *= 0.9` 作用在这个已全局化的值上，下一个 window 再 all_reduce 时又把 8 份全局值相加——每 window 膨胀 `num_ranks × decay` 倍。
**影响**: decay=0.9 时 tok_global 每 window ×7.2（实测完全吻合 8×0.9），load 数值指数爆炸，所有基于此的观察全部失真。
**修复**: all_reduce 操作在 `self.load.clone()` 上执行，`self.load` 本身仅保留本 rank 的衰减后本地历史。

### Bug 3: 异步 P2P 与集合通信的 NCCL 死锁 (async_swapper.py + controller.py)
**根因**: OEPLB 的 `AsyncSwapExecutor` 在低优先级 CUDA stream 上发起 P2P swap，通过非阻塞 `event.query()` 检查完成。但 NCCL 要求所有 rank 在 communicator 上的操作类型和顺序严格一致。如果某个 rank 的 CPU 线程先跑到下一个 window 的 `all_reduce`（默认流），而另一个 rank 的 P2P 还没真正提交给 NCCL（低优先级流调度延迟），两个 rank 在 communicator 上看到的操作类型不匹配，永久死锁。
**证据**: py-spy dump 显示全部 8 个 rank 一字不差卡在 `_decide_and_begin_swap` 的 `all_reduce` 调用。SGLang 官方 EPLB 的 `_execute_p2p_ops` 使用同步阻塞 `req.wait()` 避免此问题。
**修复**: 在发起 all_reduce 前，`try_finish(force_wait=True)` 强制阻塞等待上一轮 P2P 完成，恢复跨 rank 顺序一致性。

### Bug 4: layer_imbalance_analysis.py 硬编码 range(4)
**影响**: 8卡环境下只分析前4个 rank 的 trace 数据，漏掉一半信息。
**修复**: 改为 `range(num_ranks)`，动态适配。

## 三、算法改进

### 3.1 指数衰减 (替代清零)
每个 sync_window 结束后 `self.load *= decay_factor` 而非 `self.load.zero_()`，保留历史路由信息的指数加权移动平均，让 swap 决策基于更稳定的长期模式而非单窗口噪声。

### 3.2 路由稳定性追踪
新增 `_track_routing_stability()` 方法，计算相邻 window 的 load 分布余弦相似度。实测结果 cos_sim > 0.95，确认路由模式是长期稳定的，支撑了使用较大 sync_window + 衰减的设计决策。

### 3.3 record 采样降频
`_sample_interval = max(4, sync_window // 64)` 降低 `record_next_layer` 调用频率，减少 scheduler 关键路径上的 CPU 开销。

## 四、参数调优

### 离线模拟扫描 (180种组合, 48层全覆盖)

| 参数 | 扫描范围 | 最优值 | 备注 |
|------|---------|--------|------|
| threshold_ratio | 1.02-1.15 | **1.10** | 太低(1.02)导致噪声过度反应和大量无效 swap；太高(1.15)修正不足 |
| max_swaps_per_layer | 2-30 | **3** | >5 时 swap 量暴增，P2P 开销抵消收益；=2 时部分层修不到位 |
| sync_window | 32-128 | **128** | 小窗口(32)导致 swap 量爆炸甚至死锁；大窗口统计更稳定 |
| decay_factor | 0.85-0.95 | **0.85** | 较短记忆（保留 85% 历史），平衡响应速度和噪声抗性 |

### GPU 实测验证

| 配置 | TPS (n=3) | Delta | p-value |
|------|-----------|-------|---------|
| Baseline (无OEPLB) | 558.8±5.1 | — | — |
| t=1.05, ms=5, w=128, d=0.9 | 580.5±5.0 | +3.88% | 0.0062 |
| **t=1.08, ms=5, w=128, d=0.85** | **596.5±6.7** | **+6.76%** | **0.0020** |
| **t=1.10, ms=3, w=128, d=0.85** | **600.1±6.8** | **+7.40%** | **0.0015** |
| t=1.12, ms=2, w=128, d=0.85 | 591.7±4.8 | +5.89% | 0.0012 |

### 最终长时间验证 (n=6)

| | Baseline | OEPLB (最优) |
|---|---|---|
| 均值 | 569.03 TPS | **582.78 TPS** |
| 标准差 | 9.36 | 8.29 |
| Delta | — | **+2.42%** |
| p-value | — | **0.0228** |
| 统计显著 | — | **YES** |

## 五、关键实验发现

### 5.1 数据集路由特征决定收益
- **通用数据集** (500种prompt): 路由分散, top3专家仅占3.6%流量, per-step不均衡主要由统计噪声主导 → OEPLB收益在噪声范围内(+0.9%~+1.9%)
- **专一/集中数据集** (20种prompt重复): 路由集中, top3专家占18%流量, 不均衡更系统性更可预测 → OEPLB收益显著(+2.4%~+7.4%)

### 5.2 per-step 不均衡度实测改善
在 focused 数据集上 (真实 physical_hist 路由数据):

| Layer | Baseline | OEPLB | 改善 |
|-------|----------|-------|------|
| 0 | 1.452 | 1.190 | -18.0% |
| 12 | 1.657 | 1.376 | -17.0% |
| 24 | 1.573 | 1.389 | -11.7% |
| 36 | 1.550 | 1.389 | -10.4% |
| 42 | 1.817 | 1.444 | -20.5% |
| **平均** | | | **-15.5%** |

理论天花板 (LPT全知最优): -24.3%, 当前实现达到 ~64% 的理论上限。

### 5.3 swap 开销分析
- 首次 swap (含 prewarm): ~1000ms
- 稳态 swap: 平均 134ms/window, ~51-77 次 swap/window
- record 开销: 76.8μs/call, 累计占 scheduler 总开销 ~2s (47 windows)
- all_reduce: 14.6ms/window
- 总 OEPLB 开销: ~3.5s/run (vs ~85s 总运行时间 = 4.1%)

### 5.4 Swap 量与吞吐的非单调关系
swap 量过多(ms=15)反而 hang 或降低吞吐(P2P NVLink争用), 过少(ms=2)修正不足。
最优点在 ms=3: 精准少量 swap > 大量过度 swap。

## 六、工具产出

- `routing_tracer.py`: 独立的 ground-truth token 路由追踪器 (env var 开关, 零侵入)
- `layer_imbalance_analysis.py`: 修复了 8卡 range(4) 硬编码 bug
- 离线参数扫描框架: 基于真实路由数据的因果正确模拟, 不需要 GPU 时间

## 七、V2.1 策略验证（补充实验 2026-07-19）

### 7.1 策略改进
借鉴 V2.1 设计理念：**不限制每层 max_swaps（设为 16 作为安全阀），让每一层尽可能做到均衡**，仅用 threshold 控制哪些层需要修正。这比 V1 的 `max_swaps_per_layer=3` 更彻底——V1 的 cap 恰好限制了 swap 总量从而偶然表现不错，但本质上是在牺牲均衡质量换取稳定性。V2.1 策略用 threshold 精确控制触发范围，每个被触发的层都能完整修到位。

### 7.2 离线参数扫描（180 种配置，48 层全覆盖）
V2.1-style uncapped 最优改善：**-20.8%**（vs capped ms=3 的 -17%）

效率甜点：`threshold=1.12, window=64, decay=0.85, max_swaps=16`
- 改善 -19.9%，swap 量仅 266/window
- 收敛后稳态 swap 仅 1-6 次/window

### 7.3 GPU 实测（公平 warmup-then-measure，n=6）

| | Baseline (warmed) | OEPLB V2.1-style (warmed) |
|---|---|---|
| 数据 | [584.4, 570.1, 575.3, 566.2, 570.0, 573.6] | [605.2, 604.5, 607.8, 616.6, 615.4, 614.6] |
| 均值 | 573.27 TPS | **610.68 TPS** |
| 标准差 | 6.31 | 5.46 |
| **Delta** | | **+6.53%** |
| **p-value** | | **0.000001** |
| 重叠 | | **零重叠**（baseline max 584.4 < OEPLB min 604.5） |

### 7.4 最优配置参数
```
--pb-oeplb-threshold-ratio 1.12
--pb-oeplb-sync-window 64
--pb-oeplb-max-swaps-per-layer 16   # V2.1-style uncapped
--pb-oeplb-max-total-swap-layers 48
--pb-oeplb-cooldown-steps 5
--pb-oeplb-min-prefill-tokens 1000
decay_factor = 0.85 (hardcoded in controller.py)
```

### 7.5 关键发现：warmup 效应
OEPLB 需要约 250-300 秒（~3 个 benchmark 周期）的 swap 收敛期。在此期间 placement 从 trivial 逐步优化到稳态，吞吐从 481 TPS 逐步上升到 605+ TPS。测量 OEPLB 效果时应先发送 warmup 流量让 swap 收敛，再测正式吞吐——否则冷启动期的低吞吐会拖低整体均值，掩盖真实收益。

## 八、多数据集验证 + Kernel 时间分析（2026-07-19 续）

### 8.1 Kernel 时间 per-forward-step 公平对比

在同等负载条件下（GPU util ~62%，~795 forward steps），归一化为 per-step 的 kernel 时间：

| Category | Baseline (us/step) | OEPLB (us/step) | Delta |
|----------|--------------------|-----------------|-------|
| dispatch | 7323 | 6300 | **-14.0%** |
| **combine** | **5479** | **4015** | **-26.7%** |
| expert | 6214 | 6179 | -0.6% |
| attention | 4731 | 4695 | -0.8% |
| **TOTAL** | **25546** | **23082** | **-9.6%** |

关键确认：
- **combine 时间 -26.7%**：expert 负载更均衡后，combine 的同步等待时间大幅缩短
- **dispatch 时间 -14.0%**：同理，更均衡的 placement 减少了 dispatch 排队
- **expert 计算本身 -0.6%**：expert 计算量由 token 数决定（不取决于 placement），改善体现在通信等待上
- **总 forward step 时间 -9.6%**：解释了端到端 +6.53% TPS 提升

### 8.2 不均衡度 ratio 对比（focused dataset, fair trace）

| Category | Metric | Baseline | OEPLB | Delta |
|----------|--------|----------|-------|-------|
| dispatch | mean | 1.545 | 1.375 | -11.0% |
| dispatch | p99 | 2.622 | 1.855 | -29.3% |
| combine | mean | 1.384 | 1.272 | -8.1% |
| combine | max | 4.187 | 2.419 | -42.2% |
| expert | mean | 1.316 | 1.195 | -9.2% |
| expert | max | 2.760 | 1.933 | -30.0% |

### 8.3 多数据集对比

| Dataset | Input length | Baseline | OEPLB | Delta | p-value |
|---------|-------------|----------|-------|-------|---------|
| focused CRS (warmed, n=6) | ~1500 tok | 573.3 | **610.7** | **+6.5%** | 0.000001 |
| LongAlpaca papers (n=3) | 2500-7500 tok | 337.3 | 347.0 | +2.9% | 0.083 |
| extreme CRS (n=3) | ~5500 tok | 553.7 | 561.1 | +1.3% | 0.335 |

**核心规律**：OEPLB 收益与请求的路由集中度正相关。同域重复请求（客服/RAG 场景）收益最大，混合领域长文本次之，完全随机请求最小。

### 8.4 关于 decay_factor=0.85

- 含义：每个 sync_window 结束时 `load_history *= 0.85`
- 半衰期：`log(0.5)/log(0.85) ≈ 4.3 个 window ≈ 30 秒`
- 0.85 = 较短记忆（保留 85% 历史），平衡响应速度和噪声抗性
- 离线扫描显示 0.8-0.9 范围内差异 <0.5%，都是合理选择
