# PB-OEPLB v0.1 实验方案与预期收益分析

> **对象**：Qwen3-30B-A3B-FP8, SGLang, 单机 8×GPU, EP=8
> **目的**：验证 swap-only 版本能否在**不破坏输出**的前提下，降低 MoE 层的负载不均、缩短端到端延迟

---

## 1. 硬件与部署配置

### 目标环境（推荐）
- **单机 8×H100/H800 SXM**（NVLink 全互联，跨卡 ~450 GB/s）
- 若只有 8×A100：也可跑，MoE 计算稍慢但方案逻辑不变
- 若只有 4 卡：`--tp 4 --ep-size 4`，每卡 32 expert，一样可验证

### 启动命令

**Baseline (default)**
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --tp-size 8 --ep-size 8 \
  --enable-ep-moe --moe-a2a-backend deepep \
  --disable-cuda-graph \
  --host 0.0.0.0 --port 30000
```

**Baseline + EPLB static**（如果 SGLang 版本支持）
```bash
# 同上 + --enable-eplb --eplb-rebalance-num-iterations <大数>（禁掉重排，用初始 EPLB）
```

**Ours (pb-oeplb)**
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --tp-size 8 --ep-size 8 \
  --enable-ep-moe --moe-a2a-backend deepep \
  --disable-cuda-graph \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.25 \
  --pb-oeplb-max-swaps-per-layer 1 \
  --pb-oeplb-min-prefill-tokens 4096 \
  --pb-oeplb-cooldown-steps 500 \
  --host 0.0.0.0 --port 30000
```

---

## 2. 测试负载设计

分三档负载，覆盖不同 skew 场景：

### Workload A：合成均匀负载（sanity check）
- **目的**：确认 pb-oeplb 在无 skew 时**不引入额外开销**
- **做法**：ShareGPT 里随机采样 200 条 prompt，输入长度 512-2048 均匀分布
- **预期**：pb-oeplb 几乎不触发 swap，延迟与 baseline 持平（±2%）

### Workload B：领域集中负载（触发 skew）
- **目的**：验证 skew 场景下 pb-oeplb 有加速
- **做法**：全部使用同一领域的 prompt（比如全部代码补全、全部数学题）
- **原理**：MoE 路由有 domain-specialization 现象（Reddit 上有人测过 Qwen3-30B 单层某 expert 频次可到 5%、其他 0%），领域集中会放大 skew
- **预期**：swap 高频触发，MoE 层延迟明显下降

### Workload C：混合真实负载
- **目的**：接近真实生产
- **做法**：ShareGPT 200 条 + HumanEval 100 条 + GSM8K 100 条，随机打散
- **预期**：介于 A 和 B 之间

### 请求发送方式
用 SGLang 自带的 `bench_serving.py`：
```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 8      # req/s，模拟中等并发
```

---

## 3. 评估指标

### 3.1 正确性指标（必须先过）
| 指标 | 要求 |
|---|---|
| 相同 prompt 相同 seed 下的输出 | **bit-exact 一致**（swap 不改语义） |
| 服务不 crash | 跑 30 分钟不挂 |
| Rank 之间 mapping 表一致 | 通过日志或 assert 保证 |

### 3.2 性能指标（主要收益）
| 指标 | 说明 | 目标 |
|---|---|---|
| **MoE 层单步延迟** (ms) | forward 中 MoE 层的耗时，用 CUDA event 打点 | 相对 baseline 下降 |
| **端到端 TTFT** (Time To First Token) | 从请求到出第一个 token | 下降 |
| **端到端 TPOT** (Time Per Output Token) | 稳态 decode 的 per-token 延迟 | 下降 |
| **整体吞吐** (tokens/s) | 服务器总吞吐 | 上升 |
| **P99 延迟** | 尾延迟 | 下降更明显（skew 主要拖尾延迟） |

### 3.3 方案内部指标（诊断用）
| 指标 | 说明 |
|---|---|
| Imbalance ratio 分布 | before/after swap 的分布直方图 |
| Swap 触发频率 | 每分钟多少次 |
| Swap 迁移开销 | 每次 swap 的耗时（ms）与搬运字节数 |
| Cooldown / 阈值命中比 | 跳过 vs 触发的比例，用于调参 |

---

## 4. 预期收益分析

### 4.1 收益的理论上限

MoE 层耗时理论上受"最忙 GPU"支配（其他 GPU 在等）。假设：

- **完全均衡时**：每 GPU 处理 `T·top_k / EP` 个 token
- **skew 时**：最忙 GPU 处理 `T·top_k·(max_ratio) / EP` 个 token
- **加速上限**：`(max_ratio - 1) / max_ratio`

Qwen3-30B-A3B 在真实负载下：

| max/avg 观察值 | 潜在 MoE 加速 |
|---|---|
| 1.15（较均衡） | ~13% |
| 1.30（中度 skew，参照 DataFore Fig.17 Default） | ~23% |
| 1.50（领域集中场景） | ~33% |
| 2.00（极端 skew） | ~50% |

**MoE 层通常占 forward 总时间的 40-60%**（Qwen3 A3B 每 token 走 8/128 expert，激活占比 3.3/30.5 ≈ 11%，但 all-to-all 通信 + MoE 计算合起来在 forward 里占大头）。

### 4.2 端到端预期加速比

按"MoE 占 50% forward 时间 + swap 只能吃掉部分不均衡收益（因为不做 replicate，热点 expert 依然是 bottleneck）"折算：

| 负载 | MoE 层加速 | 端到端加速（预期） |
|---|---|---|
| Workload A（均匀） | ~0% | -1% ~ +1%（引入开销可忽略） |
| Workload B（领域集中） | 15% ~ 25% | **5% ~ 12%** |
| Workload C（混合） | 8% ~ 15% | **3% ~ 8%** |

**尾延迟收益更明显**：P99 通常比 mean 好 1.5-2×，因为 skew 拖的就是尾。

### 4.3 参考对标

- **DataFore ISCA'26 Case Study 2**（Qwen3-235B on 8×H100）：MoE 计算加速 **15.5% / 12.5%**（remap/dup 两种模式）
  - 但对方是"一次性 offline 决策 for decode"
  - 我们做"在线增量 for prefill+decode"，理论上**在稳态阶段收益会更小**（因为 baseline placement 大概率已经不算太差），**但在分布漂移场景收益会更持久**
- **DeepSeek EPLB (paper)**：均衡度提升 ~15-25%，对应吞吐提升 5-15%
  - 但 EPLB 是全量重排，我们只做 swap，收益是它的**子集**

### 4.4 Swap-only 版本的天花板（诚实评估）

**必须承认的局限**：v0.1 只做 swap 不做 replicate，意味着：

1. **单个热点 expert 依然是瓶颈**：如果一个 expert 承担了 20% 的 token，无论把它放哪张 GPU，那张 GPU 就是最忙的。swap 只能"选一个当前最闲的 GPU 接手"，不能"把这个 expert 分身到多张 GPU"。
2. **收益随 skew 强度饱和**：max_ratio 很小（<1.15）时收益小；max_ratio 很大（>2.0）时 swap 也只能缓解不能根治（需要 replicate）。
3. **甜点区**：max_ratio 在 1.2-1.6 之间时 swap 效果最好。

所以：
- **Workload B 是 swap 最能出效果的场景**
- **Workload A 主要验证"不添乱"**
- **Workload C 是"真实但保守"的收益**

---

## 5. 实验步骤（按顺序执行）

### Step 1: 正确性验证（半天）
1. 发 10 条固定 prompt（seed 固定），baseline 输出保存
2. 开 pb-oeplb 再发同样的 10 条，对比 hash
3. **必须一致**才能进入 Step 2

### Step 2: 无收益场景不添乱（半天）
1. 跑 Workload A，对比 baseline vs pb-oeplb
2. 端到端延迟差 < 2% → 通过
3. Swap 触发次数应该很少（<5 次 / 500 请求）

### Step 3: 有收益场景验证（1 天）
1. 跑 Workload B（领域集中）
2. 关注：MoE 层延迟、TTFT、TPOT、P99
3. 记录 imbalance before/after 曲线
4. 期望看到端到端 5-12% 加速

### Step 4: 真实负载（1 天）
1. 跑 Workload C
2. 记录同上指标
3. 期望端到端 3-8% 加速

### Step 5: 参数扫描（1-2 天）
- `threshold_ratio ∈ {1.15, 1.25, 1.40, 1.60}`
- `max_swaps_per_layer ∈ {1, 2}`（虽然 v0.1 建议 1，可以试试 2）
- `cooldown_steps ∈ {200, 500, 1000}`

找到 Workload C 上的最优参数组合。

---

## 6. 可能的失败模式与应对

| 失败模式 | 现象 | 应对 |
|---|---|---|
| 输出不一致 | Step 1 就挂 | 检查 mapping 表是否所有 rank 都更新；检查 send/recv 顺序；检查 FP8 scale 是否搬了 |
| 服务挂死 | 某次 swap 后 hang | send/recv 顺序两边不一致导致死锁；用 rank 编号约定顺序 |
| 收益为负 | pb-oeplb 比 baseline 慢 | swap 太频繁（cooldown 调大）；单次 swap 太贵（换 barrier 位置） |
| Skew 未消除 | imbalance after 仍然高 | 单个 expert 就承担了太多，只能靠 replicate 解决；这是 v0.1 已知局限 |
| CUDA graph 冲突 | 启动失败 | 先 `--disable-cuda-graph`，v0.2 再考虑与 cuda graph 兼容 |

---

## 7. 报告模板（跑完实验要交的东西）

```markdown
# PB-OEPLB v0.1 实验报告

## 环境
- 硬件: 8×H100 SXM, NVLink
- 模型: Qwen3-30B-A3B-FP8
- SGLang: <commit hash>
- 日期: <YYYY-MM-DD>

## 正确性
- Baseline vs pb-oeplb 输出一致性: PASS / FAIL

## 性能（表格）
| Workload | Metric | Baseline | pb-oeplb | Δ |
| A | TTFT p50 (ms) | ... | ... | ... |
| A | TPOT (ms) | ... | ... | ... |
| A | Throughput (tok/s) | ... | ... | ... |
| B | ... | ... | ... | ... |

## 观察
- Swap 触发次数: ...
- Imbalance before / after 均值: ... / ...
- 单次 swap 耗时: ... ms

## 结论
- v0.1 是否达到预期？
- 下一版 (v0.2) 需要加什么？
```

---

## 8. 给编码 AI 的一句话

> **先把 Step 1（正确性）跑通再谈收益**。swap 是最保守的重排方式，理论上不改语义、bit-exact。如果 Step 1 挂了，问题一定出在 send/recv 顺序或 mapping 表同步上，不是算法本身。收益部分即使 v0.1 只有 3-8%，也已经足够证明"在线 EPLB"这条路走得通，v0.2 加上 replicate 后收益会显著扩大。

Sources:
- [Qwen3-30B-A3B - Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [Qwen3 MoE expert profiling on Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1mceq8m/has_anyone_profiled_the_expert_specialization_in/)
- [DataFore: Patterns behind Chaos (arXiv:2510.05497)](https://arxiv.org/abs/2510.05497)
- [DeepSeek EPLB](https://github.com/deepseek-ai/EPLB)
