# EPLB Per-Layer 跨层预测 + 专家迁移 + Token Sharding 实验报告

## 目录

1. [实验环境与服务启动](#1-实验环境与服务启动)
2. [负载配置](#2-负载配置)
3. [整体执行流程](#3-整体执行流程)
4. [算法细节](#4-算法细节)
5. [通信细节与踩坑记录](#5-通信细节与踩坑记录)
6. [实现细节](#6-实现细节)
7. [实验结果](#7-实验结果)

---

## 1. 实验环境与服务启动

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | 4 × NVIDIA L20A（184.3GB/卡，SM 10.0，aarch64） |
| 互联 | NVLink（单节点内） |
| Driver | 580.95.05, CUDA 13.0 (nvcc 12.9) |
| 框架 | SGLang（editable install, /sgl-workspace/sglang） |
| 关键组件 | DeepGEMM（FP8 GEMM）、DeepEP（AlltoAll 通信）、FlashInfer（Attention） |

### 1.2 模型

**Qwen3-235B-A22B-FP8**（`/workspace/EPLB/models/Qwen/Qwen3-235B-A22B-FP8`）

| 参数 | 值 |
|------|-----|
| 架构 | qwen3_moe |
| 总参数量 / 激活参数量 | 235B / 22B |
| 层数 | 94 |
| Experts | 128 个，top-8 路由 |
| hidden_size | 4096 |
| moe_intermediate_size | 1536 |
| 权重大小（FP8） | 222.6 GB（48 个 safetensors） |

### 1.3 并行配置

TP = DP = EP = 4（4 张 GPU 各承担一个 TP/DP/EP rank），单节点部署。

- **DP=4**：每个 rank 独立处理自己的一批请求（Attention 部分）
- **EP=4**：128 个专家平均分布在 4 张卡（每卡 32 个原生 + 冗余槠位）
- **DeepEP**：MoE 层的 AlltoAll 通信后端

### 1.4 启动命令

```bash
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_TORCH_PROFILER_DIR=/workspace/EPLB/result/traces_235b_swap

python3 /workspace/EPLB/pipeline/launch_with_pipeline.py \
  --model-path /workspace/EPLB/models/Qwen/Qwen3-235B-A22B-FP8 \
  --tp 4 --dp 4 --ep-size 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --deepep-config /workspace/EPLB/deepep_config.json \
  --quantization fp8 \
  --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 32 \
  --watchdog-timeout 1200 \
  --ep-num-redundant-experts 32 \
  --port 30000 --host 0.0.0.0 \
  --trust-remote-code
```

关键参数说明：

| 参数 | 作用 |
|------|------|
| `--moe-a2a-backend deepep` | 启用 DeepEP 做 MoE 层 AlltoAll |
| `--ep-num-redundant-experts 32` | 每张卡额外预留 8 个冗余专家槠位（128+32=160 物理槠位，40/卡），供专家迁移算法使用 |
| `--deepep-config` | 自定义 DeepEP 通信用的 SM 数量（默认仅用 20/152 个 SM，配置提升到 48 个） |
| `--cuda-graph-max-bs 32` | Decode 阶段用 CUDA Graph 加速，batch size ≤32 时走 replay 路径 |
| `launch_with_pipeline.py` | 自定义启动脚本，注入本实验的 pipeline 框架（见第 6 节） |

服务不是用 `python3 -m sglang.launch_server` 直接起的，而是通过 `/workspace/EPLB/pipeline/launch_with_pipeline.py` 这个包装脚本：

1. 读取模型 `config.json` 自动探测层数/专家数/top_k
2. 初始化 `PipelineManager` 单例（初始 `enabled=False`）
3. **monkeypatch** `ModelRunner.initialize`，在原生初始化（包含全部 CUDA Graph capture/warmup）完成后才把 `pm.enabled` 置为 `True`（原因见第 5.3 节）
4. 调用 SGLang 原生的 `prepare_server_args` + `launch_server`

---

## 2. 负载配置

### 2.1 正确性验证

单条请求，验证输出语义正确：

```
POST /v1/chat/completions
{"messages": [{"role": "user", "content": "What is 12+13? Just the number."}],
 "max_tokens": 300, "temperature": 0}
→ 最终答案: 25 （正确）
```

### 2.2 性能压测

**并发压测**：128 个并发请求，一次性发出

```python
ThreadPoolExecutor(max_workers=64).map(send, range(128))
# 每个请求: max_tokens=100, temperature=0.7
```

**分波压测**（用于制造多个独立的 forward-pass boundary，压测跨层同步和权重迁移的稳定性）：

```python
for wave in range(10):
    ThreadPoolExecutor(max_workers=64).map(send, range(wave*64, wave*64+64))
    # 每波 64 个并发请求，max_tokens=100~150
```

累计测试过 128 / 320 / 384 / 640 请求规模，全部通过。

### 2.3 Trace 采集

用 SGLang 自带的 PyTorch Profiler HTTP API：

```bash
curl -X POST http://localhost:30000/start_profile -d '{"num_steps": 20}'
# ... 发送负载 ...
# trace 自动落盘到 SGLANG_TORCH_PROFILER_DIR
```

---

## 3. 整体执行流程

### 3.1 设计动机

Prefill 阶段，EP=4 下每层 MoE 执行时，各 GPU 的 expert 负载天然不均衡（不同 token 路由到不同专家，各卡忙闲不均），导致 AlltoAll 通信时快的 GPU 要等最慢的 GPU，这是通信占比高企的根本原因（详见第 7 节）。

目标：**逐层**预测下一层的路由分布 → 用预测结果调整下一层的专家物理布局（迁移/复制）→ 同时对本层的 token 做负载均衡分片，减少最慢 GPU 的等待时间。

### 3.2 分层职责划分

设计上明确拆分成两类工作，分别对应"要不要跨 GPU 同步"：

```
每一层（层内，不做跨 GPU 同步，安全嵌入 DeepEP 热路径）：
  1. 跨层路由预测：用 layer i 的 hidden_states 预测 layer i+1 的路由
  2. 本地 demand 写入：把预测结果计入 per-layer demand 矩阵（纯 GPU 操作）
  3. Token Sharding（本地，异步线程执行）：
     用【上一轮】已经全局同步好的 per-layer demand，
     结合本 rank 的 topk_ids，决定本 rank 的 token 该发去哪个 GPU

forward-pass 边界（layer_id 从 93 绕回 0，DeepEP 每层 collective
天然形成的同步点，此刻没有 DeepEP collective 在途）：
  1. Gloo AllReduce：同步整个 per-layer demand 矩阵（94×128）到全局一致
  2. Expert 迁移决策：用全局 demand 跑 LPT 算法，算出新的专家布局 + 迁移操作
  3. Metadata 一致性校验（Gloo all_gather）
  4. Gloo 权重交换：真实 P2P 搬运专家权重（CPU 中转）
  5. 更新 placement，供下一轮的逐层 token sharding 使用
```

### 3.3 完整数据流图

```
Layer i-1 结果已生效 → Layer i 的专家布局已被调整（部分专家有副本/被迁移）

Layer i 执行：
┌────────────────────────────────────────────────────────────┐
│ Gate → router_logits → TopK → actual_topk_ids               │
└─────────────────────┬────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────┐
│ [Pipeline Hook] 无条件调用（即使本地 batch 为空也调用，见 5.4） │
│                                                              │
│ 1. 跨层预测: predicted_logits = gate_input[i-1] @ gate_weight[i].T │
│    predicted_topk_ids = predicted_logits.sigmoid().topk(8)   │
│    overlap_accuracy = |predicted ∩ actual| / 8               │
│                                                              │
│ 2. 写入 per-layer demand 矩阵[layer_id] (GPU, 无同步)          │
│                                                              │
│ 3. 提交后台线程: Token Sharding                                │
│    用【上一轮 Gloo 同步好的】per-layer demand[layer_id]        │
│    + 本 rank 的 predicted_topk_ids                            │
│    → token_to_gpu 分配                                       │
└─────────────────────┬────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────┐
│ Dispatch Hook（register_pre_dispatch_hook）                  │
│ 用上一层算出的 token_to_gpu 重映射 topk_ids                    │
│ （查 placement[expert, target_gpu] 是否有副本，有则重映射）      │
└─────────────────────┬────────────────────────────────────────┘
                       ▼
              DeepEP dispatch → Expert 计算 → combine
                       │
                       ▼
              layer_id == 0? → 触发 forward-pass 边界同步
```

---

## 4. 算法细节

代码位置：`/workspace/EPLB/pipeline/algorithms.py`

### 4.1 Expert 迁移算法（`ExpertMigrationSolver`）

三阶段贪心算法，输入预测的 per-expert demand（[128] float），输出新的 `placement`（[128, 4] bool，是否在某 GPU 上有副本）和迁移操作列表。

**Phase A：统计 bookkeeping**

```
k[e] = expert e 的副本数
share[e] = demand[e] / k[e]   # 每个副本平摊的负载
gpu_load[r] = sum(share[e] for e on GPU r)
mu = total_demand / num_gpus  # 理想均衡负载
```

**Phase B：复制热专家**（`max_replicate_iters` 轮，受 `m_budget` 传输预算约束）

```python
for _ in range(max_replicate_iters):
    best_e = argmax(d[e]/k[e])   # 找负载最重的专家
    if d[best_e]/k[best_e] <= mu: break   # 已经均衡，停止
    best_r = argmin(gpu_load among GPUs with free slot)  # 找最闲的 GPU
    if 复制后不会让 best_r 变成新的最大负载: 执行复制
```

**Phase C：驱逐/迁移冷专家**（`max_migrate_iters` 轮）

```python
for _ in range(max_migrate_iters):
    if max(gpu_load) <= mu * 1.05: break  # 已经足够均衡
    overloaded_r = argmax(gpu_load)
    coldest_e = 该 GPU 上负载最小的专家
    if coldest_e 有副本: 驱逐这个副本（零成本，直接删除）
    else: 迁移到最闲的 GPU（产生 1 次 P2P 传输）
```

三种操作类型：`REPLICATE`（复制）、`EVICT`（驱逐副本）、`MIGRATE`（迁移）。

**参数**（本次实验配置）：`n_local=32`（原生专家/卡）、`n_max=8`（冗余槠位/卡）、`m_budget=159`（每轮最多传输次数）。

### 4.2 Token Sharding 算法（`TokenShardingSolver`）

两阶段算法，输入 `demand`（上一轮全局同步的 per-layer demand）+ `placement`（当前专家布局）+ 本 rank 的 `topk_ids`，输出每个 token 该发往哪个 GPU。

**Phase 1：闭式最优 makespan**

```
per_replica[e] = demand[e] / k[e]
L* = max(mu, max(per_replica))   # 理论最优的最大 GPU 负载
```

**Phase 2：Locality-first 分配**（按需求降序遍历专家，优先分给 home GPU，再分给最闲副本）

```python
for e in experts_sorted_by_demand_desc:
    for r in [home_gpu(e)] + sorted(replicas_by_load):
        assign = min(remaining_demand, L* - gpu_load[r])
        gpu_load[r] += assign; remaining -= assign
```

**Phase 3：Token→GPU 向量化分配**

```python
# 对每个 token，看它 top-8 专家分布在哪些 GPU 上，
# 选命中最多的 GPU（减少跨 GPU 通信）
score[token, gpu] = placement[topk_ids[token]].sum(dim=expert_axis)
token_to_gpu = score.argmax(dim=gpu_axis)
```

### 4.3 Dispatch Hook：把决策落地到真实路由

代码位置：`/workspace/EPLB/pipeline/dispatch_hook.py`

用 SGLang 原生的 `dispatcher.register_pre_dispatch_hook()` API，在 `topk_ids` 送入 DeepEP 之前做重映射：

```python
def remap_topk_ids(topk_ids, token_to_gpu, num_physical, num_gpus):
    lookup = 从 SGLang 的 ExpertLocationMetadata 构建
             [num_logical, num_gpus] → physical_id 或 -1（无副本）
    for token, expert in topk_ids:
        target = lookup[expert, token_to_gpu[token]]
        if target != -1: topk_ids[token, k] = target  # 命中副本，重映射
        # 否则保持原路由（fallback）
```

---

## 5. 通信细节与踩坑记录

这是本次实验中投入最多精力的部分。核心矛盾：**任何跨 4 张 GPU 的实时同步操作，只要和 DeepEP 自己的通信调度共享同一条 CUDA stream，在高并发下都有极大概率产生真死锁**——无论逻辑写得多正确。以下按时间线记录踩过的坑。

### 5.1 坑 1：每层做 NCCL AllReduce，和 DeepEP 抢 stream

最初设计是**每一层**都对预测的 demand 做一次 `torch.distributed.all_reduce`（默认 NCCL 通信组）。用 `py-spy` 抓取卡死进程的堆栈发现：

```
Rank 0: 卡在 all_reduce()
Rank 1-3: 已经过了 all_reduce，卡在 DeepEP 的 buffer.dispatch()
```

原因：我们的 all_reduce 和 DeepEP 自己的 collective 用的是同一条物理 CUDA stream 的执行队列，只要 rank 0 稍慢一点，后续所有 rank 的 DeepEP kernel 就会排在我们的 collective 后面，形成互相等待。

**尝试的修复 → 失败**：给 pipeline 的 collective 建一个独立的 **NCCL** 通信组（`torch.distributed.new_group(backend="nccl")`）。结果卡点从 `all_reduce` 变成 `predicted_topk_ids.cpu()`——说明**只要是 GPU 侧的阻塞同步操作**（不管走哪个 NCCL 通信组），只要在 DeepEP dispatch 之前插入，就会和它抢占同一条 stream 的执行队列。

### 5.2 坑 2：`torch.distributed.barrier()` 同样致命

在权重交换后加了一个 `barrier()` 想确保所有 rank 同步，同样用的默认 NCCL 通信组，同样卡死。**结论：任何用 NCCL 的额外同步点，只要嵌入 DeepEP 的逐层热路径里，都是定时炸弹。**

### 5.3 坑 3：CUDA Graph Capture 前的 dry-run warmup

SGLang 的 `cuda_graph_runner.py` 在真正 capture 前有一个隐藏的双次 dry-run（`for _ in range(2): run_once()`），这个阶段 `torch.cuda.is_current_stream_capturing()` 返回 `False`，所以我们的 hook 照常触发——在**假数据、且各 rank 天然不同步**的 warmup 阶段就执行了真实的跨 GPU 同步和权重搬运，导致死锁。

**修复**：给 pipeline 加 `enabled` 开关，默认 `False`；monkeypatch `ModelRunner.initialize`，用原生方法覆盖全部 warmup/capture 后才置 `enabled=True`（见 1.4 节的启动脚本逻辑）。

### 5.4 坑 4：Hook 调用点本身被本地条件挡住

SGLang 原生代码里，MoE 层的 hook 调用被 `hidden_states.shape[0] > 0` 挡住（本地 batch 为空就跳过）。这导致：某个 rank 某一层本地批次为空时，直接跳过了我们的 `on_moe_layer()`，包括里面的边界检测和跨 rank collective 调用——其他 rank 永远在等它。

**修复**：把 hook 调用改成无条件触发；本地为空时调用新增的 `on_empty_layer()`，仍然推进层计数、参与边界的 collective（贡献全零 tensor），保证所有 rank 永远同步调用同一组 collective。

### 5.5 坑 5：换 Gloo 通道——真正的转折点

把 demand 同步的通信组从 NCCL 换成 **Gloo**（CPU/网络后端，完全不碰 CUDA stream）。Gloo 的 `all_reduce` 物理上就不会和 DeepEP 的 NCCL collective 产生 stream 排队冲突。

轻负载（5 并发）下第一次跑通全链路推理（含真实权重交换）。但重负载（128 并发）下**权重交换**这一步仍然复现死锁：

```
DP0: 卡在 isend
DP2: 卡在 irecv
DP1, DP3: 已经完成 swap，跑到下一层的 dispatch 去了
```

### 5.6 坑 6：怀疑是 SGLang 原生 P2P 逻辑的 bug，结果被证伪

猜测：SGLang 原生的 `update_expert_weights_single_layer`（内部用 `_ChunkUtils` 做多 rank chunk 分配）在我们这种稀疏迁移模式下，可能给不同 rank 算出不对称的 P2P 操作对。于是自己写了一个**极简、可证明对称**的 diff-based 配对算法（`simple_p2p_swap.py`）：给定所有 rank 都一致的 `old_map`/`new_map`，对每个变化的物理槠位，直接找到"谁现在持有这份数据"，一对一配对，逻辑上不可能不对称。

**结果：换了算法后，死锁位置完全没变**（还是 2 个 rank 卡在 isend/irecv，另外 2 个已经走远）。这是决定性证据：**问题根本不是"算法算错了 P2P 配对"，而是任何走 NCCL/CUDA stream 的实时 P2P，只要和 DeepEP 共享物理 stream，在高负载下就会被卡住**，跟算法逻辑严谨与否无关。

### 5.7 最终方案：权重数据本身也走 Gloo

既然 Gloo 通道在所有测试中从未失败过，把**专家权重张量本身**的传输也从 NCCL P2P 改成 Gloo：

```python
# 发送方：GPU tensor → CPU tensor → Gloo isend
cpu_tensor = w[src_local].to("cpu").contiguous()
P2POp(torch.distributed.isend, cpu_tensor, dst_rank, group=gloo_group)

# 接收方：Gloo irecv → CPU buffer → 拷回 GPU
buf = torch.empty(shape, device="cpu")
P2POp(torch.distributed.irecv, buf, src_rank, group=gloo_group)
...
weights[i][dst_local].copy_(buf.to(weights[i].device))
```

代价：权重传输从 NVLink 速度降到 CPU/loopback 网络速度（本次实验中单次传输 604us~7.7ms，视 rank 而定）。收益：**物理上不可能再和 DeepEP 的 CUDA stream 产生排队冲突**。

这个方案在 640 个并发请求（10 波压测）下**全部成功，零死锁**，是本次实验最终采用的方案。

### 5.8 额外的稳健性设计

- **Metadata 一致性校验**：每次 swap 前，用 Gloo `all_gather` 核对 4 个 rank 的 `old_physical_to_logical_map` 是否严格一致，不一致就跳过本次 swap（防御性降级），同时这个 all_gather 本身也起到了对齐 4 个 rank 时序的 barrier 效果。
- **权重交换限流**：不是每次边界都做物理搬运，通过 `_swap_interval` 控制频率（决策计算每次边界都做，物理搬运可调节频率），进一步降低触发概率。
- **`torch.cuda.synchronize()`**：swap 前先排空本 rank 的 GPU 队列，减少时序偏差。

---

## 6. 实现细节

### 6.1 文件结构

```
/workspace/EPLB/pipeline/
├── __init__.py                 # 模块导出
├── cross_layer_predictor.py    # 跨层路由预测器
├── algorithms.py                # ExpertMigrationSolver + TokenShardingSolver
├── pipeline_manager.py          # 核心编排：逐层本地工作 + 边界全局同步
├── weight_swapper.py            # 权重交换封装（调用 simple_p2p_swap）
├── simple_p2p_swap.py           # Gloo 版 diff-based P2P 执行器
├── dispatch_hook.py             # SGLang pre_dispatch_hook，落地 token 重映射
└── launch_with_pipeline.py      # 启动脚本，含 warmup 屏蔽 monkeypatch
```

### 6.2 关键类

**`PipelineManager`**（单例，每个 GPU 进程各自持有一份）

- `on_moe_layer(layer_id, gate_input, gate_weight, actual_topk_ids)`：逐层调用入口
- `on_empty_layer(layer_id)`：本地 batch 为空时的降级路径
- `_on_forward_pass_boundary()`：边界同步逻辑（Gloo AllReduce → 迁移决策 → 一致性校验 → 权重交换）
- `_ensure_process_group()`：懒加载创建专用 Gloo 通信组

**`CrossLayerPredictor`**

```python
predicted_logits = prev_gate_input @ gate_weight.T   # [B, 128]
predicted_topk_ids = predicted_logits.sigmoid().topk(8).indices  # [B, 8]
overlap = |predicted_topk_ids ∩ actual_topk_ids| / 8
```

### 6.3 SGLang 侵入式修改

只改了一个文件：`/sgl-workspace/sglang/python/sglang/srt/models/qwen3_moe.py`（备份于 `.bak`）：

1. 头部加 import `pipeline` 模块
2. `forward_deepep()` 中，gate/topk 计算之后、`self.experts()` 调用之前，插入无条件的 pipeline hook 调用（含空 batch 分支）
3. 首次调用时通过 `dispatcher.register_pre_dispatch_hook()` 注册 token 重映射钩子，并注册本层 expert 权重引用供后续 P2P 使用

### 6.4 关键设计取舍

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 同步粒度 | forward-pass 边界，非每层 | DeepEP 每层强制 lockstep，边界是天然安全同步点 |
| 通信后端 | Gloo，非 NCCL | 物理隔离 CUDA stream，唯一被验证在高负载下稳定的通道 |
| 权重迁移频率 | 可配置节流（`_swap_interval`） | 降低触发频率，进一步降低风险 |
| CUDA Graph 兼容 | Pipeline 默认关闭，warmup 后才开 | 避免在假数据/未同步阶段执行真实操作 |
| 空 batch 处理 | 无条件调用 hook，用零贡献参与 collective | 避免任何 rank 静默跳过导致其他 rank 永久等待 |

---

## 7. 实验结果

### 7.1 正确性

单请求正确性验证通过：`12+13` → 模型输出 `25`。

多轮压测（128 / 320 / 384 / 640 请求规模）均 100% 成功返回，无内容异常。

### 7.2 吞吐量

| 测试场景 | 请求数 | 耗时 | 吞吐 |
|---------|--------|------|------|
| 128 并发（pipeline 全链路，含权重迁移） | 128 | 5.2s | 24.4 req/s，2439.5 tok/s |
| 10 波压测（swap_interval=1，最激进配置） | 640 | 30.8s | 20.8 req/s |
| Baseline（无 pipeline，同模型同硬件） | 128 | 24.1s | 5.3 req/s |

> 注意：pipeline 版本吞吐数字**不能直接与 baseline 简单对比**——两次测试的具体 prompt、生成长度、并发调度时机不完全相同，差异中包含了正常的 run-to-run 波动。更严谨的对比见 7.3 节的 kernel 级 trace 分析。

### 7.3 通信 Kernel 级分析（235B 模型，EP4+DeepEP，GPU 0）

| 指标 | 无 Pipeline（baseline） | 有 Pipeline（预测+迁移决策，未接入实际 swap） | 说明 |
|------|------------------------|---------------------------------------------|------|
| `notify_dispatch` 等待 | 656.8ms | 883.4ms* | *早期版本，算法未真正生效时的中间态 |
| GPU imbalance（token sharding 算法计算值） | 3.50x | 1.72x → 1.33~1.37x（本次最终版本） | 逐步优化 |
| Pipeline 自身 GPU 开销 | — | 15.0ms（占比 1.2~1.6%） | 预测+采样的 GEMM/topk 开销，几乎可忽略 |
| Dispatch remap rate | — | 30.5%~31.4% | token sharding 决策命中冗余专家副本的比例 |

### 7.4 Pipeline 内部各组件耗时（235B，fwd#40 快照，DP0）

| 组件 | 平均耗时 | 说明 |
|------|---------|------|
| 跨层预测 GPU 开销 | 每层 GEMM+topk+比较，占单层总时间 <1% |
| 每层总耗时（含逐层本地工作） | 440~508us | 完全被 DeepEP 的 dispatch/combine（数毫秒级）隐藏 |
| 边界同步耗时 | 2330~2996us | 每 forward pass 一次，含 Gloo AllReduce + 迁移决策 + 一致性校验 |
| Expert 迁移决策 | avg 165~230us/次，call_count=40 | 纯 CPU LPT 算法 |
| Token Sharding | avg 504~836us/次，call_count 900~2700+ | 纯 CPU，闲时线程池执行，被 GPU 通信时间隐藏 |
| 权重交换（Gloo） | avg 848us~7.7ms/次（视 rank 而定），call_count=2，total_swaps=4 | 4 个 rank 数量严格一致，Metadata 零分歧 |
| 跨层预测准确率 | 89.0%~89.4%（overlap accuracy） | 用 layer i 预测 layer i+1 路由的命中率 |

### 7.5 稳定性验证记录

| 测试轮次 | 并发规模 | 结果 |
|---------|---------|------|
| 单请求 | 1 | 通过 |
| 首次 128 并发 | 128 | **死锁**（NCCL P2P，坑 5/6） |
| 换 Gloo 权重传输后，128 并发 | 128 | 通过（24.4 req/s） |
| 分波压测（swap_interval=8） | 320 | 通过 |
| 分波压测（swap_interval=1，最激进） | 384 | 通过，权重交换 4/4 rank 一致 |
| 分波压测（swap_interval=1，持续 10 波） | 640 | **全部通过**，零死锁，零 metadata 分歧 |

### 7.6 结论

1. **跨层预测有效**：89%+ 的 overlap accuracy，用 layer i 的隐藏状态预测 layer i+1 的路由是可行的。
2. **Token Sharding 和 Expert 迁移的 CPU 计算开销可以被完全隐藏**：都在异步线程/边界安全点执行，不阻塞 GPU 主流水线。
3. **物理权重迁移在高并发下必须避开 NCCL/CUDA stream**：这是本次实验最重要的工程发现——不是算法问题，是分布式系统里"两个独立子系统共享同一条物理执行队列"的经典 stream-ordering 危险，唯一验证稳定的方案是把这类低频、非性能关键的同步操作换到完全独立的通信后端（Gloo）。
4. **当前瓶颈仍是 GPU 间通信本身**（notify_dispatch 等待占大头），Token Sharding 把 GPU 负载不均衡从 3.5x 降到 1.3~1.4x，理论上应该能降低通信等待时间；由于 Gloo 权重传输本身的额外开销（相对 NVLink 慢很多），当前配置下的端到端吞吐提升尚未压出干净的对比数据，是后续优化的方向。
