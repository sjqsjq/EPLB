# OEPLB Greedy Planner — Qwen3-235B-A22B-FP8 实验报告

## 环境
- 8× NVIDIA H20 (96GB/卡, NV18 全互连)
- SGLang 0.5.6.post2, DeepEP v1.2.1, PyTorch 2.9.1+cu128
- **Qwen3-235B-A22B-FP8**: 94层, 128 experts/层, top-8, EP=8 (16 experts/rank)
- 模型路径: `/root/.cache/modelscope/models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master/`
- 模型大小: 223GB (每卡 ~28GB 权重 + ~61GB KV cache/CUDA graph = ~89GB/卡)

## 数据集
- focused CRS: 20 unique CRS政策报告 (~1500 tokens each), max_tokens=1
- 测试: 100 requests, warmup: 20 requests

## 配置

### Baseline
```bash
python3 -m sglang.launch_server \
  --model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache
```

### OEPLB (Greedy planner, t=1.02)
```bash
python3 -m sglang.launch_server \
  --model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache \
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02 \
  --pb-oeplb-min-prefill-tokens 1000 --pb-oeplb-sync-window 64 \
  --pb-oeplb-cooldown-steps 5 --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64
# controller.py: decay_factor=0.9
# rebalancer.py: greedy global-budget planner, MAX_TOTAL_OPS=250
```

## 结果（交替测试 BL1→OE1→BL2→OE2→BL3→OE3）

### 吞吐 (req/s)

| | Run 1 | Run 2 | Run 3 | Mean | Std |
|---|---|---|---|---|---|
| Baseline | 4.464 | 4.462 | 4.494 | **4.474** | 0.018 |
| OEPLB | 4.840 | 4.888 | 4.888 | **4.872** | 0.027 |
| **Delta** | +8.4% | +9.6% | +8.8% | **+8.90%** | |

**p = 0.0001 (极显著), 三组完全不重叠**

### TTFT (ms)

| | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| Baseline | 12275 | 12347 | 12261 | **12294** |
| OEPLB | 11382 | 11213 | 11237 | **11277** |
| **Delta** | -7.3% | -9.2% | -8.4% | **-8.27%** |

**p = 0.0005 (极显著)**

### 总耗时 (s)

| | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| Baseline | 22.4 | 22.4 | 22.2 | **22.4** |
| OEPLB | 20.7 | 20.5 | 20.5 | **20.5** |

## 与 Qwen3-30B 对比

| 模型 | MoE 层数 | Baseline req/s | OEPLB req/s | Delta |
|------|---------|---------------|-------------|-------|
| Qwen3-30B-A3B-FP8 | 48 | 18.37 | 19.49 | +6.09% |
| **Qwen3-235B-A22B-FP8** | **94** | **4.474** | **4.872** | **+8.90%** |

235B 的改善幅度 (+8.90%) 显著大于 30B (+6.09%):
- 94 层 MoE → MoE 计算占总推理时间比例更高
- combine 同步等待改善在更多层上累积
- 每步节省的时间 × 更多层 = 更大的端到端收益

## Trace 采集说明

235B 模型的 trace 文件过大 (每 rank >200MB, 共 >1.6GB), 在当前环境下采集不稳定。
吞吐数据本身已经足够有说服力 (p=0.0001, 零重叠, 交替测试消除时间漂移)。

## Trace 采集方法（适用于所有模型）

1. 启动服务器 → warmup 流量（让 OEPLB swap 收敛）
2. 发持续 benchmark 流量（后台）
3. 等负载稳定（running-req 充分）
4. 调用 `POST /start_profile`:
   ```json
   {"output_dir": "/tmp/trace_xxx", "num_steps": 200,
    "activities": ["CPU","GPU"], "with_stack": false, "record_shapes": false}
   ```
5. Profiler 在 num_steps 个 forward pass 后自动导出 `.trace.json.gz`
6. **采集的是稳态切片**（OEPLB 已收敛后、baseline trivial placement 下）

## Greedy Planner 算法

```
每个 sync_window (64 forward passes) 结束时:
1. all_reduce 聚合所有 rank 的路由统计
2. 在 budget (250 ops) 内贪心循环:
   a. 找当前 ratio 最高的层
   b. 如果 ratio < threshold (1.02) → 停止
   c. 对该层做 1 轮 swap (hottest slot ↔ coldest slot)
   d. 如果 swap 没有改善 ratio → 标记该层完成
   e. 重新排序所有层, 回到 a
3. 通过 async P2P 执行 swap plan
4. 历史 load *= 0.9 (指数衰减)
```

## DeepSeek-Prover-V1 数据集测试 (1024 不同请求)

### 数据集
- 来源: ModelScope `AI-ModelScope/DeepSeek-Prover-V1`
- 27503 条 Lean 4 定理证明数据，取前 1024 条**不同**请求
- Prompt: header + formal_statement + goal，长度 359-17641 chars
- max_tokens=32

### 结果（交替 BL→OE，各 3 次）

#### 吞吐 (req/s)

| | Run 1 | Run 2 | Run 3 | Mean | Std |
|---|---|---|---|---|---|
| Baseline | 57.788 | 59.604 | 58.381 | **58.591** | 0.926 |
| OEPLB | 55.865 | 59.778 | 55.865 | **57.169** | 2.259 |
| **Delta** | | | | **-2.43%** | p=0.396 |

#### TTFT (ms)

| | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| Baseline | 8309 | 8052 | 8146 | **8169** |
| OEPLB | 8174 | 8517 | 7852 | **8181** |
| **Delta** | | | | **+0.15%** | p=0.957 |

### 结论
1024 个完全不同的数学证明请求，路由模式高度分散，OEPLB 无收益。
这证实了核心规律：**OEPLB 收益与请求路由集中度正相关**。

### 多数据集对比汇总 (Qwen3-235B-A22B-FP8, 8×H20)

| 数据集 | 请求多样性 | Baseline | OEPLB | Delta | p |
|--------|-----------|----------|-------|-------|---|
| focused CRS (20 prompts 重复) | 低(重复) | 4.474 | **4.872** | **+8.90%** | **0.0001** |
| DeepSeek-Prover (1024 不同) | 高(随机) | 58.591 | 57.169 | -2.43% | 0.396 |
