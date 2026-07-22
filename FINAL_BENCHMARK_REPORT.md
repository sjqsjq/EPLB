# OEPLB Greedy Planner Benchmark Report

## 实验环境
- 8× NVIDIA H20 (96GB, NV18 全互连)
- SGLang 0.5.6.post2, DeepEP v1.2.1, PyTorch 2.9.1+cu128
- Qwen3-30B-A3B-FP8, EP=8, DP=8, chunked_prefill=1024

## 数据集
- **focused CRS**: 20 unique CRS政策报告 (~1500 tokens each), 重复75次 = 1500 requests
- max_tokens=1 (纯 prefill, 无 decode)
- 每次测试前 warmup 300 requests

生成命令:
```python
import json
with open('OEPLB/benchmarks/frozen_requests_prefill_heavy.jsonl') as f:
    reqs = [json.loads(l) for l in f]
out = []
for rep in range(75):
    for r in reqs[:20]:
        r2 = dict(r); r2['id'] = f"{r['id']}_rep{rep}"; r2['max_tokens']=1; out.append(r2)
with open('/tmp/focused_out1.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
```

## 测试配置

### 公共参数 (所有配置共享)
```bash
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_HCA_LIST= \
  NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL \
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 SGLANG_JIT_DEEPGEMM_PRECOMPILE=false

COMMON="--model-path /workspace/Qwen3-30B-A3B-FP8 --tp 8 --dp 8 --ep-size 8 \
  --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache"
```

### Config A: Baseline (无 OEPLB)
```bash
python3 -m sglang.launch_server $COMMON
```

### Config B: Old planner (per-layer cap)
```bash
python3 -m sglang.launch_server $COMMON \
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.12 \
  --pb-oeplb-sync-window 64 --pb-oeplb-max-swaps-per-layer 16 \
  --pb-oeplb-max-total-swap-layers 48 --pb-oeplb-cooldown-steps 5 \
  --pb-oeplb-min-prefill-tokens 1000
# controller.py: decay_factor=0.9
```

### Config C: Greedy no-threshold (budget=300)
```bash
python3 -m sglang.launch_server $COMMON \
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.0 \
  --pb-oeplb-sync-window 64 --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-max-total-swap-layers 48 --pb-oeplb-cooldown-steps 5 \
  --pb-oeplb-min-prefill-tokens 1000
# controller.py: decay_factor=0.9
# rebalancer.py: greedy global-budget planner, MAX_TOTAL_OPS=300
```

### Config D: Greedy t=1.02 (budget=300) ← BEST
```bash
python3 -m sglang.launch_server $COMMON \
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02 \
  --pb-oeplb-sync-window 64 --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-max-total-swap-layers 48 --pb-oeplb-cooldown-steps 5 \
  --pb-oeplb-min-prefill-tokens 1000
# controller.py: decay_factor=0.9
# rebalancer.py: greedy global-budget planner, MAX_TOTAL_OPS=300
```

## 吞吐结果 (同一时段, 每配置 3 次独立启动)

| Config | Run 1 | Run 2 | Run 3 | Mean | Std | Delta% | p-value |
|--------|-------|-------|-------|------|-----|--------|---------|
| A: Baseline | 18.47 | 18.43 | 18.21 | **18.37** | 0.13 | — | — |
| B: Old planner (t=1.12,ms=16) | 18.93 | 19.11 | 18.63 | 18.89 | 0.24 | +2.83% | 0.045 |
| C: Greedy no-threshold | 19.34 | 18.98 | 18.49 | 18.94 | 0.42 | +3.08% | 0.136 |
| D: **Greedy t=1.02** | **19.39** | **19.67** | **19.41** | **19.49** | 0.16 | **+6.09%** | **0.0008** |

单位: req/s (requests per second)

## Trace 分析

### Trace 路径
- Baseline: `/tmp/trace_final_bl_out1/`
- OEPLB greedy_t102: `/tmp/trace_final_oe_out1/`

### 不均衡度

| Category | Metric | Baseline | OEPLB | Delta |
|----------|--------|----------|-------|-------|
| dispatch | mean | 1.318 | 1.302 | -1.2% |
| dispatch | max | 4.752 | 3.008 | **-36.7%** |
| combine | mean | 1.301 | 1.278 | -1.8% |
| combine | max | 3.731 | 2.259 | **-39.5%** |
| **expert** | **mean** | **1.313** | **1.197** | **-8.8%** |
| expert | std | 0.250 | 0.179 | **-28.3%** |
| expert | p99 | 2.316 | 1.739 | **-24.9%** |
| expert | max | 2.685 | 2.070 | **-22.9%** |

### Kernel 时间 (per forward step, 归一化)

| Category | Baseline | OEPLB | Delta | 说明 |
|----------|----------|-------|-------|------|
| dispatch | 4300 us | 4999 us | +16.3% | swap 改变 placement 增加跨 rank 发送 |
| **combine** | **5009 us** | **3688 us** | **-26.4%** | 均衡后同步等待大幅缩短 |
| expert | 6170 us | 6156 us | -0.2% | 计算量不变 |
| attention | 4690 us | 4707 us | +0.3% | 不受 MoE 影响 |
| **TOTAL** | **21927 us** | **21405 us** | **-2.4%** | combine 节省 > dispatch 增加 |

## Greedy Planner 算法特性

1. **全局 budget 分配**: 每次选当前 ratio 最高的层做 1 轮 swap, 不固定 per-layer cap
2. **无收益退出**: swap 后 ratio 未降低 ≥0.001 则标记该层完成, 避免无效循环
3. **全局安全限制**: MAX_TOTAL_OPS=300 防止 P2P batch 爆炸
4. **threshold 控制**: t=1.02 跳过已经足够均衡的层, 节省 P2P 带宽
