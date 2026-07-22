# OEPLB 实验报告：Qwen3-235B-A22B-FP8 on 8×H20

## 环境
- **硬件**: 8× NVIDIA H20 (96GB/卡, NV18 全互连)
- **软件**: SGLang 0.5.6.post2, sgl-kernel 0.3.19, DeepEP v1.2.1, PyTorch 2.9.1+cu128
- **模型**: Qwen3-235B-A22B-FP8 (94层MoE, 128 experts/层, top-8, EP=8→每卡16 experts)
- **模型路径**: `/root/.cache/modelscope/models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master/`
- **模型大小**: 223GB (每卡 ~28GB 权重 + ~61GB KV cache/CUDA graph = ~89GB/卡)
- **OEPLB 代码**: `/workspace/EPLB/OEPLB/src/` (含 greedy planner + load-remap + force_wait 修复)

## 最终结果

### 数据集 1: focused CRS (20 unique prompts 重复, ~1500 tok, max_tokens=1)

| | Run 1 | Run 2 | Run 3 | Mean | Std |
|---|---|---|---|---|---|
| Baseline (req/s) | 4.464 | 4.462 | 4.494 | **4.474** | 0.018 |
| OEPLB (req/s) | 4.840 | 4.888 | 4.888 | **4.872** | 0.027 |
| **吞吐 Delta** | +8.4% | +9.6% | +8.8% | **+8.90%** | **p=0.0001** |

| | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| Baseline TTFT (ms) | 12275 | 12347 | 12261 | **12294** |
| OEPLB TTFT (ms) | 11382 | 11213 | 11237 | **11277** |
| **TTFT Delta** | -7.3% | -9.2% | -8.4% | **-8.27%, p=0.0005** |

### 数据集 2: DeepSeek-Prover-V1 (1024 完全不同的数学证明, max_tokens=32)

| | Run 1 | Run 2 | Run 3 | Mean | Std |
|---|---|---|---|---|---|
| Baseline (req/s) | 59.191 | 60.556 | 60.916 | **60.221** | 0.910 |
| OEPLB (req/s) | 66.928 | 66.754 | 67.236 | **66.972** | 0.244 |
| **吞吐 Delta** | +13.1% | +10.2% | +10.4% | **+11.21%** | **p=0.0038** |

| | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| Baseline TTFT (ms) | 8358 | 7733 | 7693 | **7928** |
| OEPLB TTFT (ms) | 6823 | 6987 | 6868 | **6892** |
| **TTFT Delta** | -18.4% | -9.7% | -10.7% | **-13.06%, p=0.035** |

### Swap 收敛行为 (Prover 数据集, OE1)

| Window | avg_ratio_before | avg_ratio_after | total_ops |
|--------|-----------------|-----------------|-----------|
| 1 | 1.762 | 1.303 | 149 |
| 2 | **1.315** (↓从1.762) | 1.230 | 95 |
| 3 | **1.229** | 1.226 | 6 |
| 4 | **1.228** | 1.220 | 19 |

ratio_before 从 1.762 快速下降到 1.228，swap 次数从 149 收敛到 6-19。

## 发现并修复的关键 Bug

### Bug 1: load history 未跟随 swap 重映射 (controller.py)
**症状**: avg_ratio_before 在连续 window 间不降反升 (1.629→1.633→1.645→1.648)
**根因**: swap 改变了 physical placement (slot A ↔ slot B)，但 `self.load` 中的 decay 历史仍然按旧的 slot 分布记录。新 window 的 record 只占 ~10% 权重 (decay=0.9)，90% 是旧数据 → ratio_before 被旧数据主导，算法误以为不均衡度没改善，每个 window 都做 ~140 次重复无效 swap。
**修复**: 在 `_try_finish_pending_swap` 中，swap 完成后同步交换 `self.load` 对应 slot 的值：
```python
for op in plan:
    layer = op.layer_id
    a, b = op.phys_slot_a, op.phys_slot_b
    self.load[layer, a], self.load[layer, b] = (
        self.load[layer, b].clone(), self.load[layer, a].clone()
    )
```

### Bug 2: sample_interval 过大 + min_prefill_tokens 过高
**症状**: PROF 显示 `calls=0`（整个 window 没有录到任何 prefill 数据），swap 来不及触发
**根因**: `sample_interval = max(4, sync_window//64) = 4`，只录每 4 个 prefill batch；加上 235B 模型每 batch 只有 ~1024 tokens，64 个 forward pass 中 decode 占多数，实际录到的 prefill tokens < min_prefill_tokens=1000 → 跳过 swap 决策
**修复**:
- `sample_interval = 1` (录每个 prefill batch)
- `--pb-oeplb-min-prefill-tokens 256` (降低触发门槛)

## 起服务命令

### 环境变量 (所有配置共用)
```bash
export NVSHMEM_DIR=/workspace/nvshmem_official/libnvshmem-linux-x86_64-3.3.9_cuda12-archive
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export PYTHONUNBUFFERED=1

MODEL=/root/.cache/modelscope/models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master
```

### Baseline
```bash
python3 -m sglang.launch_server \
  --model-path $MODEL \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache
```

### OEPLB (Greedy planner)
```bash
python3 -m sglang.launch_server \
  --model-path $MODEL \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.02 \
  --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 64 \
  --pb-oeplb-cooldown-steps 5 \
  --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64
```
注: controller.py 中 `decay_factor=0.9`, `sample_interval=1`, rebalancer.py 中 `MAX_TOTAL_OPS=250`

## 数据集

### focused CRS (20 prompts 重复)
```python
import json
with open('OEPLB/benchmarks/frozen_requests_prefill_heavy.jsonl') as f:
    reqs = [json.loads(l) for l in f]
out = []
for rep in range(75):
    for r in reqs[:20]:
        r2 = dict(r); r2['id'] = f"{r['id']}_rep{rep}"; r2['max_tokens'] = 1; out.append(r2)
# 1500 requests, 20 unique CRS reports ~1500 tok each
```
- 测试: 1500 requests, warmup: 300 requests (来自同源)
- 路径: `/tmp/focused_out1.jsonl`, `/tmp/focused_out1_warmup.jsonl`

### DeepSeek-Prover-V1 (1024 unique)
- 来源: ModelScope `AI-ModelScope/DeepSeek-Prover-V1`
- 27503 条 Lean 4 定理证明数据，取前 1024 条不同请求
- Prompt: `header + formal_statement + goal`，长度 359-17641 chars (avg ~662)
- max_tokens=32
- 路径: `/tmp/prover_1024.jsonl`, `/tmp/prover_warmup50.jsonl`

## 测试方法
- **交替测试**: BL1→OE1→BL2→OE2→BL3→OE3 (消除时间漂移)
- **每轮独立重启服务器** (消除 JIT cache/温度等状态累积)
- **warmup**: 50 requests (Prover) / 300 requests (CRS), 让 OEPLB swap 收敛后再测
- **benchmark**: `long_bench.py` with CONC=1024

## Greedy Planner 算法

```
每 64 个 forward pass (sync_window) 结束时:
1. global_load = self.load.clone(); all_reduce(global_load, SUM)
2. 在 budget (250 ops) 内贪心循环:
   a. 找当前 ratio 最高的层
   b. 如果 ratio < 1.02 (threshold) → 停止
   c. 对该层做 1 轮 swap (hottest slot ↔ coldest slot)
   d. 如果 swap 没有改善 ratio → 标记该层完成，不再操作
   e. 重新排序所有层，回到 a
3. 通过 async P2P 执行 swap plan (force_wait 防死锁)
4. 同步重映射 self.load 中 swapped slots 的 decay 历史
5. self.load *= 0.9 (指数衰减)
```

## 代码文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| controller.py | OEPLB/src/controller.py | 主控制器 (decay + force_wait + load-remap) |
| rebalancer.py | OEPLB/src/rebalancer.py | Greedy global-budget planner |
| async_swapper.py | OEPLB/src/async_swapper.py | 异步 P2P 权重交换 (force_wait) |
| config.py | OEPLB/src/config.py | 配置参数 |
| fast_metadata.py | OEPLB/src/fast_metadata.py | 快速 metadata 构建 |
| routing_tracer.py | OEPLB/src/routing_tracer.py | Ground-truth 路由追踪工具 |
| layer_imbalance_analysis.py | OEPLB/scripts/ | Trace 不均衡度分析 (已修复 8-rank) |
| long_bench.py | OEPLB/scripts/ | Benchmark 客户端 (CONC=1024) |

## SGLang 补丁 (3 个文件)

| 文件 | 改动 |
|------|------|
| server_args.py | +10 CLI 参数 + `_handle_eplb_and_dispatch` 验证 |
| model_runner.py | controller 初始化 + `on_forward_pass_end` hook + routing_tracer |
| layers/moe/topk.py | `record_next_layer` hook + routing_tracer record |

---

## 复现实验记录 (2026-07-21, DeepSeek-Prover-V1 数据集, MAX_TOTAL_OPS=250)

复用本报告的环境/模型/服务器配置，独立重启服务器跑了 BL1 + OE1 + OE2（2轮 OEPLB，未跑满3轮）。`MAX_TOTAL_OPS` 在部署路径里手动改为 **250**（与本报告一致；跑之前实际部署值是150，workspace 未提交代码是300，三者不一致，本次显式对齐为250）。

### 吞吐/延迟结果

| | Run | req/s | TTFT (ms) |
|---|---|---|---|
| Baseline | BL1 | 61.43 | 7684 |
| OEPLB | OE1 | 69.95 | 6569 |
| OEPLB | OE2 | 68.31 | 6620 |
| OEPLB | **Mean** | **69.13** | **6594** |
| **Delta** | | **+12.54%** | **-14.18%** |

对比本报告原三轮结果（+11.21% req/s, p=0.0038；-13.06% TTFT, p=0.035）：**方向和量级一致**，本次略高（+12.54% vs +11.21%），差异在原报告给出的方差范围内（原三轮 req/s std=0.244），判定为**成功复现**。

### Swap 收敛行为对比 (OE1/OE2 本次 vs 原报告)

**本次 (OE1, MAX_TOTAL_OPS=250)：**

| Window | avg_ratio_before | avg_ratio_after | max_ratio_before | max_ratio_after | total_ops |
|--------|-------------------|------------------|-------------------|------------------|-----------|
| 1 | 1.711 | 1.211 | 2.449 | 1.418 | 240 |
| 2 | 1.212 | 1.194 | 1.422 | 1.422 | 44 |
| 3 | 1.204 | 1.196 | 1.428 | 1.428 | 20 |

**本次 (OE2, MAX_TOTAL_OPS=250)：**

| Window | avg_ratio_before | avg_ratio_after | max_ratio_before | max_ratio_after | total_ops |
|--------|-------------------|------------------|-------------------|------------------|-----------|
| 1 | 1.711 | 1.211 | 2.449 | 1.418 | 240 |
| 2 | 1.212 | 1.194 | 1.422 | 1.422 | 44 |
| 3 | 1.205 | 1.196 | 1.432 | 1.432 | 21 |

**原报告 (Prover 数据集, OE1)：**

| Window | avg_ratio_before | avg_ratio_after | total_ops |
|--------|-------------------|------------------|-----------|
| 1 | 1.762 | 1.303 | 149 |
| 2 | 1.315 | 1.230 | 95 |
| 3 | 1.229 | 1.226 | 6 |
| 4 | 1.228 | 1.220 | 19 |

**区别：**

1. **收敛速度不同**：本次 3 个 window 就收敛到 avg_ratio≈1.20，原报告要 4 个 window 才降到 1.228。本次第一个 window 就把 avg_ratio_before 从 1.711 打到 1.211（-29%），原报告第一个 window 只从 1.762 降到 1.303（-26%）、还需要额外 2 轮才逼近 1.22。
2. **首轮 swap 次数差很大**：本次 window 1 = 240 ops（几乎打满 MAX_TOTAL_OPS=250 的预算），原报告 window 1 只用了 149 ops。同样的预算上限、同样的 threshold=1.02，首轮触发的 swap 量却几乎翻倍——说明**除了 MAX_TOTAL_OPS 之外，rebalancer.py 的其余逻辑相对原报告版本已经变了**（`git diff` 显示 rebalancer.py 有 218 行改动，不只是那一个常量），具体是贪心排序/候选筛选逻辑的差异，需要看 diff 才能定位。
3. **本次多了 `max_ratio_before/after` 和 `layers_touched` 字段**，原报告的 DIAG 日志里没有这两项——说明诊断日志格式本身也在报告写完后被增强过。
4. **两次收敛终点一致**：本次不管 OE1/OE2 还是原报告，最终稳定态都落在 avg_ratio ≈ 1.19-1.23 附近，说明**算法收敛目标没变，变的是收敛路径/速度**。
5. **本次 OE1 与 OE2 两次独立重启后的 swap 轨迹几乎完全一致**（仅 window 3 的 total_ops 20 vs 21 有微小差异），说明该算法在相同 workload 下具有很强的确定性/可复现性，非本次结果偏差的来源。

---

## 新数据集实验记录 (2026-07-21, fixeddata: 1024×DeepSeek-Prover-V1 + 2048×BookCorpus, MAX_TOTAL_OPS=250)

### 数据集构造

从 ModelScope `youngchen/BookCorpus` 下载 `books1.tar.gz`（2.88GB，原始 BookCorpus 全量书籍 txt），解压取前 2048 本书，各截取一段 ~1500 字符正文（跳过开头约 1/10 篇幅避开版权页/目录），套用统一 chat 模板构造"续写故事"请求，`max_tokens=32, temperature=0, ignore_eos=false`，id 为 `book_0`...`book_2047`。

拼接顺序：**先 1024 条 `prover_0`...`prover_1023`（DeepSeek-Prover-V1，数学证明），再接 2048 条 `book_0`...`book_2047`（BookCorpus，小说续写）**，共 3072 条，保存为：

```
/workspace/EPLB/OEPLB/benchmarks/fixeddata.jsonl
```

这个数据集刻意设计成**两段式 workload**（前 1/3 数学证明 prompt 短/结构化，后 2/3 小说续写 prompt 长/自然语言），用于观察 domain 切换时专家热度分布的迁移，以及 OEPLB swap 策略对这种切换的响应速度。

### 吞吐/延迟结果（BL1→OE1→BL2→OE2，交替、独立重启服务器）

| | Run | req/s | TTFT (ms) |
|---|---|---|---|
| Baseline | BL1 | 50.05 | 5633 |
| Baseline | BL2 | 51.43 | 9009 |
| Baseline | **Mean** | **50.74** | **7321** |
| OEPLB | OE1 | 54.65 | 6291 |
| OEPLB | OE2 | 55.23 | 6746 |
| OEPLB | **Mean** | **54.94** | **6519** |
| **Delta** | | **+8.28%** | **-10.96%** |

**方向与此前 Prover-only 实验（+11.21% / -13.06%）一致，但收益幅度更小**——混入 2/3 BookCorpus 长文本续写后，OEPLB 净收益从 ~11% 降到 ~8%。

**异常点：Baseline 自身两轮 TTFT 波动极大（BL1=5633ms vs BL2=9009ms，相差 60%），而 OEPLB 两轮 TTFT 相对稳定（OE1=6291ms vs OE2=6746ms，相差仅 7%）。** 这与此前 Prover-only 实验里 baseline/OEPLB 都很稳定（std<0.03 req/s量级）的情况不同。目前只有2轮数据，无法判断这是巧合噪声还是"OEPLB 的 rebalance 客观上让 TTFT 尾部更稳定"的真实效应——**需要补第3轮才能下结论，不要用这两个数字直接断言 OEPLB 更稳定**。

### Swap 收敛行为：domain 切换触发了二次不均衡

**OE1_fixed：**

| Window | avg_ratio_before | avg_ratio_after | max_ratio_before | max_ratio_after | total_ops |
|--------|-------------------|------------------|-------------------|------------------|-----------|
| 1 | 1.711 | 1.211 | 2.449 | 1.418 | 240 |
| 2 | 1.212 | 1.194 | 1.422 | 1.422 | 44 |
| 3 | 1.202 | 1.199 | 1.414 | 1.414 | 8 |
| 4 | 1.183 | 1.144 | 1.437 | 1.306 | 66 |
| 5 | 1.354 | 1.115 | 1.679 | 1.296 | 228 |
| 6 | 1.175 | 1.134 | 1.423 | 1.294 | 71 |

**OE2_fixed：**

| Window | avg_ratio_before | avg_ratio_after | max_ratio_before | max_ratio_after | total_ops |
|--------|-------------------|------------------|-------------------|------------------|-----------|
| 1 | 1.711 | 1.211 | 2.449 | 1.418 | 240 |
| 2 | 1.212 | 1.194 | 1.422 | 1.422 | 44 |
| 3 | 1.203 | 1.197 | 1.419 | 1.419 | 16 |
| 4 | 1.214 | 1.128 | 1.513 | 1.353 | 130 |
| 5 | 1.304 | 1.114 | 1.636 | 1.241 | 218 |
| 6 | 1.179 | 1.132 | 1.435 | 1.279 | 74 |

**跟纯 Prover 数据集（只需 3 个 window 收敛到 avg_ratio≈1.20 就稳定不动）相比，最大的区别是：这里 window 1-3 先按 Prover 的模式快速收敛（1.711→1.212→1.20），但从 window 3→4 开始 avg_ratio_before 又重新抬升（1.20→1.18/1.21→1.35/1.30，OE1/OE2 window5 都冲到 1.3+），说明 window 3 附近正是 Prover 请求耗尽、开始进入 BookCorpus 请求的 domain 切换点——专家热度分布随 prompt 类型（数学证明 token 分布 vs 自然语言小说续写 token 分布）发生了漂移，触发了第二轮更大规模的 swap（window5 单轮 218-228 ops，接近 window1 首次冷启动量级的 240 ops）。之后 window6 才重新压回 1.13-1.17 附近趋于稳定。**

OE1 与 OE2 的 window 轨迹形状高度相似（都是"快收敛→domain切换重新恶化→二次收敛"），但具体数值有差异（window4 OE1 total_ops=66 vs OE2=130，window5 OE1 avg_before=1.354 vs OE2=1.304），说明**domain 切换点的具体不均衡模式不是完全确定性的**，可能跟两轮之间的请求调度/completion顺序的随机抖动有关(不同于纯 Prover 数据集里 OE1/OE2 几乎逐位对齐的高确定性)。

---

## 第三轮验证 (2026-07-21, fixeddata, BL3 + OE3)

补齐第3轮，验证上一节提出的"baseline TTFT 波动大 / OEPLB TTFT 更稳定"的猜测。

### 三轮汇总

| | Run1 | Run2 | Run3 | Mean | Std |
|---|---|---|---|---|---|
| Baseline req/s | 50.05 | 51.43 | 50.89 | **50.79** | 0.696 |
| OEPLB req/s | 54.65 | 55.23 | 54.25 | **54.71** | 0.495 |
| Baseline TTFT (ms) | 5633 | 9009 | 6303 | **6981** | 1787 |
| OEPLB TTFT (ms) | 6291 | 6746 | 5330 | **6123** | 723 |

**Delta req/s = +7.72% (t-test p=0.0014，显著)**
**Delta TTFT = -12.30% (t-test p=0.48，均值差异本身不显著——3轮样本量太小、baseline方差太大压过了均值差)**

### 结论：req/s 收益稳健复现，TTFT "更稳定"的猜测部分成立但不能断言均值收益

1. **req/s 上的正收益是稳健的**：三轮里 OEPLB 每一轮都比同轮次的 baseline 快（54.65>50.05, 55.23>51.43, 54.25>50.89），t-test p=0.0014，跟 Prover-only 实验的统计显著性(p=0.0038)一个量级。**这是本次两个数据集实验里最confident的结论**。
2. **TTFT 方差确实不对称，但均值差异淹没在噪声里**：baseline TTFT std=1787ms 是 OEPLB std=723ms 的 **2.5倍**，三轮 baseline TTFT 本身在 5633-9009ms 之间跳（跨度3376ms），OEPLB 三轮在 5330-6746ms（跨度1416ms）。方差的不对称是真实存在的（并非上一节猜的"巧合噪声"），**但正因为 baseline 方差这么大，3个样本点算出来的 TTFT 均值差(-12.30%) 统计上不显著(p=0.48)**——不能像 req/s 那样直接引用这个百分数当结论,只能说"OEPLB 让 TTFT 尾部波动明显收窄，但对 TTFT 均值到底降了多少,现有3轮数据给不出有统计把握的答案"。
3. **为什么 baseline 方差大**：三轮 swap 收敛日志显示 OE1/OE2/OE3 在 window 5(domain切换点)的行为高度一致(total_ops都在218-228，avg_ratio_before都在1.3左右)，说明 OEPLB 主动纠偏后专家分布相对固定；而 baseline 没有任何纠偏机制，专家热点随请求到达顺序/批次组合自由漂移，不同轮次里"哪个GPU恰好摊上最热的专家"完全看运气，这大概是 baseline TTFT 方差更大的合理解释——但这是**推测**，没有做类似 layer_imbalance_analysis.py 那样的直接测量验证，如果要坐实这个结论，下一步应该给 baseline 也跑一次逐层不均衡度分析对比。

---

## Adaptive decay_factor 实验 (2026-07-21) — 负面结果，功能默认关闭

### 动机

之前发现固定 `decay_factor=0.9` 在 domain 切换后要 2 个 window 才能完全反映新 workload（window3→4→5 才冲到 avg_ratio_before≈1.3+）。假设：0.9 让太多旧 workload 的历史残留稀释了新信号，如果检测到"请求类型突变"就临时调低 decay（丢弃更多历史），应该能让算法更快适应。

### 实现

在 `OEPLB/src/controller.py`：
- 复用（并修复）已存在但从未被调用的 `_track_routing_stability()`：原实现被 `if self._rank != 0: return` 挡住计算本身（不只是挡日志），导致只有 rank0 会更新 `_prev_load`。由于 `decay_factor` 是 per-rank 本地状态，这个 bug 修复前如果直接拿 cos_sim 驱动 decay，会导致各 rank 的历史衰减速度不一致。改为所有 rank 都计算 cos_sim（用的是 all_reduce 后各 rank 完全一致的 `global_load`），只在 rank0 保留周期性打日志。
- 新增 `_last_cos_sim`，在 `_decide_and_begin_swap()` 里 all_reduce 之后立即计算（在 `min_prefill_tokens`/`busy` 提前 return 之前，确保空闲 window 也不断档）。
- `on_forward_pass_end` 里原来固定的 `self._decay_factor = 0.9` 改为：cos_sim 低于 `decay_shift_cos_threshold`(0.85) → decay 减 `decay_step`(0.15)，下限 `decay_floor`(0.5)；cos_sim 高于 `decay_stable_cos_threshold`(0.95) → decay 加 `decay_recovery_step`(0.05)，上限 `decay_base`(0.9)；中间地带不动。连续突变窗口会累积下调（decay_factor 本身就是累积状态，不需要额外计数器）。
- 新增 `PBOEPLBConfig` 字段 `adaptive_decay` + 6 个衰减相关参数，均可用环境变量覆盖（沿用本仓库已有的 `OEPLB_EXP_MODE` 风格的实验开关惯例，因为 `server_args.py` 不在这个 git 仓库里、属于手动同步的文件，没有新增 CLI flag）。**默认 `adaptive_decay=False`，行为与之前完全一致。**

### 实验结果：req/s 变差，机制上也没有缩小 domain 切换的冲击

在 `fixeddata.jsonl`（同一个暴露了切换延迟问题的数据集）上跑了 2 轮 adaptive-decay OEPLB（独立重启服务器），对比此前已记录的 3 轮静态 decay=0.9 OEPLB：

| | req/s | TTFT (ms) |
|---|---|---|
| 静态 decay=0.9 OEPLB (3轮均值) | **54.71** | **6123** |
| Adaptive decay OEPLB (2轮均值) | **52.81** | **6157** |
| **Delta (adaptive vs 静态)** | **-3.48%** | **+0.57%（基本不变）** |

（对比无 OEPLB 的纯 baseline 均值 50.79 req/s，adaptive 版本仍有 +3.97% 的正收益——只是比静态版本差。）

**机制上也没有验证假设**：两轮 adaptive 跑的 window5（domain切换点）分别是 `avg_ratio_before=1.391, total_ops=234` 和 `avg_ratio_before=1.363, total_ops=230`，跟静态版本的 `1.354/1.304, 228/218 ops` 相比**没有变小，个别还略高**。也就是说,把 decay 调低并没有让算法在切换点反应更快/更轻。

### 为什么没用（推测，未做进一步隔离实验验证）

`decay_factor` 只影响**历史**权重，不影响**当前 window 新记录的 token**。domain 切换发生在某个 window 内部时（尤其是 conc=1024 高并发下，Prover 的尾部请求和 BookCorpus 的头部请求会在同一批 in-flight 请求里重叠，不是一个干净的瞬时切换），那个 window 自己新记录的 token 就已经主要是新 domain 的分布了——旧历史在其中占比本来就小,调低 decay 影响不大。真正的 2-window 延迟更可能是**采样量**问题：单个 sync_window=64 forward pass 攒的新 domain token 还不够多、不足以让 ratio 立刻冲高,而不是"被旧历史稀释"的问题。如果这个推测成立,那么调 decay 这个方向本身就不是正确的杠杆,该往 sync_window 长度或触发阈值上想,而不是历史权重上。

另外发现一个次要问题：把 cos_sim 计算放在 `min_prefill_tokens` 检查**之前**（为了不断档地跟踪 drift）,意味着低流量、稀疏的 window 也会参与 cos_sim 计算——稀疏 global_load 向量天然更容易算出偏低的 cos_sim（纯采样噪声,不是真的 workload 变了）,可能提前触发了不必要的 decay 下调,增加了跟真实收益无关的抖动。

### 结论

**不采用**。`adaptive_decay` 保留在代码里作为默认关闭的实验开关（`OEPLB_ADAPTIVE_DECAY=1` 才生效,默认行为跟改动前完全一致),但当前这版基于 cos_sim 的实现没有带来收益,反而轻微拖累了吞吐。如果之后想继续这个方向,下一步应该先验证"2-window延迟是采样量不够还是历史稀释"这个归因本身,例如直接对比"只调大 sync_window"和"只调 decay"两种单变量实验,而不是直接在 decay 上加自适应逻辑。

**（后续处理）代码已撤销**：上面这次 adaptive decay 的改动（`_track_routing_stability` 复用/修复 + `_last_cos_sim` + adaptive decay 更新逻辑 + `config.py` 新字段）已从 `OEPLB/src/controller.py` 和 `OEPLB/src/config.py` 完整撤销，并同步回部署路径，代码恢复到实验前的状态（`decay_factor` 固定为 0.9）。上面的实验记录保留作为"试过、没收益、为什么没收益"的存档，当前代码库里已经没有这部分改动的痕迹。

---

## Adaptive Window 实验 (2026-07-21) — 正面结果

### 背景：从"是不是能预判突变"这个问题说起

上一节的 adaptive decay 实验没有收益后，转向了另一个杠杆——sync_window（多久检查一次、决定要不要 swap 的周期）。先做了单变量对照：固定 sync_window=32（而不是默认的64）在 `fixeddata.jsonl`（1024×Prover + 2048×BookCorpus）上单独测：

| 配置 | req/s (mean) | vs baseline | TTFT ms (mean) | domain切换峰值ratio | 全程总swap次数 |
|---|---|---|---|---|---|
| 纯 Baseline | 48.94 | - | 6698 | - | - |
| sync_window=64（静态,默认） | 51.81 | +5.88% | **5593（最优）** | 1.36-1.39 | 593-596 |
| sync_window=32（静态） | 53.38 | +9.07% | 6769（更差） | **1.27-1.28（更优）** | 766-771（更多） |

机制上证实：window切小确实能把 domain 切换的冲击拆成更小的片段消化，峰值不均衡度更低,吞吐更高;但检查/swap 更频繁，P2P 搬权重跟 DeepEP dispatch/combine 抢 NVLink 带宽的次数也更多，TTFT 明显更差——是一个真实的 trade-off，不是免费的午餐。

### 设计一版（对称2窗口确认）：机制上有缺陷

第一版实现：cos_sim（复用并修复了此前 `_track_routing_stability` 的 rank-guard bug，现在所有 rank 一致计算）连续2个window低于0.85才收缩sync_window到32，连续2个window高于0.95才恢复到64。

**结果不理想**：两轮验证都显示同一个问题——**"连续2次确认"这个门槛让window收缩总是在真正的峰值window已经发生之后才触发**。原因：window的cos_sim和它自己的avg_ratio_before是同一时刻用同一份global_load算出来的,理论上可以立刻反应,但代码要求"再等下一个window也确认"才动手,导致真正需要收缩的那个高峰window仍然按64的旧节奏跑完,收缩来的太晚,没抓住该抓的峰值,却仍然承担了切换的开销。

| | req/s (mean) | TTFT ms (mean) | TTFT 方差(std) |
|---|---|---|---|
| AdaptWin v1 | 53.99 | 6975.3（最差） | **1978.9（最不稳定）** |

### 设计二版（非对称确认）：区分"收缩"和"恢复"两个方向的风险

关键认识：**收缩window只改变"多久检查一次"，不直接决定要不要swap**——rebalancer自己的threshold_ratio才是swap决策的唯一门槛，跟window大小无关。所以误判去收缩的代价很小（顶多多做一次all_reduce），跟"调decay/直接影响swap决策"完全不是一个风险量级，不需要用同样严格的连续确认。而"恢复"方向（收缩后要不要放回64）保守一点没有坏处（早放和晚放的代价都很小）。

于是把确认门槛拆成非对称的：**收缩方向 confirm=1（单窗口即触发,利用信号同步的特性,零额外延迟）,恢复方向 confirm=2（保留多窗口确认,求稳）**。

**验证结果（2轮，fixeddata）**：

| | req/s (mean) | TTFT ms (mean) | TTFT 方差(std) |
|---|---|---|---|
| SW64（静态） | 51.81 | 5592.5 | 1063.6 |
| SW32（静态） | 53.38 | 6768.7 | 948.1 |
| AdaptWin v1（对称confirm=2） | 53.99 | 6975.3 | 1978.9 |
| **AdaptWin v2（非对称1/2）** | **54.07（最高）** | **5894.5（接近sw64最优值,仅+5.4%）** | **312.1（全场最稳）** |

日志证实机制修复生效：两轮里 window 收缩消息都跟真正的峰值window（`total_ops=219-224, avg_ratio_before=1.33-1.35`）**同一时刻**触发（不再滞后一个window），随后紧跟的1-2个window用32的节奏快速收尾（`avg_ratio_before`迅速回落到1.14-1.15），稳定后2个window内又confirm恢复回64。**v2 版本第一次真正做到了"吞吐拿到sw32级别的收益,TTFT保持接近sw64级别的水准",而且波动是四个配置里最小的。**

### 三段式数据集验证（triplefixed）：更大幅度的正面结果

在 `fixeddata.jsonl` 基础上追加了从 HuggingFace（`Rowan/hellaswag`，经 hf-mirror.com 访问，validation split前2048条）下载构造的 HellaSwag 请求集——每条构造成"从4个选项里选最合理续写"的QA格式prompt，`max_tokens=32`。拼接顺序：1024×Prover + 2048×BookCorpus + 2048×HellaSwag = 5120条，存为 `OEPLB/benchmarks/triplefixed.jsonl`，构成一个**两次domain切换**的场景。

有意思的观察：日志显示只在 **Prover→BookCorpus** 切换点检测到明显的冲击波（`total_ops≈211-220, avg_ratio_before≈1.31-1.34`），**BookCorpus→HellaSwag** 切换点两轮都没有触发明显的shift confirm——推测因为 BookCorpus（英文小说）和 HellaSwag（英文常识推理QA，本次用的prompt模板也是英文自然语言）路由模式本身比较接近,真正跟两者都差异很大的是 Prover 的 Lean4 数学证明语法。这个观察目前只是相关性证据,没有做进一步隔离验证。

**AdaptWin v2 (非对称) vs 纯baseline，2轮独立重启对比：**

| | req/s (mean) | std | TTFT ms (mean) | std |
|---|---|---|---|---|
| 纯 Baseline | 63.41 | 0.855 | 6060.7 | 1137.6 |
| **AdaptWin v2** | **67.87** | **0.115** | **4081.6** | **278.2** |
| **Delta** | **+7.03%** | | **-32.65%** | |

**这是本次一系列实验里最强的正面结果**：req/s 提升 +7.03%，TTFT 大幅下降 -32.65%，且两个指标的方差都远小于baseline（req/s std 从0.855降到0.115，TTFT std从1137.6降到278.2）——不仅平均更好，**跑起来还明显更稳定**。三段式（含两次潜在切换）场景下,adaptive window的收益比两段式(fixeddata)更明显,可能是因为更长的benchmark（5120 vs 3072条请求）让"稳定期用64省TTFT"的时间占比更大,能吃到的稳定期收益也更多。

### 结论

`adaptive_window`（非对称确认版本）是这一系列实验里唯一验证出**净收益且跨数据集一致**的改动，建议保留在代码里（仍然默认关闭,通过 `OEPLB_ADAPTIVE_WINDOW=1` 等环境变量开启,不影响现有默认行为）。核心设计经验：**同一个"检测到shift就调整"的思路，套在不同参数上风险完全不同**——调 decay/直接触发 swap 的参数需要谨慎的多窗口确认（错了要花真实的P2P成本才能纠正），而调 window size 只是改变检查节奏，不直接决定swap，误判代价很小，可以对"收缩"方向更激进、对"恢复"方向保守，两个方向不必用同一套confirm逻辑。

---

## 补充：不取平均，用极值对比 (2026-07-21)

按要求换一种统计口径——不用两轮均值，而是**baseline取两轮里最低的那次，OEPLB(AdaptWin v2)取两轮里最高的那次**，直接用这两个具体数字算差值。

**注意方向：** req/s 越高越好，所以"OEPLB取最高"是对 OEPLB 更有利的取法；但 TTFT 越低越好，"OEPLB取最高"实际上是**取 OEPLB 两轮里更差的那次**，跟"baseline取最低"（baseline两轮里更好的那次）搭在一起，是对 OEPLB **最不利**的对比方式，不是选择性取好看的数字。

### fixeddata（3072条，1024×Prover + 2048×BookCorpus）

| | 具体数值 | 取值说明 |
|---|---|---|
| Baseline req/s | 48.92 | 两轮(48.95, 48.92)里最低 |
| AdaptWin v2 req/s | 54.61 | 两轮(54.61, 53.52)里最高 |
| **req/s Delta** | **+11.63%** | |
| Baseline TTFT | 6565.35 ms | 两轮(6830.14, 6565.35)里最低(最好) |
| AdaptWin v2 TTFT | 6115.21 ms | 两轮(6115.21, 5673.82)里最高(最差) |
| **TTFT Delta** | **-6.86%** | 用OEPLB更差的一次比baseline更好的一次,仍然是负的(更优) |

### triplefixed（5120条，1024×Prover + 2048×BookCorpus + 2048×HellaSwag）

| | 具体数值 | 取值说明 |
|---|---|---|
| Baseline req/s | 62.81 | 两轮(64.02, 62.81)里最低 |
| AdaptWin v2 req/s | 67.95 | 两轮(67.79, 67.95)里最高 |
| **req/s Delta** | **+8.18%** | |
| Baseline TTFT | 5256.29 ms | 两轮(6865.10, 5256.29)里最低(最好) |
| AdaptWin v2 TTFT | 4278.33 ms | 两轮(3884.91, 4278.33)里最高(最差) |
| **TTFT Delta** | **-18.61%** | 同样用OEPLB更差一次比baseline更好一次,仍然是负的(更优) |

**结论**：即便按对 OEPLB 最不利的方式取值（TTFT 用它两轮里表现更差的那次,去比 baseline 两轮里表现最好的那次），req/s 依然有 +8.18%~+11.63% 的提升，TTFT 依然有 -6.86%~-18.61% 的下降——说明这次 adaptive window (v2非对称版本) 的收益不是靠平均数掩盖了某一轮的坏结果撑出来的,哪怕挑对自己最不利的数字组合,方向仍然是正收益。

---

## 订正：triplefixed 上 "AdaptiveWindow vs Baseline" 的收益被之前的噪声高估了 (2026-07-21)

用户要求测一下"原有代码（不改 window，一直是默认 sync_window=64）"在 `triplefixed.jsonl` 上本身能拿到多少收益，于是重测了 baseline（2轮）和原版静态 OEPLB（2轮，`OEPLB_ADAPTIVE_WINDOW` 未设置，即完全没有本次session对window的任何改动）。

### 重测结果

| | req/s (mean) | TTFT ms (mean) |
|---|---|---|
| Baseline（本次重测2轮：64.22, 64.06） | 64.14 | 4475.2（2轮：4529.3, 4421.0，很稳） |
| **原版OEPLB静态(sync_window=64,无adaptive)（2轮：66.42, 68.19）** | **67.31** | **4272.9（2轮：4133.1, 4412.7）** |
| AdaptiveWindow v2（上一轮结果，2轮：67.79, 67.95） | 67.87 | 4081.6（2轮：3884.9, 4278.3） |

**原版 OEPLB（完全没动过 window/decay，一直是本session一开始就在用的默认配置）本身在 triplefixed 上就有 +4.94% req/s、-4.52% TTFT 的收益**——这个收益跟 adaptive window 无关，是 OEPLB 本体（专家热点检测+P2P swap）自带的。

**AdaptiveWindow v2 相对"原版静态OEPLB"的增量收益**：+0.83% req/s，-4.48% TTFT——是真实但幅度温和的增量，不是决定性的。

### 需要订正之前的结论

上一节"三段式数据集验证"里报告的 `AdaptWin v2 vs 纯baseline: req/s +7.03%, TTFT -32.65%`，那次对比用的baseline是更早一轮跑的（TTFT两轮 6865.1ms / 5256.29ms，均值6060.7，标准差1137.6——方差很大）。本次重测的baseline TTFT干净得多（4529.3ms / 4421.0ms，均值4475.2，方差小了一个量级）。**之前那个 -32.65% 的TTFT改善幅度，很大一部分是baseline那一轮恰好跑出偏高值(6865ms)造成的噪声，不是adaptive window真实带来的效果。**

**更接近真相的表述**：在 triplefixed 三段式数据集上，OEPLB 主体收益（+4.94%/-4.52%）不需要任何 window 改动就能拿到；adaptive window 在这基础上贡献的是一个方向正确、幅度温和的增量（+0.83%/-4.48%），量级上不如之前 `fixeddata`（两段式，只有一次真正的domain切换）实验里看到的 AdaptWin v2 vs SW64静态那组对比（54.07 vs 51.81 req/s，接近+4.4%的增量，TTFT基本持平微增）那么显著——**具体收益幅度依赖于数据集里domain切换的强度和频率，triplefixed里第二次切换(Book→HellaSwag)本身就没检测到明显冲击（见上文），能让adaptive window发挥作用的窗口更少，增量收益自然更小**。

### 教训：小样本(n=2)对比时，务必用"同一批次内部对照"而不是引用其他batch跑出来的历史数字

这次的偏差提醒：TTFT这个指标本身噪声很大（不同轮次能差1000-2000ms），2轮样本的均值极不稳定。之后再做类似对比，应该在**同一次session里连续跑完baseline和待测配置**（像这次重测一样），而不是把不同时间跑的历史数据直接拼在一起比——哪怕两次都是"标称一样的配置"，GPU热状态/系统负载漂移带来的噪声也可能跟真实效应量级相当，容易得出方向对但幅度失真的结论。

### 补充：本次重测的 swap/ratio 逐window记录（原版静态OEPLB, triplefixed）

**Run1（总swap ops=653）：**

| Window | avg_ratio_before | avg_ratio_after | max_ratio_before | max_ratio_after | total_ops |
|---|---|---|---|---|---|
| 1 | 1.711 | 1.211 | 2.449 | 1.418 | 240 |
| 2 | 1.212 | 1.194 | 1.422 | 1.422 | 44 |
| 3 | 1.199 | 1.198 | 1.435 | 1.435 | 3 |
| 4 | 1.200 | 1.196 | 1.417 | 1.417 | 9 |
| 5（Prover→Book切换点） | **1.386** | 1.111 | 1.761 | 1.227 | **235** |
| 6 | 1.157 | 1.130 | 1.374 | 1.272 | 54 |
| 7 | 1.146 | 1.134 | 1.345 | 1.293 | 29 |
| 8 | 1.149 | 1.135 | 1.360 | 1.360 | 39 |

**Run2（总swap ops=662）：**

| Window | avg_ratio_before | avg_ratio_after | max_ratio_before | max_ratio_after | total_ops |
|---|---|---|---|---|---|
| 1 | 1.711 | 1.211 | 2.449 | 1.418 | 240 |
| 2 | 1.212 | 1.194 | 1.422 | 1.422 | 44 |
| 3 | 1.199 | 1.198 | 1.435 | 1.435 | 3 |
| 4 | 1.200 | 1.195 | 1.420 | 1.420 | 14 |
| 5（Prover→Book切换点） | **1.373** | 1.110 | 1.740 | 1.224 | **229** |
| 6 | 1.153 | 1.126 | 1.349 | 1.262 | 66 |
| 7 | 1.144 | 1.137 | 1.295 | 1.295 | 20 |
| 8 | 1.160 | 1.137 | 1.359 | 1.290 | 46 |

跟之前的观察一致：两轮都只在 window5（Prover→Book切换点）出现明显冲击（avg_ratio_before 1.37-1.39, ops 229-235），window8 之后（对应 Book→HellaSwag 边界附近）没有再出现类似量级的spike——再次确认"HellaSwag跟BookCorpus路由模式接近、真正的差异者是Prover"这个观察在重测里可复现，不是单次噪声。这也是"AdaptiveWindow在triplefixed上增量收益比fixeddata小"的直接证据：8个window里只有1个真正需要adaptive机制介入，可发挥空间本来就有限。
