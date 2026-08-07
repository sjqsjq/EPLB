# OEPLB 系统设计文档

## 概述

OEPLB (Online Expert Placement Load Balancer) 是一个运行在SGLang推理服务内部的MoE专家负载均衡器。它在推理服务运行时动态调整专家物理位置，降低GPU间负载不均衡，提升吞吐。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        SGLang Server                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Scheduler   │───▶│  ModelRunner  │───▶│  MoE Layer   │      │
│  │  (per DP)    │    │  (per TP/EP) │    │  (94 layers) │      │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘      │
│                             │                   │              │
│                     ┌───────┴────────┐  ┌───────┴────────┐     │
│                     │  OEPLB         │  │  topk.py       │     │
│                     │  Controller    │◀─│  select_experts │     │
│                     │                │  │  (routing hook) │     │
│                     └───────┬────────┘  └────────────────┘     │
│                             │                                   │
│                    ┌────────┴────────────────┐                 │
│                    │                         │                 │
│              ┌─────┴──────┐          ┌───────┴───────┐         │
│              │ Rebalancer │          │ AsyncSwap     │         │
│              │ (greedy+    │          │ Executor      │         │
│              │  adaptive) │─────────▶│ (P2P weight   │         │
│              └────────────┘          │  transfer)    │         │
│                                      └───────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## 核心数据流

### 1. 路由记录 (每个forward pass)

```
topk.py::select_experts()
  │
  ├─ MoE router输出topk_ids (物理slot空间, 已经过logical→physical转换)
  │
  └─▶ Controller.record_next_layer(topk_ids)
        │
        ├─ 仅prefill batch记录 (is_prefill && !is_idle)
        │
        └─ self.load[layer_id].scatter_add_(topk_ids)  ← O(1) GPU操作
              │
              └─ load是[num_layers, num_physical_experts]的累积计数器
                 每个rank独立维护自己的局部视角
```

### 2. 决策周期 (每sync_window=16个forward pass)

```
ModelRunner.forward()尾部
  │
  └─▶ Controller.on_forward_pass_end(forward_batch)
        │
        ├─ 步骤1: _try_finish_pending_swap()  ← 检查上一轮P2P是否完成
        │    └─ 如果完成: 应用shadow buffer→live weight拷贝, 更新p2l路由表
        │
        ├─ 步骤2: 计数器 _steps_since_last_check += 1
        │    └─ 如果 < sync_window(16): return (跳过)
        │
        ├─ 步骤3: _decide_and_begin_swap()
        │    │
        │    ├─ (a) all_reduce(load) → global_load (8 ranks求和)
        │    │
        │    ├─ (b) try_build_swap_plan(global_load, p2l)
        │    │    │
        │    │    └─▶ Rebalancer (见下方详细算法)
        │    │
        │    ├─ (c) if plan为空或ops<min_swap_ops: skip (stall检测)
        │    │
        │    └─ (d) AsyncSwapExecutor.begin(plan)  ← 异步发起P2P
        │
        └─ 步骤4: load *= decay_factor(0.5)  ← 快速衰减旧信号
```

### 3. Rebalancer贪心算法 (核心创新)

```
try_build_swap_plan(global_load, p2l)
  │
  ├─ 初始化: 计算每层的不均衡度 ratio = max_rank_load / avg_rank_load
  │
  └─ 循环 (最多max_total_ops=300次):
       │
       ├─ 选ratio最高的层 best_layer
       │
       ├─ 如果ratio < threshold(1.02): 全部收敛, 退出
       │
       ├─ 在该层内搜索有效swap:
       │    │
       │    ├─ ranks_by_load_desc = 按负载排序的rank列表(高→低)
       │    ├─ ranks_by_load_asc  = 按负载排序的rank列表(低→高)
       │    │
       │    └─ 遍历 (hot_rank, cold_rank) 组合:
       │         │
       │         ├─ 获取hot_rank上未尝试的slots (按load降序)
       │         ├─ 获取cold_rank上未尝试的slots (按load升序)
       │         │
       │         ├─ ★ ADAPTIVE PAIR SELECTION ★
       │         │    │
       │         │    ├─ gap = hot_rank_sum - cold_rank_sum
       │         │    │
       │         │    ├─ if hot_slot_load <= gap:
       │         │    │    选最热slot (贪心max-delta, 快速收敛)
       │         │    │
       │         │    └─ else (会overshoot):
       │         │         选load最接近 gap/2 的slot
       │         │         (精确均衡, 避免热点互换)
       │         │
       │         ├─ 模拟swap, 计算new_ratio
       │         │
       │         ├─ if new_ratio改善 > 0.0005: 接受swap
       │         │    └─ 记录到plan, 标记slots为tried
       │         │
       │         └─ else: 标记这对slots为tried, 继续搜索
       │
       └─ 如果该层所有pair都试过仍无效: 标记exhausted, 选下一层
```

### 4. 异步P2P权重搬运

```
AsyncSwapExecutor.begin(plan)
  │
  ├─ 在独立低优先级CUDA stream上:
  │    │
  │    ├─ 为每个SwapOp分配temp buffer
  │    │
  │    └─ batch_isend_irecv(p2p_ops)  ← 一次性发射所有P2P操作
  │         ├─ rank_a: isend(weight[slot_a]) → rank_b
  │         ├─ rank_a: irecv(temp)          ← rank_b
  │         ├─ rank_b: irecv(temp)         ← rank_a
  │         └─ rank_b: isend(weight[slot_b]) → rank_a
  │
  └─ 记录completion event (非阻塞)

AsyncSwapExecutor.try_finish()
  │
  ├─ 非阻塞: event.query() → 如果未完成: return None
  │
  └─ 完成: 将temp buffer拷贝回live weight, 返回plan
       │
       └─ Controller收到plan后:
            ├─ 更新p2l路由表 (从evolving new_p2l读取)
            ├─ 更新ExpertLocationMetadata
            └─ remap load历史 (swap对应slot的计数)
```

## 关键设计决策

### 1. decay_factor = 0.5 (快速衰减)

| decay | 3窗口后旧信号残留 | 效果 |
|---|---|---|
| 0 (清零) | 0% | 数据太少, 决策质量差 |
| 0.5 | 12.5% | ★ 快速切换+足够信号量 |
| 0.9 | 73% | 跨域污染严重 |

### 2. sync_window = 16

- 每窗口~6秒(16 forward passes)
- 域切换后反应延迟<6秒
- 冷启动修正(~250 ops)在第一个窗口完成

### 3. Adaptive Pair Selection

```
场景: hot_rank_sum=183000, cold_rank_sum=145000, gap=38000

旧策略(纯贪心):
  选hot_rank最热slot(load=46200)
  swap后: cold_rank变成191200 > 原hot → overshoot! → ratio变差
  → 算法报exhausted, 实际还有638个有效pair

新策略(adaptive):
  gap/2 = 19000
  选load≈19000的slot(而非46200)
  swap后: 两rank都接近avg → ratio改善 ✓
```

### 4. 仅prefill阶段记录

- prefill: 每个token的路由决策决定后续decode阶段的expert热点
- decode: 每次只生成1个token, expert负载天然均匀
- 只记录prefill = 提前预测decode阶段的热点, 而非事后纠偏

### 5. evolving p2l更新

```
for op in plan:
    cur_a = new_p2l[layer, slot_a]  ← 从不断更新的p2l读
    cur_b = new_p2l[layer, slot_b]
    new_p2l[layer, slot_a] = cur_b   ← 交换
    new_p2l[layer, slot_b] = cur_a
```
不使用op.logical_a/b(会stale), 而是从evolving状态实时读取, 支持多op作用同一层。

## 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| sync_window | 16 | 决策周期(forward passes) |
| threshold_ratio | 1.02 | 触发swap的不均衡度阈值 |
| decay_factor | 0.5 | 负载历史衰减系数 |
| max_total_ops | 300 | 单次决策最大swap数 |
| max_swaps_per_layer | 64 | 单层最大swap数 |
| min_swap_ops | 8 | 低于此数跳过(不值得P2P开销) |

## 性能总结

| 场景 | vs Baseline | vs EPLB | 达理论最优% |
|---|---|---|---|
| L256_O1 | +13.0% | +7.0pp | 96.4% |
| L512_O1 | +18.4% | +9.4pp | 97.6% |
| L1024_O1 | +15.4% | +7.8pp | — |
| 多域prefill-heavy | +10.6% | +4.3pp | — |
