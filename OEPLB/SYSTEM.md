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

---
## 最终配置参数详解

### 核心参数（config.py + server_args.py 已同步）

| 参数 | 默认值 | 含义 | 调优依据 |
|---|---|---|---|
| sync_window | 8 | 每8个forward pass做一次决策 | 长prompt最优(8步够统计);短prompt由adaptive自动涨到32-128 |
| decay_factor | 0.5 | 负载历史每窗口衰减50% | 3窗口后旧信号只剩12.5%,快速适应域切换 |
| threshold_ratio | 1.02 | 超过1.02的不均衡度才触发swap | 接近完美均衡,只在有意义的偏差时行动 |
| max_total_ops | 300 | 单次决策最多300对swap | 冷启动实际用~250,留余量 |
| max_swaps_per_layer | 64 | 单层最多64次swap | 128个slot,最多32对,64足够 |
| min_swap_ops | 8 | 低于8对跳过(不值得P2P开销) | all_reduce+P2P固定开销~1.3ms,8对swap才划算 |
| max_total_swap_layers | 94 | 最多动94层 | Qwen3-235B有94个MoE层 |
| min_prefill_tokens | 256 | 累积够256个token才开始决策 | 防止数据太少做坏决策 |

### Adaptive Window参数（opt-in, --pb-oeplb-adaptive-window）

实际运行的Adaptive Window v2逻辑（用ratio变化驱动，不是cos_sim）：
- ratio跳变>0.03 → 窗口减半(最低8)，快速响应域切换
- ratio变化<0.003连续3次 → 窗口翻倍(最高128)，省all_reduce开销
- ratio波动0.003-0.03连续3次 → 窗口翻倍，获取更多统计量

| 参数 | 默认值 | 含义 |
|---|---|---|
| window_floor | 32 | cos_sim驱动的窗口下限(被v2的ratio逻辑覆盖) |
| window_shift_cos_threshold | 0.85 | cos_sim<0.85判定为域切换(v1逻辑,仅adaptive_window=True时用) |
| window_stable_cos_threshold | 0.95 | cos_sim>0.95判定为稳定(v1逻辑) |
| window_shift_confirm_windows | 1 | 1次低cos_sim就缩窗口 |
| window_stable_confirm_windows | 2 | 2次高cos_sim才涨窗口 |

### 推荐使用方式

```bash
# 最简配置(适合已知workload类型的场景):
--enable-pb-oeplb \
--pb-oeplb-threshold-ratio 1.02 \
--pb-oeplb-sync-window 8 \
--pb-oeplb-max-total-ops 300 \
--pb-oeplb-decay-factor 0.5
# 其余参数用默认值

# 自适应配置(适合混合workload的生产环境):
--enable-pb-oeplb \
--pb-oeplb-threshold-ratio 1.02 \
--pb-oeplb-sync-window 8 \
--pb-oeplb-max-total-ops 300 \
--pb-oeplb-decay-factor 0.5 \
--pb-oeplb-adaptive-window \
--pb-oeplb-window-floor 8
```

### 参数间的关系图

```
sync_window(8) ──────────────────────────────────────┐
  │ 每8个forward做一次决策                           │
  ▼                                                  │
decay_factor(0.5) ──── 每窗口衰减50%历史 ──────────┤
  │ 3窗口后旧信号12.5%                               │
  ▼                                                  │
threshold_ratio(1.02) ── 触发swap的阈值 ────────────┤
  │ ratio>1.02才动手                                │
  ▼                                                  │
adaptive pair selection ── 选slot策略 ───────────────┤
  │ gap大→贪心max-delta(快)                         │
  │ gap小→gap/2精确(避免overshoot)                  │
  ▼                                                  │
min_swap_ops(8) ──────── 跳过无效swap ──────────────┤
  │ ops<8不值得P2P开销                               │
  ▼                                                  │
adaptive window(可选) ── 自动调整窗口大小 ──────────┘
  │ 收敛→涨(省开销), 跳变→缩(快响应)
  │ 自动适应: 长prompt→sw=8, 短prompt→sw=32-128
```

---
## GPU内存开销对比

| 配置 | 内存(GB/卡) | vs Baseline | 说明 |
|---|---|---|---|
| Baseline(无均衡器) | 88.7 | — | auto模式, CUDA graph |
| PB-OEPLB(无冗余) | 88.7 | +0% | 仅多几MB的load张量 |
| EPLB(16冗余) | 79.8 | -10% | 总量更低是因为禁用CUDA graph省了~10GB graph buffer,但加了16冗余专家~2GB;净效果是O=256吞吐-68% |

**关键洞察**: EPLB的内存看起来更低,但实际上是因为禁用CUDA graph省了graph buffer内存,不是因为更高效——这个"省内存"的代价是decode吞吐暴跌68%。
