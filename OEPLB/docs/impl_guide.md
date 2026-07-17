# PB-OEPLB v0.1 (Swap-only) 实施指导文档

> **目标读者**：编码 AI（已配好 SGLang 环境、能启动 Qwen3-30B-A3B-FP8）
> **交付目标**：一个可跑通、可打点、可对比的最小可用实现（MVP），只做 swap，不做 replicate / migrate 相关的重型操作
> **版本**：v0.1（Swap-only, single-node, EP-only）
> **日期**：2026-07-09

---

## 0. 一句话说清主要目的

**我们要在 SGLang 里做一个"在线的、事件驱动的 expert placement 微调器"**：

> 每次 scheduler 从 prefill 切换到 decode 的边界，读一读刚才这段 prefill 里各个 expert 被 dispatch 了多少 token，如果发现某些 GPU 明显偏忙（imbalance ratio 超阈值），就在**当前 EP 布局上做少量的 expert 交换（swap）**——把某张过载 GPU 上的一个热门 expert 和某张空闲 GPU 上的一个冷门 expert 对调。对调之后，**后续 prefill 和 decode 都用新布局**，让整个 EP world 的 MoE 计算更均衡。

**它和已有工作的关系**：
- 不替代 DeepSeek EPLB（长周期全量重排），只填 EPLB 冷窗口的空
- 灵感来自 DataFore (ISCA'26 Case Study 2)，但对方是**一次性静态决策**，我们是**每次 P→D 边界都做一次增量微调**，收益覆盖 prefill+decode 双阶段

**v0.1 的范围收敛**：
- ✅ 只做 **swap**（等量对换两个 expert 的位置），不做 replicate（复制冗余副本）
- ✅ 只跑 **Qwen3-30B-A3B-FP8**，先在 **单机多卡 EP-only** 部署上验证
- ✅ 只做 **prefill → decode 边界**这一个触发点
- ❌ 不做 DP attention 场景适配（v0.2 再加）
- ❌ 不做 EPLB 共存的锁机制（v0.2 再加）
- ❌ 不做 dynamic replicate（v0.3）

---

## 1. Qwen3-30B-A3B-FP8 关键规格（决定实现细节）

| 项 | 值 | 对方案的影响 |
|---|---|---|
| 总层数 | 48 | MoE 层就是这 48 层（Qwen3 每层都是 MoE，没有 dense/moe 交错） |
| 每层 expert 数 E | **128** | 统计 tensor 是 `[48, 128]` int64 |
| top-k | **8** | dispatcher 每个 token 输出 top-8 expert id |
| 总参数 | 30.5B | 单 expert 权重 ~180 MB（FP8）/ ~360 MB（BF16） |
| 激活参数 | 3.3B | 单 token 走 8 个 expert |
| FP8 | 是 | expert 权重字节数按 FP8 算（更小、迁移更快） |

**推荐首次部署配置**（单机 8×H100/H800/A100，视你手上机器而定）：
```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-FP8 \
  --tp-size 8 \
  --ep-size 8 \
  --enable-ep-moe \
  --moe-a2a-backend deepep \
  --disable-cuda-graph      # 迁移会改 placement，v0.1 阶段先关 cuda graph 保稳
```

关键：`ep_size=8` → 每卡持有 `128/8 = 16` 个 expert。

---

## 2. 需要在 SGLang 代码里定位的关键位置

请编码 AI 先在 SGLang 源码里**找到并读懂**下面这些位置（不要动，先读）：

### 2.1 MoE 层的 topk 输出
```
python/sglang/srt/layers/moe/topk.py
python/sglang/srt/layers/moe/ep_moe/layer.py
```
关注：`select_experts(...)` 返回 `topk_ids: [num_tokens, top_k]` 的地方。这是**统计的挂点**。

### 2.2 Expert 权重的物理存放
```
python/sglang/srt/layers/moe/ep_moe/layer.py     # EPMoE 类
python/sglang/srt/models/qwen3_moe.py            # Qwen3 MoE 模块
```
关注：每层 MoE 权重被存成 `self.experts[local_expert_id]` 或类似的字典/list，local_expert_id 是**当前 rank 上的本地编号**，需要一个 `physical_to_logical` 映射表把它翻译成全局 expert id。

### 2.3 Placement / expert location 映射表
```
python/sglang/srt/managers/expert_location.py          # 如有
python/sglang/srt/managers/expert_distribution.py      # 如有
```
关注：`ExpertLocationMetadata` / `physical_to_logical_map` / `logical_to_all_physical_map` 这类结构。SGLang 里已经有一套 EPLB 用的接口，我们**复用**它。

### 2.4 Scheduler 主循环 & phase 判定
```
python/sglang/srt/managers/scheduler.py
```
关注：
- `run_batch(batch)` 里区分 `ForwardMode.EXTEND`（prefill）和 `ForwardMode.DECODE`
- `get_new_batch_prefill()` / `update_running_batch()` 决定下一 step 跑哪种模式
- 主循环里判定 "本 step 是 prefill 还是 decode" 的地方

### 2.5 EP process group
```
python/sglang/srt/distributed/parallel_state.py
```
关注：`get_moe_ep_group()` 或 `get_ep_group()`——**PB-OEPLB 所有跨 rank 通信必须用这个 group**。

---

## 3. 模块设计

新增目录：`python/sglang/srt/managers/pb_oeplb/`

```
pb_oeplb/
├── __init__.py
├── config.py            # 所有可调参数、启动开关
├── stats.py             # 统计收集器
├── rebalancer.py        # 决策 + swap 计划生成
├── swapper.py           # 实际执行 swap
└── controller.py        # 状态机 + 对外唯一入口
```

### 3.1 `config.py`

```python
from dataclasses import dataclass

@dataclass
class PBOEPLBConfig:
    enabled: bool = False
    threshold_ratio: float = 1.25      # imbalance > 1.25 才触发
    max_swaps_per_layer: int = 1       # v0.1 保守，每层最多 1 次 swap
    min_prefill_tokens: int = 4096     # 累计 token 不够则跳过
    cooldown_steps: int = 500          # 两次成功 swap 的最小间隔
    log_every_boundary: bool = True    # 每次边界都打 metrics
```

启动参数（在 `server_args.py` 里加）：
```
--enable-pb-oeplb
--pb-oeplb-threshold-ratio 1.25
--pb-oeplb-max-swaps-per-layer 1
--pb-oeplb-min-prefill-tokens 4096
--pb-oeplb-cooldown-steps 500
```

### 3.2 `stats.py`

```python
import torch
import torch.distributed as dist

class ExpertLoadStats:
    def __init__(self, num_layers: int, num_experts: int, device: str):
        self.num_layers = num_layers
        self.num_experts = num_experts
        # 累计每层每 expert 被路由到的 token 数
        self.load = torch.zeros(num_layers, num_experts,
                                dtype=torch.int64, device=device)
        self.total_tokens = 0

    def record(self, layer_id: int, topk_ids: torch.Tensor):
        """
        topk_ids: [num_tokens, top_k]，全局 expert id
        只在 prefill phase 调用
        """
        # 用 bincount 累加到 self.load[layer_id]
        flat = topk_ids.flatten()
        counts = torch.bincount(flat, minlength=self.num_experts)
        self.load[layer_id] += counts
        if layer_id == 0:  # 只在第 0 层累计 token 数，避免重复
            self.total_tokens += topk_ids.shape[0]

    def all_reduce(self, ep_group):
        """跨 EP rank 聚合"""
        dist.all_reduce(self.load, op=dist.ReduceOp.SUM, group=ep_group)

    def reset(self):
        self.load.zero_()
        self.total_tokens = 0
```

**性能关键**：`record` 每 forward step 每层都会调，必须极轻。`bincount` 是 GPU 上一次 kernel，几十 μs，可以接受。

### 3.3 `rebalancer.py`

```python
import torch
from typing import List, NamedTuple

class SwapOp(NamedTuple):
    layer_id: int
    expert_a_global_id: int   # 过载 GPU 上的热门 expert
    expert_b_global_id: int   # 空闲 GPU 上的冷门 expert
    rank_a: int
    rank_b: int

def compute_gpu_load(load_per_expert: torch.Tensor,
                     expert_to_rank: torch.Tensor,
                     num_ranks: int) -> torch.Tensor:
    """
    load_per_expert: [E], 每个 expert 的 token 数
    expert_to_rank:  [E], 每个 expert 当前所在 rank
    返回: [num_ranks]，每 GPU 的总 load
    """
    load_per_gpu = torch.zeros(num_ranks, dtype=torch.int64,
                               device=load_per_expert.device)
    load_per_gpu.scatter_add_(0, expert_to_rank, load_per_expert)
    return load_per_gpu

def try_build_swap_plan(
    global_load: torch.Tensor,          # [L, E]
    expert_to_rank: torch.Tensor,       # [L, E]，每层每 expert 当前在哪个 rank
    num_ranks: int,
    threshold_ratio: float,
    max_swaps_per_layer: int,
) -> List[SwapOp]:
    plan = []
    L, E = global_load.shape

    for l in range(L):
        load_l = global_load[l]
        e2r_l = expert_to_rank[l]

        for _ in range(max_swaps_per_layer):
            gpu_load = compute_gpu_load(load_l, e2r_l, num_ranks)
            max_load = gpu_load.max().item()
            avg_load = max(gpu_load.float().mean().item(), 1.0)
            if max_load / avg_load < threshold_ratio:
                break   # 本层已足够均衡

            rank_hot  = int(gpu_load.argmax().item())
            rank_cold = int(gpu_load.argmin().item())
            if rank_hot == rank_cold:
                break

            # 在 hot GPU 上找 load 最大的 expert
            mask_hot  = (e2r_l == rank_hot)
            mask_cold = (e2r_l == rank_cold)
            e_hot  = int((load_l * mask_hot ).argmax().item())
            # 在 cold GPU 上找 load 最小的 expert（用 max_int - load 的 trick 或直接 mask）
            cold_loads = torch.where(mask_cold, load_l,
                                     torch.full_like(load_l, 2**62))
            e_cold = int(cold_loads.argmin().item())

            plan.append(SwapOp(
                layer_id=l,
                expert_a_global_id=e_hot,
                expert_b_global_id=e_cold,
                rank_a=rank_hot,
                rank_b=rank_cold,
            ))
            # 模拟 swap 后更新 e2r_l，继续判断是否还需要下一次 swap
            e2r_l = e2r_l.clone()
            e2r_l[e_hot], e2r_l[e_cold] = rank_cold, rank_hot

    return plan
```

**注意**：v0.1 里 `max_swaps_per_layer=1`，所以里面的 for 循环最多走 1 次。留着结构方便 v0.2 调大。

### 3.4 `swapper.py`

这是最需要小心的部分。**swap 的语义 = 两个 expert 交换它们的物理 rank**。实现方式有两种：

**方案 A（v0.1 采用）：交换权重张量**
- 在 `rank_a` 和 `rank_b` 之间 `dist.send/recv` 两个 expert 的权重
- 更新每个 rank 上的 `experts[local_slot]` 指向新权重
- 更新全局 `physical_to_logical_map` 和 `logical_to_physical_map`

**方案 B（简单但需 SGLang 内部支持）：只改映射表**
- 权重不动，只改 dispatcher 用的 `logical→physical` 映射
- 但这要求所有 rank 都持有所有 expert 的权重（不成立，EP 场景下每卡只有一部分）

**所以 v0.1 走方案 A**。伪代码：

```python
import torch
import torch.distributed as dist

def execute_swap(swap_op, model, ep_group, my_rank, expert_location_meta):
    """
    每个 rank 都会调用这个函数。根据 my_rank 决定自己是发送方 / 接收方 / 旁观者。
    """
    l = swap_op.layer_id
    ra, rb = swap_op.rank_a, swap_op.rank_b
    ea, eb = swap_op.expert_a_global_id, swap_op.expert_b_global_id

    # 只有 rank_a 和 rank_b 参与实际搬运，其他 rank 只更新映射表
    if my_rank == ra:
        local_slot = expert_location_meta.global_to_local(l, ea)
        weight = model.get_expert_weight(l, local_slot)   # 返回一组 tensor（up/gate/down）
        # 先发再收，避免死锁——两边约定顺序
        for w in weight:
            dist.send(w, dst=rb, group=ep_group)
        recv_buf = model.alloc_expert_weight_buffer(l)
        for w in recv_buf:
            dist.recv(w, src=rb, group=ep_group)
        model.set_expert_weight(l, local_slot, recv_buf)

    elif my_rank == rb:
        local_slot = expert_location_meta.global_to_local(l, eb)
        recv_buf = model.alloc_expert_weight_buffer(l)
        for w in recv_buf:
            dist.recv(w, src=ra, group=ep_group)
        weight = model.get_expert_weight(l, local_slot)
        for w in weight:
            dist.send(w, dst=ra, group=ep_group)
        model.set_expert_weight(l, local_slot, recv_buf)

    # 所有 rank 更新映射表（这个必须每个 rank 都做，保证一致）
    expert_location_meta.swap(l, ea, eb)

    # barrier 确保所有 rank 都完成才继续
    dist.barrier(group=ep_group)
```

**关键实现细节**：
1. **send/recv 顺序**：两边 rank 必须约定同一顺序（比如"编号小的先发"），否则会死锁
2. **权重张量个数**：Qwen3 MoE 一个 expert 有 3 个权重（gate_proj、up_proj、down_proj），要循环发/收
3. **FP8 权重的 scale**：如果 expert 权重是 FP8，通常还有一个 fp32 scale tensor，也要一起搬
4. **映射表更新**：这是唯一一件"所有 rank 都要做"的事情，忘了就会 dispatch 错

### 3.5 `controller.py`

```python
class PBOEPLBController:
    def __init__(self, cfg, num_layers, num_experts, num_ranks,
                 my_rank, ep_group, model, expert_location_meta):
        self.cfg = cfg
        self.stats = ExpertLoadStats(num_layers, num_experts, "cuda")
        self.num_ranks = num_ranks
        self.my_rank = my_rank
        self.ep_group = ep_group
        self.model = model
        self.meta = expert_location_meta
        self.steps_since_last_swap = 10**9   # 允许第一次触发

    def on_prefill_step(self, layer_id, topk_ids):
        if not self.cfg.enabled: return
        self.stats.record(layer_id, topk_ids)

    def on_prefill_to_decode_boundary(self):
        if not self.cfg.enabled: return None
        self.steps_since_last_swap += 1

        if self.stats.total_tokens < self.cfg.min_prefill_tokens:
            self.stats.reset(); return None
        if self.steps_since_last_swap < self.cfg.cooldown_steps:
            self.stats.reset(); return None

        # 跨 EP rank 聚合
        self.stats.all_reduce(self.ep_group)

        # 当前 placement 快照
        expert_to_rank = self.meta.get_expert_to_rank_tensor()   # [L, E]

        plan = try_build_swap_plan(
            global_load=self.stats.load,
            expert_to_rank=expert_to_rank,
            num_ranks=self.num_ranks,
            threshold_ratio=self.cfg.threshold_ratio,
            max_swaps_per_layer=self.cfg.max_swaps_per_layer,
        )

        if not plan:
            self.stats.reset(); return None

        for op in plan:
            execute_swap(op, self.model, self.ep_group,
                         self.my_rank, self.meta)

        self.steps_since_last_swap = 0
        self._log_metrics(plan)
        self.stats.reset()
        return plan
```

---

## 4. 三个 Hook 点的具体接入

### Hook 1：MoE 层 topk 之后

在 `layers/moe/topk.py` 或 `ep_moe/layer.py` 的 `select_experts` 返回之后：

```python
topk_ids, topk_weights = select_experts(...)

# ---- PB-OEPLB hook ----
from sglang.srt.managers.pb_oeplb import get_controller
ctrl = get_controller()
if ctrl is not None and ctrl.is_prefill_now():
    ctrl.on_prefill_step(self.layer_id, topk_ids)
# ------------------------
```

`is_prefill_now()` 从 scheduler 那里读一个共享的 phase flag（可以是全局变量，也可以塞到 forward_batch_info 里）。

### Hook 2：Scheduler phase 切换

在 `scheduler.py` 主循环的 batch mode 判定处：

```python
prev_phase = self._pb_oeplb_last_phase
new_phase = "prefill" if batch.forward_mode.is_extend() else "decode"

if prev_phase == "prefill" and new_phase == "decode":
    ctrl = get_pb_oeplb_controller()
    if ctrl is not None:
        # 注意：必须在 forward 之前调用，避免与 forward 并发
        ctrl.on_prefill_to_decode_boundary()

self._pb_oeplb_last_phase = new_phase
```

### Hook 3：初始化

在 `ModelRunner.__init__` 或类似位置：

```python
if server_args.enable_pb_oeplb:
    from sglang.srt.managers.pb_oeplb import PBOEPLBController, PBOEPLBConfig
    ctrl = PBOEPLBController(
        cfg=PBOEPLBConfig.from_server_args(server_args),
        num_layers=model_config.num_hidden_layers,
        num_experts=model_config.num_experts,
        num_ranks=ep_size,
        my_rank=ep_rank,
        ep_group=get_moe_ep_group(),
        model=self.model,
        expert_location_meta=self.expert_location_metadata,
    )
    set_global_controller(ctrl)
```

---

## 5. 可观测性（必做，用来验证收益）

在 `controller.py` 里通过 SGLang 已有的 stdout logger 或 Prometheus counter 打这些指标：

```
pb_oeplb.boundary_events            # 累计 P→D 边界次数
pb_oeplb.swaps_executed             # 累计执行 swap 次数
pb_oeplb.swaps_skipped_below_thresh # 阈值未达标跳过次数
pb_oeplb.swaps_skipped_cooldown     # 冷却期跳过次数
pb_oeplb.swaps_skipped_insufficient # 样本不足跳过次数
pb_oeplb.imbalance_before           # 每次决策前的 imbalance ratio（per layer 或均值）
pb_oeplb.imbalance_after            # swap 后的 imbalance ratio（模拟计算）
pb_oeplb.migration_bytes_total      # 累计搬运字节数
pb_oeplb.boundary_latency_ms        # 单次 boundary 处理耗时
```

最简单：每次 boundary 用 `logger.info(json.dumps(metrics))` 打一行 JSON，之后 grep 就能画图。

---

## 6. 单元测试（模型无关）

新增 `test/srt/pb_oeplb/test_rebalancer.py`：

**测试 1：无 skew 时不生成计划**
```python
load = torch.ones(48, 128, dtype=torch.int64) * 100
e2r = torch.arange(128).repeat(48, 1) // 16   # 每 16 个 expert 一个 rank
plan = try_build_swap_plan(load, e2r, 8, 1.25, 1)
assert plan == []
```

**测试 2：单点 skew 能被正确识别并生成 swap**
```python
load = torch.ones(48, 128, dtype=torch.int64) * 10
load[0, 0] = 10000   # rank 0 上 expert 0 极热
e2r = torch.arange(128).repeat(48, 1) // 16
plan = try_build_swap_plan(load, e2r, 8, 1.25, 1)
assert len(plan) >= 1
assert plan[0].layer_id == 0
assert plan[0].expert_a_global_id == 0
```

**测试 3：swap 后模拟 imbalance 单调下降**
```python
# 应用 plan 到 e2r，再次计算 imbalance，必须 < 之前
```

---

## 7. 联调测试（端到端）

**阶段 1：功能正确性**
1. 启动 Qwen3-30B-A3B-FP8，`--enable-pb-oeplb`
2. 发一批 prompt（比如 ShareGPT 里挑 32 条 mixed length）
3. 看日志：应有 `boundary_events > 0`
4. 用 `curl /generate` 拿输出，和不开 pb-oeplb 的输出**对比 hash**——**必须一致**（swap 不改语义）
5. 如果输出不一致 → **立即停止**，检查 mapping 表更新和权重搬运顺序

**阶段 2：收益验证**
见下文实验方案。

---

## 8. Debug checklist（编码时踩坑指南）

- [ ] `all_reduce` 用 EP group，不是 world / tp group
- [ ] `bincount` 的 `minlength=num_experts` 必须显式传
- [ ] `topk_ids` 是全局 expert id 还是本地 id？确认是**全局**才能直接 bincount
- [ ] Qwen3 MoE 一个 expert 有几个权重张量？（gate + up + down）逐个搬
- [ ] FP8 权重的 scale tensor 也要搬
- [ ] send/recv 顺序两边一致（推荐 rank 小的先 send）
- [ ] `execute_swap` 完毕后**所有** rank 都要更新 mapping 表
- [ ] barrier 必须在 EP group 上
- [ ] cuda graph 先关掉（`--disable-cuda-graph`），swap 会破坏 graph capture
- [ ] 第一次 boundary 时 `total_tokens` 可能不足，跳过是对的
- [ ] statistics reset 时机：无论是否真的 swap，try 完都要 reset

---

## 9. 交付里程碑

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| M1（1-2 天） | stats + rebalancer + 单测 | 单元测试全绿 |
| M2（1-2 天） | swapper + controller + 3 个 hook 接入 | 启动服务不 crash，日志能看到 boundary event |
| M3（1 天） | 输出正确性验证 | 开/关 pb-oeplb 输出 hash 一致 |
| M4（1 天） | 打点 + 收益初测 | 拿到 baseline vs pb-oeplb 的时延对比 |

---

以上就是 v0.1 的完整实施指导。**核心原则**：先跑通、先证明"不破坏正确性"，收益优化留给 v0.2。
