# OEPLB 完整实验报告（v0.9）

> **环境**: 4×NVIDIA H20 (96GB/卡), **NV18 NVLink full mesh**（单机，非跨节点 IB）, CUDA 12.8, PyTorch 2.9.1, SGLang 0.5.6, DeepEP 1.1.0（本项目 patch 版本）
> **模型**: Qwen3-30B-A3B-FP8 (48层, 128 experts/层, top-k=8, EP=4, 32 experts/GPU)

---

## 目录

1. [项目目标与背景](#1-项目目标与背景)
2. [版本演进总表](#2-版本演进总表)
3. [v0.1→v0.5: 单机 EP-only 阶段](#3-v01v05-单机-ep-only-阶段)
4. [DeepEP NVLink 适配（关键 patch）](#4-deepep-nvlink-适配关键-patch)
5. [v0.6→v0.8: DP 支持与死锁排查](#5-v06v08-dp-支持与死锁排查)
6. [官方 EPLB 源码调研：三个关键问题](#6-官方-eplb-源码调研三个关键问题)
7. [v0.9: EPLB-style 本地计数器触发](#7-v09-eplb-style-本地计数器触发)
8. [最终验证：负载均衡效果 + 残余开销诊断](#8-最终验证负载均衡效果--残余开销诊断)
9. [已知限制与后续方向](#9-已知限制与后续方向)

---

## 1. 项目目标与背景

**PB-OEPLB**（Prefill-Boundary Online Expert Placement Load Balancer）是在 SGLang 里实现的一个**在线专家负载均衡器**：在每次 prefill→decode 边界（或周期性检查点），统计最近一段 prefill 里各 expert 被路由到的 token 数，如果发现 GPU 间负载明显不均衡，就在当前 EP 布局上做少量 expert **swap**（把过载 GPU 上的热门 expert 和空闲 GPU 上的冷门 expert 对调物理位置），让后续 forward 更均衡。

与 SGLang 官方 EPLB 的核心差异：**只做 1-vs-1 swap（不做冗余专家复制）**，且设计初衷是**只针对 prefill 阶段的路由统计做决策**（反映即将到来的 decode 阶段的真实专家热度），而非官方 EPLB 那种 prefill+decode 混合统计。

---

## 2. 版本演进总表

| 版本 | 关键变化 | 结果 |
|------|---------|------|
| v0.1 | 最小可用实现，scheduler hook 做边界检测 | 跨 rank 死锁（不同 rank 调度器独立运行，到达边界时机不同） |
| v0.2 | 边界检测移入 model_runner 的 forward path（天然跨 rank 同步） | 死锁解决 |
| v0.3 | 自建 `AsyncSwapExecutor`（低优先级 CUDA stream + event 查询），`fast_metadata.py` 向量化 metadata 重建 | 375ms→3.4ms，权重传输不阻塞主推理流 |
| v0.4 | 窗口化统计采样（sparse sampling），MIN_RECORD_TOKENS 过滤 | 单机 EP-only 场景下吞吐损耗 <1% |
| v0.5 | 采样间隔调优 | 同上，稳定基线 |
| v0.6 | **首次尝试 DP 支持**：`all_reduce(SUM)` 聚合各 DP rank 的 load | DP 模式下 warmup 阶段死锁（`busy` check 导致部分 rank 跳过 all_reduce） |
| v0.7 | 改为纯 forward-step 计数器触发（放弃 P→D boundary 语义） | 退化成官方 EPLB 的行为模式，失去 OEPLB 差异化价值，被推翻 |
| v0.8 | 三阶段 consensus（boundary MAX all_reduce → ready MAX all_reduce → load SUM all_reduce），保留 boundary 语义 | 解决死锁，但**每次 forward 2 次 all_reduce 导致 -16.6% 吞吐回归** |
| **v0.9** | 调研官方 EPLB 源码后，改用**本地 step 计数器 + 周期性单次 all_reduce**（`sync_window`），保留 prefill-only 记录语义 | **回归收窄至 -2%~-4%**，负载均衡效果实测有效 |

---

## 3. v0.1→v0.5: 单机 EP-only 阶段

早期版本（v0.1-v0.5）只支持纯 EP、无 DP attention 的部署形态。关键里程碑：

- **v0.1→v0.2 死锁修复**：v0.1 曾在 `scheduler.py` 加 P→D 边界检测 hook，但每个 rank 的 scheduler 独立运行，调用 `update_expert_location()`（集合 P2P 操作）时不同 rank 到达边界的时机不同，导致死锁。v0.2 起将边界检测移入 `model_runner.py` 的 forward 路径内部——由于 EP 场景下所有 rank 的 forward 天然同步执行，问题解决，且 scheduler.py 从此不再需要任何改动。

- **v0.3 异步 swap 架构**：诊断发现同步 P2P 传输（`batch_isend_irecv` + 阻塞 `wait()`）会卡住单线程 scheduler 事件循环整个传输时长。重写为 `AsyncSwapExecutor`：`begin()` 在独立低优先级 CUDA stream 上发起 P2P，立即返回；`try_finish()` 用非阻塞 `event.query()` 检测完成，避免 `synchronize()`。

- **v0.3 隐藏开销发现**：`ExpertLocationMetadata.init_by_mapping()`（SGLang 官方路径）内部是 `O(L×E×ep_size)` 的 Python 循环，实测 375ms。改写为 `fast_init_by_mapping()`，用向量化 GPU scatter 操作，3.4ms 完成。

- **v0.4 窗口化采样**：早期版本记录密度太高时（每个 prefill batch 都记录）在小 batch/RadixCache 命中率高的场景下产生显著 `bincount` 开销（trace 显示占比达 88.8% 的 prefill batch 是 1-token cache hit）。引入 `MIN_RECORD_TOKENS=32` 过滤 + 稀疏采样（每 K 个 eligible batch 记录一次），把统计样本分散到整个 cooldown 周期而非集中在窗口开头。

---

## 4. DeepEP NVLink 适配（关键 patch）

在为 DP+EP 混合并行做压测时，尝试用官方 DeepEP low_latency 模式（`--deepep-mode low_latency`）加速 decode，遇到硬件拓扑相关的适配问题：

**硬件事实**：4×H20 是**单机 NVLink（NV18 full mesh）互联**，不是跨节点 InfiniBand/RDMA。

**问题**：DeepEP 1.1.0 的 `low_latency_dispatch()` 在 C++ 层硬编码调用 `internode_ll::dispatch()`（`csrc/deep_ep.cpp` 内），且这条路径内的 GPU kernel（`csrc/kernels/internode_ll.cu:367`）有一句**无条件**断言：
```cpp
EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
```
即使实际数据走的是 NVLink P2P 直连拷贝（代码里 `nvshmemi_get_p2p_ptr` 非零时的分支），这句 IBGDA 状态检查依然会无条件执行，在没有 IB/RDMA 的单机拓扑下必然失败。

同时 `deep_ep/buffer.py` 里，只要 `low_latency_mode=True`，就无条件设置：
```python
os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
```
这会覆盖任何试图通过环境变量关闭 IBGDA 的尝试——**问题在代码层面，不是环境变量能绕开的**。

**修复**（对 DeepEP 1.1.0 源码的 2 处 patch，需要重新编译 `deep_ep_cpp` 扩展）：
1. `csrc/kernels/internode_ll.cu`: 注释掉 367 行的 IBGDA 断言
2. `deep_ep/buffer.py`: 改为 `os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1' if self.runtime.get_num_rdma_ranks() > 1 else '0'`（只在真的有多个 RDMA rank 时才启用 IBGDA）

配合以下环境变量强制 NVSHMEM 走 P2P/NVLink 而非 IB transport：
```bash
NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_HCA_LIST= \
NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL
```

**验证结果**（单请求, 128 tokens 输出）：

| 指标 | Low Latency (patched) | Normal | 差异 |
|------|------------------------|--------|------|
| TPOT | 6.8 ms | 78.3 ms | **11.5x** |
| TTFT | ~68 ms | ~163 ms | 2.4x |
| TPS (conc=128) | 13,209 | 1,485 | **8.9x** |

**重要限制**：`--deepep-mode low_latency` 会强制 **prefill 阶段也**走 low-latency dispatch，而其单 rank 最大 dispatch token 数有硬编码上限（`assert self.num_max_dispatch_tokens_per_rank <= 1024`）。当输入变长（如 2048 tokens）、chunked prefill batch 较大时，容易超过这个上限导致 dispatch 卡死。**长输入场景应使用 `--deepep-mode auto`**（prefill 走 normal 无限制，decode 走 low_latency 高速），这也是当前所有测试脚本采用的配置。

另外，SGLang 官方 `ExpertDistributionRecorder`（EPLB 依赖的统计模块）**不支持 `deepep_mode=auto`**（源码里只实现了 `normal`/`low_latency` 两个分支，`auto` 会直接 `raise NotImplementedError`）——这是只影响官方 EPLB 的限制，PB-OEPLB 自己的 `record_next_layer` hook 不依赖这个模块，不受影响。

---

## 5. v0.6→v0.8: DP 支持与死锁排查

引入 `--dp 4 --enable-dp-attention` 后，PB-OEPLB 的边界检测机制（原本假设"P→D boundary 在跨 rank 间天然对齐"）失效，因为 DP 模式下每个 rank 是独立 scheduler，处理不同请求，各自在不同的本地时刻经历 P→D 转换。

**v0.6（直接加 all_reduce）**：在 `_decide_and_begin_swap()` 前加一次 `all_reduce(SUM)` 聚合 `load`。问题：`async_executor.busy` 检查在 all_reduce **之前**，导致某些 rank 因为"上一次 swap 还没完成"而提前 return，跳过了这次 all_reduce，其他 rank 卡在集合操作上等待 —— **死锁**。

**v0.7（放弃 boundary，改用纯 step 计数器）**：完全抛弃边界检测，改成"每 N 个 forward 检查一次"。问题：这跟官方 EPLB 的行为完全一样了，失去了 OEPLB「只在 prefill 结束时刻决策」的设计初衷，被推翻。

**v0.8（三阶段 consensus）**：为保留边界语义，设计三段式 all_reduce：
- Phase 1（4字节）：`all_reduce(MAX)` 判断"是否有任意 rank 遇到了 P→D boundary"
- Phase 2（4字节）：`all_reduce(MAX)` 判断"是否满足 cooldown + min_tokens 条件"
- Phase 3（48KB）：`all_reduce(SUM)` 聚合 load，`busy` check 移到 all_reduce **之后**（保证集合操作永远执行）

这版解决了死锁，但引入了一个新问题：**诊断发现 DP 模式下 warmup 阶段 IDLE batch（无请求时的空 forward）没有调用 `on_forward_pass_end()`**（原代码逻辑：`if is_prefill or _pb_needs_check: call(...)`），导致各 rank 的 forward 计数器 `fwd_id` 彼此漂移不同步（例如 rank0 到 fwd_id=11 时 rank3 才到 fwd_id=8），最终在某次 all_reduce 时只有部分 rank 到达 → 死锁。**修复：`model_runner.py` 里改为无条件调用 `on_forward_pass_end()`**（每个 forward，包括 IDLE batch，都必须调用，让计数器在所有 rank 间保持同步）。

修复死锁后实测：v0.8 在 DP=4、conc=128、500 请求（输入32tok）下：**TPS 10,488，vs baseline 12,580，回归 -16.6%**。诊断：每次 forward 2 次 all_reduce（哪怕只有 4 字节）在高并发 DP 场景下的同步等待开销远超预期。

---

## 6. 官方 EPLB 源码调研：三个关键问题

为找到 v0.8 -16.6% 回归的根本解法，直接读 SGLang 官方 EPLB 源码（`sglang/srt/eplb/expert_distribution.py`, `eplb_manager.py`）回答三个问题：

**Q1: 官方 EPLB 的热路径是否零通信？**
是。`_SelectExpertsSinglePassGatherer.on_select_experts()`（`expert_distribution.py:507-511`）只做本地 `scatter_add_`，无任何通信调用。通信只在 `_StatAccumulator.dump()` 里出现（`all_reduce` 于 line 856，`broadcast` 于 line 896），而 `dump()` 只在真正触发 rebalance 时调用一次。

**Q2: 官方 EPLB 记录的是 local-expert-only 还是全局 expert 范围？**（这一点最初的假设是错的）
**不是 local-only**。当前对比配置（`--expert-distribution-recorder-mode stat` + `--deepep-mode normal`）下实际用的是 `_SelectExpertsSinglePassGatherer`，hook 点是 `topk.py` 的 `on_select_experts`——**跟 PB-OEPLB 的 `record_next_layer` 是同一个 hook 点，记录方式完全一致**（全量 expert 范围的 `scatter_add_`，天然限定在本 rank 自己的 token 批次上）。真正的 local-expert-only 记录方式（`_DeepepNormalSinglePassGatherer`，用 `local_physical_count_of_layer`）只在 `stat_approx` 近似模式下才启用，跟本项目的对比配置无关。

**Q3: EPLB 的 rebalance 触发和通信调用栈是什么？**
`EPLBManager.on_forward_pass_end()` 只是 `next(self._main_generator)`，其中：
```python
def _entrypoint(self):
    while True:
        for _ in range(self._rebalance_num_iterations):
            yield          # 纯本地 Python 循环，零通信，1000步里做999步
        yield from self.rebalance()   # 只在第1000步才做通信
```
这个纯本地整数计数器**不需要任何跨 rank 通信来保持同步**，因为 DP+EP 架构下每个 rank 每个 forward 步骤都必然调用一次这个函数（MoE all-to-all 是集合操作，强制所有 rank 参与，即使某 rank 当前无请求也要跑一次 IDLE forward）——这是一个**架构层面天然保证的隐式同步**，而不是靠通信协议显式达成的。

**结论**：v0.8 的错误在于误以为"跨 rank 一致性判断"需要显式 all_reduce consensus；实际上只要保证每个 rank 的本地计数器在同样的 forward 节奏下推进（这一点已经由架构保证），本地计数器天然就是同步的，完全不需要每步通信。

---

## 7. v0.9: EPLB-style 本地计数器触发

基于调研结论重新设计触发机制：

- **记录逻辑不变**（`record_next_layer`，已经跟官方 EPLB 一致，不需要改）
- **触发机制**：用纯本地 forward 计数器 `_steps_since_last_check`（每次 `on_forward_pass_end` 自增，不做任何通信），达到 `sync_window`（默认 64）后才检查一次
- **单次 all_reduce 兼职判断阈值**：检查点到达后，做**唯一一次** `all_reduce(SUM)` 聚合 `load` 张量；聚合结果的 `sum()` 同时用作"是否攒够 `min_prefill_tokens`"的判断依据——不需要额外的"是否就绪"共识轮次
- **保留 OEPLB 差异化**：仍然只在 prefill 阶段记录（`is_prefill` 本地判断，不需要跨 rank 一致），区别于官方 EPLB 的 prefill+decode 混合统计

通信频率对比：

| 版本 | 每 forward 通信 | ~1800 forward 总通信次数 |
|------|-----------------|--------------------------|
| v0.8 | 2 次 all_reduce（4B+4B） | ~3600 次 |
| v0.9 | 0（每 64 步 1 次 48KB all_reduce） | ~28 次 |
| 官方 EPLB | 0（每 1000 步 1 次） | ~2 次 |

---

## 8. 最终验证：负载均衡效果 + 残余开销诊断

**测试条件**：500 请求，输入精确 2048 tokens（`real_long_unique.jsonl` 源数据经 tokenizer 拼接+截断到精确 2048 token，见 `frozen_requests_in2048.jsonl`），输出 512 tokens（`ignore_eos=True`），conc=128，DP=4/EP=4，DeepEP auto 模式。

**吞吐结果**：

| 配置 | TPS | vs Baseline |
|------|-----|-------------|
| T1 Baseline | 5,522.4 | — |
| T2 PB-OEPLB v0.9 | 5,292~5,413（两次独立运行） | **-2%~-4.2%** |

**负载均衡效果验证**（对 `rebalancer.py` 加了均衡前后比例诊断日志，4 个 DP rank 完全一致的实测数据）：

| Window | 触发层数 | 均衡前 avg/max ratio | 均衡后 avg/max ratio |
|--------|---------|----------------------|----------------------|
| #3 | 40层 | 1.308 / **2.001** | 1.093 / 1.147 |
| #11 | 7层 | 1.167 / 1.182 | 1.102 / 1.148 |
| #12 | 3层 | 1.163 / 1.169 | 1.065 / 1.085 |
| #20 | 1层 | 1.156 / 1.156 | 1.053 / 1.053 |
| #28 | 9层 | 1.191 / 1.260 | 1.072 / 1.115 |

window#3 显示服务刚启动时最热的 GPU 负载达到平均值的 **2.0 倍**，swap 后压到 1.15 附近；随后每次触发时的不均衡程度逐渐收窄（说明布局在持续收敛）。**结论：swap 决策算法确实在做真实有效的负载均衡，不是摆设。**

**残余 -2%~-4% 开销的根因定位**：对比两次意外获得的数据点——

| 记录密度 | TPS | vs Baseline |
|---------|-----|-------------|
| 几乎不记录（`sample_interval` 因误配置变成100，等效于关闭记录） | 5,469.0 | -0.97% |
| 正常密度记录（`sample_interval=1`，每个 prefill batch 都在 48 层记录） | 5,292~5,413 | -2%~-4.2% |

两者的**唯一差异是记录密度**，all_reduce 频率和 swap 执行方式完全相同。差值（约 -1%~-3.2%）可以归因于 `record_next_layer` 的调用开销：每个 prefill batch 触发时，要在 48 层各做一次 `bincount` + `physical_to_logical_map` gather + `add_`。**swap 执行本身（异步 P2P，不阻塞主流）代价很小；记录阶段才是当前的主要开销来源。**

---

## 9. 已知限制与后续方向

1. **记录密度与统计精度的权衡未调优**：当前 `sample_interval` 由 `cooldown_steps // 5` 隐式决定，语义不直观。下一步应该：(a) 独立暴露 `sample_interval` 或等效参数，(b) 扫描不同采样密度下 "统计代表性 vs 记录开销" 的曲线，找到损耗更低但仍能反映真实分布的采样率。

2. **`sync_window` 和 `min_prefill_tokens` 未做参数扫描**：当前用的是 64 / 1000 的经验值，未验证是否最优。

3. **首次 swap 冲击较大**：window#3 一次性对 40+ 层同时发起 swap（源于服务刚启动时布局是 trivial 初始化，跟真实流量分布差距最大），这个"首次纠偏"的 P2P 传输量远大于后续稳态时的小幅调整。可以考虑：启动阶段用更保守的 `max_total_swap_layers` 上限，分批次执行首次大幅纠偏。

4. **DeepEP low_latency 在长输入下的限制**：`--deepep-mode low_latency` 的单 rank 最大 dispatch token 数硬上限 1024，长输入场景必须用 `auto` 模式（prefill 走 normal），这是 DeepEP 本身的限制，非 OEPLB 可控范围。

5. **官方 EPLB 对比仅完成 auto/normal 模式部分**：`--deepep-mode normal` 下 EPLB 可正常触发多次 rebalance（已验证），但 `auto` 模式下因 `ExpertDistributionRecorder` 的 `NotImplementedError` 限制无法测试 EPLB，因此当前 auto 模式下的最终对比只有 baseline vs PB-OEPLB，未包含官方 EPLB 数据点。

---

## 10. v0.10→v0.11 深度瓶颈分析：为什么 swap 有效但吞吐不涨

### 10.1 Profiling 发现

v0.9 的 `record_next_layer` 每次调用 ~800-1000μs（5-6 个 CUDA kernel launch：reshape+long+clamp+gather+bincount+add_）。v0.10 改为物理空间 `scatter_add_`（匹配 EPLB 的 `_SelectExpertsSinglePassGatherer.on_select_experts`）后降至 ~85μs，**8.5x 加速**。

v0.10 cumulative overhead breakdown（sw64, 4k input, ~12000 record calls）:
| 组件 | 耗时 | 占比 |
|------|------|------|
| record | 1006ms | 75% |
| allreduce | 155ms | 12% |
| planbuild | 91ms | 7% |
| finalize | 88ms | 7% |

### 10.2 MoE 计算占比分析

Qwen3-30B-A3B 架构：
- `shared_expert_intermediate_size = 0`（**无 shared experts**）
- routed MoE 占 forward 计算的 **~50%**（attention ~50%）
- 理论上 max_ratio=2.0 时完美均衡应带来 **33% 吞吐提升**

### 10.3 核心矛盾：不均衡在 swap 后持续回弹

对 4k 输入 LongBench 数据集的 trace 分析（sw16 清零模式，734 次 swap）：

```
window#1: max_ratio_before=1.969 → after=1.211 (43 layers, 29 swaps)
window#2: max_ratio_before=1.625 → after=1.146 (37 layers) — 又回弹了
window#3: max_ratio_before=1.574 → after=1.194 (36 layers)
...
window#8: max_ratio_before=1.219 → after=1.114 (暂稳)
window#10: max_ratio_before=1.349 → after=1.112 — 又跳回
window#14: max_ratio_before=1.573 → after=1.145 — 再次回弹到 1.57
...
window#24: max_ratio_before=1.491 → after=1.127
window#25: max_ratio_before=1.379 → after=1.149
```

**不均衡在 1.2~2.0 之间永久震荡**，swap 的效果只持续几个窗口就被新请求的路由漂移覆盖。

### 10.4 根因

不均衡的根源不在专家物理布局，而在 **router gating function 对不同输入的天然偏好**：
- 同领域的请求倾向选择同一组专家 → 创造短暂热点
- 新请求到达时热点漂移到另一组专家 → 之前的 swap 白做了
- EP=4 下每 rank 32 experts，少数专家的偏好就能创造 2x 不均衡
- Swap 永远在"追赶"变化中的热点，但追赶本身有固定开销（planbuild + finalize + P2P 传输）

这解释了为什么：
- max_ratio=2.0 降到 1.2 对吞吐影响 <3%——因为这个不均衡在 90s benchmark 中只存在于前几个 window（<5s），之后 swap 让它暂时消失，但很快又回来
- 更频繁 swap（sw8/sw16）反而更慢——因为每次 swap 有固定开销，而收益在下一个窗口就被路由漂移抹掉了

### 10.5 Swap vs Replication

| 策略 | 优势 | 劣势 | 适合场景 |
|------|------|------|---------|
| **Swap (PB-OEPLB)** | 不占额外显存、不改 dispatch | 只搬不复制，热点漂移时效果短暂 | 稳定流量、长期热点（如单一业务场景） |
| **Replication (EPLB)** | 热专家在多个 rank 上都有副本，天然抵抗漂移 | 占冗余显存（`ep-num-redundant-experts`） | 混合流量、热点快速变化 |

要达到 >10% 吞吐提升，swap-only 方案需要满足两个苛刻条件：
1. MoE 计算占 forward 时间 >30%（Qwen3-30B-A3B 满足，50%）
2. **专家热点在 swap 后保持稳定足够长时间**（当前混合流量不满足）

后续方向：考虑在 PB-OEPLB 中引入**选择性 replication**——对持续热门的少数专家做复制（而非 swap），结合 swap 处理短期热点。这需要 `ep-num-redundant-experts > 0` 的支持。
