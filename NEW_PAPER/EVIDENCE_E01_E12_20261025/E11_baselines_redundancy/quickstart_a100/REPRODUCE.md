# A100 复现指南（跨硬件三方对比：Baseline / EPLB / OEPLB）

> 硬件: 8× NVIDIA A100-SXM4-80GB (NVLink) · 模型: Qwen3-235B-A22B **BF16** (94 层, 128 专家, top-8)
> SGLang 0.5.6.post2 · torch 2.9.1+cu128 · TP8 + EP8 (每卡 16 本地专家) · Triton MoE runner · 无 DeepEP
> 日期: 2026-09-23

**前置**：先完成同目录 `ENVIRONMENT.md`（装环境 + 打两个补丁 + 启动自检）。本文档只讲「环境好了跑哪些实验」。

---

## 最快复现数据集路径

两个数据集都在仓库内、无需下载，`max_tokens=1`（纯 prefill，直接测 TTFT 吞吐）：

```
单域: OEPLB/benchmarks/final_grid/L512_O1.jsonl        # 8192 条 Lean-4 定理证明数学域, ~500tok
多域: OEPLB/benchmarks/multidomain/MD_L512_O1.jsonl    # 8192 条, 数学↔英文小说 M/E/M/E 4 段, 3 次域切换
```

想更快出结果就先只跑单域（一条命令级，见 §5）。

---

## 0. TL;DR

- A100 走 **`forward_normal` (BF16 / Triton / NCCL)**，不是 H20 的 `forward_deepep` (FP8 / DeepGEMM / DeepEP)。
- 该路径原有一个**致命 bug**：`forward_normal` 调 `self.topk(...)` 时**漏传 `expert_location_dispatch_info`**，
  导致 logical→physical 重映射不生效 → OEPLB/EPLB 搬了权重、改了元数据，但**路由压根不看** → 不均衡度**窗口间回弹**、吞吐不升反降。
- 修复（镜像 `forward_deepep` 传 dispatch info，见 `ENVIRONMENT.md` 补丁 B）后，OEPLB/EPLB 全部真正生效。
- **单域 +20.1% / 多域 +13.2%（OEPLB vs Baseline），且两个场景 OEPLB 均优于官方 EPLB。**

---

## 1. A100 实现 vs H20 实现的区别

两者共用同一个 OEPLB 控制器（`pb_oeplb/controller.py`），差异全在 **MoE 前向路径 / 通信 / 精度**：

| 维度 | H20 实现 | A100 实现 |
|---|---|---|
| MoE 前向路径 | `forward_deepep` | `forward_normal` |
| 专家并行 a2a | DeepEP (`--moe-a2a-backend deepep`) | 无专用 a2a，NCCL 常规 dispatch |
| 计算内核 | DeepGEMM (FP8 分组 GEMM) | Triton runner (`--moe-runner-backend triton`) |
| 精度 | FP8 (sm90 原生) | BF16 (sm80 无 FP8 张量核) |
| overlap schedule | 可开 | **必须 `--disable-overlap-schedule`**（否则 NCCL 死锁） |
| dp-attention | 开启 | **关闭**（见 §2） |
| P2P 权重搬运 | 可异步 | 同步 `batch_isend_irecv` |
| CUDA graph | 开 | 开 (`--cuda-graph-max-bs 64`) |
| logical→physical 重映射 | ✅ 原生传 dispatch info | ❌ 原本漏传 → **本文档修复的 bug** |

---

## 2. 为什么 A100 不开 dp-attention?

**dp-attention 的作用**：默认 TP 下 attention 是张量并行的，每卡持有 KV 头的一个分片，attention 前后要 all-reduce/all-gather。
开 dp-attention 后 attention 改为**数据并行**——每个 DP rank 用**完整** attention 处理不同 request 子集，只有 MoE/FFN 段保持专家并行。它的收益场景很特定：

1. **KV 头太少、TP 分片浪费带宽时才划算。** Qwen3-235B 用 GQA，KV 头数远小于 TP=8；TP 分片 KV 头会强制复制、浪费带宽，此时把 attention 改成 DP 更高效。
2. **它天然与 DeepEP 的 all-to-all 配对。** dp-attention 需把各 DP rank 的 token 汇聚进 EP 的 a2a 管线。A100 走 `forward_normal`（**没有 DeepEP**，只有 NCCL），两者之间的 gather/scatter 没有 DeepEP 那样的高效原生 kernel，桥接开销反而超过收益。
3. **与 `--disable-overlap-schedule` 叠加会恶化。** A100 上必须禁 overlap（否则 NCCL 死锁），通信被串行化；dp-attention 再引入一组额外集合通信，实测不稳定甚至死锁。
4. **它会改变 OEPLB 假设的 EP 布局。** 开 dp-attention 需要 `--dp-size>1`，会把 8 卡重新划分成 DP 组，而 OEPLB 控制器目前在 **dp=1** 下做负载统计与 all_reduce（日志里的 `tok global, dp=1`）。开 DP 会打乱这个假设。
5. **与 OEPLB 正交，关掉更干净。** dp-attention 平衡的是 attention/KV，OEPLB 平衡的是专家负载。关掉能**隔离 OEPLB 的效果**，并匹配 A100 上稳定的服务配置。

**一句话**：dp-attention 的收益依赖 Hopper+FP8+DeepEP 的组合；A100 的 BF16+NCCL+`forward_normal`+禁 overlap 环境里，它得不偿失且不稳定，还会干扰 OEPLB 的 EP 布局假设——所以关掉。

---

## 3. 根因 Bug 与修复

### 3.1 症状
- 不均衡度 `avg_ratio_before` **窗口间明显回弹**（修复前 window#2 回弹到 1.544，37 个窗口每次都 ~300 ops，永不收敛）。
- 开 OEPLB 反而**变慢 −3.5%**（见 §4 单域表）：swap 不改路由，只白付 all_reduce + P2P 搬运开销。

### 3.2 根因
`models/qwen3_moe.py` 的 `forward_normal` 调 `self.topk(hidden_states, router_logits)` 时**没有传 `expert_location_dispatch_info`**；
而 `forward_deepep` 正确传了 `ExpertLocationDispatchInfo.init_new(layer_id=...)`。
缺了它，`topk_ids_logical_to_physical()` 变 no-op → topk_ids 停留在 logical 编号 →
**OEPLB/EPLB 对 `physical_to_logical_map` 的一切修改都被路由忽略**，同时 token 会命中被搬走的物理槽位（静默数值错误；因 benchmark `max_tokens=1` 只测 TTFT 而未暴露）。

### 3.3 修复
见 `ENVIRONMENT.md` 补丁 B（幂等，已写入 `patch_sglang.py`，会备份 `qwen3_moe.py.preoeplbfix`）。
> 对照 H20 论文的 bug 症状相同（不均衡度回弹）但根因不同：那里是 `routed_experts_weights_of_layer` 只在 DeepSeek 上存在、Qwen 上没有导致 swap 静默 no-op；A100 这次是路由侧漏传 dispatch info。**"不均衡度回弹" 是 swap 未生效的通用信号。**

---

## 4. 复现结果

数据集均 8192 请求、`max_tokens=1`（纯 prefill）、并发 1024、双方都 `--disable-overlap-schedule`、0 错误。

### 表 1 · 单域（L512_O1，Lean-4 定理证明数学域）

| 配置 | 总时间 (s) | 吞吐 (tps) | vs Baseline |
|---|---|---|---|
| Baseline | 392.3 | 20.9 | — |
| OEPLB（修复前，bug） | 405.9 | 20.2 | −3.5% |
| **EPLB**（官方在线，0 冗余，每 50 步重排） | 348.4 | 23.5 | +12.4% |
| **OEPLB（修复后）** | **327.0** | **25.1** | **+20.1%** |

### 表 2 · 多域（MD_L512_O1，数学↔英文小说 4 段交替 M/E/M/E，3 次域切换）

| 配置 | 总时间 (s) | 吞吐 (tps) | vs Baseline |
|---|---|---|---|
| Baseline | 348.0 | 23.5 | — |
| **EPLB**（官方在线） | 324.3 | 25.3 | +7.7% |
| **OEPLB（修复后）** | **307.4** | **26.6** | **+13.2%** |

**排名（两场景一致）：OEPLB > EPLB > Baseline。** OEPLB 单域超 EPLB +6.8%，多域超 EPLB +5.1%。

### 多域自适应机制生效证据（OEPLB server 日志）
```
[PB-OEPLB-RESET]  domain shift detected: converged=1.078 -> current=1.177  → 清零负载历史重画像
[PB-OEPLB-RESET]  domain shift detected: converged=1.177 -> current=1.709  → 清零
[PB-OEPLB-WINDOW] shift confirmed -- halving sync_window 32 -> 16 -> 8      → 自适应缩窗加速再平衡
逐窗口 avg_ratio before→after: 1.730→1.176, 1.186→1.077, 1.713→1.169, 1.170→1.070,
                              1.183→1.059, 1.239→1.073, 1.273→1.061, 1.136→1.070 (ops 300→49 递减收敛，不再回弹)
```

---

## 5. 一键复现步骤

```sh
cd /workspace/EPLB/OEPLB/quickstart_a100

# 0) 打补丁（幂等；含 forward_normal dispatch-info 修复）
python3 patch_sglang.py

# 1) Baseline
sh launch_baseline_a100.sh /workspace/logs/server_baseline.log &      # 等 "fired up"
OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B \
  python3 ../scripts/run_grid_bench.py baseline ../benchmarks/final_grid/L512_O1.jsonl 1024

# 2) EPLB（官方在线）
sh launch_eplb_a100.sh /workspace/logs/server_eplb.log &
OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B \
  python3 ../scripts/run_grid_bench.py EPLB_single ../benchmarks/final_grid/L512_O1.jsonl 1024

# 3) OEPLB（修复后）
sh launch_oeplb_a100.sh /workspace/logs/server_oeplb.log &
OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B \
  python3 ../scripts/run_grid_bench.py oeplb_fixed ../benchmarks/final_grid/L512_O1.jsonl 1024

# 多域: 把上面 L512_O1.jsonl 换成 ../benchmarks/multidomain/MD_L512_O1.jsonl
```

> 每次切换 arm 前先停掉上一个 server（用 `kill`/SIGTERM，不要 `kill -9`，否则子进程变僵尸占端口）。

## 6. 产物清单
- 补丁器: `patch_sglang.py`（含 `qwen3_moe.py` forward_normal 修复 + server_args static 修复）
- 部署脚本: `deploy_oeplb.sh`（复制 OEPLB/src + diff 三个 patch 文件）
- 启动脚本: `launch_{baseline,eplb,oeplb}_a100.sh` + `env_a100.sh`
- 多域数据集: `../benchmarks/multidomain/MD_L512_O1.jsonl`（8192，M/E/M/E，3 次域切换）
- 结果: `../benchmarks/results/{baseline_noovl_a100,oeplb_a100,EPLB_single,oeplb_fixed_a100,MD_baseline,EPLB_multi,MD_oeplb_fixed}.json`
- 论文归档（含逐窗收敛 keylines）: `../../NEW_PAPER/experiments/a100_bf16_comparison/`
