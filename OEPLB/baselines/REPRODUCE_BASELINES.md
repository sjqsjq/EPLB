# §5.3.1 两张对比表的 Baseline 复现文档（硬件可移植）

> 目标读者：想在**另一台硬件**（A100 / H800 / GB200 …）上复现论文 §5.3.1
> **同域表**与**跨域表**里 5 个 baseline 的人。
> 重点覆盖两个"我们自己复现的外部静态方法"：**MoETuner**（最新，ILP1 离线）
> 和 **DataForest-Remap**（ISCA'26，prefill-guided 静态放置）。
> identity / EPLB静态 / EPLB动态 是 SGLang 原生能力，附带说明。

---

## 0. 两张表在复现什么

| 表 | 场景 | 数据集 | 采样 | 并发 | N | 口径 |
|---|---|---|---|---|---|---|
| **同域表** | prover, pinned 10× 不均衡 | `prover_256tok_out1.jsonl` | `max_new_tokens=1`（纯 prefill） | 无上限（一次 256 并发） | 256 | 3 run，**丢 r1 冷启**，取 r2/r3 中位 |
| **跨域表** | freq6，book↔prover 6 段频繁切换 | `book_4438tok_O10.jsonl` + `prover_2048tok_out1.jsonl` | `max_new_tokens=10` | `Semaphore(32)` | 1800 | 3 run，取中位 |

5 个方法（PB-OEPLB 是本文方法，不在本文档范围）：

| # | 方法 | 类型 | 放置来源 | 冗余专家 | 复现难度 |
|---|---|---|---|---|---|
| 1 | **identity** | 无放置 | 默认 contiguous | 无 | 极低 |
| 2 | **EPLB 静态** | SGLang 原生 | 录制 count → `rebalance_experts` | 16 | 低 |
| 3 | **EPLB 动态** | SGLang 原生 | `--enable-eplb` 周期重平衡 | 16 | 低 |
| 4 | **DataForest-Remap** | 外部复现 | 录制 prefill count → `rebalance_experts`（=Algorithm 2 Remap） | 无 | 中 |
| 5 | **MoETuner** | 外部复现（最新） | profile → **Gurobi ILP1** per-layer 聚类 | 无 | 高（需 Gurobi license） |

> **关键概念**：方法 2/4/5 都产出一个"专家放置"，通过 SGLang 的
> `--init-expert-location <json>` 注入。区别只在**放置是怎么算出来的**：
> - EPLB静态 / DataForest-Remap：喂 `logical_count`（94×128 token 计数），
>   SGLang 内部调 `rebalance_experts`（贪心最小负载）现算 → 二者放置算法**等价**，
>   差异仅在 DataForest **无冗余**、EPLB静态 **带 16 冗余 + `deepep_mode=normal`**。
> - MoETuner：喂 `physical_to_logical_map`（Gurobi ILP 解出的显式映射），SGLang 直接用。

---

## 1. 通用环境

### 1.1 软件版本（H20 原始环境）

```
SGLang        0.5.6.post2
torch         2.9.1+cu128
gurobipy      13.0.3        # 仅 MoETuner 需要；需有效 license
模型          Qwen3-235B-A22B-FP8 (94 层, 128 专家, top-8)
并行          TP=8 DP=8 EP=8 (--enable-dp-attention)，每卡 16 本地专家
```

### 1.2 公共环境变量（`env_235b.sh`）

```bash
export NVSHMEM_HOME=/opt/conda/lib/python3.11/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:$LD_LIBRARY_PATH"
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

> **硬件移植第一处改动**：`NVSHMEM_HOME` 路径、`NCCL_P2P_LEVEL=NVL`（NVLink）
> 依机器而定。非 NVLink 互连（如 PCIe / IB 跨节点）需相应调整或去掉。

### 1.3 两种 MoE 前向路径（决定 launch flag）

| | **H20（原始）** | **A100 / 其它 BF16 硬件** |
|---|---|---|
| MoE 路径 | `forward_deepep` | `forward_normal` |
| 精度 | FP8 | BF16 |
| A2A backend | DeepEP | NCCL |
| MoE runner | DeepGEMM | Triton |
| 需要的 flag | `--moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --quantization fp8` | **去掉**这些；`--dtype bfloat16`，用 Triton runner |

> ⚠️ **A100 路径有个致命 bug 必须先打补丁**：`forward_normal` 调 `self.topk(...)`
> 时漏传 `expert_location_dispatch_info` → logical→physical 重映射不生效 →
> 放置形同虚设。见 `quickstart_a100/ENVIRONMENT.md` 补丁 B（镜像 `forward_deepep`
> 把 dispatch info 传进去）。**不打这个补丁，所有 baseline 的放置都不生效，
> 复现出来的数字全是 identity。**

---

## 2. 数据集准备

所有数据集在 `/data/minghua/sjq/OEPLBdata/datasets/`（或迁移到新机器后改路径）。

### 2.1 同域：pinned10x fair-split（无泄漏）

`prover_256tok_out1.jsonl` 共 2048 行，确定性对半切：

```bash
cd /workspace/EPLB/OEPLB/baselines/moetuner/artifacts
SRC=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
head -n 1024 "$SRC" > pinned10x_head1024.jsonl   # profile / MoETuner ILP 输入
tail -n 1024 "$SRC" > pinned10x_tail1024.jsonl   # bench（与 profile disjoint）
```

- **profile 用 head1024，bench 用 tail1024**（fair split，防止"用测试集调放置"的泄漏）。
- 同域表里 identity/MoETuner 的 bench 直接跑 `prover_256tok_out1.jsonl` 前 N=256 行
  （见 `bench_0914.py`），pinned 变体跑 tail1024。**两者是同分布**，故可并列。

### 2.2 跨域：freq6（内联构造，无需落盘）

`freq6_bench.py` / `freq6_bench_json.py` 运行时内联拼：

```
book_4438tok_O10.jsonl[:900]    段0,2,4
prover_2048tok_out1.jsonl[:900] 段1,3,5
seg = N//6 = 300，6 段交替 → 共 1800 条
```

- 每段取前 300 条；`i%2==0` 用 book，否则 prover；`max_new_tokens=10`。
- 跨域表的 placement 全部**冻结自 prover 分布**，对 book 段错配 → 这正是要展示的
  "离线放置跨域失败"。

---

## 3. Baseline 1：identity（默认放置）

最简单的基线，也是所有 gain 的分母。

### 3.1 launch（H20）— `launch_0914_identity.sh`

```bash
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --dtype bfloat16 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.78 \
  --disable-cuda-graph --skip-server-warmup \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
```

> **baseline 统一口径**：`--disable-cuda-graph --mem-fraction-static 0.78
> --disable-radix-cache`。这三个 flag 保证 5 个方法可比（CUDA graph 会掩盖
> 重映射效果；radix cache 会跨 run 污染；mem-fraction 统一显存占用）。

### 3.2 A100 版 launch（去 DeepEP/FP8）

```bash
exec python3 -m sglang.launch_server \
  --model-path /path/to/Qwen3-235B-A22B \
  --dtype bfloat16 --tp 8 --ep-size 8 --enable-dp-attention \
  --mem-fraction-static 0.78 \
  --disable-cuda-graph --skip-server-warmup \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
# 前提：已打 §1.3 的 forward_normal dispatch-info 补丁
```

### 3.3 跑 bench

```bash
# 同域
python3 bench_0914.py \
  /data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl \
  256 identity_r1
# 跨域
python3 freq6_bench_json.py 32 1800 identity_r1
```

结果 JSON 落到 `/workspace/EPLB/OEPLB/benchmarks/results/`。跑 3 次，同域丢 r1。

---

## 4. Baseline 2 & 3：EPLB 静态 / 动态（SGLang 原生）

两者都要 **16 冗余专家 + `--deepep-mode normal`**（normal 才支持权重迁移，
但会**禁用 CUDA graph**）。

### 4.1 EPLB 静态（冻结 + redundant16）— `launch_eplb_static.sh`

```bash
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /data/models/Qwen3-235B-A22B-FP8 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.78 \
  --ep-num-redundant-experts 16 \
  --init-expert-location /workspace/logs/datafore_prover_placement.json \
  --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
```

- 静态 = 用一份 prover 分布的 `logical_count`（`datafore_prover_placement.json`）
  在 init 时算一次放置，之后**冻结不重平衡**。

### 4.2 EPLB 动态（周期全量重平衡）— `launch_eplb_dyn.sh`

```bash
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /data/models/Qwen3-235B-A22B-FP8 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.78 \
  --ep-num-redundant-experts 16 --enable-eplb \
  --eplb-rebalance-num-iterations 100 --expert-distribution-recorder-buffer-size 32 \
  --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
```

- 动态 = `--enable-eplb`，每 100 iter 全量重平衡一次（跨域表里跑了 64 次）。
- 每次重平衡阻塞推理 0.5–4.5s（论文 §5.3 的"阻塞"论据）。

---

## 5. Baseline 4：DataForest-Remap（ISCA'26，"另一个"要复现的）

> DataFore 论文两半：前半 wafer-scale 硬件改造（本集群无该硬件，不复现）；
> **后半 real-cluster prefill-guided 静态放置（Algorithm 2 Remap）= 本节**。

### 5.1 原理：Remap = 贪心最小负载放置 = SGLang `rebalance_experts`

DataForest Remap（Algorithm 2）：按 prefill 频次对专家降序，逐个塞到**当前负载
最小且还有空位**的 GPU（每 GPU `E/G` 个，无额外 slot）→ 一个 permutation。
SGLang 内置的 `rebalance_experts(tokens_per_expert=logical_count)` 做的正是这件事。
所以 **DataForest-Remap 复现 = 录 prefill count → 喂 `--init-expert-location`**。

### 5.2 步骤 A：录 prefill 路由计数

起一个带**原生 recorder** 的 identity server（`stat` 模式）：

```bash
# launch_profile.sh 关键 flag
python3 -m sglang.launch_server ... \
  --moe-a2a-backend deepep --deepep-mode normal \
  --expert-distribution-recorder-mode stat \
  --port 30000
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/path/to/dump   # .pt 落这里
```

用 HTTP 端点开/关/导出录制（容器无 curl，用 `wget`）：

```bash
wget --post-data='' -qO- http://127.0.0.1:30000/start_expert_distribution_record
#   ... 跑 profile 数据集（head1024 或 prover），结果丢弃，只要路由计数 ...
wget --post-data='' -qO- http://127.0.0.1:30000/stop_expert_distribution_record
wget --post-data='' -qO- http://127.0.0.1:30000/dump_expert_distribution_record
#   → dump 目录出现 expert_distribution_recorder_*_0.pt（rank0 已 all-reduce 全局求和）
```

### 5.3 步骤 B：转成 `logical_count` JSON

`.pt` 里 `logical_count` 形状 `[dim_extra, L=94, E=128]`，对 dim0 求和 → `[94,128]`：

```python
import torch, json, glob
f = sorted(glob.glob("dump/expert_distribution_recorder_*_0.pt"))[-1]  # 用 rank0
d = torch.load(f, map_location="cpu", weights_only=False)
lc = torch.as_tensor(d["logical_count"]).sum(dim=0).to(torch.int64)     # [94,128]
json.dump({"logical_count": lc.tolist()}, open("datafore_remap_placement.json","w"))
```

- 产出 `datafore_remap_placement.json`（键**只有** `logical_count`，94×128）。
- SGLang 加载时命中 `elif "logical_count" in data_dict:` 分支（`expert_location.py:563`）
  → `init_by_eplb` → `rebalance_experts` 现算 physical map。
- `datafore_prover_placement.json` = 同法但用 prover 全量路由录制（跨域表里的
  "冻结自 prover" oracle 放置）。

> **等价性说明**：`datafore_repro/placement_algo.py` 是 Algorithm 2 的独立 Python
> 复现（`remap_based_placement`），用于论文 Fig.17 的**模拟器**口径（+14.5% vs 论文
> +15.5%，1pp 内）。§5.3.1 两张表用的是**真实 kernel 端到端**口径，直接走 SGLang
> 的 `rebalance_experts`（与 Remap 算法等价），不经过 `placement_algo.py`。

### 5.4 步骤 C：launch + bench

```bash
# launch_datafore.sh（无冗余，deepep_mode=auto）
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /data/models/Qwen3-235B-A22B-FP8 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.78 \
  --init-expert-location /workspace/logs/datafore_remap_placement.json \
  --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
# 然后跑 §3.3 的 bench
```

- **无冗余**：不加 `--ep-num-redundant-experts`（=0）。
- **`deepep-mode auto`**（不是 normal）→ 保留 CUDA graph 兼容路径。这是 DataForest
  比 EPLB静态快的原因之一（EPLB静态被 normal 模式拖慢）。

---

## 6. Baseline 5：MoETuner（arXiv:2502.06643，最新，重点）

> MoETuner = **per-layer ILP1 聚类**：把 E=128 专家聚成 G=8 个 cluster（每 cluster
> K=16），最小化每层 `Σ_c |T_c − T_bar|`（cluster 间负载偏差）。用 **Gurobi** 解，
> 产出冻结的 `physical_to_logical_map`。无冗余、无在线适应。

三阶段流水线（`driver_moetuner.sh <variant>`，variant 用 `pinned_fair`）：

### 6.1 Phase P — profile（录 per-layer 专家 token 计数）

同 §5.2，起 `launch_profile.sh`（`--expert-distribution-recorder-mode stat`，
`--deepep-mode normal`），arm recorder，**跑 head1024**（profile split），stop+dump。

```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$BASE/artifacts/profile_pinned_fair
bash launch_profile.sh &                       # 等 "ready to roll"
wget --post-data='' -qO- http://127.0.0.1:30000/start_expert_distribution_record
#   跑 pinned10x_head1024.jsonl（结果丢弃）
wget --post-data='' -qO- http://127.0.0.1:30000/stop_expert_distribution_record
wget --post-data='' -qO- http://127.0.0.1:30000/dump_expert_distribution_record
```

### 6.2 Phase S — `.pt` → `P.npz` → Gurobi ILP1 → `placement.json`

```bash
# (1) pt → npz：抽出 P[94,128]
python3 src/pt2npz.py \
  --src-glob "$PROFILE_DIR/expert_distribution_recorder_*.pt" \
  --out      "$PROFILE_DIR/P.npz"

# (2) ILP1 求解（Gurobi，每层 TimeLimit 60s，MIPGap 0.005）
python3 src/solve_ilp.py \
  --profile "$PROFILE_DIR/P.npz" \
  --num-clusters 8 --experts-per-cluster 16 \
  --time-limit 60 --mip-gap 0.005 \
  --out "$BASE/artifacts/placement_pinned_fair.json"
```

- **ILP1 模型**（`solve_ilp.py`）：变量 `x[c,e]∈{0,1}`；约束 (1) 每专家恰属一个
  cluster，(2) 每 cluster 恰 K=16 个专家；目标 `min Σ_c d_c`，`d_c ≥ |T_c − T_bar|`
  线性化。逐层解 94 次。
- **cluster→GPU**：贪心，cluster 按总 token 降序、依次放到当前最空 GPU
  （`greedy_cluster_to_gpu`）。
- **产出**：`placement_pinned_fair.json` = `{"physical_to_logical_map": [[94×128]]}`
  + `.meta.json`（solver 统计）。
- **实测**：total solver 1015.5s（≈17min），`mean_imbalance = 1.000`（该数据集可完美均衡）。

> **Gurobi license**：`solve_ilp.py` `import gurobipy`。新机器需装 gurobipy 并配好
> license（`GRB_LICENSE_FILE` 或 token）。无 license 时可改用 CBC/HiGHS 近似，但
> 解质量与耗时不同，需在论文口径里注明。

### 6.3 Phase B — launch（注入 placement）+ bench

```bash
# launch_0914_moetuner.sh = identity 的 flag + 一行 --init-expert-location
export PLACEMENT=$BASE/artifacts/placement_pinned_fair.json
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --dtype bfloat16 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.78 --disable-cuda-graph --skip-server-warmup \
  --init-expert-location "$PLACEMENT" \
  --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache --watchdog-timeout 600
```

- MoETuner placement 命中 `physical_to_logical_map` 分支 → `init_by_mapping` 直接用，
  冻结、无重平衡、无冗余、`deepep-mode auto`（保 CUDA graph 兼容）。
- 同域跑 `bench_0914.py prover_256tok_out1.jsonl 256`；
  跨域**复用同一份 placement**（fit 于 prover）跑 `freq6_bench_json.py 32 1800`。

### 6.4 一键 driver

```bash
cd /workspace/EPLB/OEPLB/baselines/moetuner
bash driver_moetuner.sh pinned_fair     # P→S→B 全流程（profile+solve+bench）
bash driver_0914.sh identity 3          # 同域 identity 3 run
bash driver_0914.sh moetuner 3          # 同域 MoETuner 3 run
bash driver_freq6.sh identity 3         # 跨域 identity 3 run
bash driver_freq6.sh moetuner 3         # 跨域 MoETuner 3 run
```

---

## 7. Bench harness 与聚合口径

### 7.1 同域 `bench_0914.py`（= 09/14 `prover_prefill.py`）

```python
# /generate，max_new_tokens=1，temperature=0，无并发上限
# 一次 asyncio.gather 发 N=256 条；throughput = ok / wall_time (req/s)
python3 bench_0914.py <dataset.jsonl> <N=256> <label>
# → results/_0914_<label>.json  {"label","N","ok","time","tps"}
```

### 7.2 跨域 `freq6_bench_json.py`（= 09/14 `freq6_bench.py`）

```python
# /generate，max_new_tokens=10，Semaphore(32)，N=1800，6 段 book↔prover 交替
python3 freq6_bench_json.py <conc=32> <N=1800> <label>
# → results/_freq6_<label>.json
```

### 7.3 聚合规则（务必对齐，否则数字不可比）

- **同域**：3 run，**丢 r1**（冷启动，identity r1=15.7 vs r2=55.2，差 3.5×），
  取 r2/r3 **中位**。
- **跨域**：3 run 取**中位**。
- **gain 口径**：论文表里 MoETuner 的 gain 是**对齐 09/14 表的 identity=62.4**
  计算的（跨 session identity 有 ~11% run-to-run 漂移：55.7 vs 62.4）。
  **不变量是比值**：MoETuner 70.5 / identity 55.7 = **1.27×**（同 session），
  换算到 09/14 identity=62.4 → +13.0%。复现时请**同 session 内**比 identity，
  再换算，避免跨 session 漂移。

---

## 8. 硬件可移植性（迁移到新机器要改什么）

### 8.1 可直接搬运（硬件无关）

| 资产 | 说明 |
|---|---|
| `placement_pinned_fair.json`（MoETuner） | 纯 94×128 整数映射，**跨硬件通用** |
| `datafore_remap_placement.json` / `datafore_prover_placement.json` | 纯 `logical_count`，跨硬件通用 |
| 数据集 jsonl | 纯文本 prompt，跨硬件通用 |
| `bench_0914.py` / `freq6_bench_json.py` | 纯 HTTP client，跨硬件通用 |
| `src/solve_ilp.py` / `pt2npz.py` | 离线 CPU 计算（只需 Gurobi），跨硬件通用 |

> **含义**：如果你只想在**新硬件上验证放置效果**，可以把 H20 上算好的
> placement.json 直接 `scp` 过去，跳过 Phase P/S，只跑 Phase B + bench。
> 但严格复现应在新硬件上**重新 profile+solve**（路由分布随硬件不变，但
> 录制的是新机器上的实际 forward）。

### 8.2 必须改（硬件相关）

1. **launch flag**：非 DeepEP 硬件（A100/H800 BF16）去掉
   `--moe-a2a-backend deepep --deepep-mode * --moe-runner-backend deep_gemm
   --quantization fp8`，改 `--dtype bfloat16` + Triton runner。见 §1.3。
2. **模型权重**：FP8 → BF16（不同 checkpoint）。
3. **`--mem-fraction-static`**：依显存大小调（80GB A100 vs 96GB H20）。
4. **`forward_normal` dispatch-info 补丁**：A100 路径**必打**（§1.3），否则放置不生效。
5. **`env_235b.sh`**：NVSHMEM/NCCL 路径与互连类型依机器改。
6. **EP 大小**：若非 8 卡，`--ep-size` / ILP 的 `--num-clusters G` / `--experts-per-cluster K`
   要同步改（`G×K=E=128`）。

### 8.3 预期数字会变（理论预测，见 bound_curve.py）

同域 gain 随硬件的 **β** 和**死区 r_k** 变化。以 235B EP=8 L512（r_before≈1.74）为例：

| 硬件 | r_k | β | Δ_max | 预测 baseline 可优化空间 |
|---|---|---|---|---|
| H20 | 1.093 | 0.352 | 22.7% | 大（原始表：DataForest +18%, MoETuner +13%） |
| A100 | 1.093 | 0.284 | 18.3% | 中大 |
| H800 | ~1.0(通信驱动) | 0.048 | 3.6% | 小 |
| GB200 dense | 1.353 | 0.123 | 4.7% | 小 |
| GB200 sparse | 1.707 | 0.068 | 0.2% | ~0 |

> **含义**：在 H800/GB200 上，**所有静态放置 baseline 的绝对 gain 都会显著缩小**
> （因为 GEMM 被通信 overlap，消除不均衡的收益被死区吃掉）。但 baseline 之间的
> **相对排序**（PB-OEPLB > DataForest > MoETuner > EPLB静态）应保持不变——
> 这正是要在新硬件上验证的核心结论。

---

## 9. 归档路径与文件清单

### 9.1 MoETuner（`moetuner_baseline_20260916`）

```
/data/minghua/sjq/OEPLBdata/experiment_logs/moetuner_baseline_20260916/
├── EXPERIMENT_LOG.md            # 完整实验记录（结果表 + 实现注记）
├── scripts/
│   ├── driver_moetuner.sh       # P→S→B 主 driver（variant: fair/leaked/pinned_fair）
│   ├── driver_0914.sh           # 同域 identity/moetuner N-run driver
│   ├── driver_freq6.sh          # 跨域 identity/moetuner N-run driver
│   ├── launch_profile.sh        # Phase P：recorder server（deepep normal + stat）
│   ├── launch_0914_identity.sh  # Phase B：identity bench server
│   ├── launch_0914_moetuner.sh  # Phase B：moetuner bench server（+init-expert-location）
│   ├── bench_0914.py            # 同域 bench harness
│   ├── freq6_bench_json.py      # 跨域 bench harness
│   └── src/{pt2npz,solve_ilp,split_dataset}.py
├── artifacts/
│   ├── placement_pinned_fair.json{,.meta.json}   # 最终放置（可直接搬新硬件）
│   ├── P_pinned_fair.npz                          # profiled P[94,128]
│   ├── pinned10x_head1024.jsonl                   # profile split
│   └── pinned10x_tail1024.jsonl                   # bench split
├── results/                     # 各 run 的 tps JSON
└── logs/                        # 各 phase stdout
```

工作副本：`/workspace/EPLB/OEPLB/baselines/moetuner/`（含 `driver_*.sh`, `src/`, `artifacts/`）。

### 9.2 DataForest / EPLB / identity（`baseline_comparison_20260914`）

```
/data/minghua/sjq/OEPLBdata/experiment_logs/baseline_comparison_20260914/
├── launch_identity_clean.sh     # identity
├── launch_eplb_static.sh        # EPLB 静态（redundant16 + normal + prover placement）
├── launch_eplb_dyn.sh           # EPLB 动态（--enable-eplb）
├── launch_datafore.sh           # DataForest-Remap（remap placement, 无冗余, auto）
├── launch_datafore_prover.sh    # DataForest prover 冻结放置
├── launch_oeplb.sh              # PB-OEPLB（本文方法，参考用）
├── datafore_remap_placement.json    # {"logical_count":[94×128]}
├── datafore_prover_placement.json   # 同上，prover 全量
├── logical_count_94x128.npy         # 原始计数
├── prover_prefill.py / freq6_bench.py   # bench harness（09/14 原版）
└── results.json
```

DataForest Algorithm 2 独立复现（模拟器口径，Fig.17）：
`/workspace/EPLB/OEPLB/datafore_repro/{placement_algo.py, REPRODUCTION_REPORT.md, placements/}`。

---

## 10. 复现验证清单

跑完后逐项核对：

- [ ] **补丁生效**（A100）：identity 与 DataForest 的 tps 应显著不同；若几乎相等，
      说明 `forward_normal` dispatch-info 补丁没打上，放置未生效。
- [ ] **recorder 出数**：Phase P 后 dump 目录有 `expert_distribution_recorder_*_0.pt`，
      `pt2npz.py` 报 `P shape=(94,128)`，layer0 imbalance > 1（prover 是 pinned 10×）。
- [ ] **ILP 收敛**：`solve_ilp.py` 报 `mean_imbalance ≈ 1.000`（prover 可完美均衡），
      total solve ~1000s（Gurobi TimeLimit 60s/层 × 94）。
- [ ] **placement 格式**：MoETuner = `physical_to_logical_map`；DataForest = `logical_count`。
      喂错键会走错 SGLang 分支。
- [ ] **同域丢 r1**：identity r1 应远小于 r2/r3（冷启），中位取 r2/r3。
- [ ] **同 session 比 identity**：gain 用同 session identity 归一，再换算到表口径。
- [ ] **相对排序**：新硬件上应满足 PB-OEPLB ≳ DataForest > MoETuner > EPLB静态 > EPLB动态
      （同域）；跨域仅 PB-OEPLB 正、其余 ≤0（离线放置跨域失败）。
- [ ] **绝对 gain 缩小符合预期**：H800/GB200 上所有静态 baseline gain 应远小于 H20
      （死区效应，§8.3）。

---

## 附：最短复现路径（已有 placement，只想验证）

如果你已把 H20 算好的 `placement_pinned_fair.json` 和 `datafore_remap_placement.json`
搬到新机器，只想验证放置效果：

```bash
# 1. 起 identity（§3.1/3.2，按硬件选 flag），bench 3 run → identity 分母
# 2. 起 DataForest（§5.4，--init-expert-location datafore_remap_placement.json），bench
# 3. 起 MoETuner（§6.3，--init-expert-location placement_pinned_fair.json），bench
# 4. 同 session 归一算 gain，核对 §10 清单
```

跳过 Phase P/S（profile + Gurobi），最省事。严格复现才需从头 profile+solve。
