# OEPLB 快速复现指南

> 目标：在 8×H20 单机上从零重建实验环境，复现 adaptive window 实验。

## 一、硬件与基础环境

```
GPU: 8× NVIDIA H20 96GB, NVLink NV18 全互连, 无IB/RDMA
OS: Ubuntu 22.04 容器
Python: 3.11 (conda base)
CUDA: 12.8 (nvcc)
Driver: 535+ (`nvidia-smi` 验证)
CPU: 64核+, 内存: 256GB+
存储: 300GB+ (235B模型223GB + 环境 + trace)
```

H20 无 IB/RDMA，只有 NVLink——所有 DeepEP 通信都走 NVLink。

## 二、软件栈安装（按顺序）

### 2.1 系统依赖
```bash
apt-get update && apt-get install -y libibverbs-dev libnuma1 git wget
```

### 2.2 PyTorch 2.9.1 + cu128
```bash
pip install torch==2.9.1 --index-url https://mirrors.aliyun.com/pypi/simple/
# 验证: python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.9.1+cu128 True
```

### 2.3 SGLang 0.5.6.post2 + sgl-kernel 0.3.19
```bash
pip install "sglang[all]==0.5.6.post2" "sgl-kernel==0.3.19" --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 2.4 DeepEP v1.2.1（含H20 NVLink patch）
```bash
cd /workspace/build
git clone https://github.com/deepseek-ai/DeepEP.git && cd DeepEP && git checkout v1.2.1
export NVSHMEM_HOME=/opt/conda/lib/python3.11/site-packages/nvidia/nvshmem  # torch 2.9.1自带
```

**Patch 1**: `deep_ep/buffer.py` ~L96-105，IBGDA环境变量设置改为仅在有RDMA rank时执行：
```python
# 原: if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
#       os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1' ...
# 改为: 在 if 内再包一层 if self.runtime.get_num_rdma_ranks() > 1: 才设IBGDA变量
```

**Patch 2**: `csrc/kernels/internode_ll.cu` L181，注释掉IBGDA断言：
```c
// EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
```

```bash
pip install -e . --no-build-isolation

**预编译**: `python3 -m sglang.compile_deep_gemm --model /data/models/Qwen3-235B-A22B-FP8 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention --trust-remote-code`
（必须跑,否则 EPLB normal 模式会在 serving 时触发冷 JIT → 崩溃）
```

### 2.5 DeepGEMM
```bash
cd /workspace/build
git clone https://github.com/deepseek-ai/DeepGEMM.git && cd DeepGEMM && git checkout 35c4bc8
git submodule update --init --recursive
pip install -e . --no-build-isolation

**预编译**: `python3 -m sglang.compile_deep_gemm --model /data/models/Qwen3-235B-A22B-FP8 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention --trust-remote-code`
（必须跑,否则 EPLB normal 模式会在 serving 时触发冷 JIT → 崩溃）
```

### 2.6 模型下载（走ModelScope，huggingface.co不可达）
```bash
pip install modelscope
modelscope download Qwen/Qwen3-235B-A22B-FP8 --local_dir /data/models/Qwen3-235B-A22B-FP8
# 223GB，约15-30分钟
```

## 三、部署 OEPLB 到 SGLang

```bash
SGLANG_PATH=$(python3 -c "import sglang,os; print(os.path.dirname(sglang.__file__))")

# 1. 复制OEPLB核心代码
mkdir -p $SGLANG_PATH/srt/managers/pb_oeplb
cp OEPLB/src/*.py $SGLANG_PATH/srt/managers/pb_oeplb/

# 2. Patch SGLang三个文件（参考 OEPLB/README.md "SGLang 集成" 章节）
# - server_args.py: 加CLI参数(enable-pb-oeplb等)
# - model_executor/model_runner.py: initialize()创建controller, forward()尾部调on_forward_pass_end
# - layers/moe/topk.py: select_experts()后调record_next_layer
```

**注意**: 每次修改 `OEPLB/src/` 后要重新 `cp` 到 sglang 路径（没有symlink）。

### 3.1 关键 bug 修复：ep_dispatch_algorithm 强制为 static

**如果这一步漏掉，OEPLB 会静默不生效！** SGLang 0.5.6.post2 的 `server_args.py` 在 `--enable-pb-oeplb` 单独启用（不带 `--enable-eplb`、`init_expert_location=trivial`）时 **不会** 把 `ep_dispatch_algorithm` 设为 `"static"`，导致：

- `ExpertLocationDispatchInfo.init_new()` 返回 None
- `topk_ids_logical_to_physical()` 原样返回 topk_ids
- PB-OEPLB 真的做了物理权重 P2P，但 token 路由不跟着走 → imbalance 每 window 回弹到初始值 → OEPLB 无净收益

修复：编辑 `${SGLANG_PATH}/srt/server_args.py`（约 L1673），把 static 分支的条件加上 `enable_pb_oeplb`：

```python
# 修复前
if (self.enable_eplb or (self.init_expert_location != "trivial")) and (
    self.ep_dispatch_algorithm is None
):
    self.ep_dispatch_algorithm = "static"

# 修复后
if (self.enable_eplb or self.enable_pb_oeplb or (self.init_expert_location != "trivial")) and (
    self.ep_dispatch_algorithm is None
):
    self.ep_dispatch_algorithm = "static"
```

或者一行 sed：

```bash
SGLANG_PATH=$(python3 -c "import sglang,os; print(os.path.dirname(sglang.__file__))")
sed -i 's|if (self.enable_eplb or (self.init_expert_location != "trivial"))|if (self.enable_eplb or self.enable_pb_oeplb or (self.init_expert_location != "trivial"))|' $SGLANG_PATH/srt/server_args.py
```

修复后 OEPLB 在 L512_O1_realprover (prefill-dense) 上从 **+15.5% 提升到 +19.4%**（详见 `NEW_PAPER/experiments/d38_L512_O1_comparison/repro_20260915/`）。

## 四、环境变量（H20专用，所有启动都需要）

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

## 五、启动服务器

### Baseline
```bash
python3 -m sglang.launch_server \
  --model-path /data/models/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache
```

### OEPLB (推荐配置)
```bash
python3 -m sglang.launch_server \
  [同上] \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.02 \
  --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 16 \
  --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-min-swap-ops 8 \
  --pb-oeplb-max-total-ops 300 \
  --pb-oeplb-decay-factor 0.5 \
  --pb-oeplb-adaptive-window \
  --pb-oeplb-adaptive-decay \
  --pb-oeplb-window-floor 8 \
  --pb-oeplb-window-shift-confirm 2
```

**参数说明**：
- `--pb-oeplb-adaptive-window` / `--pb-oeplb-adaptive-decay`：开启论文 §4.3 的自适应窗口 + 变点时 α→0 一步清零，两者建议同时开
- `--pb-oeplb-window-floor 8`：变点收缩时窗口下界（论文 §4.3 明确 "shrink W 至 floor 8"）；默认 32 会过大
- `--pb-oeplb-window-shift-confirm 2`：连续 2 窗低 cos_sim 才确认域切换（论文 §4.2），默认 1 会误触发
- `--pb-oeplb-sync-window 16`：**初值**，adaptive-window 开启后会被动态调整到 [floor, 128]
- `--pb-oeplb-cooldown-steps` **不存在**，别照抄旧文档；RESET cooldown=3 在代码里硬编码

### 5.1 验证 OEPLB 是否真的生效（自检）

启动后必看两处：

**① server log 里 `ep_dispatch_algorithm` 必须是 `'static'`**（不能是 `None`）：

```bash
grep -oE "ep_dispatch_algorithm='?[a-zA-Z]+'?" /path/to/server.log | head -1
# 期望输出：ep_dispatch_algorithm='static'
# 如果是 ep_dispatch_algorithm=None → 3.1 的 bug 修复没做，OEPLB 无效
```

**② 应能看到 `[PB-OEPLB-DIAG]` 行，且 window N+1 的 `avg_before` ≈ window N 的 `avg_after`**（无回弹）：

```bash
grep "PB-OEPLB-DIAG" /path/to/server.log | awk '/DP0 /' | head -5
```

预期输出（L512_O1 prefill-dense）：
```
Window 1: layers_touched=93 total_ops=298 avg_before=1.74 avg_after=1.18 ...
Window 2: layers_touched=90 total_ops=225 avg_before=1.19 avg_after=1.07 ...
                                                    ^^^^ ≈ 1.18 上一窗 after，无回弹
```
如果 `avg_before` 每次都回弹到 1.55+，说明路由没同步跟着权重走 → 3.1 bug 修复失败。

## 六、数据集

### 已有数据集位置（在EPLB仓库内）
```
OEPLB/benchmarks/final_grid/       ← 最终实验用的数据集
  L256_O1.jsonl     (8192条, Prover-V1, ~250tok输入, max_tokens=1)
  L256_O64.jsonl    (8192条, 同上, max_tokens=64)
  L256_O256.jsonl   (8192条, 同上, max_tokens=256)
  L256_O1024.jsonl  (8192条, 同上, max_tokens=1024)
  L512_O*.jsonl     (8192条, Prover-V1 ~500tok)
  L1024_O*.jsonl    (4096条, BookCorpus ~1024tok)
  L2048_O*.jsonl    (2048条, BookCorpus ~2048tok)
  L4096_O*.jsonl    (1024条, BookCorpus ~4096tok)
```

### 从ModelScope重新下载原始数据
```bash
# DeepSeek-Prover-V1 (数学证明, 27503条)
modelscope download --dataset AI-ModelScope/DeepSeek-Prover-V1
# → /root/.cache/modelscope/datasets/AI-ModelScope--DeepSeek-Prover-V1/snapshots/master/dataset.jsonl

# BookCorpus (英文小说, 17868本)
modelscope download --dataset youngchen/BookCorpus
# → books1.tar.gz, 解压后 books1/epubtxt/*.txt

# HellaSwag (常识推理)
curl -o raw_hellaswag_val.jsonl https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/hellaswag/hellaswag_val.jsonl
```

### 数据集构造方法
- L=256/512: 从Prover-V1按token范围筛选自然prompt（每条独立定理）
- L=1024/2048/4096: 从BookCorpus随机截取指定字符数（~3.5字符/token）
- 输出长度通过jsonl里的 `max_tokens` 字段控制，`ignore_eos=True`(O>1时)

## 七、压测方法

```bash
cd OEPLB/scripts
python3 run_grid_bench.py <label> <dataset.jsonl> <concurrency>
# 例: python3 run_grid_bench.py test /path/to/L256_O1.jsonl 256
```

**关键注意事项**:
- 并发度必须够高才能体现OEPLB收益: L=256用conc=256, L=4096用conc=16
- warmup用/health不过模型（不污染OEPLB计数器）
- 结果输出到 `OEPLB/benchmarks/results/<label>.json`

## 八、批量实验编排

```bash
# 全面网格 (100次: 5长度×4输出×(1BL+4sw+1adaptive))
python3 run_final_sweep.py

# Adaptive vs Static Oracle对比 (60次)
python3 run_adaptive_optimal.py
```

## 九、关键文档

| 文件 | 内容 |
|------|------|
| `OEPLB/README.md` | 项目概述、SGLang集成方法、历史版本演进 |
| `OEPLB/COMPREHENSIVE_EXPERIMENT_LOG.md` | 最终实验结果汇总、代码优化记录 |
| `EPLB_VS_OEPLB_REPORT.md` | 之前的EPLB vs OEPLB对比（normal模式，旧数据，仅参考） |
| `FINAL_235B_REPORT.md` | 235B模型上的完整实验历史（adaptive window最初验证） |

## 十、当前代码状态

### OEPLB/src/ 核心文件
- `controller.py`: v0.11 + min_swap_ops + ratio-aware grow抑制(待验证)
- `config.py`: 含 min_swap_ops、adaptive_window 等全部参数
- `async_swapper.py`: 含timing instrumentation（begin()内部计时）
- `rebalancer.py`: swap plan构建（greedy最热slot↔最冷slot）
- `fast_metadata.py`: 向量化metadata重建
- `routing_tracer.py`: 路由追踪（实验用）

### SGLang patch
- `server_args.py`: 全部OEPLB CLI参数
- `model_runner.py`: controller初始化 + on_forward_pass_end
- `topk.py`: record_next_layer hook

## 十一、已知问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| DeepGEMM特定shape CUDA error | H20+特定MxNxK不兼容 | `--moe-runner-backend triton` |
| DeepEP low_latency模式crash | 无IB时IBGDA初始化失败 | buffer.py patch(见2.4) |
| EPLB+auto模式NotImplementedError | expert_distribution recorder不支持auto dispatch | EPLB只能用`--deepep-mode normal` |
| EPLB rebalance后CUDA error | rebalance更新location时kernel config异常 | EPLB的bug,偶发,重试可恢复 |
| 服务器端口残留导致重启失败 | `kill -9`后子进程变僵尸占端口 | 用`kill`(SIGTERM)而不是`kill -9` |
| 循环数据集TPS虚高 | radix cache命中跳过prefill | 必须加`--disable-radix-cache` |
| DeepGEMM首次JIT编译慢 | 首次编译无缓存 | 需要1-2分钟，后续有缓存 |
| huggingface.co不可达 | 网络限制 | 模型/数据集全部走ModelScope |
| 高并发+长输出触发watchdog timeout | 单请求耗时超过默认watchdog阈值 | 加 `--watchdog-timeout 600` |
| 服务器重启累积zombie进程 | 子进程未被完全回收 | 不影响功能，但累积影响OS响应速度，定期重启容器 |

## 十二、核心参数速查表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pb-oeplb-threshold-ratio` | 1.02 | 不均衡度阈值，ratio低于此值不触发swap |
| `--pb-oeplb-sync-window` | 16 | 每多少个forward pass检查一次不均衡度(推荐值,见REPRODUCE.md) |
| `--pb-oeplb-min-prefill-tokens` | 256 | 一个窗口内至少积累这么多prefill token才做决策 |
| `--pb-oeplb-max-swaps-per-layer` | 64 | 单层单次决策最多swap多少对slot |
| `--pb-oeplb-max-total-swap-layers` | 94 | 全局预算涉及的最大层数(Qwen3-235B有94层MoE) |
| `--pb-oeplb-max-total-ops` | 300 | 单次决策最大swap数(冷启动约用240-250) |
| `--pb-oeplb-min-swap-ops` | 8 | 低于此数跳过(不值得P2P开销) |
| `--pb-oeplb-decay-factor` | 0.5 | 负载历史每窗口衰减系数,3窗口后旧信号仅剩12.5% |
| `--pb-oeplb-adaptive-window` | flag | 开启自适应窗口(收敛/振荡自动伸缩 W) |
| `--pb-oeplb-adaptive-decay` | flag | 变点时 α→0 一步清空旧域历史 |
| `--pb-oeplb-window-floor` | 8 (论文对齐) | 收缩下界；CLI 默认 32，需显式改到 8 |
| `--pb-oeplb-window-shift-confirm` | 2 (论文对齐) | 连续 N 窗低 cos_sim 才判定变点；默认 1 |

注：`--pb-oeplb-cooldown-steps` 从未注册进 argparse，不要照抄旧文档，会报 `unrecognized arguments`。
