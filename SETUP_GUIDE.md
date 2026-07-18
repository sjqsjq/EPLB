# OEPLB 单机多卡 H20 环境搭建 & 快速验证指南

> **重要更正**（2026-07-17，基于8卡H20实机搭建踩坑反馈）：
> 本指南早期版本有多处错误，已全部修正。如果你之前按旧版操作过，请对照"已知踩坑"章节检查。

## 一、硬件要求

- **GPU**: 4×/8× NVIDIA H20 (96GB/卡)，需要 NVLink 连接（单机，非跨节点 IB）
- **CPU**: 64 核以上推荐
- **内存**: 256GB+ 系统内存
- **存储**: 100GB+ 可用空间（模型 31GB + 环境 + trace 文件）

## 二、软件环境

### 2.1 基础镜像

使用 SGLang 官方容器镜像作为起点：
```bash
docker pull lmsysorg/sglang:v0.5.6-cu128
```

核心版本要求：
| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | |
| PyTorch | 2.9.1+cu128 | 必须 ≤2.9.x；DeepEP v1.x 不兼容 torch 2.10+ |
| SGLang | 0.5.6 | |
| CUDA | 12.8+ | nvcc + runtime |

额外系统依赖（容器里可能缺）：
```bash
apt-get update && apt-get install -y libibverbs-dev libnuma1
```

### 2.2 安装 DeepEP（关键：必须用 v1.x 分支）

> **踩坑1**: `git clone` 默认拿到的是 DeepEP V2（NCCL Gin backend，需要 torch 2.10+），
> 跟本项目完全不兼容。**必须 checkout v1.2.1 tag**。

```bash
cd /workspace
git clone https://github.com/deepseek-ai/DeepEP.git deps_src/DeepEP
cd deps_src/DeepEP
git checkout v1.2.1   # ← 关键！不要用 main 分支

# 安装 NVSHMEM（DeepEP v1.x 的依赖）
# 踩坑2: pip install nvidia-nvshmem-cu12 的 wheel 版本在 nvlink -dlink 时会报
# Undefined reference，必须用官方二进制 tar 包：
wget https://developer.download.nvidia.com/compute/nvshmem/3.3.9/local_installers/nvshmem_3.3.9-1+cuda12.8_x86_64.tar.bz2
tar xf nvshmem_3.3.9-1+cuda12.8_x86_64.tar.bz2
export NVSHMEM_HOME=$(pwd)/nvshmem_3.3.9-1+cuda12.8

# 编译安装 DeepEP
# 踩坑3: pip 默认 build isolation 会让 setup.py 找不到 torch
pip install -e . --no-build-isolation
```

#### DeepEP 单机 NVLink patch（必须）

> **踩坑4**: 早期版本的 patch 描述不完整，会导致 low_latency 模式下
> `root_unique_id_opt.has_value()` 断言失败。以下是**正确的**完整 patch：

**`deep_ep/buffer.py`** 约第 96-105 行，修改思路：
- "广播 NVSHMEM unique id" 这部分对 `low_latency_mode` **必须保留执行**
- 只把"强制设置 IBGDA 环境变量"那几行改成仅在有 RDMA rank 时才执行

```python
# 原始代码:
if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
    # Enable IBGDA
    ...
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1' if self.runtime.get_num_rdma_ranks() > 1 else '0'
    os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'
    ...  # ← 这里还有广播 unique_id 的代码

# 正确的 patch（不要简单地把整个 if 块改成 >1）:
# 方法: 让 IBGDA 环境变量设置仅在 rdma_ranks > 1 时执行，
# 但 unique_id 广播和 NVSHMEM 初始化仍在 low_latency_mode 时执行。
# 具体修改取决于你的 DeepEP v1.2.1 的代码结构——
# 核心原则: 不要跳过 low_latency_mode 下的 NVSHMEM 初始化流程，
# 只跳过 IBGDA 相关的环境变量设置。
```

**`csrc/kernels/internode_ll.cu`** — 注释掉无条件的 IBGDA 断言（如有）。

### 2.3 下载模型

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B-FP8 --local-dir /workspace/Qwen3-30B-A3B-FP8
# 约 31GB
```

### 2.4 安装 OEPLB（需要手动 patch SGLang）

> **踩坑5**: 全新 pip 安装的 SGLang 0.5.6 **不包含** OEPLB 的 hook 点。
> 必须手动修改 3 个 SGLang 源文件。

#### Step 1: 部署 OEPLB 模块

```bash
SGLANG_PATH=$(python3 -c "import sglang,os; print(os.path.dirname(sglang.__file__))")
mkdir -p $SGLANG_PATH/srt/managers/pb_oeplb/

# 部署 V2.1（最新版本）
cp EPLB/OEPLB_V2/src/*.py $SGLANG_PATH/srt/managers/pb_oeplb/
```

#### Step 2: Patch SGLang 源码（3 个文件）

**文件 1: `server_args.py`** — 添加 CLI 参数
在 dataclass `ServerArgs` 中添加：
```python
enable_pb_oeplb: bool = False
pb_oeplb_threshold_ratio: float = 1.22
pb_oeplb_max_swaps_per_layer: int = 5
pb_oeplb_min_prefill_tokens: int = 1000
pb_oeplb_cooldown_steps: int = 5
pb_oeplb_max_total_swap_layers: int = 48
pb_oeplb_always_record: bool = False
pb_oeplb_sync_window: int = 128
```
在 `_handle_eplb_and_dispatch` 方法中添加：
```python
if self.enable_pb_oeplb:
    assert self.ep_size > 1, "PB-OEPLB requires ep_size > 1"
    assert not self.enable_eplb, "PB-OEPLB and EPLB cannot be enabled simultaneously"
    if self.ep_dispatch_algorithm is None:
        self.ep_dispatch_algorithm = "static"
    logger.warning("PB-OEPLB is enabled.")
```
在 argparse 部分添加对应的 `--enable-pb-oeplb` 等参数。

**文件 2: `model_executor/model_runner.py`** — 创建 controller
在 `initialize()` 方法中，模型加载完成后添加：
```python
if self.server_args.enable_pb_oeplb and (not self.is_draft_worker):
    from sglang.srt.managers.pb_oeplb.config import PBOEPLBConfig
    from sglang.srt.managers.pb_oeplb.controller import PBOEPLBController
    from sglang.srt.managers.pb_oeplb import set_pb_oeplb_controller
    cfg = PBOEPLBConfig.from_server_args(self.server_args)
    ctrl = PBOEPLBController(cfg, self)
    set_pb_oeplb_controller(ctrl)
```
在 `forward()` 方法末尾添加：
```python
from sglang.srt.managers.pb_oeplb import get_pb_oeplb_controller
ctrl = get_pb_oeplb_controller()
if ctrl is not None:
    ctrl.on_forward_pass_end(forward_batch)
```

**文件 3: `layers/moe/topk.py`** — 记录路由决策
在 `select_experts()` 返回之前添加：
```python
from sglang.srt.managers.pb_oeplb import get_pb_oeplb_controller as _get_pb_ctrl
_pb_ctrl = _get_pb_ctrl()
if _pb_ctrl is not None:
    _pb_ctrl.record_next_layer(topk_ids)
```

### 2.5 DeepGEMM 预编译（8卡必须）

> **踩坑6**: 8 个 DP rank 中只有 rank0 做 JIT 编译，如果单个 shape 编译超过
> 100 秒（DeepEP 硬编码的 `NUM_CPU_TIMEOUT_SECS`），其他 rank 等集合通信时会
> 报 `CPU recv timeout` 而整体崩溃。必须先预编译。

```bash
python3 -m sglang.compile_deep_gemm \
  --model-path /workspace/Qwen3-30B-A3B-FP8 \
  --tp 8 --trust-remote-code
# 可能需要 10-20 分钟
```

> **踩坑7**: H20 上 DeepGEMM 在 `(N=2048, K=768, num_groups=16)` 这个 shape
> （对应 8 卡 EP 下 MoE down_proj）可能报 `CUDA error: invalid configuration
> argument`。这是 DeepGEMM + H20 的已知兼容性问题，不是 OEPLB 引入的。
> **绕过方案**: 用 `--moe-runner-backend triton` 替代 `deep_gemm`。
> OEPLB 的 swap 机制跟底层 GEMM kernel 完全无关，用 triton 不影响效果验证。

## 三、快速验证

### 3.1 启动 Baseline 服务

```bash
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512

# 4卡
python3 -m sglang.launch_server \
  --model-path /workspace/Qwen3-30B-A3B-FP8 \
  --tp 4 --dp 4 --ep-size 4 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code

# 8卡（如果 DeepGEMM 有 H20 兼容问题，换 triton）
python3 -m sglang.launch_server \
  --model-path /workspace/Qwen3-30B-A3B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend triton \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code
```

### 3.2 验证推理

```bash
curl -s http://localhost:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/workspace/Qwen3-30B-A3B-FP8","prompt":"1+1=","max_tokens":10,"temperature":0}'
```

### 3.3 启动 OEPLB 服务

在启动命令中添加：
```bash
  --disable-radix-cache \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.22 \
  --pb-oeplb-min-prefill-tokens 1000 \
  --pb-oeplb-sync-window 128 \
  --pb-oeplb-cooldown-steps 5 \
  --pb-oeplb-max-total-swap-layers 48 \
  --pb-oeplb-max-swaps-per-layer 5
```

## 四、性能测试

### 4.1 准备数据

```bash
cd EPLB/OEPLB/scripts
# 循环数据集（必须配合 --disable-radix-cache 使用）
python3 -c "
import json
with open('../benchmarks/frozen_requests_prefill_heavy.jsonl') as f:
    reqs = [json.loads(l) for l in f]
out = []
for rep in range(15):
    for r in reqs:
        r2 = dict(r); r2['id'] = f\"{r['id']}_rep{rep}\"; out.append(r2)
with open('/tmp/prefill_heavy_x15.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
print(f'wrote {len(out)} requests')
"
```

### 4.2 跑吞吐测试

```bash
# 修改 long_bench.py 中 CONC = DP数 × 128
# 4卡DP=4: CONC=512
# 8卡DP=8: CONC=1024
python3 long_bench.py <label> /tmp/prefill_heavy_x15.jsonl
```

### 4.3 采集并发 Trace

```bash
# 先启动流量（后台），再启动 profiler
python3 long_bench.py traffic /tmp/prefill_heavy_x15.jsonl &
sleep 60
curl -X POST http://localhost:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/trace","num_steps":200,"activities":["CPU","GPU"],"with_stack":false,"record_shapes":false}'
# 等文件大小不再增长后再停流量
```

### 4.4 分析 Trace

```bash
python3 layer_imbalance_analysis.py <label> /tmp/trace/
```

## 五、已知踩坑汇总

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 1 | SGLang 没有 OEPLB hook | 全新 pip 安装不含 | 手动 patch 3 个文件（见 2.4） |
| 2 | DeepEP main 分支不兼容 | V2 用 NCCL Gin，需 torch 2.10+ | `git checkout v1.2.1` |
| 3 | DeepEP 编译缺依赖 | 缺 libibverbs-dev/libnuma1/NVSHMEM | apt + 官方 tar 包 |
| 4 | buffer.py patch 不完整致崩溃 | 跳过了 NVSHMEM 初始化 | 只跳 IBGDA 环境变量，保留初始化 |
| 5 | sgl_kernel 加载失败 | 缺 libnuma.so.1 | `apt-get install libnuma1` |
| 6 | 多卡 JIT 超时崩溃 | rank0 编译太慢，其他 rank 超时 | 先跑 `compile_deep_gemm` |
| 7 | DeepGEMM H20 兼容 bug | 特定 shape + H20 不兼容 | `--moe-runner-backend triton` |
| 8 | 循环数据集 TPS 虚高 | radix cache 命中 | `--disable-radix-cache` |
| 9 | profiler OOM | `with_stack=True` 长时间采集 | 必须 `with_stack=false` |
| 10 | DP 下 GPU 没喂饱 | CONC 太小 | CONC = DP数 × 128 |

## 六、8 卡扩展说明

- **EP=8**: 每卡 128/8=16 个 expert，天然不均衡度可能更高
- **CONC=1024**: 需要喂饱 8 个 DP rank
- **threshold_ratio**: 8 卡下可能需要调低（如 1.18-1.20），先跑 baseline 不均衡度分析再决定
- **moe-runner-backend**: H20 8 卡目前建议用 `triton`（绕过 DeepGEMM bug）
