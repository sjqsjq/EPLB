# PB-OEPLB 部署指南：8×H20 单机环境

> 硬件: 8× NVIDIA H20 (96GB HBM3, NV18 全互连, 无IB/RDMA)
> 验证日期: 2026-07-25, 基于 Qwen3-235B-A22B-FP8 (8卡EP) 和 Qwen3-30B-A3B-FP8 (4卡EP)

---

## 一、硬件与驱动要求

| 组件 | 要求 | 备注 |
|------|------|------|
| GPU | 8× NVIDIA H20 96GB | NVLink 全互连, NV18拓扑 |
| GPU驱动 | 535+ | `nvidia-smi` 验证 |
| CUDA | 12.8+ | `nvcc --version` 验证 |
| CPU | 64核+ | |
| 内存 | 256GB+ | |
| 存储 | 300GB+ | 235B模型223GB + 环境 + trace |

**H20特有注意事项：**
- H20 **没有IB/RDMA**，只有NVLink——所有DeepEP通信都走NVLink
- H20 的SM数量(132)比H100(132)相同，但memory bandwidth和NVLink带宽不同
- DeepGEMM在H20上的部分shape可能有兼容性问题（见"已知问题"章节）

---

## 二、软件环境搭建

### 2.1 基础镜像

```bash
docker pull lmsysorg/sglang:v0.5.6-cu128
```

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | |
| PyTorch | 2.9.1+cu128 | DeepEP v1.x 不兼容 torch 2.10+ |
| SGLang | 0.5.6.post2 | |
| CUDA | 12.8 | |
| sgl-kernel | 0.3.19 | |

```bash
# 容器内可能缺的依赖
apt-get update && apt-get install -y libibverbs-dev libnuma1
```

### 2.2 安装 DeepEP v1.2.1

**必须用v1.x分支**，main分支是V2(需torch 2.10+，不兼容)。

```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP && git checkout v1.2.1
```

**安装NVSHMEM（必须用官方tar包，不要pip install）：**
```bash
wget https://developer.download.nvidia.com/compute/nvshmem/3.3.9/local_installers/nvshmem_3.3.9-1+cuda12.8_x86_64.tar.bz2
tar xf nvshmem_3.3.9-1+cuda12.8_x86_64.tar.bz2
export NVSHMEM_HOME=$(pwd)/nvshmem_3.3.9-1+cuda12.8
```

**H20专用Patch（单机NVLink，无IB）：**

DeepEP默认假设有IB/RDMA。在纯NVLink拓扑下，low_latency模式需要patch两处:

1. `deep_ep/buffer.py` (~L96-105): IBGDA环境变量设置仅在有RDMA rank时执行，但NVSHMEM初始化和unique_id广播在low_latency_mode下必须保留
2. `csrc/kernels/internode_ll.cu`: 注释掉IBGDA断言

```bash
pip install -e . --no-build-isolation
```

### 2.3 安装 DeepGEMM

```bash
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && git checkout 35c4bc8
pip install -e .
```

### 2.4 下载模型

```bash
# 235B模型（223GB，推荐8卡）
huggingface-cli download Qwen/Qwen3-235B-A22B-FP8 \
  --revision 39eb2b06 --local-dir /path/to/Qwen3-235B-A22B-FP8
# 或 ModelScope:
# modelscope download Qwen/Qwen3-235B-A22B-FP8

# 30B模型（31GB，4卡可跑）
huggingface-cli download Qwen/Qwen3-30B-A3B-FP8 --local-dir /path/to/Qwen3-30B-A3B-FP8
```

---

## 三、部署 OEPLB

### 3.1 部署OEPLB模块到SGLang

```bash
SGLANG_PATH=$(python3 -c "import sglang,os; print(os.path.dirname(sglang.__file__))")
# 复制OEPLB核心代码
cp EPLB/OEPLB/src/*.py $SGLANG_PATH/srt/managers/pb_oeplb/
```

### 3.2 Patch SGLang（3个文件）

**文件1: `server_args.py`** — 添加CLI参数

在ServerArgs dataclass中添加:
```python
enable_pb_oeplb: bool = False
pb_oeplb_threshold_ratio: float = 1.02
pb_oeplb_max_swaps_per_layer: int = 64
pb_oeplb_max_total_swap_layers: int = 94  # Qwen3-235B有94层MoE
pb_oeplb_sync_window: int = 64
pb_oeplb_min_prefill_tokens: int = 256
pb_oeplb_cooldown_steps: int = 5
pb_oeplb_always_record: bool = False
```

在`_handle_eplb_and_dispatch`方法中添加互斥检查和dispatch algorithm设置。
在argparse部分添加对应的`--enable-pb-oeplb`等参数。

**文件2: `model_executor/model_runner.py`** — 创建controller + forward hook

**文件3: `layers/moe/topk.py`** — 记录路由决策

（具体patch代码见 `OEPLB/README.md` "SGLang 集成" 章节）

---

## 四、启动命令

### H20专用环境变量（所有配置共用）

```bash
export NVSHMEM_DIR=/path/to/nvshmem_official
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
# H20无IB，必须禁用RDMA相关
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
```

### 4.1 Baseline（无负载均衡）

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

### 4.2 OEPLB（推荐配置，235B模型）

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

### 4.3 OEPLB + Adaptive Window（domain-switch场景推荐）

在OEPLB基础上添加环境变量:
```bash
export OEPLB_ADAPTIVE_WINDOW=1
export OEPLB_WINDOW_FLOOR=32
export OEPLB_WINDOW_SHIFT_COS=0.85
export OEPLB_WINDOW_STABLE_COS=0.95
export OEPLB_WINDOW_SHIFT_CONFIRM=1
export OEPLB_WINDOW_STABLE_CONFIRM=2
```

### 4.4 SGLang官方EPLB（对比用）

```bash
python3 -m sglang.launch_server \
  --model-path $MODEL \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache \
  --ep-num-redundant-experts 16 \
  --ep-dispatch-algorithm dynamic \
  --enable-eplb --eplb-algorithm auto \
  --eplb-rebalance-num-iterations 64 \
  --expert-distribution-recorder-mode stat
```

**注意：EPLB不支持`deepep-mode=auto`（会报NotImplementedError），只能用`normal`模式。OEPLB支持auto和normal。**

---

## 五、参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pb-oeplb-threshold-ratio` | 1.02 | 不均衡度阈值，ratio低于此值不触发swap |
| `--pb-oeplb-sync-window` | 64 | 每多少个forward pass检查一次不均衡度 |
| `--pb-oeplb-min-prefill-tokens` | 256 | 一个窗口内至少积累这么多prefill token才做决策 |
| `--pb-oeplb-max-swaps-per-layer` | 64 | 单层单次决策最多swap多少对slot |
| `--pb-oeplb-max-total-swap-layers` | 94 | 全局预算涉及的最大层数 |
| `--pb-oeplb-cooldown-steps` | 5 | swap完成后冷却多少步(当前未使用) |
| `MAX_TOTAL_OPS` | 250(部署)/300(源码) | rebalancer.py中的全局swap预算，需手动改 |
| `OEPLB_DECAY_FACTOR` | 0.9 | 环境变量，指数衰减系数 |

---

## 六、测试与验证

### 6.1 数据集

预构建数据集保存在 `/data/minghua/sjq/OEPLBdata/`，命名格式: `{来源}_{输入长度}_{输出长度}.jsonl`

### 6.2 Benchmark客户端

```bash
cd EPLB/OEPLB/scripts
# 修改long_bench.py中 CONC=1024 (8卡DP=8)
python3 long_bench.py <label> <dataset.jsonl>
```

### 6.3 nsys Profiling

```bash
nsys profile --trace=cuda --sample=none --delay=145 --duration=25 \
  --output=/path/to/trace --force-overwrite=true \
  python3 -m sglang.launch_server [args...]
```

---

## 七、H20已知问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| DeepGEMM特定shape CUDA error | H20+特定MxNxK不兼容 | `--moe-runner-backend triton` |
| DeepEP low_latency模式crash | 无IB时IBGDA初始化失败 | buffer.py patch(见2.2) |
| EPLB+auto模式NotImplementedError | expert_distribution recorder不支持auto dispatch | EPLB只能用`--deepep-mode normal` |
| EPLB rebalance后CUDA error | rebalance更新location时kernel config异常 | EPLB的bug,偶发,重试可恢复 |
| 服务器端口残留导致重启失败 | `kill -9`后子进程变僵尸占端口 | 用`kill`(SIGTERM)而不是`kill -9` |
| 循环数据集TPS虚高 | radix cache命中跳过prefill | 必须加`--disable-radix-cache` |

---

## 八、与Blackwell部署的主要区别（预留）

| 维度 | H20 | Blackwell (未来) |
|------|-----|-----------------|
| 互连 | NVLink NV18 (单机) | NVLink + NVSwitch? |
| IB/RDMA | 无 | 可能有 |
| DeepEP模式 | 需patch禁用IBGDA | 可能原生支持 |
| EP规模 | 8 (单机8卡) | 可能更大(多机) |
| NVSHMEM环境变量 | 需手动设置一堆禁用项 | 可能不需要 |
| OEPLB参数 | threshold=1.02, sw=64 | 需根据新硬件重新调参 |
