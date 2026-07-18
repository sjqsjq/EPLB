# OEPLB 单机多卡 H20 环境搭建 & 快速验证指南

## 一、硬件要求

- **GPU**: 4×/8× NVIDIA H20 (96GB/卡)，需要 NVLink 连接（单机，非跨节点 IB）
- **CPU**: 64 核以上推荐
- **内存**: 256GB+ 系统内存
- **存储**: 100GB+ 可用空间（模型 31GB + 环境 + trace 文件）

## 二、软件环境

### 2.1 基础镜像（推荐）

使用 SGLang 官方容器镜像作为起点：
```bash
# SGLang v0.5.6 + CUDA 12.8 + PyTorch 2.9.1
docker pull lmsysorg/sglang:v0.5.6-cu128
# 或者如果是 ARM64:
docker pull lmsysorg/sglang:v0.5.6-cu129-arm64
```

核心版本要求：
| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11+ | |
| PyTorch | 2.9.1+cu128 | 必须支持 CUDA 12.8+ |
| SGLang | 0.5.6 | 需要已包含 EP/DP-attention 支持 |
| CUDA | 12.8+ | nvcc + runtime |
| FlashInfer | 0.6.12 | SGLang 自带 |
| DeepGEMM | (bundled) | FP8 grouped GEMM, SGLang 自带或手动编译 |

### 2.2 安装 DeepEP（关键依赖，SGLang 不自带）

DeepEP 是 DeepSeek 开源的 MoE all-to-all 通信库，OEPLB 依赖它做 token dispatch/combine。

```bash
cd /workspace
git clone https://github.com/deepseek-ai/DeepEP.git deps_src/DeepEP
cd deps_src/DeepEP
pip install -e .
```

#### DeepEP 单机 NVLink patch（必须，否则 low_latency 模式会 crash）

DeepEP 默认假设有 IB/RDMA 网络，单机纯 NVLink 需要打 2 处 patch：

**Patch 1**: `csrc/kernels/internode_ll.cu` — 注释掉无条件的 IBGDA 断言
```cpp
// 找到 assert(ibgda_enabled) 或类似断言，注释掉
```

**Patch 2**: `deep_ep/buffer.py` 约第 96-101 行 — 仅在有 RDMA rank 时才设置 IBGDA 环境变量
```python
# 原始代码:
if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1' if ...
# 改为:
if self.runtime.get_num_rdma_ranks() > 1:
    os.environ['NVSHMEM_IB_ENABLE_IBGDA'] = '1'
```

### 2.3 下载模型

```bash
# Qwen3-30B-A3B-FP8 (MoE模型, 48层, 128专家/层, top-8路由)
# 约 31GB
huggingface-cli download Qwen/Qwen3-30B-A3B-FP8 --local-dir /workspace/Qwen3-30B-A3B-FP8

# 或者从 ModelScope:
# modelscope download Qwen/Qwen3-30B-A3B-FP8 --local-dir /workspace/Qwen3-30B-A3B-FP8
```

### 2.4 安装 OEPLB

```bash
cd /workspace
git clone git@github.com:sjqsjq/EPLB.git
cd EPLB

# 部署 OEPLB 模块到 SGLang 已安装路径
SGLANG_PATH=$(python3 -c "import sglang,os; print(os.path.dirname(sglang.__file__))")
mkdir -p $SGLANG_PATH/srt/managers/pb_oeplb/

# 部署 V1 (原始版本，含 bugfix 的 async_swapper)
cp OEPLB/src/*.py $SGLANG_PATH/srt/managers/pb_oeplb/

# 或者部署 V2.1 (折中优化版本)
# cp OEPLB_V2/src/*.py $SGLANG_PATH/srt/managers/pb_oeplb/
```

**注意**: SGLang 的 `server_args.py`、`model_runner.py`、`layers/moe/topk.py` 已经包含了 OEPLB 的 hook 点（`--enable-pb-oeplb` 等 CLI 参数）。如果你的 SGLang 版本没有这些 hook，需要手动添加——参考 `OEPLB/README.md` 的 "SGLang 集成" 章节。

## 三、快速验证（5 分钟确认环境正常）

### 3.1 启动 Baseline 服务

```bash
# 单机 NVLink 环境变量（必须）
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512

# 4卡配置 (TP=4, DP=4, EP=4)
python3 -m sglang.launch_server \
  --model-path /workspace/Qwen3-30B-A3B-FP8 \
  --tp 4 --dp 4 --ep-size 4 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code

# 8卡配置 (TP=8, DP=8, EP=8) — 调整相应参数
# --tp 8 --dp 8 --ep-size 8
```

等待 "The server is fired up and ready to roll!" 出现。

### 3.2 验证推理功能

```bash
curl -s http://localhost:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/workspace/Qwen3-30B-A3B-FP8","prompt":"1+1=","max_tokens":10,"temperature":0}' | python3 -m json.tool
```

应该返回合理的文本回复。

### 3.3 启动 OEPLB 服务

```bash
# 在启动命令中加入 OEPLB 参数:
python3 -m sglang.launch_server \
  --model-path /workspace/Qwen3-30B-A3B-FP8 \
  --tp 4 --dp 4 --ep-size 4 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.22 \
  --pb-oeplb-min-prefill-tokens 1000 \
  --pb-oeplb-sync-window 128 \
  --pb-oeplb-cooldown-steps 5 \
  --pb-oeplb-max-total-swap-layers 48 \
  --pb-oeplb-max-swaps-per-layer 5
```

日志中应该看到:
```
[PB-OEPLB] Init v0.11 ... thresh=1.22, min_tok=1000, sync_window=128
# 或 V2.1:
[OEPLB-V2.1] Init: ... threshold_ratio=1.22, record_interval=32 (jittered)
```

## 四、性能测试

### 4.1 准备测试数据

项目自带多个 frozen 数据集:
```
EPLB/OEPLB/benchmarks/
├── frozen_requests_prefill_heavy.jsonl   # 500条, 4096in/32out — prefill主导场景
├── frozen_requests_longbench_4k.jsonl    # 500条, 长上下文(6k-24k字符)
├── frozen_requests_domain_clustered.jsonl # 500条, 域聚类
└── frozen_requests_real_production.jsonl  # 500条, 真实生产数据
```

循环数据集做长跑(**必须加 --disable-radix-cache**):
```python
import json
with open('OEPLB/benchmarks/frozen_requests_prefill_heavy.jsonl') as f:
    reqs = [json.loads(l) for l in f]
out = []
for rep in range(15):
    for r in reqs:
        r2 = dict(r); r2['id'] = f"{r['id']}_rep{rep}"; out.append(r2)
with open('/tmp/prefill_heavy_x15.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
# 7500 requests
```

### 4.2 跑吞吐测试

```bash
cd EPLB/OEPLB/scripts
# 修改 long_bench.py 中的 CONC 为 DP数×128:
# 4卡DP=4: CONC=512
# 8卡DP=8: CONC=1024
python3 long_bench.py <label> /tmp/prefill_heavy_x15.jsonl
```

### 4.3 采集 Trace（并发流量下）

在吞吐测试运行期间，通过 HTTP API 启动 profiler:
```bash
# 启动 profiler（必须 with_stack=false，否则会 OOM）
curl -X POST http://localhost:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/trace_output","num_steps":200,"activities":["CPU","GPU"],"with_stack":false,"record_shapes":false}'

# 等待自动停止（num_steps=200 个 forward pass 后自动导出 .trace.json.gz）
# 检查文件是否生成:
ls /tmp/trace_output/*.gz
```

### 4.4 分析 Trace

```bash
cd EPLB/OEPLB/scripts
python3 layer_imbalance_analysis.py <label> /tmp/trace_output/
```

输出 dispatch/combine/expert 的逐层逐step不均衡度(mean/max)。

## 五、已知陷阱（必读）

1. **循环数据集必须加 `--disable-radix-cache`**，否则第2轮起会命中缓存跳过 prefill，TPS 虚高 2-3 倍
2. **profiler `with_stack=True` 在长时间+高并发下会 OOM**，必须用 `with_stack=false`
3. **profiler `num_steps` 是总 forward_ct**（prefill+decode都算），不是 prefill 窗口数
4. **`--enable-layerwise-nvtx-marker` 不会产生 trace 事件**——它用 NVTX range，不进 chrome trace
5. **DP=N 时 conc 应设为 128×N**，否则每个 DP rank 只跑 128/N 个请求，GPU 远没饱和

## 六、8 卡扩展说明

8 卡 H20 相比 4 卡的关键差异:
- **EP=8**: 每卡只有 128/8=16 个 expert（vs 4卡的 32个），不均衡可能更严重
- **DP=8**: 需要 CONC=1024 才能喂饱所有 rank
- **NVLink 拓扑**: 8 卡可能不是全连接（取决于 H20 NVLink 桥接拓扑），P2P swap 延迟可能不均匀
- **threshold_ratio 可能需要调整**: 更多 rank 意味着天然不均衡度更高（max/avg 更容易偏离 1.0），可能需要把 threshold 调低（如 1.18-1.20）

建议：先跑 baseline 不均衡度分析（`layer_imbalance_analysis.py`），看 8 卡下的天然 expert imbalance mean 是多少，再决定 threshold 设多少。
