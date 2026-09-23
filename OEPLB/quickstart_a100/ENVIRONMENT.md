# A100 环境搭建（8×A100-80GB / BF16 / 非-DeepEP）

> 目标：在 8×A100-80GB 单机上从零把 OEPLB 跑起来。
> A100（sm80）**没有 FP8 张量核**，因此走 **`forward_normal` 路径：BF16 权重 + Triton MoE runner + NCCL 常规 dispatch**，
> **不需要编译 DeepEP，也不需要 DeepGEMM**——比 H20 环境明显更简单。

## 一、硬件与基础环境（实测）

```
GPU: 8× NVIDIA A100-SXM4-80GB, NVLink 全互连, 无 IB/RDMA
OS:  Linux 容器
Python: 3.11 (conda base)
CUDA:   12.8 (torch 自带 cu128)
CPU: 64核+, 内存: 256GB+
存储: 500GB+ (235B BF16 模型 ~440GB + 环境)
```

与 H20 相比，A100 环境的核心区别：**无 FP8、无 DeepEP、无 DeepGEMM**。参见 `REPRODUCE.md` §1 的完整对照表。

## 二、软件栈安装（比 H20 少两大件）

### 2.1 系统依赖
```bash
apt-get update && apt-get install -y libnuma1 git wget
```

### 2.2 PyTorch 2.9.1 + cu128
```bash
pip install torch==2.9.1 --index-url https://mirrors.aliyun.com/pypi/simple/
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 期望: 2.9.1+cu128 True
```

### 2.3 SGLang 0.5.6.post2 + sgl-kernel 0.3.19
```bash
pip install "sglang[all]==0.5.6.post2" "sgl-kernel==0.3.19" --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 2.4 —— 不需要 DeepEP / DeepGEMM ——
A100 用 `--moe-runner-backend triton` + NCCL a2a，**跳过 H20 的 DeepEP v1.2.1 编译和 DeepGEMM 编译**。
这是 A100 环境相对 H20 最省事的地方（H20 那两步各要 build 几分钟、还要打 NVLink patch）。

### 2.5 模型下载（BF16，非 FP8）
```bash
pip install modelscope
modelscope download Qwen/Qwen3-235B-A22B --local_dir /workspace/models/Qwen3-235B-A22B
# 注意：是 BF16 版 Qwen3-235B-A22B（约 440GB），不是 H20 用的 -FP8 版。
```

## 三、部署 OEPLB 到 SGLang

`deploy_oeplb.sh` 会把 OEPLB 核心代码复制进 SGLang，并对比三个 patch 文件：

```bash
cd /workspace/EPLB/OEPLB/quickstart_a100
sh deploy_oeplb.sh            # 复制 OEPLB/src/*.py -> sglang/srt/managers/pb_oeplb/，并 diff server_args/model_runner/topk
```

然后打两个必需补丁（`patch_sglang.py` **幂等**，可反复运行；会备份原文件）：

```bash
python3 patch_sglang.py
```

`patch_sglang.py` 打的两个补丁：

### 补丁 A（与 H20 相同）：server_args.py 强制 `ep_dispatch_algorithm='static'`
`--enable-pb-oeplb` 单独启用时，SGLang 0.5.6.post2 不会自动把 dispatch 设成 static，导致 logical→physical
重映射被跳过、OEPLB 静默不生效。补丁把 static 分支条件补上 `enable_pb_oeplb`。

### 补丁 B（A100 专有，**关键 bug**）：qwen3_moe.py `forward_normal` 补传 dispatch info
A100 走 `forward_normal`，它调 `self.topk(...)` 时**原本漏传 `expert_location_dispatch_info`**，
导致 `topk_ids_logical_to_physical()` 变 no-op → OEPLB/EPLB 搬了权重但路由不跟着走 → 不均衡度**窗口间回弹**、
吞吐不升反降。补丁镜像 `forward_deepep` 补传：

```python
# PB-OEPLB FIX: forward_normal 也传 dispatch info，让非-DeepEP 路径应用 logical->physical 重映射
topk_output = self.topk(
    hidden_states, router_logits,
    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id),
)
```

> 补丁 B 对 **EPLB 和 OEPLB 都必需**（两者都靠 `update_expert_location` 改 `physical_to_logical_map`）。
> 根因分析见 `REPRODUCE.md` §3。

## 四、环境变量（A100 专用）

`env_a100.sh`（每个启动脚本都会 source 它）：

```sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_LEVEL=NVL      # 走 NVLink
export NCCL_IB_DISABLE=1       # 无 IB
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B
export MODEL_PATH=/workspace/models/Qwen3-235B-A22B
```

注意：**没有** H20 那一堆 `NVSHMEM_*` 变量——因为不用 DeepEP。

## 五、启动服务器

三个启动脚本已在本文件夹，直接用（都会 source `env_a100.sh`）：

```bash
sh launch_baseline_a100.sh /workspace/logs/server_baseline.log &   # identity 基线
sh launch_eplb_a100.sh     /workspace/logs/server_eplb.log &       # 官方在线 EPLB
sh launch_oeplb_a100.sh    /workspace/logs/server_oeplb.log &      # OEPLB（修复后）
```

三个脚本共享的 **A100 关键 flag（与 H20 不同）**：

| flag | 原因 |
|---|---|
| `--moe-runner-backend triton` | A100 无 FP8，用 Triton BF16 grouped GEMM 代替 DeepGEMM |
| `--disable-overlap-schedule` | **必须**，否则 A100 上 NCCL 死锁 |
| 无 `--enable-dp-attention` | **A100 关闭 dp-attention**，原因见 `REPRODUCE.md` §2 |
| 无 `--moe-a2a-backend deepep` | 无 DeepEP，用 NCCL 常规 dispatch |
| `--mem-fraction-static 0.88` | BF16 模型更大，显存吃紧 |
| `--cuda-graph-max-bs 64` | A100 上的稳定上限 |
| `--attention-backend flashinfer` | A100 注意力后端 |
| `--disable-radix-cache` | 循环数据集下防止 prefix cache 命中虚高 TPS |
| `--watchdog-timeout 600` | 高并发长 prefill 防误杀 |

OEPLB 额外参数（`--pb-oeplb-*`）与 H20 一致，含 `--pb-oeplb-adaptive-window`、`--pb-oeplb-window-floor 8`、
`--pb-oeplb-window-shift-confirm 2`；`adaptive-decay` 在 A100 脚本里通过环境变量 `OEPLB_ADAPTIVE_DECAY=1` 开启（非 CLI flag）。

### 5.1 自检（OEPLB 是否真生效）
```bash
grep -oE "ep_dispatch_algorithm='?[a-zA-Z]+'?" /workspace/logs/server_oeplb.log | head -1
# 期望: ep_dispatch_algorithm='static'（若为 None → 补丁 A 没打上）

grep -E "PB-OEPLB-(DIAG|RESET|WINDOW)" /workspace/logs/server_oeplb.log | head
# 期望: 不均衡度 avg_before 不回弹（若每窗都回弹到初值 → 补丁 B 没打上）
```

## 六、下一步

环境跑通后，照同目录 `REPRODUCE.md` 复现单域 +20.1% / 多域 +13.2% 的三方对比（Baseline / EPLB / OEPLB）。
