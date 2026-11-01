#!/bin/bash
export NVSHMEM_HOME=/opt/conda/lib/python3.11/site-packages/nvidia/nvshmem
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:$LD_LIBRARY_PATH
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python3 -m sglang.launch_server --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --dtype bfloat16 --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.78 --disable-cuda-graph --skip-server-warmup \
  --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache --watchdog-timeout 600
