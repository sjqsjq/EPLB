#!/bin/bash
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
exec python3 -m sglang.launch_server --model-path /workspace/Qwen3-30B-A3B-FP8 --tp 4 --dp 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --port 30000 --host 0.0.0.0 --trust-remote-code --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.05 --pb-oeplb-min-prefill-tokens 1000 --pb-oeplb-sync-window 256 --pb-oeplb-cooldown-steps 5 --pb-oeplb-max-total-swap-layers 48 --pb-oeplb-max-swaps-per-layer 8
