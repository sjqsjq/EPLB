#!/bin/bash
set -u
. /workspace/EPLB/oeplb_env.sh
export CUDA_HOME=/opt/conda/envs/oeplb_qwen35/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH=$(python3 -c "import torch; print(torch.__path__[0])")/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
exec python3 -m sglang.launch_server --model-path /data/models/Qwen3-30B-A3B-FP8 --tp 4 --dp 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --port 30000 --host 0.0.0.0 --trust-remote-code --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.05 --pb-oeplb-min-prefill-tokens 1000 --pb-oeplb-sync-window 256 --pb-oeplb-max-total-swap-layers 48 --pb-oeplb-max-swaps-per-layer 8
