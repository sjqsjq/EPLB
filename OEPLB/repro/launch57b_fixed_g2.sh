#!/bin/bash
. /workspace/logs/env_235b.sh
# $1 = path to physical_to_logical_map json
export CUDA_VISIBLE_DEVICES=0,1
exec python3 -m sglang.launch_server \
  --model-path /data/models/Qwen2-57B-A14B-Instruct \
  --tp 2 --dp 2 --ep-size 2 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm --quantization fp8 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 \
  --init-expert-location "$1"
