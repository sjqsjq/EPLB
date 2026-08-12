#!/bin/bash
# $1 = path to physical_to_logical_map json
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /data/models/Qwen2-57B-A14B-Instruct \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm --quantization fp8 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 \
  --init-expert-location "$1"
