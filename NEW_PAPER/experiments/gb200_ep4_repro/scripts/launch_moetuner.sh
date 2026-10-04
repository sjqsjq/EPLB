#!/bin/bash
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /workspace/models/Qwen3-235B-A22B-FP8 \
  --tp 4 --dp 4 --ep-size 4 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --init-expert-location /workspace/EPLB/OEPLB/baselines/moetuner/artifacts/placement_moetuner_fresh_ep4.json \
  --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
