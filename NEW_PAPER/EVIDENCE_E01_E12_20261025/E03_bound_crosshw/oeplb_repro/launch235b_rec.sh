#!/bin/bash
. /workspace/logs/env_235b.sh
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/workspace/logs/recdump235
mkdir -p $SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
exec python3 -m sglang.launch_server \
  --model-path /data/models/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal \
  --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 \
  --disable-cuda-graph \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 \
  --expert-distribution-recorder-mode stat \
  --expert-distribution-recorder-buffer-size 4000
