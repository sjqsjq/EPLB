#!/bin/bash
# h20-placement-fullservice-v3 arm B launcher (SETUP-DERIVED, hash-frozen).
# Verbatim copy of /workspace/logs/launch235b_identity.sh with exactly one
# added flag: --init-expert-location "$1" (the calibration-derived ILP map).
# $1 = json with {"physical_to_logical_map": 94x128}
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 \
  --init-expert-location "$1"
