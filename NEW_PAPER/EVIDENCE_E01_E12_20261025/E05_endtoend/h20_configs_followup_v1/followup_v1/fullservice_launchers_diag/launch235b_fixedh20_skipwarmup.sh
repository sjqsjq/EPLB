#!/bin/bash
# DIAGNOSTIC ONLY (pre-v3 validation, not part of any frozen campaign).
# Verbatim copy of fullservice_launchers_v2/launch235b_fixedh20.sh with one
# added flag: --skip-server-warmup. Tests the hypothesis that the fx boot
# failure is the warmup forward on a relocated map (historical precedent
# server0914_moetuner.log booted fixed maps with skip_server_warmup=True).
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
  --skip-server-warmup \
  --init-expert-location "$1"
