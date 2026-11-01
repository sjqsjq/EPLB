#!/bin/bash
# MoETuner benchmark launcher: identity code path + SGLang native init_expert_location.
# Usage:
#   PLACEMENT=/path/to/placement.json ./launch_moetuner.sh
. /workspace/logs/env_235b.sh
PLACEMENT="${PLACEMENT:?PLACEMENT env var must be set to placement.json path}"
if [ ! -f "$PLACEMENT" ]; then
    echo "[launch_moetuner] ERROR: placement file not found: $PLACEMENT" >&2
    exit 1
fi
echo "[launch_moetuner] init_expert_location=$PLACEMENT"
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --init-expert-location "$PLACEMENT" \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
