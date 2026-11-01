#!/bin/bash
. /workspace/logs/env_235b.sh
export OEPLB_ADAPTIVE_DECAY=1
export OEPLB_BIAS_CORRECT=1
exec python3 -m sglang.launch_server --model-path /workspace/models/Qwen3-235B-A22B-FP8 \
  --tp 4 --dp 4 --ep-size 4 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode normal \
  --moe-runner-backend deep_gemm --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache --watchdog-timeout 600 \
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02 --pb-oeplb-min-prefill-tokens 256 --pb-oeplb-sync-window 16 \
  --pb-oeplb-max-total-swap-layers 94 --pb-oeplb-max-swaps-per-layer 64 --pb-oeplb-min-swap-ops 8 \
  --pb-oeplb-max-total-ops 300 --pb-oeplb-decay-factor 0.5 --pb-oeplb-adaptive-window \
  --pb-oeplb-window-floor 8 --pb-oeplb-window-shift-confirm 2
