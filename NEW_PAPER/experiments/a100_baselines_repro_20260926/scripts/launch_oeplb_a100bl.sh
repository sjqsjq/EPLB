#!/bin/sh
# PB-OEPLB under the §5.3.1 baseline口径 (disable-cuda-graph, mem 0.88, triton, no dp-attn)
set -e
. "$(dirname "$0")/env_a100.sh"
export OEPLB_ADAPTIVE_DECAY=1   # 变点时 α→0 一步清零(env-only, 非 CLI flag)
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --dtype bfloat16 --tp 8 --ep-size 8 \
  --moe-runner-backend triton --attention-backend flashinfer \
  --mem-fraction-static 0.88 --context-length 8192 \
  --max-running-requests 256 \
  --disable-cuda-graph --disable-overlap-schedule \
  --disable-radix-cache --watchdog-timeout 600 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.02 \
  --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 16 \
  --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-min-swap-ops 8 \
  --pb-oeplb-max-total-ops 300 \
  --pb-oeplb-decay-factor 0.5 \
  --pb-oeplb-adaptive-window \
  --pb-oeplb-window-floor 8 \
  --pb-oeplb-window-shift-confirm 2
