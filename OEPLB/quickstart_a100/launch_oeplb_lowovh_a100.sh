#!/bin/sh
. "$(dirname "$0")/env_a100.sh"
LOG=${1:-/workspace/logs/server_oeplb_low.log}
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 --ep-size 8 --moe-runner-backend triton \
  --mem-fraction-static 0.88 --cuda-graph-max-bs 64 \
  --max-running-requests 256 --context-length 4096 \
  --attention-backend flashinfer --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 --disable-overlap-schedule \
  --host 0.0.0.0 --port 30000 \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.05 \
  --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 32 \
  --pb-oeplb-max-total-swap-layers 48 \
  --pb-oeplb-max-swaps-per-layer 16 \
  --pb-oeplb-min-swap-ops 8 \
  --pb-oeplb-max-total-ops 64 \
  --pb-oeplb-decay-factor 0.5 \
  > "$LOG" 2>&1
