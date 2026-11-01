#!/bin/sh
. "$(dirname "$0")/env_a100.sh"
# adaptive-decay (α→0 one-step reset at change points) is env-only, not a CLI flag
export OEPLB_ADAPTIVE_DECAY=1
LOG=${1:-/workspace/logs/server_oeplb.log}
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 --ep-size 8 \
  --moe-runner-backend triton \
  --mem-fraction-static 0.88 \
  --cuda-graph-max-bs 64 \
  --max-running-requests 256 \
  --context-length 4096 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --disable-radix-cache \
  --watchdog-timeout 600 \
  --disable-overlap-schedule \
  --host 0.0.0.0 --port 30000 \
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
  --pb-oeplb-window-shift-confirm 2 \
  > "$LOG" 2>&1
