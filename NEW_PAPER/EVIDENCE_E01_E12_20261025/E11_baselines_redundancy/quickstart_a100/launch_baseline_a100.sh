#!/bin/sh
. "$(dirname "$0")/env_a100.sh"
LOG=${1:-/workspace/logs/server_baseline.log}
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
  > "$LOG" 2>&1
