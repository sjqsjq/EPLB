#!/bin/sh
set -e
. /workspace/EPLB/OEPLB/baselines/a100/env_a100.sh
EXTRA=""
[ -n "${SCAN_PLC:-}" ] && EXTRA="--init-expert-location $SCAN_PLC"
[ -n "${SCAN_CHUNK:-}" ] && EXTRA="$EXTRA --chunked-prefill-size $SCAN_CHUNK"
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --dtype bfloat16 --tp 8 --ep-size 8 \
  --moe-runner-backend triton --attention-backend flashinfer \
  --mem-fraction-static 0.88 --context-length 8192 --max-running-requests 256 \
  --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache \
  --watchdog-timeout 600 --port 30000 --host 0.0.0.0 --trust-remote-code $EXTRA
