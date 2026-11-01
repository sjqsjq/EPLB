#!/bin/sh
set -e
. /workspace/EPLB/OEPLB/baselines/a100/env_a100.sh
EXTRA=""
[ -n "${SCAN_PLC:-}" ] && EXTRA="--init-expert-location $SCAN_PLC"
exec python3 -m sglang.launch_server \
  --model-path /workspace/models/Qwen2-57B-A14B-Instruct \
  --dtype bfloat16 --tp ${SCAN_EP:?} --ep-size ${SCAN_EP:?} \
  --moe-runner-backend triton --attention-backend flashinfer \
  --mem-fraction-static 0.85 --context-length 4096 --max-running-requests 256 \
  --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache \
  --watchdog-timeout 600 --port 30000 --host 0.0.0.0 --trust-remote-code $EXTRA
