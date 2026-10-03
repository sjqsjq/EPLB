#!/bin/sh
# η实验臂: 固定W=16(关adaptive-window,让窗口持续触发), ETA_THR/ETA_OPS 外部注入
set -e
. /workspace/EPLB/OEPLB/baselines/a100/env_a100.sh
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --dtype bfloat16 --tp 8 --ep-size 8 \
  --moe-runner-backend triton --attention-backend flashinfer \
  --mem-fraction-static 0.88 --context-length 8192 --max-running-requests 256 \
  --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache \
  --watchdog-timeout 600 --port 30000 --host 0.0.0.0 --trust-remote-code \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio ${ETA_THR:?} \
  --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 16 \
  --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-min-swap-ops 8 \
  --pb-oeplb-max-total-ops ${ETA_OPS:?} \
  --pb-oeplb-decay-factor 0.5
