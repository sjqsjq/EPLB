#!/bin/sh
set -e
. /workspace/EPLB/OEPLB/baselines/a100/env_a100.sh
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/workspace/logs/a100_recorder_dump
mkdir -p "$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --dtype bfloat16 --tp 8 --ep-size 8 \
  --moe-runner-backend triton --attention-backend flashinfer \
  --mem-fraction-static 0.88 --context-length 8192 \
  --max-running-requests 256 \
  --disable-cuda-graph --disable-overlap-schedule \
  --disable-radix-cache --watchdog-timeout 600 \
  --expert-distribution-recorder-mode stat \
  --port 30000 --host 0.0.0.0 --trust-remote-code
