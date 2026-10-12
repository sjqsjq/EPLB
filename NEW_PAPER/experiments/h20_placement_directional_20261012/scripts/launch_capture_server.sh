#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL_PATH:-/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8}"
MAP="${IDENTITY_MAP:-$ROOT/placements/identity_service_map.json}"
TRIGGER="${CAPTURE_TRIGGER:-$ROOT/inputs/capture_trigger.json}"
PORT="${PORT:-30000}"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "missing model config under $MODEL" >&2
  exit 1
fi
if [[ ! -f "$MAP" ]]; then
  echo "missing identity map $MAP" >&2
  exit 1
fi

source /workspace/logs/env_235b.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1
export H20_PLACEMENT_CAPTURE_TRIGGER="$TRIGGER"
export PYTHONPATH="$ROOT/scripts/capture_site${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal \
  --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 \
  --mem-fraction-static 0.8 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --watchdog-timeout 600 \
  --port "$PORT" --host 0.0.0.0 --trust-remote-code \
  --init-expert-location "$MAP"
