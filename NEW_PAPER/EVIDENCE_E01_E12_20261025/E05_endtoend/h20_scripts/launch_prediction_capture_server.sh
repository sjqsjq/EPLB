#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIGGER="${PREDICTION_CAPTURE_TRIGGER:-$ROOT/inputs/prediction_v1/capture_trigger.json}"
PROTOCOL="${PREDICTION_CAPTURE_PROTOCOL:-$ROOT/configs/followup_v1/prediction_capture_protocol.json}"
VERIFIER="$ROOT/scripts/verify_prediction_capture_protocol.py"
PORT="${PORT:-30000}"

if [[ ! -f "$PROTOCOL" ]]; then
  echo "missing prediction capture protocol $PROTOCOL" >&2
  exit 1
fi
python3 "$VERIFIER" --protocol "$PROTOCOL"
FROZEN_MODEL="$(python3 - "$PROTOCOL" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["model_path"])
PY
)"
FROZEN_MAP="$(python3 - "$PROTOCOL" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["identity_map"]["path"])
PY
)"
FROZEN_TRIGGER="$(python3 - "$PROTOCOL" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["current_trigger_path"])
PY
)"
MODEL="${MODEL_PATH:-$FROZEN_MODEL}"
MAP="${IDENTITY_MAP:-$FROZEN_MAP}"
if [[ "$(realpath "$MODEL")" != "$(realpath "$FROZEN_MODEL")" ]]; then
  echo "MODEL_PATH differs from the frozen prediction model" >&2
  exit 1
fi
if [[ "$(realpath "$MAP")" != "$(realpath "$FROZEN_MAP")" ]]; then
  echo "IDENTITY_MAP differs from the frozen prediction mapping" >&2
  exit 1
fi
if [[ "$TRIGGER" != "$FROZEN_TRIGGER" ]]; then
  echo "capture trigger path differs from the frozen protocol" >&2
  exit 1
fi
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
export H20_PREDICTION_CAPTURE_TRIGGER="$TRIGGER"
export PYTHONPATH="$ROOT/scripts/prediction_capture_site${PYTHONPATH:+:$PYTHONPATH}"
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
