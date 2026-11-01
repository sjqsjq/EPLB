#!/bin/bash
# CD-X0 capture server: identical flags to launch_prediction_capture_server.sh
# (frozen prediction-capture configuration), CD-X0 capture site and trigger path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIGGER="${CDX0_CAPTURE_TRIGGER:?set CDX0_CAPTURE_TRIGGER}"
MODEL="/root/models/ms_cache/models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master"
MAP="$ROOT/placements/identity_service_map.json"
PORT="${PORT:-30000}"
EXPECTED_IDENTITY_SHA="$(python3 - "$ROOT/configs/followup_v1/prediction_capture_protocol.json" <<'PY'
import json,sys,pathlib,hashlib
proto=json.loads(pathlib.Path(sys.argv[1]).read_text())
rec=proto["identity_map"]
p=pathlib.Path(rec["path"])
h=hashlib.sha256()
with p.open("rb") as f:
    for c in iter(lambda: f.read(1<<20), b""): h.update(c)
assert h.hexdigest()==rec["sha256"], "frozen identity map record mismatch"
print(rec["sha256"])
PY
)"
ACTUAL_SHA="$(sha256sum "$MAP" | cut -d' ' -f1)"
[[ "$ACTUAL_SHA" == "$EXPECTED_IDENTITY_SHA" ]] || { echo "identity map sha differs from frozen record" >&2; exit 1; }
[[ -f "$MODEL/config.json" ]] || { echo "missing model config" >&2; exit 1; }
source /workspace/logs/env_235b.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1
export H20_PREDICTION_CAPTURE_TRIGGER="$TRIGGER"
export PYTHONPATH="$ROOT/scripts/cdx0_capture_site${PYTHONPATH:+:$PYTHONPATH}"
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
