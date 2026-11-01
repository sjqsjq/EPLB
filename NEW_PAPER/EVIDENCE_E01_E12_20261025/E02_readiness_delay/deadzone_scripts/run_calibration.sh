#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 TOTAL_TOKENS [extra deadzone_layer.py args...]" >&2
  exit 2
fi
TOKENS=$1
shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1
export PYTHONUNBUFFERED=1
python "$ROOT/scripts/build_delay_extension.py"
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  "$ROOT/scripts/deadzone_layer.py" \
  --total-tokens "$TOKENS" \
  --conditions-json "$ROOT/scripts/calibration_conditions.json" \
  --warmup 10 --blocks 5 --iterations-per-block 100 \
  --delay-calibration-repetitions 0 \
  --output "$ROOT/results/calibration_N${TOKENS}.json" \
  "$@"
