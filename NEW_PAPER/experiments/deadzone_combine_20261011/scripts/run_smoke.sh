#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1
export PYTHONUNBUFFERED=1

python "$ROOT/scripts/build_delay_extension.py"
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  "$ROOT/scripts/deadzone_layer.py" \
  --total-tokens 256 \
  --conditions-json "$ROOT/scripts/smoke_conditions.json" \
  --warmup 3 \
  --blocks 1 \
  --iterations-per-block 2 \
  --delay-calibration-repetitions 5 \
  "$@"
