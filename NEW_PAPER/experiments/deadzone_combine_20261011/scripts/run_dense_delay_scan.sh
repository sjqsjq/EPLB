#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 TOTAL_TOKENS [extra deadzone_layer.py args...]" >&2
  exit 2
fi

TOKENS=$1
shift
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAL="$ROOT/results/calibration_N${TOKENS}.json"
PLAN="$ROOT/results/delay_dense_plan_N${TOKENS}.json"
RESULT="$ROOT/results/delay_dense_N${TOKENS}.json"

case "$TOKENS" in
  256)
    # The coarse scan located H/C0 near 0.25.
    RATIOS="0,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.40,0.48"
    ;;
  4096)
    # The coarse scan located H/C0 near 0.12.
    RATIOS="0,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20"
    ;;
  *)
    echo "dense ratios are defined only for TOTAL_TOKENS=256 or 4096" >&2
    exit 2
    ;;
esac

if [[ ! -f "$CAL" ]]; then
  echo "missing calibration: $CAL" >&2
  exit 1
fi

python "$ROOT/scripts/make_delay_scan.py" \
  --calibration "$CAL" \
  --output "$PLAN" \
  --ratios "$RATIOS" \
  --no-all-rank-control

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1
export PYTHONUNBUFFERED=1

python "$ROOT/scripts/build_delay_extension.py"
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  "$ROOT/scripts/deadzone_layer.py" \
  --total-tokens "$TOKENS" \
  --conditions-json "$PLAN" \
  --warmup 10 --blocks 5 --iterations-per-block 100 \
  --delay-calibration-repetitions 20 \
  --output "$RESULT" \
  "$@"
