#!/usr/bin/env bash
set -euo pipefail

TOKENS=${1:-4096}
if [[ "$TOKENS" != 4096 ]]; then
  echo "representative timeline protocol is fixed at TOTAL_TOKENS=4096" >&2
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NSYS=${NSYS:-/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nsys}
PLAN_DIR="$ROOT/results/timeline_plans_N${TOKENS}"
TRACE_DIR="$ROOT/traces/N${TOKENS}"

if [[ ! -x "$NSYS" ]]; then
  echo "Nsight Systems CLI not executable: $NSYS" >&2
  exit 1
fi
python "$ROOT/scripts/make_timeline_conditions.py" \
  --calibration "$ROOT/results/calibration_N${TOKENS}.json" \
  --output-dir "$PLAN_DIR"
mkdir -p "$TRACE_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export SGLANG_CI_DISABLE_MOE_FUSED_FUNC=1
export PYTHONUNBUFFERED=1
python "$ROOT/scripts/build_delay_extension.py"

DEFAULT_NAMES=(
  timeline-original-zero
  timeline-original-inside
  timeline-original-outside
  timeline-barrier-zero
  timeline-barrier-inside
  timeline-barrier-outside
)
if [[ $# -gt 1 ]]; then
  NAMES=("${@:2}")
else
  NAMES=("${DEFAULT_NAMES[@]}")
fi

for NAME in "${NAMES[@]}"
do
  case "$NAME" in
    timeline-original-zero|timeline-original-inside|timeline-original-outside|\
    timeline-barrier-zero|timeline-barrier-inside|timeline-barrier-outside) ;;
    *) echo "unknown timeline condition: $NAME" >&2; exit 2 ;;
  esac
  torchrun --standalone --nnodes=1 --nproc-per-node=8 --no-python \
    "$ROOT/scripts/nsys_rank_wrapper.sh" \
    "$NSYS" "$TRACE_DIR/$NAME" \
    python "$ROOT/scripts/deadzone_layer.py" \
    --total-tokens "$TOKENS" \
    --conditions-json "$PLAN_DIR/$NAME.json" \
    --warmup 2 --blocks 1 --iterations-per-block 1 \
    --delay-calibration-repetitions 0 \
    --profile-capture-condition "$NAME" \
    --output "$ROOT/results/${NAME}_N${TOKENS}.json"

  REPORT_COUNT=$(find "$TRACE_DIR" -maxdepth 1 -type f \
    -name "${NAME}-rank*.nsys-rep" -size +0c | wc -l)
  if [[ "$REPORT_COUNT" -ne 8 ]]; then
    echo "$NAME produced $REPORT_COUNT nonempty reports, expected 8" >&2
    exit 1
  fi
done
