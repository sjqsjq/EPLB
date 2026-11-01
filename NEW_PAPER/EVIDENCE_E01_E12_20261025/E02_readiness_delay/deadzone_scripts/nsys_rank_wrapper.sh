#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 NSYS OUTPUT_PREFIX COMMAND [ARGS...]" >&2
  exit 2
fi

NSYS=$1
OUTPUT_PREFIX=$2
shift 2

if [[ -z "${LOCAL_RANK:-}" ]]; then
  echo "LOCAL_RANK is not set; launch this wrapper through torchrun" >&2
  exit 2
fi

exec "$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --export=sqlite \
  --force-overwrite=true \
  --output="${OUTPUT_PREFIX}-rank${LOCAL_RANK}" \
  "$@"
