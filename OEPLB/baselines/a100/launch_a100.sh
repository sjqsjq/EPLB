#!/bin/sh
# Usage: sh launch_a100.sh <method>
# method: identity | eplb_static | eplb_dyn | datafore | moetuner
set -e
. "$(dirname "$0")/env_a100.sh"
M=${1:?method required}

COMMON="--model-path $MODEL_PATH \
  --dtype bfloat16 --tp 8 --ep-size 8 \
  --moe-runner-backend triton --attention-backend flashinfer \
  --mem-fraction-static 0.88 --context-length 8192 \
  --max-running-requests 256 \
  --disable-cuda-graph --disable-overlap-schedule \
  --disable-radix-cache --watchdog-timeout 600 \
  --port 30000 --host 0.0.0.0 --trust-remote-code"

case "$M" in
  identity)
    EXTRA="" ;;
  eplb_static)
    EXTRA="--ep-num-redundant-experts 16 --init-expert-location $PLACEMENT_PROVER" ;;
  eplb_dyn)
    EXTRA="--ep-num-redundant-experts 16 --enable-eplb \
      --eplb-rebalance-num-iterations 100 --expert-distribution-recorder-buffer-size 32" ;;
  datafore)
    EXTRA="--init-expert-location $PLACEMENT_PROVER" ;;
  moetuner)
    EXTRA="--init-expert-location $PLACEMENT_MOETUNER" ;;
  *) echo "unknown method: $M"; exit 1 ;;
esac

echo "[launch_a100/$M] flags: $COMMON $EXTRA"
exec python3 -m sglang.launch_server $COMMON $EXTRA
