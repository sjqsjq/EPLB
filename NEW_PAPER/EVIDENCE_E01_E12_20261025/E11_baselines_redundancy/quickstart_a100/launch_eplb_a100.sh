#!/bin/sh
. "$(dirname "$0")/env_a100.sh"
# Official SGLang EPLB baseline (online periodic global rebalance).
# Fair vs OEPLB: both online, 0 redundant experts, dispatch=static (auto).
# Difference: EPLB recomputes a full-model placement every N forward passes
# via init_by_eplb(logical_count) + update_expert_location (heavy);
# OEPLB does incremental per-window greedy swaps + changepoint/adaptive-window.
# Requires the forward_normal dispatch-info fix to actually affect routing.
LOG=${1:-/workspace/logs/server_eplb.log}
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 --ep-size 8 \
  --moe-runner-backend triton \
  --mem-fraction-static 0.88 \
  --cuda-graph-max-bs 64 \
  --max-running-requests 256 \
  --context-length 4096 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --disable-radix-cache \
  --watchdog-timeout 600 \
  --disable-overlap-schedule \
  --host 0.0.0.0 --port 30000 \
  --enable-eplb \
  --ep-num-redundant-experts 0 \
  --eplb-rebalance-num-iterations 50 \
  --eplb-min-rebalancing-utilization-threshold 1.0 \
  > "$LOG" 2>&1
