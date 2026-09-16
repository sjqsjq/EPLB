#!/bin/bash
# MoETuner profile launcher: identity layout + record P/R via env-var hook.
. /workspace/logs/env_235b.sh
export MOETUNER_PROFILE=1
export MOETUNER_PROFILE_DIR=${MOETUNER_PROFILE_DIR:-/workspace/EPLB/OEPLB/baselines/moetuner/artifacts/profile_run}
export MOETUNER_PROFILE_CHUNK_FWD=${MOETUNER_PROFILE_CHUNK_FWD:-256}
mkdir -p "$MOETUNER_PROFILE_DIR"
echo "[launch_profile] MOETUNER_PROFILE_DIR=$MOETUNER_PROFILE_DIR"
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
