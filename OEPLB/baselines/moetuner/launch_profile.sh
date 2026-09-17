#!/bin/bash
# MoETuner profile launcher: identity layout + SGLang native recorder.
# Recorder is armed via SGLang HTTP endpoints from the driver, not here.
. /workspace/logs/env_235b.sh
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR:-/workspace/EPLB/OEPLB/baselines/moetuner/artifacts/profile_run}
mkdir -p "$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
echo "[launch_profile] dump dir=$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
cd "$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal \
  --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --expert-distribution-recorder-mode stat \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600
