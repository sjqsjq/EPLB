#!/bin/bash
# h20-placement-fullservice-v1 SETUP-ONLY profiling launcher (untimed).
# Verbatim identity launcher + expert-distribution recorder (stat mode).
# Recorded logical routing counts are placement-independent (router gate
# weights are frozen); identity placement is used so no swaps occur.
. /workspace/logs/env_235b.sh
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/workspace/logs/recdump
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 \
  --expert-distribution-recorder-mode stat
