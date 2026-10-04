#!/bin/bash
. /workspace/logs/env_235b.sh
# 冗余专家(->num_groups=36)但不带 --enable-eplb/recorder, 避免 warmup 期集合通信 desync.
# GEMM shape 只由 ep-num-redundant-experts 决定, 与 enable-eplb 无关.
exec python3 -m sglang.compile_deep_gemm \
  --model-path /workspace/models/Qwen3-235B-A22B-FP8 \
  --tp 4 --dp 4 --ep-size 4 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.78 \
  --ep-num-redundant-experts 16 \
  --trust-remote-code --timeout 2400 --port 30000 --host 0.0.0.0
