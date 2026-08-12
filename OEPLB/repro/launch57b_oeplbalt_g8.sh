#!/bin/bash
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \
  --model-path /data/models/Qwen2-57B-A14B-Instruct \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm --quantization fp8 \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 \
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.10 --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 64 --pb-oeplb-decay-factor 0.5 \
  --pb-oeplb-max-total-swap-layers 28 --pb-oeplb-max-swaps-per-layer 32 \
  --pb-oeplb-min-swap-ops 8 --pb-oeplb-max-total-ops 300
