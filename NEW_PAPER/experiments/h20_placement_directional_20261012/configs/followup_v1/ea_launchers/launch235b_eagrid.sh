#!/bin/bash
# E-A calibration-grid / smoke launcher template (env-driven, sha-locked).
# Required env: EA_W (sync-window), EA_ALPHA (decay-factor).
# Optional env: EA_ADAPTIVE_WINDOW=1, EA_ADAPTIVE_DECAY=1 (default 0/omitted).
# All other flags are byte-identical to the v3 online launcher
# (/workspace/logs/launch235b_oeplb.sh, sha 46860da0...): threshold-ratio 1.02,
# min-prefill-tokens 256, swap budgets, window-floor 8, window-shift-confirm 2,
# dead_zone_ratio default 0.0 -> auto 1.096 identical across all arms (C6),
# calibrate_adaptive_sensitivity default False (C7: warm-start/tier unreachable).
. /workspace/logs/env_235b.sh
EAFLAGS=""
[ "${EA_ADAPTIVE_WINDOW:-0}" = "1" ] && EAFLAGS="$EAFLAGS --pb-oeplb-adaptive-window"
[ "${EA_ADAPTIVE_DECAY:-0}" = "1" ] && EAFLAGS="$EAFLAGS --pb-oeplb-adaptive-decay"
exec python3 -m sglang.launch_server \
  --model-path /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm \
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 \
  --port 30000 --host 0.0.0.0 --trust-remote-code \
  --disable-radix-cache --watchdog-timeout 600 \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.02 \
  --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window ${EA_W:?EA_W required} \
  --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-min-swap-ops 8 \
  --pb-oeplb-max-total-ops 300 \
  --pb-oeplb-decay-factor ${EA_ALPHA:?EA_ALPHA required} \
  --pb-oeplb-window-floor 8 \
  --pb-oeplb-window-shift-confirm 2 \
  $EAFLAGS
