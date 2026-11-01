#!/bin/bash
# Five-arm campaign launcher template (env-driven; sha-locked in the frozen
# ea_v1_campaign_protocol.json). Extends launch235b_eagrid.sh with EXACTLY ONE
# addition: EA_PLANNER_BI=1 -> OEPLB_PLANNER_BEST_IMPROVEMENT=1 (P11 arm uses
# planner rule best_improvement_rank_pair; default 0/unset -> default rule
# heuristic_first_improvement). Server flags below are byte-identical to
# launch235b_eagrid.sh (hence to the v3 online launcher): threshold-ratio 1.02,
# min-prefill-tokens 256, swap budgets, window-floor 8, window-shift-confirm 2,
# dead_zone_ratio default 0.0 -> auto 1.096 identical across all arms (C6),
# calibrate_adaptive_sensitivity default False (C7: warm-start/tier unreachable).
. /workspace/logs/env_235b.sh
EAFLAGS=""
[ "${EA_ADAPTIVE_WINDOW:-0}" = "1" ] && EAFLAGS="$EAFLAGS --pb-oeplb-adaptive-window"
[ "${EA_ADAPTIVE_DECAY:-0}" = "1" ] && EAFLAGS="$EAFLAGS --pb-oeplb-adaptive-decay"
[ "${EA_PLANNER_BI:-0}" = "1" ] && export OEPLB_PLANNER_BEST_IMPROVEMENT=1
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
