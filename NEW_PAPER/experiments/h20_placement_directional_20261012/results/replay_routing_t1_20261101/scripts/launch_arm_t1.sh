#!/bin/bash
# launch_arm_t1.sh - replay_routing_t1_20261101 six-condition launcher.
# BYTE-SPLICE of launch_arm_g3.sh (frozen G3 mother): BASE, PB flags,
# pb_env_common (V21 frozen knobs), pb_env_G3 (SP fixed), launch_server are
# verbatim mother bytes. PRE-REGISTERED deltas (PROTOCOL_t1 sec.3):
#   1) BASE deepep-mode auto replaced by normal (author D1 path A; graph off
#      flag untouched).
#   2) case arms ORIG/GAP32/GAP128/GAP300/MD300/BI300 replace G3_PF/G3_ALL;
#      per-arm budget via bash substitution of --pb-oeplb-max-total-ops 300;
#      OEPLB_PAIR_RULE explicit; OEPLB_RECORD_ALL_PHASE=1; BI env per config.
#   3) ORIG = BASE only (no PB), trivial init.
# Usage: ART=<artifact_dir> launch_arm_t1.sh ARM
set -u
ARM=${1:?usage: launch_arm_g3.sh ARM}
ART=${ART:?ART must be set}
. /workspace/logs/env_235b.sh
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
BASE="--model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache --watchdog-timeout 600"
PB="--enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02 --pb-oeplb-min-prefill-tokens 256 --pb-oeplb-sync-window 16 --pb-oeplb-max-total-swap-layers 94 --pb-oeplb-max-swaps-per-layer 64 --pb-oeplb-min-swap-ops 8 --pb-oeplb-max-total-ops 300 --pb-oeplb-decay-factor 0.0 --pb-oeplb-window-floor 8 --pb-oeplb-window-shift-confirm 2"

pb_env_common(){
  export EA_ALPHA=0.0
  unset EA_ADAPTIVE_WINDOW EA_ADAPTIVE_DECAY EA_PLANNER_BI
  export OEPLB_WINDOW_CONTROLLER=21
  # FROZEN fitted knobs (FREEZE_record_v2p1.md; do not edit)
  export OEPLB_V21_KAPPA=3875.1768798828125
  export OEPLB_V21_Z_LO=22.643858727592757 OEPLB_V21_Z_HI=314.14010272527486 OEPLB_V21_Z_URGENT=942.4203081758246
  export OEPLB_V21_THETA_R_LOW=0.010095552687949321 OEPLB_V21_THETA_R_ACT=0.011395643112507271
  export OEPLB_V21_N_MIN=12320768
  export OEPLB_V21_M_SHRINK=2 OEPLB_V21_M_GROW=3
  export OEPLB_V21_DWELL_SHRINK=2 OEPLB_V21_DWELL_GROW=2
  export OEPLB_V21_U_MAX=4 OEPLB_V21_W_UNCERTAIN_TARGET=16
  export OEPLB_V21_URGENT_EPISODE_CAP=3 OEPLB_V21_BETA=0.5
  export OEPLB_V21_LOG=$ART/v2p1_rank0.jsonl
  export EA_W=16
  export OEPLB_PROBE_MODE=off
}
pb_env_G3(){
  pb_env_common
  export OEPLB_SP_MODE=fixed OEPLB_SP_S=1 OEPLB_SP_W=16 \
         OEPLB_SP_SEED=20261018 OEPLB_SP_REV=0
}

launch_server(){ if [ "${LAUNCH_DRY:-0}" = 1 ]; then echo "DRY python3 -m sglang.launch_server $*"; exit 0; fi
  exec python3 -m sglang.launch_server "$@"; }

case "$ARM" in
  ORIG)
    launch_server $BASE;;
  GAP32)
    pb_env_G3
    export OEPLB_RECORD_ALL_PHASE=1 OEPLB_PAIR_RULE=gap
    launch_server $BASE ${PB/--pb-oeplb-max-total-ops 300/--pb-oeplb-max-total-ops 32};;
  GAP128)
    pb_env_G3
    export OEPLB_RECORD_ALL_PHASE=1 OEPLB_PAIR_RULE=gap
    launch_server $BASE ${PB/--pb-oeplb-max-total-ops 300/--pb-oeplb-max-total-ops 128};;
  GAP300)
    pb_env_G3
    export OEPLB_RECORD_ALL_PHASE=1 OEPLB_PAIR_RULE=gap
    launch_server $BASE $PB;;
  MD300)
    pb_env_G3
    export OEPLB_RECORD_ALL_PHASE=1 OEPLB_PAIR_RULE=maxdelta
    launch_server $BASE $PB;;
  BI300)
    pb_env_G3
    export OEPLB_RECORD_ALL_PHASE=1 OEPLB_PAIR_RULE=gap OEPLB_PLANNER_BEST_IMPROVEMENT=1
    launch_server $BASE $PB;;
  *)
    echo "FATAL unknown arm $ARM" >&2; exit 9;;
esac
