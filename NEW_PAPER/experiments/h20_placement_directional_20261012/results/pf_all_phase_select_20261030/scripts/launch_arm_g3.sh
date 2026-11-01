#!/bin/bash
# launch_arm_g3.sh - pf_all_phase_select_20261030 two-arm launcher.
# Cloned from launch_arm_pc.sh (PBS archive, frozen) by BYTE-SPLICE: BASE,
# PB flag block, pb_env_common (frozen V21 knobs) and launch_server are
# verbatim bytes of the mother file; no manual re-typing. PRE-REGISTERED
# deltas (PROTOCOL_g3.md sec.2):
#   1) arms G3_PF / G3_ALL replace the six PBS arms; identity initial
#      layout (NO --init-expert-location); BASE + PB unchanged (frozen
#      threshold-ratio 1.02 flag + auto dead_zone => effective thr
#      1.0962402945, production aperture).
#   2) pb_env_G3 = pb_env_common + FIXED SP mode: OEPLB_SP_MODE=fixed
#      SP_S=1 SP_W=16 SP_REV=0 (PD-12 config.py:243 rejects rev1 unless
#      adaptive) SP_SEED=20261018; SP_H unset => H=W (config.py:371
#      default 0). Adaptive-only knobs NOT set (env_clear in runner).
#   3) SOLE ARM DELTA: OEPLB_RECORD_ALL_PHASE=0 (G3_PF) vs 1 (G3_ALL)
#      (config.py:373; PD-13 gate controller.py:460-465).
#   4) PL placement block removed (campaign has no placements).
# Usage: ART=<artifact_dir> launch_arm_g3.sh ARM   (G3_PF | G3_ALL)
set -u
ARM=${1:?usage: launch_arm_g3.sh ARM}
ART=${ART:?ART must be set}
. /workspace/logs/env_235b.sh
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
BASE="--model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache --watchdog-timeout 600"
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
  G3_PF)
    pb_env_G3
    export OEPLB_RECORD_ALL_PHASE=0
    launch_server $BASE $PB;;
  G3_ALL)
    pb_env_G3
    export OEPLB_RECORD_ALL_PHASE=1
    launch_server $BASE $PB;;
  *)
    echo "FATAL unknown arm $ARM" >&2; exit 9;;
esac
