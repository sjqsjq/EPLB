#!/bin/bash
# launch_arm.sh — final_plan_20261019 unified per-arm launcher.
# Usage: ART=<artifact_dir> launch_arm.sh ARM [params]
#   I                     identity: base only (no PB, no EPLB)
#   E ALGO R PERIOD       sglang EPLB (--enable-eplb; dispatch/init/chunk/
#                         threshold/buffer = THIS VERSION's defaults, recorded
#                         in C_configs/EPLB_MECHANISM_NOTES.md — no hidden search)
#   P                     PB-OEPLB-final (rev1 frozen, verbatim env_common)
#   F S H W               PB-fixed (SP_MODE=fixed, SP_REV=0, F_dense precedent
#                         + OEPLB_SP_H; W_EA adaptation stays ACTIVE per
#                         ARM_FLAGS.md pre-registered interpretation)
#   A1|A2|A3|A4           P + exactly one ablation flag
set -u
ARM=${1:?usage: launch_arm.sh ARM [params]}
ART=${ART:?ART must be set}
. /workspace/logs/env_235b.sh
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
BASE="--model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 --port 30000 --host 0.0.0.0 --trust-remote-code --disable-radix-cache --watchdog-timeout 600"
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
pb_env_P(){
  pb_env_common
  export OEPLB_SP_MODE=adaptive OEPLB_SP_S=1 OEPLB_SP_W=16 \
         OEPLB_SP_SEED=20261018 OEPLB_SP_QUIET_STREAK=3 \
         OEPLB_SP_RECOVER_MIN_CYCLES=2 OEPLB_SP_REV=1
}

launch_server(){ if [ "${LAUNCH_DRY:-0}" = 1 ]; then echo "DRY python3 -m sglang.launch_server $*"; exit 0; fi
  exec python3 -m sglang.launch_server "$@"; }

case "$ARM" in
  I)
    launch_server $BASE;;
  E)
    ALGO=${2:?E needs ALGO}; R=${3:?E needs R}; PERIOD=${4:?E needs PERIOD}
    # PRE-REGISTERED deviation (EPLB_MECHANISM_NOTES.md sec.6, 2026-10-18):
    # this sglang build's expert-distribution recorder (mode 'stat', auto-set by
    # --enable-eplb) is implemented ONLY for deepep-mode normal/low_latency;
    # 'auto' raises NotImplementedError (eplb/expert_distribution.py:314).
    # Sealed precedent: d38_L512_O1_comparison EPLB baseline (2026-09-10) ran
    # this exact stack with --deepep-mode normal. E arm therefore uses normal;
    # ALL other arms keep the rev1 sealed base (auto). Fairness caveat goes to
    # the limitations section of the freeze doc and final report.
    BASE_E=${BASE/--deepep-mode auto/--deepep-mode normal}
    launch_server $BASE_E --enable-eplb \
      --eplb-algorithm "$ALGO" --ep-num-redundant-experts "$R" \
      --eplb-rebalance-num-iterations "$PERIOD";;
  P)
    pb_env_P
    launch_server $BASE $PB;;
  F)
    S=${2:?F needs S}; H=${3:?F needs H}; W=${4:?F needs W}
    pb_env_common
    export OEPLB_SP_MODE=fixed OEPLB_SP_S=$S OEPLB_SP_H=$H OEPLB_SP_W=$W OEPLB_SP_REV=0
    launch_server $BASE $PB;;
  A1)
    pb_env_P; export OEPLB_RECORD_ALL_PHASE=1
    launch_server $BASE $PB;;
  A2)
    pb_env_P; export OEPLB_PAIR_RULE=maxdelta
    launch_server $BASE $PB;;
  A3)
    pb_env_P; export OEPLB_SP_FORCE_S=1
    launch_server $BASE $PB;;
  A4)
    pb_env_P; export OEPLB_NO_COMMIT=1
    launch_server $BASE $PB;;
  *)
    echo "FATAL unknown arm $ARM" >&2; exit 9;;
esac
