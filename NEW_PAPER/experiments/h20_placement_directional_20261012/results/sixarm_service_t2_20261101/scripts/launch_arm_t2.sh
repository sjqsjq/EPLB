#!/bin/bash
# launch_arm_t2.sh - sixarm_service_t2_20261101 six-arm launcher.
# BYTE-SPLICE of launch_arm_xd.sh (frozen XD mother): BASE composition, PB
# flags, pb_env_common, pb_env_P, ES/ED/MT/DF/P case arms verbatim mother
# bytes. PRE-REGISTERED deltas (PROTOCOL_t2 sec.1, author D1 path A):
#   1) BASE deepep-mode auto replaced by normal for ALL arms (ES/ED
#      substitution lines become no-ops, kept byte-identical); graph off
#      flag untouched.
#   2) PL -> this campaign placements dir (files copied from XD, shas
#      identical to XD frozen lock, asserted at generation).
#   3) P arm RG=0 only (queue passes 0; XD precedent params=0).
# Usage: ART=<artifact_dir> launch_arm_t2.sh ARM [params]
#   I | ES | ED | MT | DF | P RG
set -u
ARM=${1:?usage: launch_arm_xd.sh ARM [params]}
ART=${ART:?ART must be set}
. /workspace/logs/env_235b.sh
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
PL=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101/placements
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
  ES)
    # EPLB static (sglang native): E3 mixed-domain logical_count injected via
    # init_by_eplb/rebalance_experts; R=16; deepep normal (weight-migration
    # support; recorder/eplb NotImplementedError on auto); graph off uniform.
    BASE_ES=${BASE/--deepep-mode auto/--deepep-mode normal}
    launch_server $BASE_ES --ep-num-redundant-experts 16 \
      --init-expert-location $PL/datafore_e3_placement.json;;
  ED)
    # EPLB dynamic (sglang native): frozen E*=e5 (FINAL_SELECTION.json):
    # algorithm auto, R=16, period 256. E3 actual trigger count verified from
    # old logs: 16 rebalance starts per boot (m_E3_E_b3/b5).
    BASE_ED=${BASE/--deepep-mode auto/--deepep-mode normal}
    launch_server $BASE_ED --enable-eplb \
      --eplb-algorithm auto --ep-num-redundant-experts 16 \
      --eplb-rebalance-num-iterations 256;;
  MT)
    # MoETuner-style: E3 profile P_e3.npz -> offline Gurobi ILP1 per-layer
    # clustering (physical_to_logical_map -> init_by_mapping); no redundancy;
    # auto. Solve cost single-listed in placement meta + LEDGER.
    launch_server $BASE --init-expert-location $PL/placement_e3_fair.json;;
  DF)
    # DataForest-style (ISCA26 Algorithm 2 Remap = greedy least-loaded,
    # executed by sglang rebalance_experts on the SHARED E3 logical_count,
    # byte-identical to ES profile file per author key point 2); no
    # redundancy; auto.
    launch_server $BASE --init-expert-location $PL/datafore_e3_remap_placement.json;;
  P)
    RG=${2:?P needs RG}
    if [ "$RG" != "0" ] && [ "$RG" != "1" ]; then echo "FATAL RG must be 0 or 1" >&2; exit 9; fi
    pb_env_P
    export OEPLB_SP_RG=$RG
    launch_server $BASE $PB;;
  *)
    echo "FATAL unknown arm $ARM" >&2; exit 9;;
esac
