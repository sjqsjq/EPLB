#!/bin/bash
# launch_arm_pc.sh - prover_code_switch_20261027 six-arm launcher.
# Cloned from launch_arm_xd.sh (4d007710, frozen). PRE-REGISTERED deltas
# (PROTOCOL_pc.md sec.2) - placement material reverts to PAPER 5.3.1
# prover-frozen lineage + ED period per paper + author-specified workload:
#   1) $PL -> this campaign placements dir (prover-frozen paper placements;
#      shas in code_lock/placement_shas.env, written once at campaign open).
#   2) ES: datafore_prover_placement.json (paper launch_eplb_static.sh:
#      normal + R16 + init-expert-location prover-frozen logical_count).
#   3) DF: datafore_prover_placement.json (paper launch_datafore_prover.sh:
#      auto, no redundancy, same prover-frozen file).
#   4) MT: placement_pinned_fair.json (paper driver_freq6.sh ->
#      launch_0914_moetuner.sh: auto + init-expert-location, no redundancy).
#   5) ED: --eplb-rebalance-num-iterations 100 (paper value; e3 campaign
#      used 256) + --expert-distribution-recorder-buffer-size 32 (paper).
#   6) P arm: frozen P0 AERO-base rev1 RG=0 (author: new-design scheme);
#      env block byte-identical to launch_arm_xd.sh.
# Everything else (model, BASE composition, PB flags, frozen v21 knobs,
# env_common, ready protocol, uniform --disable-cuda-graph + mem 0.8) is
# verbatim from launch_arm_xd.sh.
# Usage: ART=<artifact_dir> launch_arm_pc.sh ARM [params]
#   I | ES | ED | MT | DF | P RG
set -u
ARM=${1:?usage: launch_arm_pc.sh ARM [params]}
ART=${ART:?ART must be set}
. /workspace/logs/env_235b.sh
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
PL=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/prover_code_switch_20261027/placements
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
    # EPLB static (paper lineage): prover-frozen logical_count injected via
    # init_by_eplb/rebalance_experts; R=16; deepep normal (weight-migration
    # support; recorder/eplb NotImplementedError on auto); graph off uniform.
    BASE_ES=${BASE/--deepep-mode auto/--deepep-mode normal}
    launch_server $BASE_ES --ep-num-redundant-experts 16 \
      --init-expert-location $PL/datafore_prover_placement.json;;
  ED)
    # EPLB dynamic (paper lineage): normal + R16 + enable-eplb + algorithm
    # auto + period 100 (paper launch_eplb_dyn.sh value) + recorder buffer 32.
    BASE_ED=${BASE/--deepep-mode auto/--deepep-mode normal}
    launch_server $BASE_ED --enable-eplb \
      --eplb-algorithm auto --ep-num-redundant-experts 16 \
      --eplb-rebalance-num-iterations 100 \
      --expert-distribution-recorder-buffer-size 32;;
  MT)
    # MoETuner-style (paper lineage): prover fair-split ILP1 placement
    # (physical_to_logical_map -> init_by_mapping); no redundancy; auto.
    launch_server $BASE --init-expert-location $PL/placement_pinned_fair.json;;
  DF)
    # DataForest-style (paper lineage): ISCA26 Algorithm 2 Remap arm ran on
    # datafore_prover_placement.json (auto, no redundancy); sglang
    # rebalance_experts executes greedy least-loaded at init.
    launch_server $BASE --init-expert-location $PL/datafore_prover_placement.json;;
  P)
    RG=${2:?P needs RG}
    if [ "$RG" != "0" ] && [ "$RG" != "1" ]; then echo "FATAL RG must be 0 or 1" >&2; exit 9; fi
    pb_env_P
    export OEPLB_SP_RG=$RG
    launch_server $BASE $PB;;
  *)
    echo "FATAL unknown arm $ARM" >&2; exit 9;;
esac
