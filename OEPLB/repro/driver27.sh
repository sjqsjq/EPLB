#!/bin/bash
# Validate the three new mechanisms on the two configurations where the eta
# analysis says each failure mode lives:
#   8-GPU / L256    -- condition A (swap/headroom = 1.26, eta 29%): dead zone
#   4-GPU / ShareGPT-- condition B (swap/headroom = 0.09, eta 6%):  bias + gate
# Offline replay predicts the homogeneous L256 windows are untouched by the bias
# correction and that ShareGPT loses ~45% of its decisions, so any change on
# L256 must come from the dead zone and any change on ShareGPT from the gate.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
D=/data/minghua/sjq/OEPLBdata/datasets
DS256=$D/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
DSSHARE=$D/sharegpt/sharegpt_natural_20k.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup env $3 bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 300); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d27] $2 FAILED TO BOOT"; return 1; }
  echo "[d27] $2 ready"
}
runarm () {   # $1=tag $2=launch $3=dataset $4=envstring
  for RD in 1 2; do
    boot "$2" ${1}_r$RD "$4" || continue
    cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d27_${1}_r$RD $3 256 > /dev/null
    L=$LOGD/server57b_${1}_r$RD.log
    echo "[d27] $1 r$RD done | decisions=$(grep -c 'swap(s) done' $L) | nogap=$(grep -c 'no swap' $L) | gated=$(grep -c 'noise/margin' $L) | budget=$(grep -c 'swap budget' $L)"
  done
}
# --- 8 GPU / L256: condition A ---
runarm g8_base   "$LOGD/launch57b_oeplb_g8.sh" $DS256 ""
runarm g8_dz     "$LOGD/launch57b_oeplb_g8.sh" $DS256 "OEPLB_DEAD_ZONE_RATIO=1.099"
runarm g8_dz_bud "$LOGD/launch57b_oeplb_g8.sh" $DS256 "OEPLB_DEAD_ZONE_RATIO=1.099 OEPLB_SWAP_BUDGET_FRAC=0.02"
runarm g8_all    "$LOGD/launch57b_oeplb_g8.sh" $DS256 "OEPLB_DEAD_ZONE_RATIO=1.099 OEPLB_BIAS_CORRECT=1 OEPLB_BIAS_GATE=0.5 OEPLB_SWAP_BUDGET_FRAC=0.02"
# --- 4 GPU / ShareGPT: condition B (22.6 min per run, 2 arms only) ---
runarm g4s_base  "$LOGD/launch57b_oeplb_g4.sh" $DSSHARE ""
runarm g4s_all   "$LOGD/launch57b_oeplb_g4.sh" $DSSHARE "OEPLB_DEAD_ZONE_RATIO=1.032 OEPLB_BIAS_CORRECT=1 OEPLB_BIAS_GATE=0.5"
pkill -9 -f "$PAT"; echo D27_DONE
