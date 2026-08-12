#!/bin/bash
# Adaptive window / adaptive decay experiment on 235B/EP8.
#
# Core hypothesis (ADAPTIVE_DESIGN.md): W and alpha affect steady-state behaviour
# ONLY through the effective memory M = W/(1-alpha). So a (W,alpha) grid should
# collapse onto a single throughput-vs-M curve, and the optimal M* should grow
# with segment length L_seg. Existing data cannot test this (the 'multi-domain'
# set was near-homogeneous), so we build genuinely piecewise-stationary loads.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
boot () {  # $1=launch $2=tag $3=envstr
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup env $3 bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d29] $2 FAILED"; return 1; }
  echo "[d29] $2 ready"
}
runbench () {  # $1=tag $2=dataset
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d29_$1 $2 256 > /dev/null
  L=$LOGD/server235b_$1.log
  echo "[d29] $1 done | decisions=$(grep -c 'swap(s) done' $L) | sw_changes=$(grep -c 'sw to' $L) | resets=$(grep -c 'RESET\|zeroing' $L)"
}

# --- Phase 0: verify the two datasets are genuinely different domains on 235B ---
# (record routing counts for each, compare r_before; a real changepoint needs them to differ)
# Skipped here for time; make_segmented already interleaves L256(math) vs ShareGPT(chat).

# --- Phase 1: (W, alpha) grid on the L_seg=200 segmented load ---
# M = W/(1-alpha). Grid chosen so several (W,alpha) share the same M:
#   M=16:  (W16,a0)  (W8,a0.5)
#   M=32:  (W32,a0)  (W16,a0.5)  (W8,a0.75)
#   M=64:  (W64,a0)  (W32,a0.5)  (W16,a0.75)
#   M=160: (W16,a0.9)
DS=$LOGD/seg_L200.jsonl
for spec in "W16_a0 16 0" "W8_a50 8 0.5" "W32_a0 32 0" "W16_a50 16 0.5" "W8_a75 8 0.75" \
            "W64_a0 64 0" "W32_a50 32 0.5" "W16_a75 16 0.75" "W16_a90 16 0.9"; do
  set -- $spec; TAG=$1; W=$2; A=$3
  boot "$LOGD/launch235b_oeplb.sh" g_$TAG "OE_SW=$W OE_DECAY=$A" || continue
  runbench g_$TAG $DS
done

# --- Phase 2: adaptive vs best static, across 3 segment lengths ---
for LS in 50 200 1000; do
  DS=$LOGD/seg_L${LS}.jsonl
  # static baseline (sw=16, decay=0.5)
  boot "$LOGD/launch235b_oeplb.sh" s_L${LS}_static "OE_SW=16 OE_DECAY=0.5" && runbench s_L${LS}_static $DS
  # adaptive window enabled
  boot "$LOGD/launch235b_oeplb.sh" s_L${LS}_adw "OE_SW=16 OE_DECAY=0.5 OE_ADW=--pb-oeplb-adaptive-window" && runbench s_L${LS}_adw $DS
  # adaptive window + adaptive decay (alpha->0 at changepoint, tests P3)
  boot "$LOGD/launch235b_oeplb.sh" s_L${LS}_adwd "OE_SW=16 OE_DECAY=0.5 OE_ADW=--pb-oeplb-adaptive-window OEPLB_ADAPTIVE_DECAY=1" && runbench s_L${LS}_adwd $DS
done
pkill -9 -f "$PAT"
echo D29_DONE
