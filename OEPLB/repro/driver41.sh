#!/bin/bash
# M* peak search at intermediate segment length L500.
# d34 (L200): throughput worsened with M (optimal at smallest M=16).
# d31 (L1000): throughput improved with M up to 64 (optimal not yet reached).
# So the optimal M* shifts from <16 (L200) to >64 (L1000). At L500 we expect an
# INTERIOR minimum, confirming the theory that a finite optimal M exists and
# grows with L_seg (M* ~ sqrt(L_seg)).
# W=16 fixed, alpha varies -> M = W/(1-alpha) in {16,32,64,128,256}.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=$LOGD/segp_L500.jsonl
BUDGET=0.10
boot () {
  pkill -9 -f "$PAT" 2>/dev/null; sleep 5
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 1500 ] && break; sleep 3
  done
  sleep 8
  setsid nohup env $3 bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d41] $2 FAILED"; return 1; }
  echo "[d41] $2 ready"
}
run () {
  cd /workspace/EPLB/OEPLB/scripts && timeout 700 python3 run_grid_bench.py _d41_$1 $DS 256 > /dev/null 2>&1
  echo "[d41] $1 done"
}
boot "$LOGD/launch235b_fixed.sh $LOGD/plc235b_bal.json" d41_base "" && run d41_base
for spec in "M16 16 0" "M32 16 0.5" "M64 16 0.75" "M128 16 0.875" "M256 16 0.9375"; do
  set -- $spec; TAG=$1; W=$2; A=$3
  boot "$LOGD/launch235b_oeplb.sh" d41_$TAG "OE_SW=$W OE_DECAY=$A OEPLB_SWAP_BUDGET_FRAC=$BUDGET" || continue
  run d41_$TAG
done
pkill -9 -f "$PAT" 2>/dev/null; echo D41_DONE
