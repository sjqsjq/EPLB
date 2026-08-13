#!/bin/bash
# M* latency-side test: use segp_L200 (short segments) with larger M values.
# d31 on segp_L1000 showed monotonic improvement with M (no peak); segp_L200
# should show the peak where the latency cost (slow response to changepoints)
# starts dominating the variance cost.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=$LOGD/segp_L200.jsonl
BUDGET=0.10
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup env $3 bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d34] $2 FAILED"; return 1; }
  echo "[d34] $2 ready"
}
run () {
  cd /workspace/EPLB/OEPLB/scripts && timeout 600 python3 run_grid_bench.py _d34_$1 $DS 256 > /dev/null 2>&1
  L=$LOGD/server235b_$1.log
  D=$(grep "DIAG" $L 2>/dev/null|grep -c "DP0")
  echo "[d34] $1 done | decisions=$D"
}
# Reference (no OEPLB)
boot "$LOGD/launch235b_fixed.sh $LOGD/plc235b_bal.json" d34_base "" && run d34_base
# M = 16, 32, 64, 128, 256 (all with W=16, varying alpha)
for spec in "M16_a0 16 0" "M32_a50 16 0.5" "M64_a75 16 0.75" "M128_a875 16 0.875" "M256_a9375 16 0.9375"; do
  set -- $spec; TAG=$1; W=$2; A=$3
  boot "$LOGD/launch235b_oeplb.sh" d34_$TAG "OE_SW=$W OE_DECAY=$A OEPLB_SWAP_BUDGET_FRAC=$BUDGET" || continue
  run d34_$TAG
done
pkill -9 -f "$PAT"; echo D34_DONE
