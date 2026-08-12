#!/bin/bash
# The balancer decides on an UNWEIGHTED per-window imbalance estimate, so tiny
# batches (whose ratio is inflated by sampling variance and which cost no wall
# time) can trigger swaps that pay P2P blocking for nothing.  min-prefill-tokens
# defaults to 256 while the median window carries ~9e5 tokens.  Raise the gate
# and see whether eta (29% on 8-GPU 57B) improves.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 300); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d26] $2 FAILED TO BOOT"; return 1; }
  echo "[d26] $2 ready"
}
# arm A: default gate 256 (paper's config).  arm B: gate at 1e5 tokens.
# arm C: gate 1e5 AND threshold raised to the measured knee r_k=1.099.
for RD in 1 2; do
  for A in "256_102 256 1.02" "1e5_102 100000 1.02" "1e5_rk 100000 1.099"; do
    set -- $A; TAG=$1; GATE=$2; TH=$3
    sed -e "s/--pb-oeplb-min-prefill-tokens 256/--pb-oeplb-min-prefill-tokens $GATE/" \
        -e "s/--pb-oeplb-threshold-ratio 1.02/--pb-oeplb-threshold-ratio $TH/" \
        $LOGD/launch57b_oeplb_g8.sh > $LOGD/launch57b_oeplb_g8_$TAG.sh
    boot "$LOGD/launch57b_oeplb_g8_$TAG.sh" g8_$TAG'_r'$RD || continue
    cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d26_${TAG}_r$RD $DS 256 > /dev/null
    echo "[d26] round $RD $TAG done  swaps=$(grep -c 'swap(s) done' $LOGD/server57b_g8_${TAG}_r$RD.log)"
  done
done
pkill -9 -f "$PAT"; echo D26_DONE
