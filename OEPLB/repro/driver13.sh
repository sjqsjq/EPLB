#!/bin/bash
# EP=4 counterpart of driver12: does the hinge knee r_k move with EP size?
# Routing counts are EP-independent, so counts57b.json is reused.
LOGD=/workspace/logs
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct

boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d13] $2 FAILED TO BOOT"; return 1; }
  echo "[d13] $2 ready"
}

python3 $LOGD/gen_placement.py $LOGD/counts57b.json 4 $LOGD/plc57b_g4 1.08 1.15 1.25 1.40 | tee $LOGD/plc57b_g4.txt

for RD in 1 2; do
for P in bal r108 r115 r125 r140 conc; do
  F=$LOGD/plc57b_g4_$P.json
  [ -f $F ] || { echo "[d13] skip $P"; continue; }
  boot "$LOGD/launch57b_fixed_g4.sh $F" g4_${P}_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d13_${P}_r$RD $DS 256 > /dev/null
  echo "[d13] round $RD $P done"
done
  boot "$LOGD/launch57b_base_g4.sh" g4_id_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d13_id_r$RD $DS 256 > /dev/null
  echo "[d13] round $RD identity done"
done
pkill -9 -f "$PAT"
echo D13_DONE
