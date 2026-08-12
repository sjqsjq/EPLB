#!/bin/bash
# 30B/EP4 T(r) sweep: the model that showed NEGATIVE gain in the paper.
# If beta < 0 (dispatch-dominated), this is the first case where our model
# correctly predicts "don't enable the balancer".
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-30B-A3B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server30b_$2.log 2>&1 &
  for i in $(seq 1 300); do grep -q "ready to roll" $LOGD/server30b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server30b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server30b_$2.log || { echo "[d28] $2 FAILED TO BOOT"; return 1; }
  echo "[d28] $2 ready"
}
# Phase 1: counts already recorded offline into counts30b.json (from recdump30b/*.pt)
[ -f $LOGD/counts30b.json ] || { echo "counts30b.json missing"; echo D28_DONE; exit 1; }
echo "[d28] using existing counts30b.json"
# Phase 2: generate placements
python3 $LOGD/gen_placement.py $LOGD/counts30b.json 4 $LOGD/plc30b 1.10 1.20 1.40 1.60 | tee $LOGD/plc30b.txt
# Phase 3: sweep
for RD in 1 2; do
  for P in bal r110 r120 r140 r160 conc; do
    F=$LOGD/plc30b_$P.json; [ -f $F ] || continue
    boot "$LOGD/launch30b_fixed_g4.sh $F" ${P}_r$RD || continue
    cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d28_${P}_r$RD $DS 256 > /dev/null
    echo "[d28] round $RD $P done"
  done
  boot "$LOGD/launch30b_base_g4.sh" id_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d28_id_r$RD $DS 256 > /dev/null
  echo "[d28] round $RD identity done"
done
pkill -9 -f "$PAT"
echo "[d28] === FIT ==="; python3 $LOGD/r_avg.py $LOGD/counts30b.json 4
echo D28_DONE
