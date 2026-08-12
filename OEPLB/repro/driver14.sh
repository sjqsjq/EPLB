#!/bin/bash
# 235B T(r) sweep -- closes the biggest open item in the bound model: the paper
# borrows r_k=1.10 from the 57B/8-GPU sweep, which drives the L512 system
# efficiency above 100% and therefore cannot be right.
LOGD=/workspace/logs
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8

boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 10
  setsid nohup bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 300); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d14] $2 FAILED TO BOOT"; return 1; }
  echo "[d14] $2 ready"
}

# ---- phase 1: record the routing distribution ----
if [ ! -f $LOGD/counts235b.json ]; then
  boot "$LOGD/launch235b_rec.sh" rec || exit 1
  python3 $LOGD/dump_counts235.py /dev/null start
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d14_rec $DS 256 > /dev/null
  echo "[d14] record load done"
  python3 $LOGD/dump_counts235.py $LOGD/counts235b.json stop
fi
[ -f $LOGD/counts235b.json ] || { echo "[d14] no counts, abort"; exit 1; }

# ---- phase 2: placements ----
python3 $LOGD/gen_placement.py $LOGD/counts235b.json 8 $LOGD/plc235b 1.20 1.40 1.60 1.75 | tee $LOGD/plc235b.txt

# ---- phase 3: sweep ----
for RD in 1 2; do
for P in bal r120 r140 r160 r175 conc; do
  F=$LOGD/plc235b_$P.json
  [ -f $F ] || { echo "[d14] skip $P"; continue; }
  boot "$LOGD/launch235b_fixed.sh $F" sw_${P}_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d14_${P}_r$RD $DS 256 > /dev/null
  echo "[d14] round $RD $P done"
done
  boot "$LOGD/launch_baseline.sh" sw_id_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d14_id_r$RD $DS 256 > /dev/null
  echo "[d14] round $RD identity done"
done
pkill -9 -f "$PAT"
echo D14_DONE
