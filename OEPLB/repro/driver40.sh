#!/bin/bash
# Strengthen the 30B negative-gain / budget-recovery result (was n=2, thin).
# Comprehensive: baseline(identity) + default(1.02) + deadzone(1.031) + deadzone+budget,
# n=3 each. Confirms beta=+0.207 positive bound but default negative, budget recovers.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-30B-A3B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
boot () {
  pkill -9 -f "$PAT" 2>/dev/null; sleep 5
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 1500 ] && break; sleep 2
  done
  sleep 5
  setsid nohup env $3 bash $1 > $LOGD/server30b_$2.log 2>&1 &
  for i in $(seq 1 200); do grep -q "ready to roll" $LOGD/server30b_$2.log && break; grep -q "Traceback" $LOGD/server30b_$2.log && break; sleep 3; done
  grep -q "ready to roll" $LOGD/server30b_$2.log || { echo "[d40] $2 FAILED"; return 1; }
  echo "[d40] $2 ready"
}
for RD in 1 2 3; do
  boot "$LOGD/launch30b_base_g4.sh" b40_bl_r$RD "" && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d40_bl_r$RD $DS 256 > /dev/null) && echo "[d40] baseline r$RD done"
  boot "$LOGD/launch30b_oeplb_g4.sh" b40_dflt_r$RD "" && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d40_dflt_r$RD $DS 256 > /dev/null) && echo "[d40] default r$RD done"
  boot "$LOGD/launch30b_oeplb_g4.sh" b40_dz_r$RD "OEPLB_DEAD_ZONE_RATIO=1.031" && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d40_dz_r$RD $DS 256 > /dev/null) && echo "[d40] deadzone r$RD done"
  boot "$LOGD/launch30b_oeplb_g4.sh" b40_dzb_r$RD "OEPLB_DEAD_ZONE_RATIO=1.031 OEPLB_SWAP_BUDGET_FRAC=0.02" && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d40_dzb_r$RD $DS 256 > /dev/null) && echo "[d40] deadzone+budget r$RD done"
done
pkill -9 -f "$PAT" 2>/dev/null; echo D40_DONE
