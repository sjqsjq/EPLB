#!/bin/bash
# Re-measure §5.8 reproducibility: 3 independent cold starts on 235B L512_O1.
# The old data (22603/22885/22850) is from /tmp-era and cannot be traced to L512.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $LOGD/launch235b_oeplb.sh > $LOGD/server235b_rep_r$1.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_rep_r$1.log && break; grep -q "Traceback" $LOGD/server235b_rep_r$1.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_rep_r$1.log || { echo "[d32] r$1 FAILED"; return 1; }
  echo "[d32] r$1 ready"
}
for R in 1 2 3; do
  boot $R || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d32_rep_r$R $DS 256 > /dev/null
  echo "[d32] r$R done"
done
pkill -9 -f "$PAT"; echo D32_DONE
