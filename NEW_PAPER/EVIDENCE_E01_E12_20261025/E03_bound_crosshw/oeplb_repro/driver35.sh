#!/bin/bash
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/multi_domain/prefill_heavy_universal.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d35] $2 FAILED"; return 1; }
  echo "[d35] $2 ready"
}
for RD in 1 2; do
  boot "$LOGD/launch235b_fixed.sh $LOGD/plc235b_bal.json" md16k_bl_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d35_bl_r$RD $DS 256 > /dev/null) && echo "[d35] baseline r$RD done"
  boot "$LOGD/launch235b_oeplb.sh" md16k_oe_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d35_oe_r$RD $DS 256 > /dev/null) && echo "[d35] oeplb r$RD done"
done
pkill -9 -f "$PAT"; echo D35_DONE
