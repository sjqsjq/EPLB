#!/bin/bash
# Re-measure multi-domain 235B: the paper's +14.0% gives eta=118%>100%, likely
# inflated (n=1, dataset from /tmp). The multi-domain dataset is still on disk.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/multi_domain/multidomain_v2_out1.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d33] $2 FAILED"; return 1; }
  echo "[d33] $2 ready"
}
# baseline (no OEPLB): launch235b_fixed with bal placement
for RD in 1 2; do
  boot "$LOGD/launch235b_fixed.sh $LOGD/plc235b_bal.json" md_bl_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d33_bl_r$RD $DS 256 > /dev/null) && \
    echo "[d33] baseline r$RD done"
done
# OEPLB with dead-zone threshold
for RD in 1 2; do
  boot "$LOGD/launch235b_oeplb.sh" md_oe_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d33_oe_r$RD $DS 256 > /dev/null) && \
    echo "[d33] oeplb r$RD done"
done
pkill -9 -f "$PAT"; echo D33_DONE
