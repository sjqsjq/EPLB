#!/bin/bash
# Multi-domain 235B with IDENTITY baseline (the correct headline comparison).
# d35 used the static-optimal 'bal' placement as baseline, which measured the
# adaptation benefit (OEPLB vs static-optimal) but NOT the headline gain. This
# gives OEPLB vs identity on the same prefill_heavy_universal set.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/multi_domain/prefill_heavy_universal.jsonl
boot () {
  pkill -9 -f "$PAT" 2>/dev/null; sleep 5
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 1500 ] && break
    sleep 3
  done
  sleep 8
  setsid nohup bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d39] $2 FAILED"; return 1; }
  echo "[d39] $2 ready"
}
for RD in 1 2; do
  boot "$LOGD/launch235b_identity.sh" md39_bl_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d39_bl_r$RD $DS 256 > /dev/null) && echo "[d39] baseline(identity) r$RD done"
  boot "$LOGD/launch235b_oeplb.sh" md39_oe_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d39_oe_r$RD $DS 256 > /dev/null) && echo "[d39] oeplb r$RD done"
done
pkill -9 -f "$PAT" 2>/dev/null; echo D39_DONE
