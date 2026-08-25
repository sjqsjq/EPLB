#!/bin/bash
# Prover single-domain 235B, identity baseline + OEPLB, 2 rounds each.
# Robust startup: longer cleanup wait + GPU-memory check before boot.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
boot () {
  pkill -9 -f "$PAT" 2>/dev/null; sleep 5
  # wait until GPU memory is freed (<1000MiB on all)
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 1500 ] && break
    sleep 3
  done
  sleep 8
  setsid nohup bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Scheduler hit an exception" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d38] $2 FAILED"; return 1; }
  echo "[d38] $2 ready"
}
for RD in 1 2; do
  boot "$LOGD/launch235b_identity.sh" pv38_bl_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d38_bl_r$RD $DS 256 > /dev/null) && echo "[d38] baseline r$RD done"
  boot "$LOGD/launch235b_oeplb.sh" pv38_oe_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d38_oe_r$RD $DS 256 > /dev/null) && echo "[d38] oeplb r$RD done"
done
pkill -9 -f "$PAT" 2>/dev/null; echo D38_DONE
