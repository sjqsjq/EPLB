#!/bin/bash
# prover_256tok_out1 dataset — swap 掉之前的 L512_O1_realprover_n8192
LOGD=/workspace/logs
PAT="sglang.launch_server"
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl

boot () {
  pkill -9 -f "$PAT" 2>/dev/null; sleep 5
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 1500 ] && break
    sleep 3
  done
  sleep 8
  setsid nohup bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do
    grep -q "ready to roll" $LOGD/server235b_$2.log && break
    grep -q "Scheduler hit an exception" $LOGD/server235b_$2.log && break
    sleep 4
  done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[p256] $2 FAILED"; return 1; }
  echo "[p256] $2 ready"
}

for RD in 1 2; do
  boot "$LOGD/launch235b_identity.sh" p256_bl_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _p256_bl_r$RD $DS 256 > /dev/null) && echo "[p256] baseline r$RD done"
done

for RD in 1 2; do
  boot "$LOGD/launch235b_oeplb.sh" p256_oe_r$RD && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _p256_oe_r$RD $DS 256 > /dev/null) && echo "[p256] oeplb-adaptive r$RD done"
done

pkill -9 -f "$PAT" 2>/dev/null; echo P256_DONE
