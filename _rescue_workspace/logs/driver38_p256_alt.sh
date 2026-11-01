#!/bin/bash
# prover_256tok_out1 dataset — 交替运行 BL↔OE，每次都 boot 新 server
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
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[p256alt] $2 FAILED"; return 1; }
  echo "[p256alt] $2 ready"
}

run_bench () {
  local kind=$1  # bl|oe
  local rd=$2
  local launcher=$3
  local tag="p256alt_${kind}_r${rd}"
  boot "$launcher" $tag && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _${tag} $DS 256 > /dev/null) && \
    echo "[p256alt] ${kind} r${rd} done"
}

# Round 1
run_bench bl 1 "$LOGD/launch235b_identity.sh"
run_bench oe 1 "$LOGD/launch235b_oeplb.sh"
# Round 2
run_bench bl 2 "$LOGD/launch235b_identity.sh"
run_bench oe 2 "$LOGD/launch235b_oeplb.sh"

pkill -9 -f "$PAT" 2>/dev/null; echo P256_ALT_DONE
