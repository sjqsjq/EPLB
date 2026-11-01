#!/bin/bash
# 复现论文 §1: L512_O1_realprover, 8192 req, conc=256, alt BL/OE 2 轮
LOGD=/workspace/logs
PAT="sglang.launch_server"
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
export OEPLB_MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8

boot () {
  pkill -9 -f "$PAT" 2>/dev/null; sleep 8
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 2000 ] && break
    sleep 3
  done
  sleep 12
  setsid nohup bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 360); do
    grep -q "ready to roll" $LOGD/server235b_$2.log && break
    grep -q "Scheduler hit an exception" $LOGD/server235b_$2.log && { echo "[d38] $2 CRASH"; return 1; }
    sleep 4
  done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d38] $2 FAILED"; return 1; }
  sleep 6
  echo "[d38] $2 ready"
}

run_bench () {
  local kind=$1
  local rd=$2
  local launcher=$3
  local tag="d38repro_${kind}_r${rd}"
  boot "$launcher" $tag && \
    (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _${tag} $DS 256 > /workspace/logs/bench_${tag}.log 2>&1) && \
    echo "[d38] ${kind} r${rd} done"
}

run_bench bl 1 "$LOGD/launch235b_identity.sh"
run_bench oe 1 "$LOGD/launch235b_oeplb.sh"
run_bench bl 2 "$LOGD/launch235b_identity.sh"
run_bench oe 2 "$LOGD/launch235b_oeplb.sh"

pkill -9 -f "$PAT" 2>/dev/null; echo D38_REPRO_DONE
