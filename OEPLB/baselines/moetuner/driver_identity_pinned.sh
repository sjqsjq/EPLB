#!/bin/bash
set -euo pipefail
LOGD=/workspace/logs
BASE=/workspace/EPLB/OEPLB/baselines/moetuner
BENCHDIR=/workspace/EPLB/OEPLB/scripts
DS=$BASE/artifacts/pinned10x_tail1024.jsonl

wait_ready() {
  local logfile=$1
  for i in $(seq 1 200); do
    if grep -q "The server is fired up" "$logfile" 2>/dev/null; then return 0; fi
    if grep -q "Traceback\|Error" "$logfile" 2>/dev/null; then return 1; fi
    sleep 3
  done
  return 1
}

for r in 1 2; do
  tag="identity_pinned_r${r}"
  echo "[identity/pinned] booting $tag"
  setsid nohup bash /workspace/logs/launch235b_identity.sh > $LOGD/server235b_${tag}.log 2>&1 < /dev/null &
  PID=$!
  wait_ready $LOGD/server235b_${tag}.log
  echo "[identity/pinned] $tag ready"
  ( cd $BENCHDIR && python3 run_grid_bench.py "_${tag}" $DS 256 > $BASE/logs/bench_${tag}.log 2>&1 )
  echo "[identity/pinned] r${r} done"
  kill -TERM -$PID 2>/dev/null || true
  wait $PID 2>/dev/null || true
  sleep 10
done
echo "[identity/pinned] ALL_DONE"
