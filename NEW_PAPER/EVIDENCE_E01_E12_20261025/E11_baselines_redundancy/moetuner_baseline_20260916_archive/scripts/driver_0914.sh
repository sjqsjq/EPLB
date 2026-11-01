#!/bin/bash
set -euo pipefail
LOGD=/workspace/logs
BASE=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
KIND=${1:-identity}   # identity | moetuner
NRUNS=${2:-3}

wait_ready() {
  local logfile=$1
  for i in $(seq 1 300); do
    if grep -q "The server is fired up" "$logfile" 2>/dev/null; then return 0; fi
    if grep -q "Traceback\|Error" "$logfile" 2>/dev/null; then
      if ! grep -q "The server is fired up" "$logfile" 2>/dev/null; then
        # ignore transient errors after fire-up, but here fire-up not yet
        :
      fi
    fi
    sleep 3
  done
  return 1
}

if [ "$KIND" = "moetuner" ]; then
  export PLACEMENT=$BASE/artifacts/placement_pinned_fair.json
  SCRIPT=$BASE/launch_0914_moetuner.sh
else
  SCRIPT=$BASE/launch_0914_identity.sh
fi

echo "[0914/$KIND] booting server..."
setsid nohup bash $SCRIPT > $LOGD/server0914_${KIND}.log 2>&1 < /dev/null &
PID=$!
if ! wait_ready $LOGD/server0914_${KIND}.log; then
  echo "[0914/$KIND] server failed to start; tail:"; tail -50 $LOGD/server0914_${KIND}.log; exit 1
fi
echo "[0914/$KIND] server ready pid=$PID"
sleep 5

for r in $(seq 1 $NRUNS); do
  LABEL="${KIND}_r${r}"
  echo "[0914/$KIND] run $r/$NRUNS ..."
  python3 $BASE/bench_0914.py $DS 256 "$LABEL" 2>&1 | tee $BASE/logs/bench_0914_${LABEL}.log
  sleep 3
done

echo "[0914/$KIND] shutting down server pid=$PID"
kill -TERM -$PID 2>/dev/null || true
sleep 5
kill -KILL -$PID 2>/dev/null || true
echo "[0914/$KIND] DONE"
