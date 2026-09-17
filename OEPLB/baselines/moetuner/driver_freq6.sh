#!/bin/bash
set -euo pipefail
LOGD=/workspace/logs
BASE=/workspace/EPLB/OEPLB/baselines/moetuner
KIND=${1:-identity}
NRUNS=${2:-3}
CONC=32
N=1800

wait_ready() {
  local logfile=$1
  for i in $(seq 1 300); do
    if grep -q "The server is fired up" "$logfile" 2>/dev/null; then return 0; fi
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

echo "[freq6/$KIND] booting server..."
setsid nohup bash $SCRIPT > $LOGD/serverfreq6_${KIND}.log 2>&1 < /dev/null &
PID=$!
wait_ready $LOGD/serverfreq6_${KIND}.log || { echo "boot failed"; tail -30 $LOGD/serverfreq6_${KIND}.log; exit 1; }
echo "[freq6/$KIND] server ready pid=$PID"
sleep 5

for r in $(seq 1 $NRUNS); do
  LABEL="${KIND}_r${r}"
  echo "[freq6/$KIND] run $r/$NRUNS ..."
  python3 $BASE/freq6_bench_json.py $CONC $N "$LABEL" 2>&1 | tee $BASE/logs/bench_freq6_${LABEL}.log
  sleep 3
done

kill -TERM -$PID 2>/dev/null || true
sleep 5
kill -KILL -$PID 2>/dev/null || true
echo "[freq6/$KIND] DONE"
