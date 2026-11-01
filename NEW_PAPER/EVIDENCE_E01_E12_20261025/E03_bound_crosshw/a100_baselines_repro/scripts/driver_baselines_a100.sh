#!/bin/bash
# Usage: bash driver_baselines_a100.sh <method> <table:0914|freq6> [nruns]
set -uo pipefail
METHOD=$1
TABLE=${2:-0914}
NRUNS=${3:-3}
A100=/workspace/EPLB/OEPLB/baselines/a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
LOGD=/workspace/logs
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
mkdir -p "$LOGD" "$A100/logs"

SLOG=$LOGD/server_a100_${TABLE}_${METHOD}.log
: > "$SLOG"

# wait until no sglang server is running (port free)
for i in $(seq 1 60); do
  pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break
  echo "[driver] waiting for old server to exit..."; sleep 5
done

echo "[a100/$METHOD/$TABLE] booting server..."
setsid nohup bash "$A100/launch_a100.sh" "$METHOD" > "$SLOG" 2>&1 < /dev/null &
PID=$!

ready=0
for i in $(seq 1 600); do   # up to 30 min
  if grep -q "The server is fired up" "$SLOG" 2>/dev/null; then ready=1; break; fi
  kill -0 $PID 2>/dev/null || break
  sleep 3
done
if [ $ready -ne 1 ]; then
  echo "[a100/$METHOD/$TABLE] BOOT FAILED; last 40 lines:"; tail -40 "$SLOG"
  kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null
  exit 1
fi
echo "[a100/$METHOD/$TABLE] server ready (pid=$PID), benching..."
sleep 5

for r in $(seq 1 $NRUNS); do
  L="${METHOD}_r${r}"
  echo "[a100/$METHOD/$TABLE] run $r/$NRUNS ..."
  if [ "$TABLE" = "0914" ]; then
    python3 "$MB/bench_0914.py" "$DS" 256 "$L" 2>&1 | tee "$A100/logs/bench_0914_${L}.log"
  else
    python3 "$MB/freq6_bench_json.py" 32 1800 "$L" 2>&1 | tee "$A100/logs/bench_freq6_${L}.log"
  fi
  sleep 3
done

echo "[a100/$METHOD/$TABLE] shutting down..."
kill -TERM -$PID 2>/dev/null || true
sleep 8
kill -KILL -$PID 2>/dev/null || true
echo "[a100/$METHOD/$TABLE] DONE"
