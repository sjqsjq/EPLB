#!/bin/bash
set -uo pipefail
LAT=/workspace/EPLB/OEPLB/baselines/a100/latency
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B

run_arm_lat() {
  local ARM=$1 LS=$2 F6RUNS=$3
  local SLOG=/workspace/logs/server_a100_lat_${ARM}.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  setsid nohup bash $LS > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0 dead=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
    sleep 3
  done
  [ $ready -ne 1 ] && { echo "BOOT_FAIL lat/$ARM"; tail -25 "$SLOG"; kill -9 $PID 2>/dev/null; return 1; }
  sleep 5
  for r in 1 2 3; do
    python3 /workspace/EPLB/OEPLB/scripts/run_grid_bench.py lat_${ARM}_sd_r${r} $LAT/lat_prover256.jsonl 256 2>&1 | tail -1
  done
  for r in $(seq 1 $F6RUNS); do
    python3 /workspace/EPLB/OEPLB/scripts/run_grid_bench.py lat_${ARM}_f6_r${r} $LAT/lat_freq6.jsonl 32 2>&1 | tail -1
  done
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[lat/$ARM] DONE"
}

run_arm_lat identity $LAT/launch_wrap_identity.sh 1
run_arm_lat datafore $LAT/launch_wrap_datafore.sh 1
run_arm_lat moetuner $LAT/launch_wrap_moetuner.sh 1
run_arm_lat eplb_dyn $LAT/launch_wrap_eplb_dyn.sh 2
run_arm_lat oeplb    $LAT/launch_wrap_oeplb.sh 2
echo STEP5_DONE
