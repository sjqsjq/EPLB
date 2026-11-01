#!/bin/bash
set -uo pipefail
LAT=/workspace/EPLB/OEPLB/baselines/a100/latency
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B
while ! grep -q STEP6_DONE /workspace/logs/step6.log 2>/dev/null; do sleep 60; done
sleep 20
SLOG=/workspace/logs/server_a100_lat_oeplb.log; : > "$SLOG"
for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
setsid nohup bash $LAT/launch_wrap_oeplb.sh > "$SLOG" 2>&1 < /dev/null &
PID=$!; ready=0; dead=0
for i in $(seq 1 600); do
  grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
  if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
  sleep 3
done
[ $ready -ne 1 ] && { echo BOOT_FAIL; tail -25 "$SLOG"; exit 1; }
sleep 5
for r in 1 2 3; do python3 /workspace/EPLB/OEPLB/scripts/run_grid_bench.py lat_oeplb_sd_r${r} $LAT/lat_prover256.jsonl 256 2>&1 | tail -1; done
for r in 1 2; do python3 /workspace/EPLB/OEPLB/scripts/run_grid_bench.py lat_oeplb_f6_r${r} $LAT/lat_freq6.jsonl 32 2>&1 | tail -1; done
kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
echo STEP5B_DONE
