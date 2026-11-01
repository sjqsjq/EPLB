#!/bin/bash
set -uo pipefail
T=/workspace/EPLB/OEPLB/theory_a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl

run_eta2() {  # $1=label $2=thr $3=ops $4=W
  local L=$1 SLOG=/workspace/logs/server_a100_eta2_$1.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  export ETA_THR=$2 ETA_OPS=$3 ETA_W=$4
  setsid nohup bash $T/launch_eta2_a100.sh > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0 dead=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
    sleep 3
  done
  [ $ready -ne 1 ] && { echo "BOOT_FAIL $L"; tail -25 "$SLOG"; kill -9 $PID 2>/dev/null; return 1; }
  sleep 5
  for r in 1 2 3; do
    python3 $MB/bench_0914.py "$DS" 2048 "eta2${L}_r${r}" 2>&1 | tee $T/logs/eta2_${L}_r${r}.log
    sleep 3
  done
  echo "[$L] swap窗口数(TP0 DIAG): $(grep 'PB-OEPLB-DIAG' $SLOG | grep -c TP0)"
  grep "PB-OEPLB-DIAG" "$SLOG" | grep TP0 | tail -2
  grep -E "swap\(s\) done" "$SLOG" | grep TP0 | awk -F'done \\\\(' '{print $2}' | cut -d')' -f1 | sort | uniq -c | tail -5
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[eta2/$L] DONE"
}

echo "===== η-sustained 臂1: churn-lite (thr1.02, ops300, W=4) ====="
run_eta2 churn 1.02 300 4
echo "===== η-sustained 臂2: gated (thr1.20, ops300, W=4) ====="
run_eta2 gated 1.20 300 4
echo STEP3B_DONE
