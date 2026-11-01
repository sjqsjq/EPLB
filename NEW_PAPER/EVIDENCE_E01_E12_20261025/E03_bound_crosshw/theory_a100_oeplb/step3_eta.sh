#!/bin/bash
set -uo pipefail
T=/workspace/EPLB/OEPLB/theory_a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl

run_eta() {  # $1=label $2=thr $3=ops
  local L=$1 SLOG=/workspace/logs/server_a100_eta_$1.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  export ETA_THR=$2 ETA_OPS=$3
  setsid nohup bash $T/launch_eta_a100.sh > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0 dead=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
    sleep 3
  done
  [ $ready -ne 1 ] && { echo "BOOT_FAIL $L"; tail -25 "$SLOG"; kill -9 $PID 2>/dev/null; return 1; }
  sleep 5
  for r in 1 2 3 4 5; do
    python3 $MB/bench_0914.py "$DS" 256 "eta${L}_r${r}" 2>&1 | tee $T/logs/eta_${L}_r${r}.log
    sleep 3
  done
  echo "[$L] swap 统计: $(grep -c 'swap(s) done' $SLOG) 次窗口, 总ops=$(grep -oE '[0-9]+ swap' $SLOG | awk '{s+=$1}END{print s}')"
  grep "PB-OEPLB-DIAG" "$SLOG" | grep TP0 | tail -3
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[eta/$L] DONE"
}

echo "===== η臂1: thr=1.02 无预算(churn) ====="; run_eta churn 1.02 100000
echo "===== η臂2: thr=1.02 预算256 =====";      run_eta bud256 1.02 256
echo "===== η臂3: thr=1.06 预算300(死区门控) ====="; run_eta dzgated 1.06 300
echo STEP3_DONE
