#!/bin/bash
# 验证: 持续负载下 A(thr1.02) 逐步打磨尾部趋近 C; B(thr1.093) 死区内冻结不追
set -uo pipefail
RES=/workspace/EPLB/OEPLB/benchmarks/results
A100=/workspace/EPLB/OEPLB/baselines/a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
N=2048

run_arm() {  # $1..$k-3: launch cmd; then label, nruns, slog
  local NR=$2 PREF=$1 SLOG=$3; shift 3
  : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  setsid nohup bash "$@" > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    kill -0 $PID 2>/dev/null || break
    sleep 3
  done
  [ $ready -ne 1 ] && { echo "BOOT_FAIL $PREF"; tail -25 "$SLOG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1; }
  sleep 5
  for r in $(seq 1 $NR); do
    python3 $MB/bench_0914.py "$DS" $N "${PREF}_r${r}" 2>&1 | tee $A100/logs/bench_sust_${PREF}_r${r}.log
    cp $RES/_0914_${PREF}_r${r}.json $RES/_0914_a100_sust_${PREF}_r${r}.json 2>/dev/null
    grep "PB-OEPLB-DIAG" "$SLOG" | grep "TP0" | tail -1 | cut -c1-200
    sleep 3
  done
  echo "--- $PREF DIAG trajectory (TP0) ---"
  grep "PB-OEPLB-DIAG" "$SLOG" | grep "TP0" | awk '{for(i=1;i<=NF;i++) if($i~/total_ops|avg_ratio|max_ratio/) printf "%s ", $i; print ""}'
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[$PREF] DONE"; sleep 15
}

echo "===== ARM A: OEPLB default(thr=1.02) N=2048 x6 ====="
run_arm aseq 6 /workspace/logs/server_a100_sust_aseq.log $A100/launch_oeplb_a100bl.sh
echo "===== ARM B: OEPLB deadzone(thr=1.093) N=2048 x4 ====="
run_arm bseq 4 /workspace/logs/server_a100_sust_bseq.log $A100/launch_oeplb_deadzone_a100.sh
echo "===== ARM C: OEPLB hybrid(init+guard) N=2048 x3 ====="
run_arm hseq 3 /workspace/logs/server_a100_sust_hseq.log $A100/launch_oeplb_hybrid_a100.sh
echo "===== ARM DF: DataForest(天花板参照) N=2048 x3 ====="
run_arm dseq 3 /workspace/logs/server_a100_sust_dseq.log $A100/launch_a100.sh datafore
echo PHASE6_DONE
