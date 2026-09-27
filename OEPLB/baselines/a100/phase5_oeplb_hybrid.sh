#!/bin/bash
set -uo pipefail
RES=/workspace/EPLB/OEPLB/benchmarks/results
A100=/workspace/EPLB/OEPLB/baselines/a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
for T in 0914 freq6; do
  SLOG=/workspace/logs/server_a100_${T}_oeplb_hybrid.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  setsid nohup bash $A100/launch_oeplb_hybrid_a100.sh > "$SLOG" 2>&1 < /dev/null &
  PID=$!
  ready=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    kill -0 $PID 2>/dev/null || break
    sleep 3
  done
  [ $ready -ne 1 ] && { echo "BOOT_FAIL $T"; tail -30 "$SLOG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; continue; }
  grep -oE "init_expert_location from [a-z_]+" "$SLOG" | head -1
  sleep 5
  NR=3; [ "$T" = "0914" ] && NR=5
  for r in $(seq 1 $NR); do
    if [ "$T" = "0914" ]; then
      python3 $MB/bench_0914.py "$DS" 256 "oeplbhyb_r${r}" 2>&1 | tee $A100/logs/bench_0914_oeplbhyb_r${r}.log
    else
      python3 $MB/freq6_bench_json.py 32 1800 "oeplbhyb_r${r}" 2>&1 | tee $A100/logs/bench_freq6_oeplbhyb_r${r}.log
    fi
    cp $RES/_${T}_oeplbhyb_r${r}.json $RES/_${T}_a100_oeplbhyb_r${r}.json
    sleep 3
  done
  grep -E "PB-OEPLB-(RESET|WINDOW)" "$SLOG" | tail -4
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[phase5/$T] DONE"
  sleep 15
done
echo PHASE5_DONE
