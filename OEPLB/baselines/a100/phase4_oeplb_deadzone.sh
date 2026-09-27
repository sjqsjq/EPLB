#!/bin/bash
set -uo pipefail
RES=/workspace/EPLB/OEPLB/benchmarks/results
A100=/workspace/EPLB/OEPLB/baselines/a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
SLOG=/workspace/logs/server_a100_0914_oeplb_dz.log; : > "$SLOG"
for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
setsid nohup bash $A100/launch_oeplb_deadzone_a100.sh > "$SLOG" 2>&1 < /dev/null &
PID=$!
ready=0
for i in $(seq 1 600); do
  grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
  kill -0 $PID 2>/dev/null || break
  sleep 3
done
[ $ready -ne 1 ] && { echo BOOT_FAIL; tail -30 "$SLOG"; exit 1; }
sleep 5
for r in 1 2 3 4 5; do
  python3 $MB/bench_0914.py "$DS" 256 "oeplbdz_r${r}" 2>&1 | tee $A100/logs/bench_0914_oeplbdz_r${r}.log
  cp $RES/_0914_oeplbdz_r${r}.json $RES/_0914_a100_oeplbdz_r${r}.json
  sleep 3
done
echo "--- steady-state ops check (should drop toward 0) ---"
grep -E "PB-OEPLB-DIAG" "$SLOG" | grep "TP0" | tail -6
grep -cE "PB-OEPLB-SWAP|batch_isend" "$SLOG" | xargs echo "swap-ish lines:"
kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
echo PHASE4_DONE
