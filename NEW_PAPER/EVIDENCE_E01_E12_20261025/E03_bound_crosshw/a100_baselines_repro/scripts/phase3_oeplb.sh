#!/bin/bash
set -uo pipefail
RES=/workspace/EPLB/OEPLB/benchmarks/results
A100=/workspace/EPLB/OEPLB/baselines/a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
LOGD=/workspace/logs
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl

echo "[phase3] waiting for PHASE2_DONE..."
while ! grep -q PHASE2_DONE $LOGD/phase2.log 2>/dev/null; do sleep 60; done
sleep 30

for T in 0914 freq6; do
  SLOG=$LOGD/server_a100_${T}_oeplb.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  echo "[phase3/oeplb/$T] booting..."
  setsid nohup bash $A100/launch_oeplb_a100bl.sh > "$SLOG" 2>&1 < /dev/null &
  PID=$!
  ready=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    kill -0 $PID 2>/dev/null || break
    sleep 3
  done
  if [ $ready -ne 1 ]; then echo "[phase3/oeplb/$T] BOOT FAILED"; tail -40 "$SLOG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; continue; fi
  echo "[phase3/oeplb/$T] ready; 自检:"
  grep -oE "ep_dispatch_algorithm='?[a-zA-Z]+'?" "$SLOG" | head -1
  sleep 5
  for r in 1 2 3; do
    L="oeplb_r${r}"
    if [ "$T" = "0914" ]; then
      python3 $MB/bench_0914.py "$DS" 256 "$L" 2>&1 | tee $A100/logs/bench_0914_${L}.log
    else
      python3 $MB/freq6_bench_json.py 32 1800 "$L" 2>&1 | tee $A100/logs/bench_freq6_${L}.log
    fi
    cp $RES/_${T}_${L}.json $RES/_${T}_a100_oeplb_r${r}.json 2>/dev/null
    sleep 3
  done
  # OEPLB 机制生效证据
  grep -cE "PB-OEPLB" "$SLOG" | xargs echo "[phase3/oeplb/$T] PB-OEPLB log lines:"
  grep -E "PB-OEPLB-(RESET|WINDOW)" "$SLOG" | tail -6
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[phase3/oeplb/$T] DONE"
  sleep 15
done
echo PHASE3_DONE
