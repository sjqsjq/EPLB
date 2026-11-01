#!/bin/bash
# 校准 N=2048 持续负载协议的动态范围: identity 参照
set -uo pipefail
while ! grep -q PHASE6_DONE /workspace/logs/phase6.log 2>/dev/null; do sleep 45; done
sleep 20
RES=/workspace/EPLB/OEPLB/benchmarks/results
A100=/workspace/EPLB/OEPLB/baselines/a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
SLOG=/workspace/logs/server_a100_sust_iseq.log; : > "$SLOG"
for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
setsid nohup bash $A100/launch_a100.sh identity > "$SLOG" 2>&1 < /dev/null &
PID=$!
ready=0
for i in $(seq 1 600); do
  grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
  kill -0 $PID 2>/dev/null || break
  sleep 3
done
[ $ready -ne 1 ] && { echo BOOT_FAIL; tail -25 "$SLOG"; exit 1; }
sleep 5
for r in 1 2 3; do
  python3 $MB/bench_0914.py "$DS" 2048 "iseq_r${r}" 2>&1 | tee $A100/logs/bench_sust_iseq_r${r}.log
  cp $RES/_0914_iseq_r${r}.json $RES/_0914_a100_sust_iseq_r${r}.json 2>/dev/null
  sleep 3
done
kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
echo PHASE7_DONE
