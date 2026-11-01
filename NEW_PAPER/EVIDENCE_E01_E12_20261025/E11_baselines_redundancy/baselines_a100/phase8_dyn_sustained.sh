#!/bin/bash
# EPLB动态持续负载(让重排真正触发) + MoETuner持续负载补全
set -uo pipefail
RES=/workspace/EPLB/OEPLB/benchmarks/results
A100=/workspace/EPLB/OEPLB/baselines/a100
MB=/workspace/EPLB/OEPLB/baselines/moetuner
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
N=2048

run_arm() {
  local PREF=$1 NR=$2 SLOG=$3; shift 3
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
    sleep 3
  done
  echo "--- $PREF rebalance 统计 ---"
  grep -E "EPLBManager" "$SLOG" | grep TP0 | grep -c "rebalance start" || true
  grep -E "EPLBManager.*rebalance end" "$SLOG" | grep TP0 | tail -3
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[$PREF] DONE"; sleep 15
}

echo "===== EPLB动态 sustained N=2048 x5 ====="
run_arm dyseq 5 /workspace/logs/server_a100_sust_dyseq.log $A100/launch_a100.sh eplb_dyn
echo "===== MoETuner sustained N=2048 x3 ====="
run_arm mseq 3 /workspace/logs/server_a100_sust_mseq.log $A100/launch_a100.sh moetuner
echo PHASE8_DONE
