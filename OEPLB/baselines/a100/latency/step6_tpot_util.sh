#!/bin/bash
set -uo pipefail
LAT=/workspace/EPLB/OEPLB/baselines/a100/latency
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B
echo "[step6] waiting STEP5_DONE..."
while ! grep -q STEP5_DONE /workspace/logs/step5.log 2>/dev/null; do sleep 60; done
sleep 20

python3 - <<'PY'
import json
src=[json.loads(l) for l in open("/workspace/EPLB/OEPLB/baselines/a100/latency/lat_prover256.jsonl")]
with open("/workspace/EPLB/OEPLB/baselines/a100/latency/lat_prover256_o64.jsonl","w") as f:
    for r in src:
        r["max_tokens"]=64; r["ignore_eos"]=True
        f.write(json.dumps(r)+"\n")
print("o64 dataset ready:", len(src))
PY

run_arm6() {
  local ARM=$1 LS=$2
  local SLOG=/workspace/logs/server_a100_lat6_${ARM}.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  setsid nohup bash $LS > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0 dead=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
    sleep 3
  done
  [ $ready -ne 1 ] && { echo "BOOT_FAIL lat6/$ARM"; tail -25 "$SLOG"; kill -9 $PID 2>/dev/null; return 1; }
  sleep 5
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 2 > $LAT/util_${ARM}_sd64.log &
  local SM1=$!
  for r in 1 2; do
    python3 /workspace/EPLB/OEPLB/scripts/run_grid_bench.py lat6_${ARM}_sd64_r${r} $LAT/lat_prover256_o64.jsonl 256 2>&1 | tail -1
  done
  kill $SM1 2>/dev/null
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 5 > $LAT/util_${ARM}_f6.log &
  local SM2=$!
  python3 /workspace/EPLB/OEPLB/scripts/run_grid_bench.py lat6_${ARM}_f6_r1 $LAT/lat_freq6.jsonl 32 2>&1 | tail -1
  kill $SM2 2>/dev/null
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[lat6/$ARM] DONE"
}

run_arm6 identity $LAT/launch_wrap_identity.sh
run_arm6 datafore $LAT/launch_wrap_datafore.sh
run_arm6 moetuner $LAT/launch_wrap_moetuner.sh
run_arm6 eplb_dyn $LAT/launch_wrap_eplb_dyn.sh
run_arm6 oeplb    $LAT/launch_wrap_oeplb.sh
echo STEP6_DONE
