#!/bin/bash
set -uo pipefail
T=/workspace/EPLB/OEPLB/theory_a100
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
echo "[step1] waiting STEP0_DONE..."
while ! grep -q STEP0_DONE /workspace/logs/step0.log 2>/dev/null; do sleep 30; done
sleep 20

echo "[step1] generating placements from A100-native counts..."
cd /workspace/EPLB/OEPLB
python3 repro/gen_placement.py $T/data/counts235b_a100.json 8 $T/placements/plcA100 \
  1.02 1.06 1.10 1.15 1.22 1.30 1.40 1.55 | tee /workspace/logs/plcA100.txt

run_tag() {
  local TAG=$1 PLC=${2:-}
  local SLOG=/workspace/logs/server_a100_scan_${TAG}.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  export SCAN_PLC="$PLC"
  setsid nohup bash $T/launch_scan_a100.sh > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0 dead=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
    sleep 3
  done
  if [ $ready -ne 1 ]; then echo "BOOT_FAIL $TAG"; tail -25 "$SLOG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1; fi
  sleep 5
  for r in 1 2 3 4 5; do
    SAVE=1; [ $r -eq 1 ] && SAVE=0    # 丢 r1(冷启动), r2-r5 落盘
    python3 $T/bench_scan.py "$DS" 256 "$TAG" $r $SAVE 2>&1 | tee $T/logs/scan_${TAG}_r${r}.log
    sleep 3
  done
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[scan/$TAG] DONE"
}

run_tag id
for t in bal r102 r106 r110 r115 r122 r130 r140 r155 conc; do
  run_tag $t $T/placements/plcA100_${t}.json
done

echo "[step1] fitting hinge..."
R_BEFORE=$(python3 -c "
import json, numpy as np
C=np.array(json.load(open('$T/data/counts235b_a100.json'))['counts'],dtype=float)
print('%.4f'%np.mean([(C[l].reshape(8,16).sum(1).max())/(C[l].sum()/8) for l in range(94)]))")
echo "R_BEFORE(A100 identity) = $R_BEFORE"
cd /workspace/EPLB/OEPLB
python3 repro/fit_f3.py $R_BEFORE 1.07 id,r130 plcA100.txt _scanA100_ | tee $T/logs/fit_f3_a100.txt
echo STEP1_DONE
