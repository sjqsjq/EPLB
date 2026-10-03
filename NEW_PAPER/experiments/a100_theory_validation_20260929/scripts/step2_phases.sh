#!/bin/bash
set -uo pipefail
T=/workspace/EPLB/OEPLB/theory_a100
DS256=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
DS494=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_long494tok_out1.jsonl

run_tag() {  # $1=tag $2=placement $3=ds $4=prefix $5=chunk(opt)
  local TAG=$1 PLC=$2 DS=$3 PFX=$4 CHUNK=${5:-}
  local SLOG=/workspace/logs/server_a100_${PFX}${TAG}.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  export SCAN_PLC="$PLC" SCAN_CHUNK="$CHUNK"
  setsid nohup bash $T/launch_scan2_a100.sh > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0 dead=0
  for i in $(seq 1 600); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
    sleep 3
  done
  if [ $ready -ne 1 ]; then echo "BOOT_FAIL $PFX$TAG"; tail -25 "$SLOG"; kill -TERM -$PID 2>/dev/null; sleep 5; kill -KILL -$PID 2>/dev/null; return 1; fi
  sleep 5
  for r in 1 2 3 4 5; do
    SAVE=1; [ $r -eq 1 ] && SAVE=0
    python3 $T/bench_scan.py "$DS" 256 "$TAG" $r $SAVE "$PFX" 2>&1 | tee $T/logs/${PFX}${TAG}_r${r}.log
    sleep 3
  done
  kill -TERM -$PID 2>/dev/null; sleep 8; kill -KILL -$PID 2>/dev/null
  echo "[$PFX$TAG] DONE"
}

echo "===== B1: 494tok 协议鲁棒性(6臂) ====="
run_tag bal  $T/placements/plcA100_bal.json  $DS494 _scanL494_
run_tag r110 $T/placements/plcA100_r110.json $DS494 _scanL494_
run_tag r122 $T/placements/plcA100_r122.json $DS494 _scanL494_
run_tag r130 $T/placements/plcA100_r130.json $DS494 _scanL494_
run_tag r155 $T/placements/plcA100_r155.json $DS494 _scanL494_
run_tag id   ""                              $DS494 _scanL494_
cd /workspace/EPLB/OEPLB
python3 repro/fit_f3.py 1.716 1.07 id,r130 plcA100.txt _scanL494_ | tee $T/logs/fit_f3_L494.txt

echo "===== B2: chunk4096 机制判别(7臂) ====="
run_tag bal  $T/placements/plcA100_bal.json  $DS256 _scanC4K_ 4096
run_tag r106 $T/placements/plcA100_r106.json $DS256 _scanC4K_ 4096
run_tag r110 $T/placements/plcA100_r110.json $DS256 _scanC4K_ 4096
run_tag r115 $T/placements/plcA100_r115.json $DS256 _scanC4K_ 4096
run_tag r122 $T/placements/plcA100_r122.json $DS256 _scanC4K_ 4096
run_tag r140 $T/placements/plcA100_r140.json $DS256 _scanC4K_ 4096
run_tag id   ""                              $DS256 _scanC4K_ 4096
python3 repro/fit_f3.py 1.716 1.07 id plcA100.txt _scanC4K_ | tee $T/logs/fit_f3_C4K.txt
echo STEP2_DONE
