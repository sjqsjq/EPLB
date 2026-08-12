#!/bin/bash
# One profile per config at its BALANCED placement (that is T_flat), to test
#   H1: beta = B/T_flat == routed-expert GEMM share of wall time
#   H2: B*(r_k-1) == overlappable part of dispatch/combine
# If both hold, beta and r_k become one-profile quantities instead of 14-run sweeps.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
PDIR=/workspace/logs/prof; mkdir -p $PDIR
DS256=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
DS512=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 300); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d23] $2 FAILED TO BOOT"; return 1; }
  echo "[d23] $2 ready"
}
prof () {  # $1=tag $2=dataset
  mkdir -p $PDIR/$1
  curl -s -X POST http://127.0.0.1:30000/start_profile \
       -H 'Content-Type: application/json' \
       -d "{\"output_dir\":\"$PDIR/$1\",\"num_steps\":8,\"activities\":[\"GPU\"]}" ; echo
  cd /workspace/EPLB/OEPLB/scripts && timeout 300 python3 run_grid_bench.py _d23_$1 $2 64 > /dev/null 2>&1
  sleep 25
  echo "[d23] $1 profiled: $(ls $PDIR/$1 | wc -l) files"
}
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
boot "$LOGD/launch57b_fixed_g8.sh $LOGD/plc57b_bal.json"    p_g8  && prof g8  $DS256
boot "$LOGD/launch57b_fixed_g4.sh $LOGD/plc57b_g4_bal.json" p_g4  && prof g4  $DS256
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
boot "$LOGD/launch235b_fixed.sh $LOGD/plc235b_bal.json"     p_235 && prof 235 $DS512
pkill -9 -f "$PAT"
for t in g8 g4 235; do python3 $LOGD/parse_trace.py $PDIR/$t/*.gz $PDIR/$t/*.json 2>/dev/null | head -12; done
echo D23_DONE
