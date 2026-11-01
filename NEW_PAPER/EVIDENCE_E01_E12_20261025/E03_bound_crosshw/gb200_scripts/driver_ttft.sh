#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
T1=/workspace/data/t1_prover256_grid.jsonl; T2=/workspace/data/t2_freq6_grid.jsonl
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
declare -A L=([identity]=$LOGD/launch_identity.sh [oeplb]=$LOGD/launch_oeplb.sh)
ts(){ date '+%H:%M:%S'; }
for tag in identity oeplb; do
  echo "[$(ts)] $tag boot"
  bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
  rm -f $LOGD/srv_$tag.log; setsid bash ${L[$tag]} > $LOGD/srv_$tag.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_$tag.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_$tag.log && break; sleep 8; done
  grep -q "ready to roll" $LOGD/srv_$tag.log || { echo "[$(ts)] $tag BOOTFAIL"; continue; }
  sleep 5; cd /workspace/EPLB/OEPLB/scripts
  python3 $RG ttft_${tag}_t1w $T1 256 >/dev/null 2>&1   # warmup(收敛/预热)
  python3 $RG ttft_${tag}_t1 $T1 256 > $LOGD/ttft_${tag}_t1.log 2>&1
  echo "[$(ts)] $tag t1 done"
  python3 $RG ttft_${tag}_t2 $T2 32 > $LOGD/ttft_${tag}_t2.log 2>&1
  echo "[$(ts)] $tag t2 done"
done
bash $LOGD/killall.sh >/dev/null 2>&1
echo "[$(ts)] TTFT_DONE"
