#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
DS=/workspace/data/t1_tail1024_O1_grid.jsonl
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
declare -A L=([oeplb_bc]=$LOGD/launch_oeplb_bc.sh [oeplb_wide]=$LOGD/launch_oeplb_wide.sh)
ts(){ date '+%H:%M:%S'; }
for tag in oeplb_bc oeplb_wide; do
  echo "[$(ts)] === $tag boot ==="
  bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
  rm -f $LOGD/srv_$tag.log; setsid bash ${L[$tag]} > $LOGD/srv_$tag.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_$tag.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_$tag.log && break; sleep 8; done
  grep -q "ready to roll" $LOGD/srv_$tag.log || { echo "[$(ts)] $tag BOOTFAIL"; continue; }
  echo "[$(ts)] $tag ready"; sleep 4; cd /workspace/EPLB/OEPLB/scripts
  for w in 1 2 3 4 5; do python3 $RG sd_${tag}_w$w $DS 256 >/dev/null 2>&1; done
  for r in 1 2 3; do python3 $RG sd_${tag}_r$r $DS 256 > $LOGD/sd_${tag}_r$r.log 2>&1; done
  echo "[$(ts)] $tag done; DIAG ops序列:"; grep "PB-OEPLB-DIAG" $LOGD/srv_$tag.log|grep "DP0 "|tail -6|grep -oE "total_ops=[0-9]+ avg_ratio_before=[0-9.]+ avg_ratio_after=[0-9.]+"
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] OEFIX_DONE"
