#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
DS=/workspace/data/t1_tail1024_O1_grid.jsonl
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
declare -A L=([identity]=$LOGD/launch_identity.sh [moetuner]=$LOGD/launch_moetuner.sh
 [datafore]=$LOGD/launch_datafore.sh [oeplb_def]=$LOGD/launch_oeplb.sh [oeplb_aggr]=$LOGD/launch_oeplb_aggr.sh)
declare -A W=([identity]=1 [moetuner]=1 [datafore]=1 [oeplb_def]=4 [oeplb_aggr]=4)
ts(){ date '+%H:%M:%S'; }
for tag in identity moetuner datafore oeplb_def oeplb_aggr; do
  echo "[$(ts)] === $tag boot ==="
  bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
  rm -f $LOGD/srv_$tag.log; setsid bash ${L[$tag]} > $LOGD/srv_$tag.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_$tag.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_$tag.log && break; sleep 8; done
  grep -q "ready to roll" $LOGD/srv_$tag.log || { echo "[$(ts)] $tag BOOTFAIL"; continue; }
  echo "[$(ts)] $tag ready"; sleep 4; cd /workspace/EPLB/OEPLB/scripts
  for w in $(seq ${W[$tag]}); do python3 $RG sd_${tag}_w$w $DS 256 >/dev/null 2>&1; done
  echo "[$(ts)] $tag warmup(${W[$tag]}) done"
  for r in 1 2 3; do python3 $RG sd_${tag}_r$r $DS 256 > $LOGD/sd_${tag}_r$r.log 2>&1; done
  echo "[$(ts)] $tag timed done"
  case $tag in oeplb_*) echo "  DIAG尾:"; grep "PB-OEPLB-DIAG" $LOGD/srv_$tag.log|grep "DP0 "|tail -2|grep -oE "avg_ratio_before=[0-9.]+ avg_ratio_after=[0-9.]+";; esac
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] SAMEDOMAIN_DONE"
