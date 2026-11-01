#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
T1=/workspace/data/t1_tail1024_O1_grid.jsonl
XD1=/workspace/data/xd_freq6_O1_grid.jsonl; XD10=/workspace/data/xd_freq6_O10_grid.jsonl
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
declare -A L=([identity]=$LOGD/launch_identity.sh [datafore]=$LOGD/launch_datafore.sh
 [eplb_static]=$LOGD/launch_eplb_static.sh [eplb_dyn]=$LOGD/launch_eplb.sh
 [oeplb]=$LOGD/launch_oeplb.sh [moetuner]=$LOGD/launch_moetuner.sh)
ts(){ date '+%H:%M:%S'; }
for tag in identity datafore eplb_static moetuner eplb_dyn oeplb; do
  echo "[$(ts)] === $tag boot ==="
  bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
  rm -f $LOGD/srv_$tag.log; setsid bash ${L[$tag]} > $LOGD/srv_$tag.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_$tag.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_$tag.log && break; sleep 8; done
  grep -q "ready to roll" $LOGD/srv_$tag.log || { echo "[$(ts)] $tag BOOTFAIL"; continue; }
  echo "[$(ts)] $tag ready"; sleep 5; cd /workspace/EPLB/OEPLB/scripts
  case $tag in oeplb|eplb_dyn) python3 $RG rm_${tag}_t1w $T1 256 >/dev/null 2>&1;; esac
  python3 $RG rm_${tag}_t1 $T1 256 > $LOGD/rm_${tag}_t1.log 2>&1; echo "[$(ts)] $tag t1(O=1) done"
  python3 $RG rm_${tag}_xd1 $XD1 32 > $LOGD/rm_${tag}_xd1.log 2>&1; echo "[$(ts)] $tag xd_O1 done"
  python3 $RG rm_${tag}_xd10 $XD10 32 > $LOGD/rm_${tag}_xd10.log 2>&1; echo "[$(ts)] $tag xd_O10 done"
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] REMEASURE_DONE"
