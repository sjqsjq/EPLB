#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
DSS="mmlu gsm8k humaneval cmmlu csqa arc_easy obqa prover256"
ts(){ date '+%H:%M:%S'; }
run_all(){ # $1=tag $2=launchscript $3=nwarm
  echo "[$(ts)] === $1 boot ==="
  bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
  rm -f $LOGD/srv_$1.log; setsid bash $2 > $LOGD/srv_$1.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_$1.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_$1.log && break; sleep 8; done
  grep -q "ready to roll" $LOGD/srv_$1.log || { echo "[$(ts)] $1 BOOTFAIL"; return 1; }
  echo "[$(ts)] $1 ready"; sleep 4; cd /workspace/EPLB/OEPLB/scripts
  for ds in $DSS; do
    echo "[$(ts)] >>> $1 $ds START"
    for w in $(seq $3); do python3 $RG $1_${ds}_w$w /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1; done
    for r in 1 2; do python3 $RG $1_${ds}_r$r /workspace/data/sd/$ds.jsonl 256 > $LOGD/md_${1}_${ds}_r$r.log 2>&1; done
    echo "[$(ts)] >>> $1 $ds DONE"
  done
}
run_all mid /workspace/logs/launch_identity.sh 1
run_all moe /workspace/logs/launch_oeplb.sh 2
bash $LOGD/killall.sh >/dev/null 2>&1
echo "[$(ts)] MULTIDOMAIN_DONE"
