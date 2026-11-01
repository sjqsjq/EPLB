#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
A=/workspace/data/best_prover512_O1.jsonl; B=/workspace/data/tpot_mmlu_O64.jsonl
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
ts(){ date '+%H:%M:%S'; }
runu(){ ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits|tr '\n' ' '; echo; sleep 1; done > $4 ) & local sp=$!
  python3 $RG $1 $2 $3 > $LOGD/bc_$1.log 2>&1; kill $sp 2>/dev/null; wait $sp 2>/dev/null; }
boot(){ bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3; rm -f $LOGD/srv_bc_$1.log
  setsid bash $2 > $LOGD/srv_bc_$1.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_bc_$1.log && return 0; grep -qi "Scheduler hit an exception" $LOGD/srv_bc_$1.log && return 1; sleep 8; done; return 1; }
for cfg in A B; do
  [ $cfg = A ] && DS=$A || DS=$B
  for arm in identity oeplb; do
    echo "[$(ts)] === cfg$cfg $arm boot ==="
    boot bc_${cfg}_${arm} $( [ $arm = identity ] && echo $LOGD/launch_identity.sh || echo $LOGD/launch_oeplb.sh ) || { echo "[$(ts)] $cfg $arm BOOTFAIL"; continue; }
    echo "[$(ts)] $cfg $arm ready"; sleep 4; cd /workspace/EPLB/OEPLB/scripts
    nw=1; [ $arm = oeplb ] && nw=3
    for w in $(seq $nw); do python3 $RG bcw_${cfg}_${arm}_$w $DS 256 >/dev/null 2>&1; done
    for r in 1 2; do runu bc_${cfg}_${arm}_r$r $DS 256 $LOGD/utilbc_${cfg}_${arm}_r$r.csv; done
    echo "[$(ts)] cfg$cfg $arm done"
  done
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] BESTCONFIG_DONE"
