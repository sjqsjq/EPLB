#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
DSS="mmlu gsm8k humaneval cmmlu csqa arc_easy obqa prover256"
ts(){ date '+%H:%M:%S'; }
bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
rm -f $LOGD/srv_idredo.log; setsid bash $LOGD/launch_identity.sh > $LOGD/srv_idredo.log 2>&1 </dev/null &
for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_idredo.log && break; sleep 8; done
grep -q "ready to roll" $LOGD/srv_idredo.log || { echo "BOOTFAIL"; exit 1; }
echo "[$(ts)] identity ready"; sleep 4; cd /workspace/EPLB/OEPLB/scripts
for ds in $DSS; do
  python3 $RG idr_${ds}_w /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1
  for r in 1 2 3; do python3 $RG idr_${ds}_r$r /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1; done
  echo "[$(ts)] identity $ds done"
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] ID_REDO_DONE"
