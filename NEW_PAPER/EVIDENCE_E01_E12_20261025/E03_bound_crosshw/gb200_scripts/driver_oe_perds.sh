#!/bin/bash
# 每数据集 fresh boot OEPLB, 消除carryover; window1 before-ratio=该集真实identity不均衡度
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
DSS="mmlu gsm8k humaneval cmmlu csqa arc_easy obqa prover256"
ts(){ date '+%H:%M:%S'; }
for ds in $DSS; do
  echo "[$(ts)] === OEPLB fresh boot for $ds ==="
  bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
  rm -f $LOGD/srv_oe_$ds.log; setsid bash $LOGD/launch_oeplb.sh > $LOGD/srv_oe_$ds.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_oe_$ds.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_oe_$ds.log && break; sleep 8; done
  grep -q "ready to roll" $LOGD/srv_oe_$ds.log || { echo "[$(ts)] $ds BOOTFAIL"; continue; }
  sleep 4; cd /workspace/EPLB/OEPLB/scripts
  echo "[$(ts)] $ds ready; warmup×4"
  for w in 1 2 3 4; do python3 $RG oefd_${ds}_w$w /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1; done
  for r in 1 2 3; do python3 $RG oefd_${ds}_r$r /workspace/data/sd/$ds.jsonl 256 > $LOGD/oefd_${ds}_r$r.log 2>&1; done
  rb=$(grep "PB-OEPLB-DIAG" $LOGD/srv_oe_$ds.log|grep "DP0 "|head -1|grep -oE "avg_ratio_before=[0-9.]+"|head -1)
  ra=$(grep "PB-OEPLB-DIAG" $LOGD/srv_oe_$ds.log|grep "DP0 "|tail -1|grep -oE "avg_ratio_after=[0-9.]+"|head -1)
  echo "[$(ts)] $ds DONE  真实r_before($rb) r_after($ra)"
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] OE_PERDS_DONE"
