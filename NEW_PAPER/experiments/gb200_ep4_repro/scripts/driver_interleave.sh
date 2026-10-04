#!/bin/bash
# 每数据集: identity与OEPLB背靠背同session交错测, 消除session间系统漂移
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
DSS="prover256 humaneval gsm8k mmlu cmmlu obqa arc_easy csqa"
ts(){ date '+%H:%M:%S'; }
boot(){ bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3; rm -f $LOGD/srv_$2.log
  setsid bash $1 > $LOGD/srv_$2.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_$2.log && return 0; grep -qi "Scheduler hit an exception" $LOGD/srv_$2.log && return 1; sleep 8; done; return 1; }
for ds in $DSS; do
  echo "[$(ts)] ===== $ds ====="
  boot $LOGD/launch_identity.sh il_id_$ds && { cd /workspace/EPLB/OEPLB/scripts
    python3 $RG il_id_${ds}_w /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1
    for r in 1 2 3; do python3 $RG il_id_${ds}_r$r /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1; done
    echo "[$(ts)] $ds identity done"; } || echo "[$(ts)] $ds id BOOTFAIL"
  boot $LOGD/launch_oeplb.sh il_oe_$ds && { cd /workspace/EPLB/OEPLB/scripts
    for w in 1 2 3; do python3 $RG il_oe_${ds}_w$w /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1; done
    for r in 1 2 3; do python3 $RG il_oe_${ds}_r$r /workspace/data/sd/$ds.jsonl 256 >/dev/null 2>&1; done
    rb=$(grep "PB-OEPLB-DIAG" $LOGD/srv_il_oe_$ds.log|grep "DP0 "|head -1|grep -oE "avg_ratio_before=[0-9.]+"|head -1)
    echo "[$(ts)] $ds oeplb done  r_before($rb)"; } || echo "[$(ts)] $ds oe BOOTFAIL"
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] INTERLEAVE_DONE"
