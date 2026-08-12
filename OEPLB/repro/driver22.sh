#!/bin/bash
# Appendix F's 4-GPU numbers were n=1 per arm; L512_O1's +4.7% re-measured to
# +2.39% (driver20).  The other two exceed their own bounds (115%, 140%), so
# re-measure them the same way: 2 rounds per arm, alternating, fresh boot each.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
D=/data/minghua/sjq/OEPLBdata/datasets
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d22] $2 FAILED TO BOOT"; return 1; }
  echo "[d22] $2 ready"
}
run () {  # $1=tag $2=dataset
  for RD in 1 2; do for A in base oeplb; do
    boot "$LOGD/launch57b_${A}_g4.sh" $1_${A}_r$RD || continue
    cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d22_$1_${A}_r$RD $2 256 > /dev/null
    echo "[d22] $1 round $RD $A done"
    [ $A = oeplb ] && grep -o "avg_ratio_before=[0-9.]*" $LOGD/server57b_$1_oeplb_r$RD.log|head -1
  done; done
}
run multi $D/multi_domain/multidomain_v2_out1.jsonl
run share $D/sharegpt/sharegpt_natural_20k.jsonl
pkill -9 -f "$PAT"; echo D22_DONE
