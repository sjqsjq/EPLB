#!/bin/bash
# Appendix F reports +4.7% on 4-GPU L512_O1, but driver17 measured that dataset's
# own r_before=1.1125, which with r_k=1.032 and f_sens=0.369 caps the gain at
# ~2.75%.  The +4.7% was a single run per arm.  Re-measure with 2 rounds each.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d20] $2 FAILED TO BOOT"; return 1; }
  echo "[d20] $2 ready"
}
for RD in 1 2; do for A in base oeplb; do
  boot "$LOGD/launch57b_${A}_g4.sh" l512_${A}_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d20_${A}_r$RD $DS 256 > /dev/null
  echo "[d20] round $RD $A done"
  [ $A = oeplb ] && grep -o "avg_ratio_before=[0-9.]*" $LOGD/server57b_l512_oeplb_r$RD.log|head -1
done; done
pkill -9 -f "$PAT"; echo D20_DONE
