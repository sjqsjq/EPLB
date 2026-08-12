#!/bin/bash
# 4th (beta, r_k) point: same model, EP=2.  Pre-registered prediction in NOTES.md
# (r_k ~= 1.00-1.01, beta in [0.34,0.45], ceiling 0.4-0.8%, OEPLB gain ~= 0).
# identity r_avg(ep=2)=1.0210 already sits at/below the EP=4 and EP=8 knees, so
# this also tests whether the model correctly says "no headroom here".
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 300); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d24] $2 FAILED TO BOOT"; return 1; }
  echo "[d24] $2 ready"
}
for RD in 1 2; do
  for P in bal r105 r110 r120 r140 conc; do
    F=$LOGD/plc57b_g2_$P.json; [ -f $F ] || continue
    boot "$LOGD/launch57b_fixed_g2.sh $F" g2_${P}_r$RD || continue
    cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d24_${P}_r$RD $DS 256 > /dev/null
    echo "[d24] round $RD $P done"
  done
  boot "$LOGD/launch57b_base_g2.sh" g2_id_r$RD \
    && (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d24_id_r$RD $DS 256 > /dev/null) \
    && echo "[d24] round $RD identity done"
  boot "$LOGD/launch57b_oeplb_g2.sh" g2_oe_r$RD \
    && (cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d24_oe_r$RD $DS 256 > /dev/null) \
    && echo "[d24] round $RD oeplb done" \
    && grep -o "avg_ratio_before=[0-9.]*" $LOGD/server57b_g2_oe_r$RD.log|head -1
done
pkill -9 -f "$PAT"; echo D24_DONE
