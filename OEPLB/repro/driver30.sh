#!/bin/bash
# Does the dead-zone threshold turn 30B from net-negative to positive?
# driver28 measured 30B beta=+0.20 (POSITIVE, not <0 as the paper claimed), and
# r_k=1.051. The paper's negative 30B gains (-2.6~-3.9%) were on the OLD default
# threshold 1.02 (inside the dead zone -> endless swaps -> overhead > headroom).
# Arms: (A) default 1.02, (B) dead-zone 1.051, (C) 1.051 + swap budget.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-30B-A3B-FP8
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup env $3 bash $1 > $LOGD/server30b_$2.log 2>&1 &
  for i in $(seq 1 300); do grep -q "ready to roll" $LOGD/server30b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server30b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server30b_$2.log || { echo "[d30] $2 FAILED"; return 1; }
  echo "[d30] $2 ready"
}
run () {
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d30_$1 $DS 256 > /dev/null
  L=$LOGD/server30b_$1.log
  echo "[d30] $1 done | decisions=$(grep -c 'swap(s) done' $L)"
}
for RD in 1 2; do
  boot "$LOGD/launch30b_base_g4.sh" bl_r$RD "" && run bl_r$RD
  boot "$LOGD/launch30b_oeplb_g4.sh" oe_dflt_r$RD "" && run oe_dflt_r$RD
  boot "$LOGD/launch30b_oeplb_g4.sh" oe_dz_r$RD "OEPLB_DEAD_ZONE_RATIO=1.051" && run oe_dz_r$RD
  boot "$LOGD/launch30b_oeplb_g4.sh" oe_dzb_r$RD "OEPLB_DEAD_ZONE_RATIO=1.051 OEPLB_SWAP_BUDGET_FRAC=0.02" && run oe_dzb_r$RD
done
pkill -9 -f "$PAT"; echo D30_DONE
