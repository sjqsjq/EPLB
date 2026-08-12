#!/bin/bash
# The 4-GPU rows paired r_before (L256) with a measured gain (ShareGPT 20K).
# driver13's `id` arm already gives the 4-GPU L256 baseline (139.06 s, n=2), so
# one OEPLB-on arm on the SAME dataset makes the row self-consistent and lets
# the +2.29% bound be compared against a gain measured on the same workload.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl

boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d18] $2 FAILED TO BOOT"; return 1; }
  echo "[d18] $2 ready"
}

for RD in 1 2; do
  boot "$LOGD/launch57b_oeplb_g4.sh" g4_oe_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d18_oe_r$RD $DS 256 > /dev/null
  echo "[d18] round $RD oeplb done"
  # runtime r_before/r_after on this exact dataset+config, for the model's inputs
  grep -o "avg_ratio_before=[0-9.]*" $LOGD/server57b_g4_oe_r$RD.log | tail -3
  grep -o "avg_ratio_after=[0-9.]*"  $LOGD/server57b_g4_oe_r$RD.log | tail -3
  grep -c "swap" $LOGD/server57b_g4_oe_r$RD.log
done
pkill -9 -f "$PAT"
echo D18_DONE
