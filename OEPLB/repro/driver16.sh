#!/bin/bash
# Does the dead zone r_k move with load?  Three placements (below-knee, at the
# natural r, and the concentrated endpoint) at two concurrencies.  The
# discriminating statistic is the low-r fraction
#     q = (T(r=1.148) - T(r=1.010)) / (T(r=1.550) - T(r=1.010))
# which equals (1.148-1.010)/(1.550-1.010) = 0.256 if T(r) is a straight line,
# and is much smaller if a dead zone exists.
LOGD=/workspace/logs
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
SFX=server; PAT="sglang.launch_$SFX"

boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d16] $2 FAILED TO BOOT"; return 1; }
  echo "[d16] $2 ready"
}

for RD in 1 2; do
for P in bal r122 conc; do
  boot "$LOGD/launch57b_fixed_g8.sh $LOGD/plc57b_$P.json" ld_${P}_r$RD || continue
  for C in 64 512; do
    cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d16_${P}_c${C}_r$RD $DS $C > /dev/null
    echo "[d16] round $RD $P conc=$C done"
  done
done
done
pkill -9 -f "$PAT"
echo D16_DONE
