#!/bin/bash
# f_MoE identification: sweep a forced imbalance ratio r with the balancer OFF,
# fit T(r) = A + B*r, and derive f = B*r_b/(A+B*r_b) per configuration.
LOGD=/workspace/logs
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct

boot () {  # $1=launch cmdline  $2=tag
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d12] $2 FAILED TO BOOT"; return 1; }
  echo "[d12] $2 ready"
}

# ---- phase 1: record the routing distribution under the real load ----
if [ ! -f $LOGD/counts57b.json ]; then
  boot "$LOGD/launch57b_rec_g8.sh" rec || exit 1
  python3 $LOGD/dump_counts.py /dev/null start
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d12_rec $DS 256 > /dev/null
  echo "[d12] record load done"
  python3 $LOGD/dump_counts.py $LOGD/counts57b.json stop
fi
[ -f $LOGD/counts57b.json ] || { echo "[d12] no counts, abort"; exit 1; }

# ---- phase 2: build placements at target imbalance ratios ----
python3 $LOGD/gen_placement.py $LOGD/counts57b.json 8 $LOGD/plc57b 1.10 1.22 1.35 1.50 | tee $LOGD/plc57b.txt

# ---- phase 3: sweep, 2 interleaved rounds ----
for RD in 1 2; do
for P in bal r110 r122 r135 r150 conc; do
  F=$LOGD/plc57b_$P.json
  [ -f $F ] || { echo "[d12] skip $P (no json)"; continue; }
  boot "$LOGD/launch57b_fixed_g8.sh $F" sw_${P}_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d12_${P}_r$RD $DS 256 > /dev/null
  echo "[d12] round $RD $P done"
done
  boot "$LOGD/launch57b_base_g8.sh" sw_id_r$RD || continue
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d12_id_r$RD $DS 256 > /dev/null
  echo "[d12] round $RD identity done"
done
pkill -9 -f "$PAT"
echo D12_DONE
