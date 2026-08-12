#!/bin/bash
# Numerical-equivalence / task-accuracy across a real expert migration.
# Design: probe the SAME server instance before and after a load that triggers
# PB-OEPLB swaps.  A baseline arm (no swaps possible) supplies the null: any
# pre/post drift it shows is measurement noise, not an effect of moving weights.
LOGD=/workspace/logs
M=/data/models/Qwen2-57B-A14B-Instruct
DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L256_O1_realprover_n16384.jsonl
SFX=server; PAT="sglang.launch_$SFX"

boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d15] $2 FAILED TO BOOT"; return 1; }
  echo "[d15] $2 ready"
}

arm () {  # $1=launch script  $2=arm name  $3=trial
  T=$2_t$3
  boot "$1" eq_$T || return 1
  python3 $LOGD/gsm8k_probe.py  $LOGD/gsm_${T}_pre.json  $M 200
  python3 $LOGD/corpus_probe.py $LOGD/cp_${T}_pre.json   $M 200
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d15_${T} $DS 256 > /dev/null
  SW=$(grep -c 'swap(s) done\|issued .* swap' $LOGD/server57b_eq_$T.log)
  ERR=$(grep -c 'P2P failed\|P2P swap failed' $LOGD/server57b_eq_$T.log)
  echo "[d15] $T load done: swap-log-lines=$SW p2p-errors=$ERR"
  python3 $LOGD/gsm8k_probe.py  $LOGD/gsm_${T}_post.json $M 200
  python3 $LOGD/corpus_probe.py $LOGD/cp_${T}_post.json  $M 200
  echo "[d15] $T done"
}

for t in 1 2; do
  arm "$LOGD/launch57b_oeplb_g8.sh" oeplb $t
  arm "$LOGD/launch57b_base_g8.sh"  base  $t
done
pkill -9 -f "$PAT"
echo D15_DONE
