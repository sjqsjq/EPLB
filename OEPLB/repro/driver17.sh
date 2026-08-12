#!/bin/bash
# The three appendix-F scenarios were reported with r_before=1.113, a value
# measured on L256.  Record each dataset's own routing counts (one boot, three
# recordings) so their r_before / headroom is measured rather than borrowed.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
D=/data/minghua/sjq/OEPLBdata/datasets

boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d17] $2 FAILED TO BOOT"; return 1; }
  echo "[d17] $2 ready"
}

boot "$LOGD/launch57b_rec_g8.sh" rec17 || { echo D17_DONE; exit 1; }

rec () {   # $1=tag  $2=dataset  $3=concurrency
  python3 $LOGD/dump_counts.py $LOGD/counts57b_$1.json start
  cd /workspace/EPLB/OEPLB/scripts && python3 run_grid_bench.py _d17_$1 $2 $3 > /dev/null
  python3 $LOGD/dump_counts.py $LOGD/counts57b_$1.json stop
  echo "[d17] $1 recorded"
}

rec l512  $D/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl 256
rec multi $D/multi_domain/multidomain_v2_out1.jsonl                            256
rec share $D/sharegpt/sharegpt_natural_20k.jsonl                               256

pkill -9 -f "$PAT"
for T in l512 multi share; do
  [ -f $LOGD/counts57b_$T.json ] && python3 $LOGD/r_avg.py $LOGD/counts57b_$T.json 8 4
done
echo D17_DONE
