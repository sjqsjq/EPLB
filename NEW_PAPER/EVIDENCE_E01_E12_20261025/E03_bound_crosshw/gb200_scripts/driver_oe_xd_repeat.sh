#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
XD1=/workspace/data/xd_freq6_O1_grid.jsonl; XD10=/workspace/data/xd_freq6_O10_grid.jsonl
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
ts(){ date '+%H:%M:%S'; }
bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
rm -f $LOGD/srv_oeplb.log; setsid bash $LOGD/launch_oeplb.sh > $LOGD/srv_oeplb.log 2>&1 </dev/null &
for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_oeplb.log && break; sleep 8; done
grep -q "ready to roll" $LOGD/srv_oeplb.log || { echo "BOOTFAIL"; exit 1; }
echo "[$(ts)] oeplb ready"; sleep 5; cd /workspace/EPLB/OEPLB/scripts
# 预热让OEPLB在跨域上充分自适应
python3 $RG oexd_w1 $XD1 32 >/dev/null 2>&1
python3 $RG oexd_w2 $XD10 32 >/dev/null 2>&1
echo "[$(ts)] warmup done"
for r in 1 2 3; do python3 $RG oerep_xd1_r$r $XD1 32 > $LOGD/oerep_xd1_r$r.log 2>&1; echo "[$(ts)] xd1 r$r done"; done
for r in 1 2 3; do python3 $RG oerep_xd10_r$r $XD10 32 > $LOGD/oerep_xd10_r$r.log 2>&1; echo "[$(ts)] xd10 r$r done"; done
bash $LOGD/killall.sh >/dev/null 2>&1
echo "[$(ts)] OE_XD_REPEAT_DONE"
