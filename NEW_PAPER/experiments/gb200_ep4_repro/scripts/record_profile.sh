#!/bin/bash
LOGD=/workspace/logs; OE=/workspace/EPLB/OEPLB
ts(){ date '+%H:%M:%S'; }
mkdir -p $LOGD/recdump; rm -f $LOGD/recdump/*.pt
bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
rm -f $LOGD/srv_profile.log
setsid bash $LOGD/launch_profile.sh > $LOGD/srv_profile.log 2>&1 </dev/null &
for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_profile.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_profile.log && { echo "BOOTFAIL"; exit 1; }; sleep 8; done
grep -q "ready to roll" $LOGD/srv_profile.log || { echo "NOTREADY"; exit 1; }
echo "[$(ts)] profile server ready; recorder mode:"; grep -o "expert_distribution_recorder_mode='[a-z]*'" $LOGD/srv_profile.log|head -1
sleep 3
python3 $OE/repro/dump_counts.py $LOGD/counts_gb200_prover256.json start
echo "[$(ts)] recording started; 跑 head1024..."
cd $OE/scripts; export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
python3 run_grid_bench.py profile_head1024 /workspace/data/prover256_head1024.jsonl 256 > $LOGD/profile_run.log 2>&1
echo "[$(ts)] head1024 done; stop+dump"
python3 $OE/repro/dump_counts.py $LOGD/counts_gb200_prover256.json stop
echo "[$(ts)] RECORD_DONE"
bash $LOGD/killall.sh >/dev/null 2>&1
