#!/bin/bash
LOGD=/workspace/logs; ts(){ date '+%H:%M:%S'; }
bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3
rm -f $LOGD/srv_moetuner.log
setsid bash $LOGD/launch_moetuner.sh > $LOGD/srv_moetuner.log 2>&1 </dev/null &
for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_moetuner.log && break; grep -qi "Scheduler hit an exception" $LOGD/srv_moetuner.log && break; sleep 8; done
grep -q "ready to roll" $LOGD/srv_moetuner.log || { echo "[$(ts)] MOETUNER BOOTFAIL"; exit 1; }
echo "[$(ts)] moetuner ready"; grep -o "ep_dispatch_algorithm='[a-z]*'" $LOGD/srv_moetuner.log|head -1
sleep 5
cd /workspace/EPLB/OEPLB/baselines/moetuner
for r in w1 w2; do python3 bench_0914.py /workspace/data/prover_256tok_out1.jsonl 256 t1_moetuner_$r >/dev/null 2>&1; done
for r in 1 2 3 4 5; do python3 bench_0914.py /workspace/data/prover_256tok_out1.jsonl 256 t1_moetuner_r$r >/dev/null 2>&1; done
echo "[$(ts)] moetuner table1(tps) done"
for r in 1 2; do python3 $LOGD/freq6_local.py 32 1800 t2_moetuner_r$r > $LOGD/freq6_moetuner_r$r.log 2>&1; done
echo "[$(ts)] moetuner table2(tps) done"
cd /workspace/EPLB/OEPLB/scripts; export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
python3 run_grid_bench.py ttft_moetuner_t1w /workspace/data/t1_prover256_grid.jsonl 256 >/dev/null 2>&1
python3 run_grid_bench.py ttft_moetuner_t1 /workspace/data/t1_prover256_grid.jsonl 256 > $LOGD/ttft_moetuner_t1.log 2>&1
python3 run_grid_bench.py ttft_moetuner_t2 /workspace/data/t2_freq6_grid.jsonl 32 > $LOGD/ttft_moetuner_t2.log 2>&1
echo "[$(ts)] moetuner TTFT/TPOT done"
bash $LOGD/killall.sh >/dev/null 2>&1
echo "[$(ts)] MOETUNER_DONE"
