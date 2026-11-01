#!/bin/bash
# wait_and_rerun_t2.sh - fires queue_t2_rerun.sh (sha16=a7e2aaf40afb7577) after master_run_t1t2 exits
T2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101
while pgrep -f "master_run_t1t[2]" > /dev/null 2>&1; do sleep 30; done
sleep 45
echo "[waiter] master gone, firing rerun $(date -u +%FT%TZ)" >> $T2/rerun.log
bash $T2/scripts/queue_t2_rerun.sh >> $T2/rerun.log 2>&1
echo "[waiter] rerun exited rc=$? $(date -u +%FT%TZ)" >> $T2/rerun.log
