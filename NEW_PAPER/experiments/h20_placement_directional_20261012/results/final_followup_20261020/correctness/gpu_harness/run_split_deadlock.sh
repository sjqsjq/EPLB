#!/bin/bash
# Round-2 offline mechanism proof: single-group (expect HANG) vs per-peer (expect PASS).
set -u
cd "$(dirname "$0")"
LOG_D=split_deadlock_logs; mkdir -p $LOG_D
for MODE in single perpeer; do
  echo "=== MODE=$MODE start $(date -u +%FT%TZ) ===" | tee -a $LOG_D/summary.log
  timeout 900 torchrun --nproc_per_node=8 --master_port=29617 \
      test_split_deadlock_gpu.py --mode $MODE --iters 40 --timeout 180 \
      > $LOG_D/$MODE.log 2>&1
  RC=$?
  LAST=$(grep -c "\[progress\]" $LOG_D/$MODE.log)
  HUNG=$(grep -icE "Watchdog caught|timed out|Timeout" $LOG_D/$MODE.log)
  echo "=== MODE=$MODE rc=$RC progress_lines=$LAST watchdog_hits=$HUNG end $(date -u +%FT%TZ) ===" | tee -a $LOG_D/summary.log
  tail -3 $LOG_D/$MODE.log | tee -a $LOG_D/summary.log
done
echo "ALL_DONE $(date -u +%FT%TZ)" | tee -a $LOG_D/summary.log
