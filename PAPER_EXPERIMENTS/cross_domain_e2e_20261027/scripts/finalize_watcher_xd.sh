#!/bin/sh
# finalize_watcher_xd.sh -- cron-independent safety net.
# Waits for ALL_DONE row in LEDGER, then runs analyze_xd.py --final and writes marker.
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
LOG=$X/logs/finalize_watcher.log
echo "WATCHER_START $(date -u +%FT%TZ) pid=$$" >> $LOG
i=0
while [ $i -lt 120 ]; do
  if grep -q "ALL_DONE" $X/LEDGER_xd.tsv 2>/dev/null; then
    echo "ALL_DONE_DETECTED $(date -u +%FT%TZ) iter=$i" >> $LOG
    sleep 15
    cd $X/analysis || exit 1
    python3 analyze_xd.py --final >> $LOG 2>&1
    rc=$?
    echo "FINAL_ANALYZE rc=$rc $(date -u +%FT%TZ)" >> $LOG
    echo "rc=$rc ts=$(date -u +%FT%TZ) analyze_sha=7f62f26544127f8b55a9a663da3969bedb448d8de502956824e0cc5bc847c6fd" > $X/analysis/FINAL_ANALYZE_DONE
    exit 0
  fi
  i=$((i+1))
  sleep 20
done
echo "WATCHER_TIMEOUT_NO_ALL_DONE $(date -u +%FT%TZ)" >> $LOG
