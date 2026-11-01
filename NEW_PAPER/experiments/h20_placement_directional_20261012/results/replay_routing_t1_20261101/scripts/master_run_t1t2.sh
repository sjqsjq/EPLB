#!/bin/bash
# master_run_t1t2.sh - sequential rc-gated execution: T1 queue -> T2 smoke -> T2 formal.
# Any gate failure => STOP (no silent continuation). Author authorization 2026-11-01.
set -u
T1=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/replay_routing_t1_20261101
T2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101
echo "[master] T1_QUEUE start $(date -u +%FT%TZ)"
bash $T1/scripts/queue_t1.sh
RC1=$?
echo "[master] T1_QUEUE rc=$RC1 $(date -u +%FT%TZ)"
if [ $RC1 -ne 0 ]; then echo "[master] STOP_after_T1 rc=$RC1 $(date -u +%FT%TZ)"; exit $RC1; fi
echo "[master] T2_SMOKE start $(date -u +%FT%TZ)"
bash $T2/scripts/queue_t2_smoke.sh
RC2=$?
echo "[master] T2_SMOKE rc=$RC2 $(date -u +%FT%TZ)"
if [ $RC2 -ne 0 ]; then echo "[master] STOP_after_T2_smoke rc=$RC2 $(date -u +%FT%TZ)"; exit $RC2; fi
echo "[master] T2_FORMAL start $(date -u +%FT%TZ)"
bash $T2/scripts/queue_t2_formal.sh
RC3=$?
echo "[master] MASTER_END rc_t1=$RC1 rc_t2smoke=$RC2 rc_t2formal=$RC3 $(date -u +%FT%TZ)"
exit $RC3
