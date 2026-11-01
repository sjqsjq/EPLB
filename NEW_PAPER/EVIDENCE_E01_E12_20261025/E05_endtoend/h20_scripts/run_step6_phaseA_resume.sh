#!/bin/bash
# Step-6 Phase A RESUME (deviation memo step6_deviation_infra_portrace.md):
# W1 grid, W2 grid, then W0_f32 infra re-run (s1r). W0_f08/f16 done; f64 manual-archived.
source "$(dirname "$0")/step6_common.sh"
CLOG=$S6/phaseA.log
verify_hashes || { echo "HASH ABORT" | tee -a "$CLOG"; exit 1; }
echo "[$(date -Is)] PHASE_A_RESUME start (9 cells: W1x4, W2x4, W0_f32 re-run)" | tee -a "$CLOG"
for cond in W1 W2; do
  for W in 8 16 32 64; do
    run_start "phaseA/${cond}_f$(printf %02d $W)__s1" "s6a_${cond}_f${W}" "$cond" fixed "$W" "$CLOG"
  done
done
run_start "phaseA/W0_f32__s1r" "s6a_W0_f32r" W0 fixed 32 "$CLOG"
echo "[$(date -Is)] PHASE_A_DONE" | tee -a "$CLOG"
