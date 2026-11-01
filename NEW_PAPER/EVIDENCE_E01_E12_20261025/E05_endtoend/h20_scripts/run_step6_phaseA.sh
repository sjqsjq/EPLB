#!/bin/bash
# Step-6 Phase A: F-local calibration grids. 12 starts, fixed order (protocol §4).
source "$(dirname "$0")/step6_common.sh"
CLOG=$S6/phaseA.log
verify_hashes || { echo "HASH ABORT" | tee -a "$CLOG"; exit 1; }
echo "[$(date -Is)] PHASE_A start (12 cells)" | tee -a "$CLOG"
for cond in W0 W1 W2; do
  for W in 8 16 32 64; do
    run_start "phaseA/${cond}_f$(printf %02d $W)__s1" "s6a_${cond}_f${W}" "$cond" fixed "$W" "$CLOG"
  done
done
echo "[$(date -Is)] PHASE_A_DONE" | tee -a "$CLOG"
