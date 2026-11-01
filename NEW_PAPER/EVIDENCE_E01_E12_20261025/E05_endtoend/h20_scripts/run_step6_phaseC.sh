#!/bin/bash
# Step-6 Phase C: mechanism ablations, config-only on frozen code (protocol §4).
# Order: W0-M2off, W0-M3off, W0-M1p, W2-M2off, W2-M3off, W2-M1p. 1 start each, exploratory.
source "$(dirname "$0")/step6_common.sh"
CLOG=$S6/phaseC.log
verify_hashes || { echo "HASH ABORT" | tee -a "$CLOG"; exit 1; }
echo "[$(date -Is)] PHASE_C start (6 cells)" | tee -a "$CLOG"
run_abl(){ # <cond> <name> then ABL_* in env
  run_start "phaseC/$1_$2__s1" "s6c_$1_$2" "$1" v2 16 "$CLOG"
}
ABL_THETA_LOW=1000000000000000000 ABL_THETA_ACT=0 run_abl W0 M2off
ABL_DWELL_S=0 ABL_DWELL_G=0 ABL_M_SHRINK=1 ABL_M_GROW=1 ABL_CAP=99 run_abl W0 M3off
ABL_BETA=0.0 run_abl W0 M1p
ABL_THETA_LOW=1000000000000000000 ABL_THETA_ACT=0 run_abl W2 M2off
ABL_DWELL_S=0 ABL_DWELL_G=0 ABL_M_SHRINK=1 ABL_M_GROW=1 ABL_CAP=99 run_abl W2 M3off
ABL_BETA=0.0 run_abl W2 M1p
echo "[$(date -Is)] PHASE_C_DONE" | tee -a "$CLOG"
