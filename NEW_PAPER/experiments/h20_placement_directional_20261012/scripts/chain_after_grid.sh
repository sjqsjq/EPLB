#!/bin/bash
# Waits for the frozen E-A calibration grid to finish, then runs the
# refuse-on-failure pipeline: grid analysis/selection -> P11 deployment +
# campaign code lock -> campaign protocol freeze. STOPS before launching the
# five-arm campaign (launch is a deliberate step after verifying selection).
# Any step failure => CHAIN_FAIL logged, marker not written, chain exits.
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
GLOG=$EXP/results/ea_v1/calibration_grid/campaign.log
LOG=$EXP/results/ea_v1/pipeline_chain.log
READY=$EXP/results/ea_v1/PIPELINE_READY
log(){ echo "[$(date -Is)] $*" >> "$LOG"; }
log "CHAIN_START waiting for GRID_DONE"
while true; do
  if grep -q "GRID_DONE" "$GLOG" 2>/dev/null; then break; fi
  if ! pgrep -f "run_ea_calibration_grid.sh" >/dev/null 2>&1; then
    sleep 30
    if ! grep -q "GRID_DONE" "$GLOG" 2>/dev/null; then
      log "CHAIN_BLOCKED grid driver gone without GRID_DONE (crash? retired starts must NOT be retried; awaiting operator)"
      exit 1
    fi
    break
  fi
  sleep 60
done
log "GRID_DONE detected; waiting 60s for driver teardown"
sleep 60
log "STEP1 analyzer"
python3 $EXP/scripts/analyze_ea_calibration_grid.py >> "$LOG" 2>&1 || { log "CHAIN_FAIL analyzer rc=$?"; exit 1; }
log "STEP2 deploy P11 + campaign code lock"
python3 $EXP/scripts/deploy_p11_and_lock.py >> "$LOG" 2>&1 || { log "CHAIN_FAIL deploy rc=$?"; exit 1; }
log "STEP3 freeze campaign protocol"
python3 $EXP/scripts/freeze_ea_campaign_protocol.py >> "$LOG" 2>&1 || { log "CHAIN_FAIL freeze rc=$?"; exit 1; }
touch "$READY"
log "CHAIN_DONE PIPELINE_READY written; campaign launch awaits operator verification"
