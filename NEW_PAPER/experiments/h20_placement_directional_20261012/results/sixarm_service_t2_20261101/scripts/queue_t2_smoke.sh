#!/bin/bash
# queue_t2_smoke.sh - path A smoke, STRICT: any nonzero rc aborts (PROTOCOL_t2 sec.3).
# Covers plain-arm x normal (I) and PB-adaptive-arm x normal (P0). Zero formal data.
set -u
T2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101
SMOKE=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/pf_all_phase_select_20261030/B_data/streams/prover_512_smoke.jsonl
SMOKESHA=7d575c9c68075a52bbc2f068ccfa5dbef323ae23e84c81d56f729737af057fba
runs(){
  echo "[queue_t2_smoke] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $T2/scripts/run_t2_boot.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
  RC=$?
  if [ $RC -ne 0 ]; then echo "[queue_t2_smoke] SMOKE_FAIL $1 rc=$RC $(date -u +%FT%TZ)"; printf "%s\tqueue\tSMOKE_FAIL at %s rc=%s\n" "$(date -u +%FT%TZ)" "$1" "$RC" >> /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101/LEDGER_t2.tsv; exit $RC; fi
}
runs PROV_I_s0 I "" "$SMOKE" "$SMOKESHA" 512 256 600
runs PROV_P0_s0 P "0" "$SMOKE" "$SMOKESHA" 512 256 600
echo "[queue_t2_smoke] T2_SMOKE_END $(date -u +%FT%TZ)"
printf "%s\tqueue\tT2_SMOKE_END 2_boots\n" "$(date -u +%FT%TZ)" >> /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101/LEDGER_t2.tsv
