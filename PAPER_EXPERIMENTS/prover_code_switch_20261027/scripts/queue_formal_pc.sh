#!/bin/bash
# queue_formal_pc.sh - 24 formal boots in locked rotation order (same
# rotation scheme as cross_domain_e2e_20261027 ORDER_TABLE: each arm takes
# each position across rounds where possible).
# rc=2 -> GUARD_ABORT stop queue; rc=1 -> log+continue (operator triage
# within INFRA budget, never silently dropped). PROTOCOL_pc.md sec.5.
set -u
PC=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/prover_code_switch_20261027
DS=$PC/B_data/streams/pc_prover_code_switch.jsonl
DSSHA=01e33f82de54238da2ad342036998fe31a244c519339bfec91983ac45c773f69
run(){
  echo "[queue] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $PC/scripts/run_pc_boot.sh "$1" "$2" "$3" "$DS" "$DSSHA" 4096 32 1800
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> $PC/LEDGER_pc.tsv; exit 2; fi
  if [ $RC -ne 0 ]; then echo "[queue] $1 rc=$RC logged continue $(date -u +%FT%TZ)"; fi
}
run ES_r1 ES ""
run ED_r1 ED ""
run MT_r1 MT ""
run DF_r1 DF ""
run P0_r1 P "0"
run I_r1 I ""
run ED_r2 ED ""
run MT_r2 MT ""
run DF_r2 DF ""
run P0_r2 P "0"
run I_r2 I ""
run ES_r2 ES ""
run MT_r3 MT ""
run DF_r3 DF ""
run P0_r3 P "0"
run I_r3 I ""
run ES_r3 ES ""
run ED_r3 ED ""
run DF_r4 DF ""
run P0_r4 P "0"
run I_r4 I ""
run ES_r4 ES ""
run ED_r4 ED ""
run MT_r4 MT ""
echo "[queue] QUEUE_PC_COMPLETE $(date -u +%FT%TZ)"
printf "%s\tqueue\tQUEUE_PC_COMPLETE 24_formal_boots\n" "$(date -u +%FT%TZ)" >> $PC/LEDGER_pc.tsv
