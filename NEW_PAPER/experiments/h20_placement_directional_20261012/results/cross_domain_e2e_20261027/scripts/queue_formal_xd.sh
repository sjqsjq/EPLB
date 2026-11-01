#!/bin/bash
# queue_formal_xd.sh - 24 formal boots in locked ORDER_TABLE order.
# Generated programmatically from ORDER_TABLE.json (24 run lines asserted).
# rc=2 -> GUARD_ABORT stop queue; rc=1 -> log+continue (operator triage
# within INFRA budget, never silently dropped). PROTOCOL sec.6/8.
set -u
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
DS=$F/B_data/streams/e3_multidomain_switch.jsonl
DSSHA=d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879
run(){
  echo "[queue] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $X/scripts/run_xd_boot.sh "$1" "$2" "$3" "$DS" "$DSSHA" 8192 256 900
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> $X/LEDGER_xd.tsv; exit 2; fi
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
echo "[queue] ALL_DONE $(date -u +%FT%TZ)"
printf "%s\tqueue\tALL_DONE 24_formal_boots\n" "$(date -u +%FT%TZ)" >> $X/LEDGER_xd.tsv
