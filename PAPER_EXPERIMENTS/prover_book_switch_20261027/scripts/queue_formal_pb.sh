#!/bin/bash
# queue_formal_pb.sh - 12 formal boots (6 arms x 2 rounds, author: run only
# twice then joint analysis). Rotation continues the pc campaign scheme:
# r1: ES ED MT DF P0 I | r2: ED MT DF P0 I ES.
# rc=2 -> GUARD_ABORT stop queue; rc=1 -> log+continue (INFRA budget <=2
# zero-data restarts, never silently dropped). PROTOCOL_pb.md sec.5.
set -u
PB=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/prover_book_switch_20261027
DS=$PB/B_data/streams/pb_prover_book_switch.jsonl
DSSHA=2da3467263d4f52889ea2931fa84d9fb244ed3dfabe136fddb5c1b6feb7bf904
run(){
  echo "[queue] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $PB/scripts/run_pb_boot.sh "$1" "$2" "$3" "$DS" "$DSSHA" 3048 32 1800
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> $PB/LEDGER_pb.tsv; exit 2; fi
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
echo "[queue] QUEUE_PB_COMPLETE $(date -u +%FT%TZ)"
printf "%s\tqueue\tQUEUE_PB_COMPLETE 12_formal_boots\n" "$(date -u +%FT%TZ)" >> $PB/LEDGER_pb.tsv
