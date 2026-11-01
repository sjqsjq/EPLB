#!/bin/bash
# queue_bsk.sh - 12 EXPLORATORY boots (6 arms x 2 rounds) single-domain pure book.
# Author single-domain trial directive 2026-10-27. Arms: P0 (baseline AERO-base),
# P0B (bias_gate 0.5), P0S (swap_budget_frac 0.08), P0F (both), ED, DF.
# Rotation: r1 P0 P0B P0S P0F ED DF | r2 P0B P0S P0F ED DF P0.
# rc=2 -> GUARD_ABORT stop queue; rc=1 -> log+continue (INFRA budget <=2
# zero-data restarts, never silently dropped). PROTOCOL_bsk.md sec.5.
set -u
BSK=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/book_single_knobs_20261027
DS=$BSK/B_data/streams/bs_book_single.jsonl
DSSHA=2e9fe867cf2cb317ea56492d0297e09f8b5303d08a9276dec81058ffda8d156a
run(){
  echo "[queue] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $BSK/scripts/run_bsk_boot.sh "$1" "$2" "$3" "$DS" "$DSSHA" 2000 32 1800
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> $BSK/LEDGER_bsk.tsv; exit 2; fi
  if [ $RC -ne 0 ]; then echo "[queue] $1 rc=$RC logged continue $(date -u +%FT%TZ)"; fi
}
run P0_r1 P "0"
run P0B_r1 P0B ""
run P0S_r1 P0S ""
run P0F_r1 P0F ""
run ED_r1 ED ""
run DF_r1 DF ""
run P0B_r2 P0B ""
run P0S_r2 P0S ""
run P0F_r2 P0F ""
run ED_r2 ED ""
run DF_r2 DF ""
run P0_r2 P "0"
echo "[queue] QUEUE_BSK_COMPLETE $(date -u +%FT%TZ)"
printf "%s\tqueue\tQUEUE_BSK_COMPLETE 12_exploratory_boots\n" "$(date -u +%FT%TZ)" >> $BSK/LEDGER_bsk.tsv
