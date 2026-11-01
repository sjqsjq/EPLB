#!/bin/bash
# queue_b2p.sh - 4 formal boots, book->prover static-layout sensitivity.
# block1: OLD(book profile) -> REF(prover profile); block2: REF -> OLD.
# Author directive 2026-10-30 (checklist items 1-12+14; group1 flipped to
# book->prover priority). rc=2 -> GUARD_ABORT stop queue; rc=1 -> log and
# continue (INFRA budget <=2 zero-data restarts, never silently dropped).
set -u
CB=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/book_prover_static_20261030
DS=$CB/B_data/streams/b2p_prover_first2048.jsonl
DSSHA=f8e948f32e483579eb127fba33aa3b324b5cd14114f8cf46d65768e243286c77
run(){
  echo "[queue] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $CB/scripts/run_b2p_boot.sh "$1" "$2" "$3" "$DS" "$DSSHA" 2048 32 1800
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> $CB/LEDGER_b2p.tsv; exit 3; fi
  if [ $RC -ne 0 ]; then echo "[queue] $1 rc=$RC logged continue $(date -u +%FT%TZ)"; fi
}
run OLD_b1 B2P_OLD ""
run REF_b1 B2P_REF ""
run REF_b2 B2P_REF ""
run OLD_b2 B2P_OLD ""
echo "[queue] QUEUE_B2P_DONE 4 boots $(date -u +%FT%TZ)"
printf "%s\tqueue\tQUEUE_B2P_DONE 4_formal_boots\n" "$(date -u +%FT%TZ)" >> $CB/LEDGER_b2p.tsv
