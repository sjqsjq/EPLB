#!/bin/bash
# queue_t2_pbd_bp.sh - AUTHOR-DIRECTED 2026-11-02: PBD scenario rerun on book-first corrected stream.
# Author: original design = book-to-prover. Frozen PROTOCOL_t2.md sec.2 declared prover-to-book and the
# formal stream matched that protocol; per author instruction the corrected-order stream was materialized
# (block swap, row bytes preserved, multiset identical verified, sha in code_lock/stream_bp_sha.env) and
# 2 rounds (r4/r5, 12 boots) are rerun. Frozen docs NOT edited; correction logged in LEDGER_t2.tsv;
# formal 36 boots untouched. rc=2 GUARD_ABORT stop; rc=1 log+continue.
set -u
T2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101
PBDBP=$T2/B_data/streams/pbd_book_prover_double_out32_ie.jsonl
PBDBPSHA=7b42568a121b05e1054d4a60de8588db23e0d179357a503c7f3d2993b47a132a
run(){
  echo "[queue_t2_bp] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $T2/scripts/run_t2_boot.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue_t2_bp] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue_bp\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> $T2/LEDGER_t2.tsv; exit 2; fi
  if [ $RC -ne 0 ]; then echo "[queue_t2_bp] $1 rc=$RC logged continue $(date -u +%FT%TZ)"; fi
}
run PBD_BP_DF_r4 DF "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_P0_r4 P "0" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_I_r4 I "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_ES_r4 ES "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_ED_r4 ED "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_MT_r4 MT "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_P0_r5 P "0" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_I_r5 I "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_ES_r5 ES "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_ED_r5 ED "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_MT_r5 MT "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
run PBD_BP_DF_r5 DF "" "$PBDBP" "$PBDBPSHA" 5344 256 1800
echo "[queue_t2_bp] BP_QUEUE_END 12boots $(date -u +%FT%TZ)"
printf "%s\tqueue_bp\tBP_QUEUE_END 12boots\n" "$(date -u +%FT%TZ)" >> $T2/LEDGER_t2.tsv
