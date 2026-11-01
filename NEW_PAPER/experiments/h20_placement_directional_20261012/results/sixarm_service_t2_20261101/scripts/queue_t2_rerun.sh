#!/bin/bash
# queue_t2_rerun.sh - zero_data rerun of 10 first_attempt failures caused by DEFECT_FIX
# (launcher line19 PL= prefix lost in splice; fixed 15:43:55Z new_sha16=ad0ae540cff81805).
# First-attempt boot dirs archived to boots_fail1/ before each rerun. Order = ORDER_TABLE order.
# rc=2 GUARD_ABORT stop; rc=1 log+continue.
set -u
T2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101
PROV=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/pf_all_phase_select_20261030/B_data/streams/prover_1024_out32_ie.jsonl
PROVSHA=0f8b5a50f0a59b24ed0fa251a065e4ebf19cb51c45c86ce48ed7e0a00c5a4ca3
PBDO=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101/B_data/streams/pbd_prover_book_double_out32_ie.jsonl
PBDOSHA=cdd1d8643812e695c357e4b5548f9ec742f28d866dced9099931e38d2461cdee
FAILDIR=$T2/boots_fail1
mkdir -p $FAILDIR
runr(){
  echo "[queue_t2_rerun] $1 arm=$2 start $(date -u +%FT%TZ)"
  if [ -d $T2/boots/$1 ]; then rm -rf $FAILDIR/$1; cp -r $T2/boots/$1 $FAILDIR/$1; fi
  bash $T2/scripts/run_t2_boot.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue_t2_rerun] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue_rerun\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> $T2/LEDGER_t2.tsv; exit 2; fi
  if [ $RC -ne 0 ]; then echo "[queue_t2_rerun] $1 rc=$RC logged continue $(date -u +%FT%TZ)"; fi
}
runr PROV_ES_r1 ES "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_MT_r1 MT "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_DF_r1 DF "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_MT_r2 MT "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_DF_r2 DF "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_ES_r2 ES "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_MT_r3 MT "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_DF_r3 DF "" "$PROV" "$PROVSHA" 1024 256 1800
runr PROV_ES_r3 ES "" "$PROV" "$PROVSHA" 1024 256 1800
runr PBD_ES_r1 ES "" "$PBDO" "$PBDOSHA" 5344 256 1800
echo "[queue_t2_rerun] RERUN_QUEUE_END 10boots $(date -u +%FT%TZ)"
printf "%s\tqueue_rerun\tRERUN_QUEUE_END 10boots\n" "$(date -u +%FT%TZ)" >> $T2/LEDGER_t2.tsv
