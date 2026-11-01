#!/bin/bash
# queue_t2_formal.sh - 36 formal boots in frozen ORDER_TABLE_t2.json order.
# Generated programmatically from ORDER_TABLE (36 run lines asserted).
# rc=2 GUARD_ABORT stop; rc=1 log+continue (INFRA <=2, never silently dropped).
set -u
T2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101
PROV=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/pf_all_phase_select_20261030/B_data/streams/prover_1024_out32_ie.jsonl
PROVSHA=0f8b5a50f0a59b24ed0fa251a065e4ebf19cb51c45c86ce48ed7e0a00c5a4ca3
PBDO=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101/B_data/streams/pbd_prover_book_double_out32_ie.jsonl
PBDOSHA=cdd1d8643812e695c357e4b5548f9ec742f28d866dced9099931e38d2461cdee
run(){
  echo "[queue_t2] $1 arm=$2 start $(date -u +%FT%TZ)"
  bash $T2/scripts/run_t2_boot.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"
  RC=$?
  if [ $RC -eq 2 ]; then echo "[queue_t2] GUARD_ABORT at $1 rc=2 $(date -u +%FT%TZ)"; printf "%s\tqueue\tGUARD_ABORT at %s rc=2\n" "$(date -u +%FT%TZ)" "$1" >> /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101/LEDGER_t2.tsv; exit 2; fi
  if [ $RC -ne 0 ]; then echo "[queue_t2] $1 rc=$RC logged continue $(date -u +%FT%TZ)"; fi
}
run PROV_ES_r1 ES "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_ED_r1 ED "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_MT_r1 MT "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_DF_r1 DF "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_P0_r1 P "0" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_I_r1 I "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_ED_r2 ED "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_MT_r2 MT "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_DF_r2 DF "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_P0_r2 P "0" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_I_r2 I "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_ES_r2 ES "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_MT_r3 MT "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_DF_r3 DF "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_P0_r3 P "0" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_I_r3 I "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_ES_r3 ES "" "$PROV" "$PROVSHA" 1024 256 1800
run PROV_ED_r3 ED "" "$PROV" "$PROVSHA" 1024 256 1800
run PBD_ES_r1 ES "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_ED_r1 ED "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_MT_r1 MT "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_DF_r1 DF "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_P0_r1 P "0" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_I_r1 I "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_ED_r2 ED "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_MT_r2 MT "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_DF_r2 DF "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_P0_r2 P "0" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_I_r2 I "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_ES_r2 ES "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_MT_r3 MT "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_DF_r3 DF "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_P0_r3 P "0" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_I_r3 I "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_ES_r3 ES "" "$PBDO" "$PBDOSHA" 5344 256 1800
run PBD_ED_r3 ED "" "$PBDO" "$PBDOSHA" 5344 256 1800
echo "[queue_t2] T2_QUEUE_END $(date -u +%FT%TZ)"
printf "%s\tqueue\tT2_QUEUE_END 36_formal_boots\n" "$(date -u +%FT%TZ)" >> /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/sixarm_service_t2_20261101/LEDGER_t2.tsv
