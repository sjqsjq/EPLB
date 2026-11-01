#!/bin/bash
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
L=$G/correctness/gpu_jobs/LEDGER.tsv
run_server() {
  local S=$(date -u +%FT%TZ) RC V
  bash $G/correctness/gpu_jobs/run_server_job.sh "$@"
  RC=$?
  [ $RC -eq 0 ] && V=PASS || V=FAIL
  echo -e "$1\t$S\t$(date -u +%FT%TZ)\t$RC\t$V" >> $L
  [ $V = PASS ]
}
run_server J2_E3_P P "" e3_multidomain_switch.jsonl 8192 256 3600 4 || { echo HALTED_AT=J2 >> $L; exit 1; }
run_server J3_E4_F F "1 16 16" e4_fullduplex.jsonl 2048 256 3600 6 || { echo HALTED_AT=J3 >> $L; exit 1; }
run_server J6_E4_P P "" e4_fullduplex.jsonl 2048 256 3600 6 || { echo HALTED_AT=J6 >> $L; exit 1; }
echo -e "ALL\t-\t$(date -u +%FT%TZ)\t0\tQUEUE_PASS" >> $L
