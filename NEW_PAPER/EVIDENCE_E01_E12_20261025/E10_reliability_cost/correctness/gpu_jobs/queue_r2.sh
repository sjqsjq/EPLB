#!/bin/bash
# ROUND-2 live verification queue (v2 code, FIX-3). Halt on first FAIL.
# Round-2 ledger, separate from the spent phase-2 6-job ledger. <=4 server jobs.
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
L=$G/correctness/gpu_jobs/LEDGER_r2.tsv
echo -e "job\tstart_utc\tend_utc\trc_job\trc_extra\tverdict" > $L
echo "# ROUND-2 verification. deployed async_swapper sha: $(sha256sum /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/async_swapper.py | cut -c1-16)" >> $L
echo "# deployed controller sha: $(sha256sum /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/controller.py | cut -c1-16)" >> $L
run_server() {
  local S=$(date -u +%FT%TZ) RC RCX V
  bash $G/correctness/gpu_jobs/run_server_job.sh "$@"
  RC=$?
  python3 $G/correctness/gpu_jobs/assert_r2_extra.py "$G/correctness/gpu_jobs/$1" "$1"
  RCX=$?
  [ $RC -eq 0 ] && [ $RCX -eq 0 ] && V=PASS || V=FAIL
  echo -e "$1\t$S\t$(date -u +%FT%TZ)\t$RC\t$RCX\t$V" >> $L
  [ "$V" = PASS ]
}
run_server R2_P_E3 P ""        e3_multidomain_switch.jsonl 8192 256 3600 4 || { echo HALTED_AT=R2_P_E3 >> $L; exit 1; }
run_server R2_F_E3 F "1 16 16" e3_multidomain_switch.jsonl 8192 256 3600 4 || { echo HALTED_AT=R2_F_E3 >> $L; exit 1; }
run_server R2_P_E4 P ""        e4_fullduplex.jsonl         2048 256 3600 6 || { echo HALTED_AT=R2_P_E4 >> $L; exit 1; }
run_server R2_F_E4 F "1 16 16" e4_fullduplex.jsonl         2048 256 3600 6 || { echo HALTED_AT=R2_F_E4 >> $L; exit 1; }
echo -e "ALL\t-\t$(date -u +%FT%TZ)\t0\t0\tQUEUE_PASS" >> $L
