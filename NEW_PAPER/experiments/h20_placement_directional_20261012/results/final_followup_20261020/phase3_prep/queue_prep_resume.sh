#!/bin/bash
# Phase-3 prep resume queue (v2 code, ADDENDUM A1 of PROTOCOL.md).
# Code version (frozen v2): async_swapper 2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f
#                           controller     4964cfc8e8af397a (unchanged from v1)
# B-arm only: B1r (B1 slot re-execution under v2), B2, B3, B4, then pooled eligibility.
# Halt-on-fail; ledger appends to LEDGER_prep.tsv; prep budget after this queue: 9/12.
set -u
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
L=$G/phase3_prep/LEDGER_prep.tsv
# guard: deployed code must be exactly the frozen v2
SHA=$(sha256sum /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/async_swapper.py | cut -d' ' -f1)
if [ "$SHA" != "2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f" ]; then
  echo -e "ABORT_v2_sha_mismatch\t$(date -u +%FT%TZ)\t-\t1\tFAIL" >> $L; exit 1
fi
echo -e "# RESUME_v2 async=2e4652a0306cb315 controller=4964cfc8e8af397a start=$(date -u +%FT%TZ)" >> $L
run() {
  local S=$(date -u +%FT%TZ) RC V
  bash $G/phase3_prep/run_prep_boot.sh "$@"
  RC=$?
  [ $RC -eq 0 ] && V=PASS || V=FAIL
  echo -e "$1\t$S\t$(date -u +%FT%TZ)\t$RC\t$V" >> $L
  [ $V = PASS ]
}
E3=e3_multidomain_switch.jsonl; E4=e4_fullduplex.jsonl
run prep3_B1r_P_E3 P "" $E3 8192 256 3600 pb 4 || { echo HALTED_AT=B1r >> $L; exit 1; }
run prep3_B2_P_E3  P "" $E3 8192 256 3600 pb 4 || { echo HALTED_AT=B2  >> $L; exit 1; }
run prep3_B3_P_E4  P "" $E4 2048 256 3600 pb 6 || { echo HALTED_AT=B3  >> $L; exit 1; }
run prep3_B4_P_E4  P "" $E4 2048 256 3600 pb 6 || { echo HALTED_AT=B4  >> $L; exit 1; }
python3 $G/phase3_prep/extract_eligibility_resume.py || { echo HALTED_AT=ELIGIBILITY >> $L; exit 1; }
echo -e "PREP_ALL_v2\t-\t$(date -u +%FT%TZ)\t0\tQUEUE_PASS" >> $L
