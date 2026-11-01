#!/bin/bash
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
L=$G/phase3_prep/LEDGER_prep.tsv
[ -f $L ] || echo -e "label\tstart_utc\tend_utc\trc\tverdict" > $L
run() {
  local S=$(date -u +%FT%TZ) RC V
  bash $G/phase3_prep/run_prep_boot.sh "$@"
  RC=$?
  [ $RC -eq 0 ] && V=PASS || V=FAIL
  echo -e "$1\t$S\t$(date -u +%FT%TZ)\t$RC\t$V" >> $L
  [ $V = PASS ]
}
E3=e3_multidomain_switch.jsonl; E4=e4_fullduplex.jsonl
run prep3_C1_I_E3      I ""           $E3 8192 256 3600 plain 0 || { echo HALTED_AT=C1 >> $L; exit 1; }
run prep3_C2_I_E4      I ""           $E4 2048 256 3600 plain 0 || { echo HALTED_AT=C2 >> $L; exit 1; }
# freeze SLO reference from C boots BEFORE any comparison arm
python3 $G/phase3_prep/extract_slo.py || { echo HALTED_AT=SLO_FREEZE >> $L; exit 1; }
run prep3_A1_Enative_E3 E "auto 16 256" $E3 8192 256 3600 plain 0 || { echo HALTED_AT=A1 >> $L; exit 1; }
run prep3_A2_Enative_E4 E "auto 16 256" $E4 2048 256 3600 plain 0 || { echo HALTED_AT=A2 >> $L; exit 1; }
run prep3_B1_P_E3      P ""           $E3 8192 256 3600 pb 4 || { echo HALTED_AT=B1 >> $L; exit 1; }
run prep3_B2_P_E3      P ""           $E3 8192 256 3600 pb 4 || { echo HALTED_AT=B2 >> $L; exit 1; }
run prep3_B3_P_E4      P ""           $E4 2048 256 3600 pb 6 || { echo HALTED_AT=B3 >> $L; exit 1; }
run prep3_B4_P_E4      P ""           $E4 2048 256 3600 pb 6 || { echo HALTED_AT=B4 >> $L; exit 1; }
python3 $G/phase3_prep/extract_eligibility.py
echo -e "PREP_ALL\t-\t$(date -u +%FT%TZ)\t0\tQUEUE_PASS" >> $L
