#!/bin/bash
# Author-directed diag queue (README.md): E3 P vs I on v2, then on pre-fix original, restore v2.
set -u
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
D=$G/author_diag_20261021
DEP=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
LOCK=$G/code_lock/deployed_pb_oeplb
V2=$G/freeze/code_fixed_v2/deployed_pb_oeplb
L=$D/LEDGER_adhoc.tsv
V2_A=2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f
V2_C=4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697
ORIG_A=93fa4ccdd653443a13d3b3f4975dd3b2f5dcddad71efedbd86a257f0ed0d12b9
ORIG_C=c6d2f76454679dffd9b41eb0d8fa1d98f6765eb72e02cfaeb372ad85dbd84f7e
[ -f $L ] || echo -e "label\tstart_utc\tend_utc\trc\ttps\tcode" > $L
restore_v2() {
  cp $V2/async_swapper.py $DEP/async_swapper.py
  cp $V2/controller.py $DEP/controller.py
  rm -rf $DEP/__pycache__
  local a=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1) c=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
  if [ "$a" = "$V2_A" ] && [ "$c" = "$V2_C" ]; then
    echo -e "RESTORE_v2\t-\t$(date -u +%FT%TZ)\t0\t-\tOK" >> $L
  else
    echo -e "RESTORE_v2\t-\t$(date -u +%FT%TZ)\t1\t-\tFAIL_a=${a:0:16}_c=${c:0:16}" >> $L
  fi
}
trap restore_v2 EXIT
sha_now() { sha256sum $DEP/async_swapper.py | cut -d' ' -f1; }
A0=$(sha_now)
[ "$A0" = "$V2_A" ] || { echo "ABORT: deployed not v2 at start ($A0)" >> $L; exit 2; }
echo "# ADHOC_START $(date -u +%FT%TZ) v2 verified" >> $L
run() {
  local LAB=$1 ARM=$2 PAR=$3 MODE=$4 MS=$5 CODE=$6
  local S=$(date -u +%FT%TZ) RC TPS
  bash $D/run_adhoc_boot.sh "$LAB" "$ARM" "$PAR" "$MODE" "$MS"
  RC=$?
  TPS=$(python3 -c "import json;print(round(json.load(open('$D/boots/$LAB/$LAB.json'))['tps'],2))" 2>/dev/null || echo NA)
  echo -e "$LAB\t$S\t$(date -u +%FT%TZ)\t$RC\t$TPS\t$CODE" >> $L
}
# 1) current fixed system (v2)
run adhoc_P_v2_E3  P "" pb 4 v2
run adhoc_I_v2_E3  I "" plain 0 v2
# 2) swap to pre-fix ORIGINAL (old-campaign code; known deadlock risk — contained, artifacts kept)
cp $LOCK/async_swapper.py $DEP/async_swapper.py
cp $LOCK/controller.py $DEP/controller.py
rm -rf $DEP/__pycache__
AS=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); CS=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
if [ "$AS" != "$ORIG_A" ] || [ "$CS" != "$ORIG_C" ]; then echo -e "ABORT_orig_swap\t-\t$(date -u +%FT%TZ)\t1\t-\tFAIL" >> $L; exit 2; fi
echo "# SWAPPED_TO_ORIG $(date -u +%FT%TZ) a=${AS:0:16} c=${CS:0:16}" >> $L
run adhoc_P_orig_E3 P "" pb 4 orig
run adhoc_I_orig_E3 I "" plain 0 orig
# restore happens via trap EXIT
echo "# ADHOC_DONE $(date -u +%FT%TZ)" >> $L
exit 0
