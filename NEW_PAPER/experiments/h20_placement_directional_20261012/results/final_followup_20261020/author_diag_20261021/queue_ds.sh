#!/bin/bash
# Author-directed diag #4: E3/E4 campaign streams under PAPER-ERA server flags, 2x2 matrix
# (code: v2 / gh_pristine) x (ds: E3 / E4), plus paper-flags identity references.
# Campaign-flags numbers already exist (adhoc_P_v2_E3=44.1, adhoc_I_v2_E3=42.8, adhoc_P_gh_E3=38.5,
# phase-4 formal E4: I=2840.25 P=2613.95). Separate author_diag ledger; v2 restore at exit.
set -u
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
D=$G/author_diag_20261021
DEP=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
GH=$D/gh_pristine_pkg
L=$D/LEDGER_adhoc.tsv
V2_A=2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f
V2_C=4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697
GH_A=93fa4ccdd653443a13d3b3f4975dd3b2f5dcddad71efedbd86a257f0ed0d12b9
GH_C=ada04359c2a7ffa2abb2594525c3605c28cfcaa57419497e9e6466d856127df9
GH_CFG=232ecbd529d0b109dfed93242c7299770e89e8918fe613e716eab34a8586d4ea
BK=/tmp/dep_pre_ds_backup
A0=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); C0=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
[ "$A0" = "$V2_A" ] && [ "$C0" = "$V2_C" ] || { echo -e "# DS_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tdeployed_not_v2" >> $L; exit 2; }
pgrep -f 'run_adhoc_boot\.sh|run_f6_boot\.sh|run_ds_boot\.sh|queue_gh\.sh|queue_f6\.sh' >/dev/null && { echo -e "# DS_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tanother_queue_running" >> $L; exit 2; }
echo "# DS_START $(date -u +%FT%TZ) v2 verified" >> $L
restore_v2_full() {
  rm -rf $DEP; cp -a $BK $DEP; rm -rf $DEP/__pycache__
  local a=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1) c=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
  if [ "$a" = "$V2_A" ] && [ "$c" = "$V2_C" ]; then echo -e "RESTORE_v2_ds\t-\t$(date -u +%FT%TZ)\t0\t-\tOK" >> $L
  else echo -e "RESTORE_v2_ds\t-\t$(date -u +%FT%TZ)\t1\t-\tFAIL_a=${a:0:16}_c=${c:0:16}" >> $L; fi
}
run() {
  local LAB=$1 LCH=$2 DS=$3 CODE=$4
  local S=$(date -u +%FT%TZ) RC TPS
  bash $D/run_ds_boot.sh "$LAB" "$LCH" "$DS" "$CODE"; RC=$?
  TPS=$(python3 -c "import json;print(json.load(open('$D/ds_boots/$LAB/ds_result.json'))['tps'])" 2>/dev/null || echo NA)
  echo -e "$LAB\t$S\t$(date -u +%FT%TZ)\t$RC\t$TPS\t$CODE" >> $L
}
run e3_I_paper  paper_plain E3 v2
run e4_I_paper  paper_plain E4 v2
run e3_v2_paper paper_pb    E3 v2
run e4_v2_paper paper_pb    E4 v2
rm -rf $BK; cp -a $DEP $BK || { echo -e "# DS_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tbackup_failed" >> $L; exit 2; }
trap restore_v2_full EXIT
find $DEP -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp $GH/__init__.py $GH/async_swapper.py $GH/config.py $GH/controller.py \
   $GH/fast_metadata.py $GH/per_request_tracer.py $GH/rebalancer.py $GH/routing_tracer.py $DEP/
rm -rf $DEP/__pycache__
AS=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); CS=$(sha256sum $DEP/controller.py | cut -d' ' -f1); GS=$(sha256sum $DEP/config.py | cut -d' ' -f1); N=$(ls $DEP | wc -l)
if [ "$AS" != "$GH_A" ] || [ "$CS" != "$GH_C" ] || [ "$GS" != "$GH_CFG" ] || [ "$N" != "8" ]; then
  echo -e "# DS_SWAP_FAIL\t-\t$(date -u +%FT%TZ)\t1\t-\ta=${AS:0:16}_c=${CS:0:16}_n=$N" >> $L; exit 2
fi
echo "# DS_SWAPPED_TO_GH $(date -u +%FT%TZ) a=${AS:0:16} c=${CS:0:16}" >> $L
run e3_gh_paper paper_pb E3 gh_pristine
run e4_gh_paper paper_pb E4 gh_pristine
echo "# DS_DONE $(date -u +%FT%TZ)" >> $L
exit 0
