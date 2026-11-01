#!/bin/bash
# Author-directed FINAL CONFIRMATION on freq6: identity first, then v2 (new), then gh_pristine (old).
# Paper-flags cells x3 each + campaign-flags control cells x1 each (incl. gh+camp, never tested before).
# Ad-hoc ledger only; formal budget untouched; unconditional v2 restore via trap.
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
BK=/tmp/dep_pre_final_backup
for i in $(seq 1 240); do
  pgrep -f 'queue_rep\.sh|queue_ds\.sh|queue_adhoc\.sh|run_ds_boot|run_f6_boot' >/dev/null 2>&1 && { sleep 10; continue; }
  break
done
sleep 15
A0=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); C0=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
[ "$A0" = "$V2_A" ] && [ "$C0" = "$V2_C" ] || { echo -e "# FINAL_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tnot_v2" >> $L; exit 2; }
echo "# FINAL_START $(date -u +%FT%TZ) v2 verified" >> $L
restore_v2_full() {
  rm -rf $DEP; cp -a $BK $DEP; rm -rf $DEP/__pycache__
  local a=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1) c=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
  if [ "$a" = "$V2_A" ] && [ "$c" = "$V2_C" ]; then echo -e "RESTORE_v2_final\t-\t$(date -u +%FT%TZ)\t0\t-\tOK" >> $L
  else echo -e "RESTORE_v2_final\t-\t$(date -u +%FT%TZ)\t1\t-\tFAIL_a=${a:0:16}_c=${c:0:16}" >> $L; fi
}
runf6() {
  local LAB=$1 LCH=$2 CODE=$3 MS=$4
  local S=$(date -u +%FT%TZ) RC TPS
  bash $D/run_f6_boot.sh "$LAB" "$LCH" "$CODE" "$MS"; RC=$?
  TPS=$(python3 -c "import json;r=json.load(open('$D/f6_boots/$LAB/f6_result.json'));print('%.2f/%.2f'%(r['tps_run1'] or -1,r['tps_run2'] or -1))" 2>/dev/null || echo NA)
  echo -e "$LAB\t$S\t$(date -u +%FT%TZ)\t$RC\t$TPS\t$CODE" >> $L
}
# --- identity FIRST (author order), v2-deployed, PB off ---
runf6 f6_I_paper_f1 paper_plain v2 0
runf6 f6_I_paper_f2 paper_plain v2 0
runf6 f6_I_paper_f3 paper_plain v2 0
# --- new system (v2) paper flags x3 ---
runf6 f6_v2_P_paper_f1 paper_pb v2 1
runf6 f6_v2_P_paper_f2 paper_pb v2 1
runf6 f6_v2_P_paper_f3 paper_pb v2 1
# --- campaign-flags controls on v2 deploy ---
runf6 f6_I_camp_f1  camp_I v2 0
runf6 f6_v2_P_camp_f1 camp_P v2 1
# --- swap to gh_pristine (old system) ---
rm -rf $BK; cp -a $DEP $BK || { echo -e "# FINAL_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tbackup_failed" >> $L; exit 2; }
trap restore_v2_full EXIT
find $DEP -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp $GH/__init__.py $GH/async_swapper.py $GH/config.py $GH/controller.py \
   $GH/fast_metadata.py $GH/per_request_tracer.py $GH/rebalancer.py $GH/routing_tracer.py $DEP/
rm -rf $DEP/__pycache__
AS=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); CS=$(sha256sum $DEP/controller.py | cut -d' ' -f1); GS=$(sha256sum $DEP/config.py | cut -d' ' -f1); N=$(ls $DEP | wc -l)
if [ "$AS" != "$GH_A" ] || [ "$CS" != "$GH_C" ] || [ "$GS" != "$GH_CFG" ] || [ "$N" != "8" ]; then
  echo -e "# FINAL_SWAP_FAIL\t-\t$(date -u +%FT%TZ)\t1\t-\ta=${AS:0:16}_n=$N" >> $L; exit 2
fi
echo "# FINAL_SWAPPED_TO_GH $(date -u +%FT%TZ)" >> $L
# --- old system (gh_pristine) paper flags x3 + campaign control x1 (first time ever on freq6) ---
runf6 f6_gh_P_paper_f1 paper_pb gh_pristine 1
runf6 f6_gh_P_paper_f2 paper_pb gh_pristine 1
runf6 f6_gh_P_paper_f3 paper_pb gh_pristine 1
runf6 f6_gh_P_camp_f1  camp_P gh_pristine 0
echo "# FINAL_DONE $(date -u +%FT%TZ)" >> $L
exit 0
