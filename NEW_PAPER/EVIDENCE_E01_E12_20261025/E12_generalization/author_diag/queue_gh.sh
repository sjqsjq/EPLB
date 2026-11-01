#!/bin/bash
# Author-directed diag queue #2 (author correction 2026-10-21): the "pristine original" system is
# GitHub origin/main OEPLB/src (git@github.com:sjqsjq/EPLB.git, never pushed over, never modified):
# 8-file package, async_swapper=93fa4ccd (identical to code_lock), controller=ada04359, config=232ecbd5.
# Runs E3 P-vs-I direct throughput on the FULL pristine package, then UNCONDITIONALLY restores v2.
# Pool separation: adhoc_* labels, gh_pristine code column, NOT in any formal/prep budget pool.
# Disclosure: P arm under pristine code executes the pristine controller's own logic with the same
# CLI flags (launch_arm.sh ART=P); campaign v2.1 env knobs (OEPLB_SP_MODE/WINDOW_CONTROLLER/V21_*)
# are inert on pristine code by construction.
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
BK=/tmp/dep_pre_gh_backup

# ---- phase 0: wait for in-flight adhoc queue (adhoc_I_orig_E3 + its trap restore) to finish ----
for i in $(seq 1 240); do
  pgrep -f 'queue_adhoc\.sh|run_adhoc_boot\.sh' > /dev/null 2>&1 || break
  sleep 10
done
sleep 20  # let trap-restore settle
grep -q 'RESTORE_v2' $L || { echo -e "# GH_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tno_RESTORE_v2_row" >> $L; exit 2; }
A0=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1)
C0=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
if [ "$A0" != "$V2_A" ] || [ "$C0" != "$V2_C" ]; then
  echo -e "# GH_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tdeployed_not_v2_a=${A0:0:16}_c=${C0:0:16}" >> $L; exit 2
fi

restore_v2_full() {
  rm -rf $DEP
  cp -a $BK $DEP
  rm -rf $DEP/__pycache__
  local a=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1) c=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
  if [ "$a" = "$V2_A" ] && [ "$c" = "$V2_C" ]; then
    echo -e "RESTORE_v2_full\t-\t$(date -u +%FT%TZ)\t0\t-\tOK" >> $L
  else
    echo -e "RESTORE_v2_full\t-\t$(date -u +%FT%TZ)\t1\t-\tFAIL_a=${a:0:16}_c=${c:0:16}" >> $L
  fi
}

# ---- phase 1: back up FULL deployed dir, swap in exactly the 8 pristine files ----
rm -rf $BK
cp -a $DEP $BK || { echo -e "# GH_ABORT\t-\t$(date -u +%FT%TZ)\t1\t-\tbackup_failed" >> $L; exit 2; }
trap restore_v2_full EXIT
find $DEP -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp $GH/__init__.py $GH/async_swapper.py $GH/config.py $GH/controller.py \
   $GH/fast_metadata.py $GH/per_request_tracer.py $GH/rebalancer.py $GH/routing_tracer.py $DEP/
rm -rf $DEP/__pycache__
AS=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1)
CS=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
GS=$(sha256sum $DEP/config.py | cut -d' ' -f1)
N=$(ls $DEP | wc -l)
if [ "$AS" != "$GH_A" ] || [ "$CS" != "$GH_C" ] || [ "$GS" != "$GH_CFG" ] || [ "$N" != "8" ]; then
  echo -e "# GH_SWAP_FAIL\t-\t$(date -u +%FT%TZ)\t1\t-\ta=${AS:0:16}_c=${CS:0:16}_cfg=${GS:0:16}_n=$N" >> $L
  exit 2   # trap restores v2
fi
echo "# SWAPPED_TO_GH_PRISTINE $(date -u +%FT%TZ) a=${AS:0:16} c=${CS:0:16} cfg=${GS:0:16} n=$N" >> $L

run() {
  local LAB=$1 ARM=$2 PAR=$3 MODE=$4 MS=$5 CODE=$6
  local S=$(date -u +%FT%TZ) RC TPS
  bash $D/run_adhoc_boot.sh "$LAB" "$ARM" "$PAR" "$MODE" "$MS"
  RC=$?
  TPS=$(python3 -c "import json;print(round(json.load(open('$D/boots/$LAB/$LAB.json'))['tps'],2))" 2>/dev/null || echo NA)
  echo -e "$LAB\t$S\t$(date -u +%FT%TZ)\t$RC\t$TPS\t$CODE" >> $L
}

# ---- phase 2: E3 direct-throughput boots on pristine code ----
run adhoc_P_gh_E3 P "" pb 4 gh_pristine
run adhoc_I_gh_E3 I "" plain 0 gh_pristine
# restore happens via trap EXIT
echo "# GH_QUEUE_DONE $(date -u +%FT%TZ)" >> $L
exit 0
