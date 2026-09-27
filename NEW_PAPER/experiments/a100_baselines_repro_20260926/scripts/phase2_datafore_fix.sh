#!/bin/bash
# Phase 2: after run_all completes — collect results, re-run datafore with the
# PROVER-ALIGNED placement (archived datafore_remap_placement.json is distribution-
# mismatched: cos=0.31 vs bench profile P; datafore_prover_placement.json cos=0.9998),
# then restore git-tracked H20 archive files.
set -uo pipefail
RES=/workspace/EPLB/OEPLB/benchmarks/results
A100=/workspace/EPLB/OEPLB/baselines/a100

echo "[phase2] waiting for phase1 ALL_BASELINES_DONE..."
while ! grep -q ALL_BASELINES_DONE /workspace/logs/run_all_a100.log 2>/dev/null; do sleep 60; done
sleep 30

echo "[phase2] collecting phase1 bare-name results -> a100_ names"
for m in moetuner eplb_static eplb_dyn; do
  for r in 1 2 3; do
    [ -f $RES/_0914_${m}_r${r}.json ]  && cp $RES/_0914_${m}_r${r}.json  $RES/_0914_a100_${m}_r${r}.json
    [ -f $RES/_freq6_${m}_r${r}.json ] && cp $RES/_freq6_${m}_r${r}.json $RES/_freq6_a100_${m}_r${r}.json
  done
done

echo "[phase2] switching datafore placement -> PLACEMENT_PROVER (corrected)"
sed -i 's|EXTRA="--init-expert-location \$PLACEMENT_DATAFORE" ;;|EXTRA="--init-expert-location $PLACEMENT_PROVER" ;;|' $A100/launch_a100.sh
grep -n "datafore" -A1 $A100/launch_a100.sh | head -4

for T in 0914 freq6; do
  bash $A100/driver_baselines_a100.sh datafore $T 3 || echo "!!!! FAILED datafore $T"
  sleep 15
done

echo "[phase2] collecting corrected datafore results"
for r in 1 2 3; do
  cp $RES/_0914_datafore_r${r}.json  $RES/_0914_a100_datafore_r${r}.json 2>/dev/null
  cp $RES/_freq6_datafore_r${r}.json $RES/_freq6_a100_datafore_r${r}.json 2>/dev/null
done

echo "[phase2] restoring git-tracked H20 originals"
cd /workspace/EPLB
git checkout -- OEPLB/benchmarks/results/_0914_identity_r1.json OEPLB/benchmarks/results/_0914_identity_r2.json OEPLB/benchmarks/results/_0914_identity_r3.json \
  OEPLB/benchmarks/results/_freq6_identity_r1.json OEPLB/benchmarks/results/_freq6_identity_r2.json OEPLB/benchmarks/results/_freq6_identity_r3.json \
  OEPLB/benchmarks/results/_0914_moetuner_r1.json OEPLB/benchmarks/results/_0914_moetuner_r2.json OEPLB/benchmarks/results/_0914_moetuner_r3.json \
  OEPLB/benchmarks/results/_freq6_moetuner_r1.json OEPLB/benchmarks/results/_freq6_moetuner_r2.json OEPLB/benchmarks/results/_freq6_moetuner_r3.json
git status -s OEPLB/benchmarks/results/ | head
echo PHASE2_DONE
