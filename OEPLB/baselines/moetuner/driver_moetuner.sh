#!/bin/bash
# MoETuner baseline full driver.
#
# Runs 3 phases, launching/killing SGLang server between each.  Assumes
# launch_profile.sh, launch_moetuner.sh, solve_ilp.py, and the two dataset
# splits already exist alongside this script.
#
# Phase P (profile-run):  identity launch with MOETUNER_PROFILE=1, bench on
#                         profile-side of the corpus (throwaway result).
# Phase S (solve):        run Gurobi ILP1 over the collected .npz.
# Phase B (benchmark):    launch with --init-expert-location <placement>,
#                         bench 2 rounds on the benchmark-side of the corpus.
#
# Called twice from outside: once for FAIR split (head-profile / tail-bench),
# once for LEAKED variant (full profile / full bench).
#
# usage:  driver_moetuner.sh <variant>
#         where variant is "fair" or "leaked"

set -u
VARIANT="${1:?variant: fair or leaked}"

BASE=/workspace/EPLB/OEPLB/baselines/moetuner
LOGD=/workspace/logs
PAT="sglang.launch_server"
BENCHDIR=/workspace/EPLB/OEPLB/scripts
FULL_DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
HEAD_DS=$BASE/artifacts/L512_O1_realprover_head2048.jsonl
TAIL_DS=$BASE/artifacts/L512_O1_realprover_tail6144.jsonl

case "$VARIANT" in
  fair)
    PROFILE_DS=$HEAD_DS
    BENCH_DS=$TAIL_DS
    ;;
  leaked)
    PROFILE_DS=$FULL_DS
    BENCH_DS=$FULL_DS
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2; exit 1;;
esac

PROFILE_DIR=$BASE/artifacts/profile_${VARIANT}
PLACEMENT=$BASE/artifacts/placement_${VARIANT}.json
mkdir -p $PROFILE_DIR $BASE/logs

boot () {
  local script=$1; local tag=$2
  pkill -9 -f "$PAT" 2>/dev/null; sleep 8
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 2000 ] && break
    sleep 3
  done
  sleep 12
  setsid nohup bash $script > $LOGD/server235b_${tag}.log 2>&1 &
  for i in $(seq 1 360); do
    grep -q "ready to roll" $LOGD/server235b_${tag}.log && break
    grep -q "Scheduler hit an exception" $LOGD/server235b_${tag}.log && { echo "[mt/$VARIANT] $tag CRASH"; return 1; }
    sleep 4
  done
  grep -q "ready to roll" $LOGD/server235b_${tag}.log || { echo "[mt/$VARIANT] $tag boot FAILED"; return 1; }
  sleep 6
  echo "[mt/$VARIANT] $tag ready"
}

# --- Phase P: profile ---
echo "[mt/$VARIANT] Phase P: profile $PROFILE_DS"
export MOETUNER_PROFILE_DIR=$PROFILE_DIR
boot $BASE/launch_profile.sh "profile_${VARIANT}" || exit 1
( cd $BENCHDIR && python3 run_grid_bench.py "_mt_${VARIANT}_profile" $PROFILE_DS 256 > $BASE/logs/bench_profile_${VARIANT}.log 2>&1 )
echo "[mt/$VARIANT] profile bench done"
# graceful kill so the tracer flushes on exit (SGLang default trap handles it)
pkill -f "$PAT" 2>/dev/null; sleep 20
pkill -9 -f "$PAT" 2>/dev/null; sleep 5

# Merge chunks: last chunk written contains cumulative P/R.  Rank 0 is
# what we need for ILP1 (P/R are identical across ranks in identity layout).
echo "[mt/$VARIANT] Phase S: solve ILP1"
LAST_CHUNK=$(ls -1 $PROFILE_DIR/rank0_chunk*.npz 2>/dev/null | sort -V | tail -1)
if [ -z "$LAST_CHUNK" ]; then
    # No chunks (profile too small).  Force a manual flush.
    echo "[mt/$VARIANT] no chunks found -- profile ran too short.  Check $PROFILE_DIR"
    ls $PROFILE_DIR
    exit 1
fi
echo "[mt/$VARIANT] using profile: $LAST_CHUNK"
python3 $BASE/src/solve_ilp.py \
  --profile $LAST_CHUNK \
  --num-clusters 8 \
  --experts-per-cluster 16 \
  --time-limit 60 \
  --mip-gap 0.005 \
  --out $PLACEMENT \
  2>&1 | tee $BASE/logs/ilp_solve_${VARIANT}.log
[ -f $PLACEMENT ] || { echo "solve failed"; exit 1; }

# --- Phase B: bench ---
run_bench () {
    local rd=$1; local tag="mt_${VARIANT}_r${rd}"
    export PLACEMENT
    boot $BASE/launch_moetuner.sh $tag || return 1
    ( cd $BENCHDIR && python3 run_grid_bench.py "_${tag}" $BENCH_DS 256 > $BASE/logs/bench_${tag}.log 2>&1 )
    echo "[mt/$VARIANT] r${rd} done"
}

echo "[mt/$VARIANT] Phase B: benchmark on $BENCH_DS"
run_bench 1
run_bench 2

pkill -9 -f "$PAT" 2>/dev/null
echo "[mt/$VARIANT] ALL_DONE"
