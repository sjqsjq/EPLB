#!/bin/bash
# MoETuner baseline driver — SGLang-native path (no custom hooks).
#
# Phase P: launch identity server with --expert-distribution-recorder-mode stat.
#          curl /start_expert_distribution_record, run bench on profile split,
#          curl /stop and /dump, collect the *.pt files.
# Phase S: pt2npz -> solve_ilp -> placement.json (physical_to_logical_map only)
# Phase B: launch with --init-expert-location placement.json (no pb-oeplb),
#          bench 2 rounds on the benchmark split.
#
# usage: driver_moetuner.sh <fair|leaked>
set -u
VARIANT="${1:?variant: fair or leaked}"

BASE=/workspace/EPLB/OEPLB/baselines/moetuner
LOGD=/workspace/logs
PAT="python3 -m sglang.launch_server"
BENCHDIR=/workspace/EPLB/OEPLB/scripts
FULL_DS=/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl
HEAD_DS=$BASE/artifacts/L512_O1_realprover_head2048.jsonl
TAIL_DS=$BASE/artifacts/L512_O1_realprover_tail6144.jsonl

PINNED_HEAD=$BASE/artifacts/pinned10x_head1024.jsonl
PINNED_TAIL=$BASE/artifacts/pinned10x_tail1024.jsonl
case "$VARIANT" in
  fair)       PROFILE_DS=$HEAD_DS; BENCH_DS=$TAIL_DS ;;
  leaked)     PROFILE_DS=$FULL_DS; BENCH_DS=$FULL_DS ;;
  pinned_fair) PROFILE_DS=$PINNED_HEAD; BENCH_DS=$PINNED_TAIL ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

PROFILE_DIR=$BASE/artifacts/profile_${VARIANT}
PLACEMENT=$BASE/artifacts/placement_${VARIANT}.json
mkdir -p $PROFILE_DIR $BASE/logs
rm -f $PROFILE_DIR/expert_distribution_recorder_*.pt

boot () {
  local script=$1; local tag=$2
  pkill -9 -f "$PAT" 2>/dev/null; sleep 8
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
    [ "$used" -lt 2000 ] && break
    sleep 3
  done
  sleep 12
  setsid nohup bash $script > $LOGD/server235b_${tag}.log 2>&1 </dev/null &
  for i in $(seq 1 360); do
    grep -q "ready to roll" $LOGD/server235b_${tag}.log && break
    grep -q "Scheduler hit an exception" $LOGD/server235b_${tag}.log && { echo "[mt/$VARIANT] $tag CRASH"; return 1; }
    sleep 4
  done
  grep -q "ready to roll" $LOGD/server235b_${tag}.log || { echo "[mt/$VARIANT] $tag boot FAILED"; return 1; }
  sleep 6
  echo "[mt/$VARIANT] $tag ready"
}

# ---- Phase P: profile ----
echo "[mt/$VARIANT] Phase P: profile $PROFILE_DS"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$PROFILE_DIR
boot $BASE/launch_profile.sh "profile_${VARIANT}" || exit 1

# arm the recorder
echo "[mt/$VARIANT] curl /start_expert_distribution_record"
wget --post-data='' -qO- http://127.0.0.1:30000/start_expert_distribution_record | head -c 200; echo

# run the profile-side bench (result thrown away — we only care about routing counts)
( cd $BENCHDIR && python3 run_grid_bench.py "_mt_${VARIANT}_profile" $PROFILE_DS 256 > $BASE/logs/bench_profile_${VARIANT}.log 2>&1 )
echo "[mt/$VARIANT] profile bench done"

echo "[mt/$VARIANT] curl /stop_expert_distribution_record"
wget --post-data='' -qO- http://127.0.0.1:30000/stop_expert_distribution_record | head -c 200; echo
echo "[mt/$VARIANT] curl /dump_expert_distribution_record"
wget --post-data='' -qO- http://127.0.0.1:30000/dump_expert_distribution_record | head -c 200; echo
sleep 3

# graceful kill
pkill -f "$PAT" 2>/dev/null; sleep 20
pkill -9 -f "$PAT" 2>/dev/null; sleep 5

ls -la $PROFILE_DIR/expert_distribution_recorder_*.pt 2>&1 | head -5
PT_COUNT=$(ls $PROFILE_DIR/expert_distribution_recorder_*.pt 2>/dev/null | wc -l)
if [ "$PT_COUNT" -eq 0 ]; then
    echo "[mt/$VARIANT] no .pt produced -- profile FAILED"; exit 1
fi

# ---- Phase S: solve ----
echo "[mt/$VARIANT] Phase S: pt->npz->ILP1"
python3 $BASE/src/pt2npz.py \
  --src-glob "$PROFILE_DIR/expert_distribution_recorder_*.pt" \
  --out $PROFILE_DIR/P.npz 2>&1 | tee $BASE/logs/pt2npz_${VARIANT}.log

python3 $BASE/src/solve_ilp.py \
  --profile $PROFILE_DIR/P.npz \
  --num-clusters 8 \
  --experts-per-cluster 16 \
  --time-limit 60 \
  --mip-gap 0.005 \
  --out $PLACEMENT \
  2>&1 | tee $BASE/logs/ilp_solve_${VARIANT}.log
[ -f $PLACEMENT ] || { echo "solve failed"; exit 1; }

# ---- Phase B: benchmark ----
run_bench () {
    local rd=$1; local tag="mt_${VARIANT}_r${rd}"
    export PLACEMENT
    boot $BASE/launch_moetuner.sh $tag || return 1
    ( cd $BENCHDIR && python3 run_grid_bench.py "_${tag}" $BENCH_DS 256 > $BASE/logs/bench_${tag}.log 2>&1 )
    echo "[mt/$VARIANT] r${rd} done"
}

run_bench_retry () {
    local rd=$1
    for attempt in 1 2; do
        if run_bench $rd; then return 0; fi
        echo "[mt/$VARIANT] r${rd} attempt ${attempt} failed, retrying..."
        pkill -9 -f "$PAT" 2>/dev/null; sleep 10
    done
    return 1
}

echo "[mt/$VARIANT] Phase B: benchmark on $BENCH_DS"
run_bench_retry 1
run_bench_retry 2

pkill -9 -f "$PAT" 2>/dev/null
echo "[mt/$VARIANT] ALL_DONE"
