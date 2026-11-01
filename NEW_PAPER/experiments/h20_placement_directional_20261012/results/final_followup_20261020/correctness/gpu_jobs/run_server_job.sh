#!/bin/bash
# One GPU correctness job: real server, original fault path.
# Usage: run_server_job.sh JOB ARM "PARAMS" STREAM NREQ CONC BENCH_TIMEOUT MIN_SWAPS
JOB=$1; ARM=$2; PARAMS=$3; STREAM=$4; NREQ=$5; CONC=$6; BENCH_TIMEOUT=$7; MIN_SWAPS=$8
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
ART=$G/correctness/gpu_jobs/$JOB
mkdir -p "$ART"
echo "{\"job\":\"$JOB\",\"arm\":\"$ARM\",\"params\":\"$PARAMS\",\"stream\":\"$STREAM\",\"nreq\":$NREQ,\"conc\":$CONC,\"bench_timeout\":$BENCH_TIMEOUT,\"min_swaps\":$MIN_SWAPS,\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/job_meta.json"

# clean environment (same as old campaign attempt_cell)
env_clear() { unset $(env | grep -oE '^(OEPLB|EA)_[A-Z0-9_]+' | tr '\n' ' ') 2>/dev/null; }
kill_server_quiet() {
  pkill -9 -f 'sglang.launch_server' 2>/dev/null; pkill -9 -f 'sglang::' 2>/dev/null
  for i in $(seq 1 90); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 1000 ] && return 0
    sleep 4
  done
  return 1
}
env_clear; kill_server_quiet
echo "[$JOB] launch arm=$ARM params=$PARAMS $(date -u +%FT%TZ)"
env_clear
ART=$ART setsid nohup bash $F/scripts/launch_arm.sh $ARM $PARAMS > "$ART/server.log" 2>&1 &
SPID=$!; echo $SPID > "$ART/server.pid"
READY=no
for i in $(seq 1 225); do
  grep -q 'ready to roll' "$ART/server.log" 2>/dev/null && { READY=yes; break; }
  grep -qE 'Traceback|CUDA out of memory|RuntimeError' "$ART/server.log" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || break
  sleep 4
done
echo "[$JOB] ready=$READY after $((i*4))s $(date -u +%FT%TZ)"
if [ $READY != yes ]; then
  kill_server_quiet
  python3 $G/correctness/gpu_jobs/assert_server_job.py "$ART" "$JOB" "$NREQ" "$MIN_SWAPS" --no-bench
  exit 1
fi
sleep 5
echo "[$JOB] bench start $(date -u +%FT%TZ)"
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout $BENCH_TIMEOUT \
  python3 bench_client_v2.py $JOB $F/B_data/streams/$STREAM $CONC) > "$ART/bench.log" 2>&1
BRC=$?
echo "[$JOB] bench rc=$BRC $(date -u +%FT%TZ)"
sleep 20   # let teardown flushes land
kill_server_quiet
python3 $G/correctness/gpu_jobs/assert_server_job.py "$ART" "$JOB" "$NREQ" "$MIN_SWAPS"
ARC=$?
echo "{\"job\":\"$JOB\",\"bench_rc\":$BRC,\"assert_rc\":$ARC,\"end\":\"$(date -u +%FT%TZ)\"}" >> "$ART/job_meta.json.end"
exit $ARC
