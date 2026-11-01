#!/bin/bash
# Usage: run_prep_boot.sh LABEL ARM "PARAMS" STREAM NREQ CONC BENCH_TIMEOUT MODE MIN_SWAPS
LABEL=$1; ARM=$2; PARAMS=$3; STREAM=$4; NREQ=$5; CONC=$6; BENCH_TIMEOUT=$7; MODE=$8; MIN_SWAPS=${9:-0}
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
ART=$G/phase3_prep/boots/$LABEL
mkdir -p "$ART"
echo "{\"label\":\"$LABEL\",\"arm\":\"$ARM\",\"params\":\"$PARAMS\",\"stream\":\"$STREAM\",\"nreq\":$NREQ,\"conc\":$CONC,\"bench_timeout\":$BENCH_TIMEOUT,\"mode\":\"$MODE\",\"min_swaps\":$MIN_SWAPS,\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
env_clear() { unset $(env | grep -oE '^(OEPLB|EA)_[A-Z0-9_]+' | tr '\n' ' ') 2>/dev/null; }
kill_server_quiet() {
  pkill -9 -f 'sglang.launch_server' 2>/dev/null; pkill -9 -f 'sglang::' 2>/dev/null
  for i in $(seq 1 90); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 1000 ] && return 0
    sleep 4
  done; return 1
}
env_clear; kill_server_quiet; env_clear
ART=$ART setsid nohup bash $F/scripts/launch_arm.sh $ARM $PARAMS > "$ART/server.log" 2>&1 &
SPID=$!; echo $SPID > "$ART/server.pid"
READY=no
for i in $(seq 1 225); do
  grep -q 'ready to roll' "$ART/server.log" 2>/dev/null && { READY=yes; break; }
  grep -qE 'Traceback|CUDA out of memory|RuntimeError' "$ART/server.log" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || break
  sleep 4
done
echo "[$LABEL] ready=$READY after $((i*4))s $(date -u +%FT%TZ)"
if [ $READY != yes ]; then
  kill_server_quiet
  python3 $G/phase3_prep/assert_prep_boot.py "$ART" "$LABEL" "$NREQ" "$MODE" "$MIN_SWAPS" --no-bench
  exit 1
fi
sleep 5
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout $BENCH_TIMEOUT \
  python3 bench_client_v2.py $LABEL $F/B_data/streams/$STREAM $CONC) > "$ART/bench.log" 2>&1
BRC=$?
echo "[$LABEL] bench rc=$BRC $(date -u +%FT%TZ)"
sleep 15
kill_server_quiet
python3 $G/phase3_prep/assert_prep_boot.py "$ART" "$LABEL" "$NREQ" "$MODE" "$MIN_SWAPS"
ARC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc\":$BRC,\"assert_rc\":$ARC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
exit $ARC
