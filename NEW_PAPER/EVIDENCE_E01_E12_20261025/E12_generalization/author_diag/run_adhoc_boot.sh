#!/bin/bash
# Usage: run_adhoc_boot.sh LABEL ARM "PARAMS" MODE MIN_SWAPS
LABEL=$1; ARM=$2; PARAMS=$3; MODE=$4; MIN_SWAPS=${5:-0}
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
E3=$F/B_data/streams/e3_multidomain_switch.jsonl
DSSHA=d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879
ART=$G/author_diag_20261021/boots/$LABEL
mkdir -p "$ART"
[ "$(sha256sum $E3 | cut -d' ' -f1)" = "$DSSHA" ] || { echo "{\"error\":\"stream_sha\"}" > $ART/BOOT_ABORT.json; exit 2; }
DEP=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
ASHA=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); CSHA=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
echo "{\"label\":\"$LABEL\",\"arm\":\"$ARM\",\"mode\":\"$MODE\",\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
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
  python3 $G/phase3_prep/assert_prep_boot.py "$ART" "$LABEL" 8192 "$MODE" "$MIN_SWAPS" --no-bench
  echo "{\"label\":\"$LABEL\",\"outcome\":\"BOOT_FAIL\",\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
  exit 1
fi
sleep 5
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout 1200 \
  python3 bench_client_v2.py $LABEL $E3 256) > "$ART/bench.log" 2>&1
BRC=$?
echo "[$LABEL] bench rc=$BRC $(date -u +%FT%TZ)"
sleep 15
kill_server_quiet
python3 $G/phase3_prep/assert_prep_boot.py "$ART" "$LABEL" 8192 "$MODE" "$MIN_SWAPS" 2>/dev/null
ARC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc\":$BRC,\"assert_rc\":$ARC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
exit $BRC
