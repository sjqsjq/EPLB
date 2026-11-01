#!/bin/bash
# Usage: run_abl_boot.sh LABEL ARM "PARAMS" DS_PATH DS_SHA NREQ CONC BENCH_TIMEOUT
# abl_prover_sd_20261026 phase-2 formal boot.
# Cloned from phase4_core/run_supp_boot.sh (frozen pattern) with deltas:
#   - artifacts under abl_prover_sd_20261026/boots/
#   - launcher + bench-client sha guards added (recorded in boot_meta)
#   - assert_prep_boot.py (min-swap gated) REPLACED by mech_check_p2.py
#     (completeness-gated only; mechanism facts never affect rc — PROTOCOL §3 clause 6)
# AUTHORIZATION REQUIRED before any execution (standing directive; see queue_abl.sh header).
LABEL=$1; ARM=$2; PARAMS=$3; DS=$4; DSSHA=$5; NREQ=$6; CONC=$7; BENCH_TIMEOUT=$8
A=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/abl_prover_sd_20261026
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
ART=$A/boots_p2/$LABEL
mkdir -p "$ART"
# 1) stream integrity (frozen dataset sha; DATASET_FREEZE.json)
ACTUAL_SHA=$(sha256sum "$DS" | cut -d" " -f1)
if [ "$ACTUAL_SHA" != "$DSSHA" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"stream_sha_mismatch\",\"expected\":\"$DSSHA\",\"actual\":\"$ACTUAL_SHA\"}" > "$ART/BOOT_ABORT.json"
  exit 2
fi
# 2) code version guard (frozen code_fixed_v2)
ASHA=$(sha256sum /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/async_swapper.py | cut -d" " -f1)
CSHA=$(sha256sum /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/controller.py | cut -d" " -f1)
if [ "$ASHA" != "2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f" ] || \
   [ "$CSHA" != "4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"code_sha_mismatch\",\"async\":\"$ASHA\",\"controller\":\"$CSHA\"}" > "$ART/BOOT_ABORT.json"
  exit 2
fi
# 3) launcher + client guards (frozen phase-4 tools, byte-identical)
LSHA=$(sha256sum $F/scripts/launch_arm.sh | cut -d" " -f1)
BSHA=$(sha256sum $F/scripts/bench_client_v2.py | cut -d" " -f1)
if [ "$LSHA" != "c2f6b9dd6e57c64d48379b8bf7b6c012df0bd9c48c9bd254f5ca16299d5f8ede" ] || \
   [ "$BSHA" != "066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"tool_sha_mismatch\",\"launch_arm\":\"$LSHA\",\"bench_client\":\"$BSHA\"}" > "$ART/BOOT_ABORT.json"
  exit 2
fi
echo "{\"label\":\"$LABEL\",\"purpose\":\"phase2_formal\",\"arm\":\"$ARM\",\"params\":\"$PARAMS\",\"ds\":\"$DS\",\"ds_sha256\":\"$ACTUAL_SHA\",\"nreq\":$NREQ,\"conc\":$CONC,\"bench_timeout\":$BENCH_TIMEOUT,\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"launch_arm_sha\":\"$LSHA\",\"bench_client_sha\":\"$BSHA\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
env_clear() { unset $(env | grep -oE "^(OEPLB|EA)_[A-Z0-9_]+" | tr "\n" " ") 2>/dev/null; }
kill_server_quiet() {
  pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null
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
  grep -q "ready to roll" "$ART/server.log" 2>/dev/null && { READY=yes; break; }
  grep -qE "Traceback|CUDA out of memory|RuntimeError" "$ART/server.log" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || break
  sleep 4
done
echo "[$LABEL] ready=$READY after $((i*4))s $(date -u +%FT%TZ)"
if [ $READY != yes ]; then
  kill_server_quiet
  python3 $A/mech_check_p2.py "$ART" "$LABEL" "$NREQ" --no-bench
  exit 1
fi
sleep 5
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout $BENCH_TIMEOUT \
  python3 bench_client_v2.py $LABEL "$DS" $CONC) > "$ART/bench.log" 2>&1
BRC=$?
echo "[$LABEL] bench rc=$BRC $(date -u +%FT%TZ)"
sleep 15
kill_server_quiet
python3 $A/mech_check_p2.py "$ART" "$LABEL" "$NREQ"
MRC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc\":$BRC,\"mech_check_rc\":$MRC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
exit $MRC
