#!/bin/bash
# Usage: run_rge2e_boot.sh LABEL ARM "PARAMS" DS_PATH DS_SHA NREQ CONC BENCH_TIMEOUT
# recovery_admission_e2e_20261027 boot runner.
# Cloned from abl_prover_sd_20261026/run_abl_boot_p2.sh (frozen phase-2
# pattern). Deltas (PROTOCOL.md sec.2/6): campaign paths; guards extended
# to sp_state (AERO-RG patched sha) + window_v2 + launch_arm_rge2e.sh +
# three placement artifacts; bracket-safe pkill patterns; per-boot LEDGER
# rows; policy field (AERO-base/AERO-RG) recorded in boot_meta.
# AUTHORIZATION: author task directive 2026-10-26 (prep <=3, formal 4x7=28,
# zero-data INFRA restarts <=2). Facts-only mech check; no auto-retries.
LABEL=$1; ARM=$2; PARAMS=$3; DS=$4; DSSHA=$5; NREQ=$6; CONC=$7; BENCH_TIMEOUT=$8
A2=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
PBPKG=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
ART=$A2/boots/$LABEL
mkdir -p "$ART"
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$A2/LEDGER_rge2e.tsv"; }
# 1) stream integrity (frozen x4 stream)
ACTUAL_SHA=$(sha256sum "$DS" | cut -d" " -f1)
if [ "$ACTUAL_SHA" != "$DSSHA" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"stream_sha_mismatch\",\"expected\":\"$DSSHA\",\"actual\":\"$ACTUAL_SHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT stream_sha_mismatch"; exit 2
fi
# 2) code guards: frozen async_swapper/controller/window_v2 + AERO-RG sp_state
ASHA=$(sha256sum $PBPKG/async_swapper.py | cut -d" " -f1)
CSHA=$(sha256sum $PBPKG/controller.py | cut -d" " -f1)
SSHA=$(sha256sum $PBPKG/sp_state.py | cut -d" " -f1)
WSHA=$(sha256sum $PBPKG/window_v2.py | cut -d" " -f1)
if [ "$ASHA" != "2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f" ] || \
   [ "$CSHA" != "4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697" ] || \
   [ "$SSHA" != "525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466" ] || \
   [ "$WSHA" != "810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"code_sha_mismatch\",\"async\":\"$ASHA\",\"controller\":\"$CSHA\",\"sp_state\":\"$SSHA\",\"window_v2\":\"$WSHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT code_sha_mismatch sp_state=$SSHA window_v2=$WSHA"; exit 2
fi
# 3) tool guards: campaign launcher + frozen bench client
LSHA=$(sha256sum $A2/scripts/launch_arm_rge2e.sh | cut -d" " -f1)
BSHA=$(sha256sum $F/scripts/bench_client_v2.py | cut -d" " -f1)
if [ "$LSHA" != "c2c272594f1851df2ff829f493f8717bc2072fdf2db30fe2b710975899916e18" ] || \
   [ "$BSHA" != "066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"tool_sha_mismatch\",\"launcher\":\"$LSHA\",\"bench_client\":\"$BSHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT tool_sha_mismatch"; exit 2
fi
# 4) placement guards (static arms material, reuse-locked)
PL1=$(sha256sum $A2/placements/datafore_prover_placement.json | cut -d" " -f1)
PL2=$(sha256sum $A2/placements/datafore_remap_placement.json | cut -d" " -f1)
PL3=$(sha256sum $A2/placements/placement_pinned_fair.json | cut -d" " -f1)
if [ "$PL1" != "c1787ad19f16ef5d840d5a7c7e364ecf23fb3caafba682e56c0f21991a298f29" ] || \
   [ "$PL2" != "07f913570a9c7e0518e7ef39718b38e9f93405678a0a0eb2a7c3adab81e5d85e" ] || \
   [ "$PL3" != "299cea283e93efe69d3aec8ab71f210908d8b262d45f602079ed3daebe2a2a60" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"placement_sha_mismatch\",\"prover\":\"$PL1\",\"remap\":\"$PL2\",\"moetuner\":\"$PL3\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT placement_sha_mismatch"; exit 2
fi
POLICY="n-a"
if [ "$ARM" = "P" ]; then
  if [ "$PARAMS" = "1" ]; then POLICY="AERO-RG"; else POLICY="AERO-base"; fi
fi
echo "{\"label\":\"$LABEL\",\"purpose\":\"rge2e\",\"arm\":\"$ARM\",\"params\":\"$PARAMS\",\"policy\":\"$POLICY\",\"ds\":\"$DS\",\"ds_sha256\":\"$ACTUAL_SHA\",\"nreq\":$NREQ,\"conc\":$CONC,\"bench_timeout\":$BENCH_TIMEOUT,\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"code_sp_state\":\"$SSHA\",\"code_window_v2\":\"$WSHA\",\"launch_arm_rge2e_sha\":\"$LSHA\",\"bench_client_sha\":\"$BSHA\",\"placement_prover\":\"$PL1\",\"placement_remap\":\"$PL2\",\"placement_moetuner\":\"$PL3\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
ledger "$LABEL" "START arm=$ARM policy=$POLICY"
env_clear() { unset $(env | grep -oE "^(OEPLB|EA)_[A-Z0-9_]+" | tr "\n" " ") 2>/dev/null; }
kill_server_quiet() {
  pkill -9 -f "sglang[.]launch[_]server" 2>/dev/null; pkill -9 -f "sglang[:][:]" 2>/dev/null
  for i in $(seq 1 90); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 1000 ] && return 0
    sleep 4
  done; return 1
}
env_clear; kill_server_quiet; env_clear
ART=$ART setsid nohup bash $A2/scripts/launch_arm_rge2e.sh $ARM $PARAMS > "$ART/server.log" 2>&1 &
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
  python3 $A2/scripts/mech_check_rge2e.py "$ART" "$LABEL" "$NREQ" --no-bench
  ledger "$LABEL" "READY_FAIL arm=$ARM policy=$POLICY"
  exit 1
fi
sleep 5
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout $BENCH_TIMEOUT \
  python3 bench_client_v2.py $LABEL "$DS" $CONC) > "$ART/bench.log" 2>&1
BRC=$?
echo "[$LABEL] bench rc=$BRC $(date -u +%FT%TZ)"
sleep 15
kill_server_quiet
python3 $A2/scripts/mech_check_rge2e.py "$ART" "$LABEL" "$NREQ"
MRC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc\":$BRC,\"mech_check_rc\":$MRC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
ledger "$LABEL" "DONE arm=$ARM policy=$POLICY bench_rc=$BRC mech_rc=$MRC"
exit $MRC
