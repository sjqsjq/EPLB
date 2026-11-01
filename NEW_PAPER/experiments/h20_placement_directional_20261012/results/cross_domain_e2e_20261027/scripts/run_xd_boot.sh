#!/bin/bash
# Usage: run_xd_boot.sh LABEL ARM "PARAMS" DS_PATH DS_SHA NREQ CONC BENCH_TIMEOUT
# cross_domain_e2e_20261027 boot runner.
# Cloned from recovery_admission_e2e_20261027/run_rge2e_boot.sh (frozen
# pattern). Deltas (PROTOCOL.md sec.2/4/6): campaign paths; frozen E3 stream;
# launcher = launch_arm_xd.sh (placement-path-only delta from rge2e launcher);
# placement guards read frozen shas from code_lock/placement_shas.env (written
# once at profile freeze, never modified; missing lock = BOOT_ABORT); boot_meta
# records per-boot placement shas for mech_check cross-verification.
# AUTHORIZATION: author confirmation 2026-10-27 (prep <=3, formal 24,
# zero-data INFRA restarts <=2). Facts-only mech check; no auto-retries.
LABEL=$1; ARM=$2; PARAMS=$3; DS=$4; DSSHA=$5; NREQ=$6; CONC=$7; BENCH_TIMEOUT=$8
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
PBPKG=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
ART=$X/boots/$LABEL
mkdir -p "$ART"
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$X/LEDGER_xd.tsv"; }
# 1) stream integrity (frozen E3 four-domain stream)
ACTUAL_SHA=$(sha256sum "$DS" | cut -d" " -f1)
if [ "$ACTUAL_SHA" != "$DSSHA" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"stream_sha_mismatch\",\"expected\":\"$DSSHA\",\"actual\":\"$ACTUAL_SHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT stream_sha_mismatch"; exit 2
fi
# 2) code guards: frozen async_swapper/controller/window_v2 + patched sp_state
ASHA=$(sha256sum $PBPKG/async_swapper.py | cut -d" " -f1)
CSHA=$(sha256sum $PBPKG/controller.py | cut -d" " -f1)
SSHA=$(sha256sum $PBPKG/sp_state.py | cut -d" " -f1)
WSHA=$(sha256sum $PBPKG/window_v2.py | cut -d" " -f1)
if [ "$ASHA" != "2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f" ] || \
   [ "$CSHA" != "4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697" ] || \
   [ "$SSHA" != "525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466" ] || \
   [ "$WSHA" != "810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"code_sha_mismatch\",\"async\":\"$ASHA\",\"controller\":\"$CSHA\",\"sp_state\":\"$SSHA\",\"window_v2\":\"$WSHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT code_sha_mismatch"; exit 2
fi
# 3) tool guards: campaign launcher + frozen bench client
LSHA=$(sha256sum $X/scripts/launch_arm_xd.sh | cut -d" " -f1)
BSHA=$(sha256sum $F/scripts/bench_client_v2.py | cut -d" " -f1)
if [ "$LSHA" != "4d007710e9a3a05dcbbec1d5d499a875c51a930793add469526a369cb5014d34" ] || \
   [ "$BSHA" != "066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"tool_sha_mismatch\",\"launcher\":\"$LSHA\",\"bench_client\":\"$BSHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT tool_sha_mismatch"; exit 2
fi
# 4) placement guards (frozen E3 mixed-domain calibration profile)
LOCK=$X/code_lock/placement_shas.env
if [ ! -f "$LOCK" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"placement_lock_missing\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT placement_lock_missing"; exit 2
fi
. "$LOCK"
if [ -z "${PL_ES:-}" ] || [ -z "${PL_DF:-}" ] || [ -z "${PL_MT:-}" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"placement_lock_incomplete\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT placement_lock_incomplete"; exit 2
fi
PL1=$(sha256sum $X/placements/datafore_e3_placement.json | cut -d" " -f1)
PL2=$(sha256sum $X/placements/datafore_e3_remap_placement.json | cut -d" " -f1)
PL3=$(sha256sum $X/placements/placement_e3_fair.json | cut -d" " -f1)
if [ "$PL1" != "$PL_ES" ] || [ "$PL2" != "$PL_DF" ] || [ "$PL3" != "$PL_MT" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"placement_sha_mismatch\",\"es\":\"$PL1\",\"df\":\"$PL2\",\"mt\":\"$PL3\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT placement_sha_mismatch"; exit 2
fi
POLICY="n-a"
if [ "$ARM" = "P" ]; then
  if [ "$PARAMS" = "1" ]; then POLICY="AERO-RG"; else POLICY="AERO-base"; fi
fi
echo "{\"label\":\"$LABEL\",\"purpose\":\"xd_e3\",\"arm\":\"$ARM\",\"params\":\"$PARAMS\",\"policy\":\"$POLICY\",\"ds\":\"$DS\",\"ds_sha256\":\"$ACTUAL_SHA\",\"nreq\":$NREQ,\"conc\":$CONC,\"bench_timeout\":$BENCH_TIMEOUT,\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"code_sp_state\":\"$SSHA\",\"code_window_v2\":\"$WSHA\",\"launch_arm_xd_sha\":\"$LSHA\",\"bench_client_sha\":\"$BSHA\",\"placement_es\":\"$PL1\",\"placement_df\":\"$PL2\",\"placement_mt\":\"$PL3\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
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
ART=$ART setsid nohup bash $X/scripts/launch_arm_xd.sh $ARM $PARAMS > "$ART/server.log" 2>&1 &
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
  python3 $X/scripts/mech_check_xd.py "$ART" "$LABEL" "$NREQ" --no-bench
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
python3 $X/scripts/mech_check_xd.py "$ART" "$LABEL" "$NREQ"
MRC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc\":$BRC,\"mech_check_rc\":$MRC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
ledger "$LABEL" "DONE arm=$ARM policy=$POLICY bench_rc=$BRC mech_rc=$MRC"
exit $MRC
