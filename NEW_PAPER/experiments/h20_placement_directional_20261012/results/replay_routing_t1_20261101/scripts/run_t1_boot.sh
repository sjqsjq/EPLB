#!/bin/bash
# Usage: run_t1_boot.sh LABEL ARM DS_PATH DS_SHA NREQ CONC BENCH_TIMEOUT
# replay_routing_t1_20261101 boot runner. BYTE-SPLICE of run_xd_boot.sh (frozen
# XD mother). PRE-REGISTERED deltas (PROTOCOL_t1 sec.5/6):
#   1) campaign paths; launcher launch_arm_t1.sh (new sha guard);
#   2) args renumbered (no PARAMS in T1); POLICY=t1-ARM;
#   3) placement guard block removed (T1 has no placements; layouts are
#      offline-derived, archived in analysis/layouts);
#   4) bench runs TWICE on the same frozen stream (short repeats):
#      labels LABEL_rep1 / LABEL_rep2 (X caliber, PROTOCOL_t1 sec.1);
#   5) mech_check_t1.py (facts-only: ready, path A args, issued-ops dedup
#      per G3 verified caliber, both rep bench ok==nreq).
# AUTHORIZATION: author 2026-11-01 (T1<=1.3h GPU within 7.5h cap).
# Facts-only mech check; no auto-retries; ready_s = engineering overhead.
LABEL=$1; ARM=$2; DS=$3; DSSHA=$4; NREQ=$5; CONC=$6; BENCH_TIMEOUT=$7; PARAMS=""
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/replay_routing_t1_20261101
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
PBPKG=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
ART=$X/boots/$LABEL
mkdir -p "$ART"
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$X/LEDGER_t1.tsv"; }
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
LSHA=$(sha256sum $X/scripts/launch_arm_t1.sh | cut -d" " -f1)
BSHA=$(sha256sum $F/scripts/bench_client_v2.py | cut -d" " -f1)
if [ "$LSHA" != "c00ee3cd7550a04a7e9bd3a18e47fa886c2c5a7ebf0ce58a761bf849111349d6" ] || \
   [ "$BSHA" != "066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c" ]; then
  echo "{\"label\":\"$LABEL\",\"error\":\"tool_sha_mismatch\",\"launcher\":\"$LSHA\",\"bench_client\":\"$BSHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT tool_sha_mismatch"; exit 2
fi
# 4) placements: none in T1 (offline-derived layouts; PROTOCOL_t1 sec.6).
POLICY="t1-$ARM"
echo "{\"label\":\"$LABEL\",\"purpose\":\"t1_replay\",\"arm\":\"$ARM\",\"ds\":\"$DS\",\"ds_sha256\":\"$ACTUAL_SHA\",\"nreq\":$NREQ,\"conc\":$CONC,\"bench_timeout\":$BENCH_TIMEOUT,\"reps\":2,\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"code_sp_state\":\"$SSHA\",\"code_window_v2\":\"$WSHA\",\"launch_arm_t1_sha\":\"$LSHA\",\"bench_client_sha\":\"$BSHA\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
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
ART=$ART setsid nohup bash $X/scripts/launch_arm_t1.sh $ARM > "$ART/server.log" 2>&1 &
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
  python3 $X/scripts/mech_check_t1.py "$ART" "$LABEL" "$NREQ" --no-bench
  ledger "$LABEL" "READY_FAIL arm=$ARM policy=$POLICY"
  exit 1
fi
sleep 5
BRC=0
for REP in 1 2; do
  SL="${LABEL}_rep${REP}"
  (cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout $BENCH_TIMEOUT \
    python3 bench_client_v2.py "$SL" "$DS" $CONC) > "$ART/bench_$SL.log" 2>&1
  RRC=$?
  echo "[$LABEL] rep=$REP bench rc=$RRC $(date -u +%FT%TZ)"
  if [ $RRC -ne 0 ]; then BRC=1; fi
  sleep 10
done
sleep 15
kill_server_quiet
python3 $X/scripts/mech_check_t1.py "$ART" "$LABEL" "$NREQ"
MRC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc\":$BRC,\"mech_check_rc\":$MRC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
ledger "$LABEL" "DONE arm=$ARM policy=$POLICY bench_rc=$BRC mech_rc=$MRC"
exit $MRC
