#!/bin/bash
# Usage: run_g3_boot.sh LABEL ARM DS DSSHA NREQ CONC BENCH_TIMEOUT S1LABEL [DS2 DSSHA2 NREQ2 S2LABEL]
# pf_all_phase_select_20261030 boot runner. Cloned from run_pb_boot.sh
# (PBS archive, frozen) by BYTE-SPLICE: code guards (4 frozen pb_oeplb
# shas), env_clear/kill_server_quiet, ready loop are verbatim mother
# bytes. Deltas (PROTOCOL_g3.md sec.4/6): campaign paths + LEDGER_g3;
# launcher = launch_arm_g3.sh (sha below, computed at materialization);
# placement guards REPLACED by stream-lock membership guard (frozen shas
# in code_lock/stream_shas.env, written once); optional second bench
# segment per boot (smoke covers both tasks in one boot, per-task
# accounting via segment_meta.jsonl with v2p1/server.log line offsets);
# mech_check_pb.py (PBS-scoped) replaced by check_g3_boot.py modes
# smoke|formal|readyfail. Facts-only; no auto-retries.
LABEL=$1; ARM=$2; DS=$3; DSSHA=$4; NREQ=$5; CONC=$6; BENCH_TIMEOUT=$7; S1LABEL=$8
DS2=${9:-}; DSSHA2=${10:-}; NREQ2=${11:-0}; S2LABEL=${12:-}
G3=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/pf_all_phase_select_20261030
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
PBPKG=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
ART=$G3/boots/$LABEL
mkdir -p "$ART"
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$G3/LEDGER_g3.tsv"; }
# 1) stream integrity: every bench segment stream must match its frozen sha
ACTUAL_SHA=$(sha256sum "$DS" | cut -d" " -f1)
if [ "$ACTUAL_SHA" != "$DSSHA" ]; then
  echo "{\"error\":\"stream_sha_mismatch\",\"expected\":\"$DSSHA\",\"actual\":\"$ACTUAL_SHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT stream_sha_mismatch"; exit 2
fi
if [ -n "$DS2" ]; then
  ACTUAL_SHA2=$(sha256sum "$DS2" | cut -d" " -f1)
  if [ "$ACTUAL_SHA2" != "$DSSHA2" ]; then
    echo "{\"error\":\"stream2_sha_mismatch\",\"expected\":\"$DSSHA2\",\"actual\":\"$ACTUAL_SHA2\"}" > "$ART/BOOT_ABORT.json"
    ledger "$LABEL" "ABORT stream2_sha_mismatch"; exit 2
  fi
fi
# 2) code guards: frozen async_swapper/controller/window_v2 + patched sp_state (verbatim mother block)
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
LSHA=$(sha256sum $G3/scripts/launch_arm_g3.sh | cut -d" " -f1)
BSHA=$(sha256sum $F/scripts/bench_client_v2.py | cut -d" " -f1)
if [ "$LSHA" != "c12c374b78e18b42e0153cca526bcc98995c3f0555ea8cf66d7dd48cdc860009" ] || \
   [ "$BSHA" != "066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c" ]; then
  echo "{\"error\":\"tool_sha_mismatch\",\"launcher\":\"$LSHA\",\"bench_client\":\"$BSHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT tool_sha_mismatch"; exit 2
fi
# 4) stream-lock membership guard (replaces PBS placement guards)
LOCK=$G3/code_lock/stream_shas.env
if [ ! -f "$LOCK" ]; then
  echo "{\"error\":\"stream_lock_missing\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT stream_lock_missing"; exit 2
fi
. "$LOCK"
inlock(){ [ "$1" = "$SM_ARC_FULL" ] || [ "$1" = "$SM_PROVER_FULL" ] || [ "$1" = "$SM_ARC_SMOKE" ] || [ "$1" = "$SM_PROVER_SMOKE" ]; }
if ! inlock "$DSSHA"; then
  echo "{\"error\":\"stream_not_in_lock\",\"sha\":\"$DSSHA\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT stream_not_in_lock"; exit 2
fi
if [ -n "$DS2" ] && ! inlock "$DSSHA2"; then
  echo "{\"error\":\"stream2_not_in_lock\",\"sha\":\"$DSSHA2\"}" > "$ART/BOOT_ABORT.json"
  ledger "$LABEL" "ABORT stream2_not_in_lock"; exit 2
fi
echo "{\"label\":\"$LABEL\",\"purpose\":\"g3_phase_select\",\"arm\":\"$ARM\",\"ds\":\"$DS\",\"ds_sha256\":\"$ACTUAL_SHA\",\"nreq\":$NREQ,\"conc\":$CONC,\"bench_timeout\":$BENCH_TIMEOUT,\"s1label\":\"$S1LABEL\",\"ds2\":\"$DS2\",\"ds2_sha256\":\"${ACTUAL_SHA2:-}\",\"nreq2\":$NREQ2,\"s2label\":\"$S2LABEL\",\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"code_sp_state\":\"$SSHA\",\"code_window_v2\":\"$WSHA\",\"launch_arm_g3_sha\":\"$LSHA\",\"bench_client_sha\":\"$BSHA\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
ledger "$LABEL" "START arm=$ARM"
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
ART=$ART setsid nohup bash $G3/scripts/launch_arm_g3.sh $ARM > "$ART/server.log" 2>&1 &
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
  python3 $G3/scripts/check_g3_boot.py "$ART" "$LABEL" readyfail
  ledger "$LABEL" "READY_FAIL arm=$ARM"
  exit 1
fi
sleep 5
seg(){ # $1=seglabel $2=ds $3=nreq
  SL=$1; SDS=$2; SN=$3
  V1=$(wc -l < "$ART/v2p1_rank0.jsonl" 2>/dev/null || echo 0)
  G1=$(wc -l < "$ART/server.log" 2>/dev/null || echo 0)
  T1=$(date -u +%FT%TZ)
  (cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout $BENCH_TIMEOUT \
    python3 bench_client_v2.py "$SL" "$SDS" $CONC) > "$ART/bench_$SL.log" 2>&1
  BRC=$?
  T2=$(date -u +%FT%TZ)
  V2=$(wc -l < "$ART/v2p1_rank0.jsonl" 2>/dev/null || echo 0)
  G2=$(wc -l < "$ART/server.log" 2>/dev/null || echo 0)
  echo "{\"seg\":\"$SL\",\"ds\":\"$SDS\",\"nreq\":$SN,\"bench_rc\":$BRC,\"start\":\"$T1\",\"end\":\"$T2\",\"v2p1_before\":$V1,\"v2p1_after\":$V2,\"slog_before\":$G1,\"slog_after\":$G2}" >> "$ART/segment_meta.jsonl"
  echo "[$LABEL] seg $SL bench rc=$BRC $T2"
  return $BRC
}
ANYBRC=0
seg "$S1LABEL" "$DS" "$NREQ" || ANYBRC=1
if [ -n "$DS2" ]; then
  sleep 5
  seg "$S2LABEL" "$DS2" "$NREQ2" || ANYBRC=1
fi
sleep 15
kill_server_quiet
if [[ "$LABEL" == SM_* ]]; then MODE=smoke; else MODE=formal; fi
python3 $G3/scripts/check_g3_boot.py "$ART" "$LABEL" "$MODE"
MRC=$?
echo "{\"label\":\"$LABEL\",\"any_bench_rc\":$ANYBRC,\"check_rc\":$MRC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
ledger "$LABEL" "DONE arm=$ARM any_bench_rc=$ANYBRC check_rc=$MRC"
exit $MRC
