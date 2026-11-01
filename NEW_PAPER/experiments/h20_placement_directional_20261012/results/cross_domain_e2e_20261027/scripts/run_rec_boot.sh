#!/bin/bash
# run_rec_boot.sh - prep profile-recording boot driver (cross_domain_e2e_20261027).
# Full E3 stream (8192 req, conc 256) against identity base + per_token expert
# distribution recorder; graceful SIGTERM so the recorder dumps its .pt.
# AUTHORIZATION: author confirmation 2026-10-27 (prep <=3). Prep boot #1.
set -u
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
PBPKG=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
DS=$F/B_data/streams/e3_multidomain_switch.jsonl
ART=$X/prep/record_boot
L=$X/LEDGER_xd.tsv
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$L"; }
mkdir -p "$ART"
# guards
ACTUAL_SHA=$(sha256sum "$DS" | cut -d" " -f1)
[ "$ACTUAL_SHA" = "d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879" ] || { ledger rec_boot "ABORT stream_sha_mismatch $ACTUAL_SHA"; exit 2; }
ASHA=$(sha256sum $PBPKG/async_swapper.py | cut -d" " -f1)
CSHA=$(sha256sum $PBPKG/controller.py | cut -d" " -f1)
SSHA=$(sha256sum $PBPKG/sp_state.py | cut -d" " -f1)
WSHA=$(sha256sum $PBPKG/window_v2.py | cut -d" " -f1)
BSHA=$(sha256sum $F/scripts/bench_client_v2.py | cut -d" " -f1)
LSHA=$(sha256sum $X/scripts/launch_rec_e3.sh | cut -d" " -f1)
if [ "$ASHA" != "2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f" ] || \
   [ "$CSHA" != "4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697" ] || \
   [ "$SSHA" != "525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466" ] || \
   [ "$WSHA" != "810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74" ] || \
   [ "$BSHA" != "066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c" ]; then
  ledger rec_boot "ABORT code_sha_mismatch A=$ASHA C=$CSHA S=$SSHA W=$WSHA B=$BSHA"; exit 2
fi
echo "{\"label\":\"rec_e3\",\"purpose\":\"profile_recording\",\"ds_sha256\":\"$ACTUAL_SHA\",\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"code_sp_state\":\"$SSHA\",\"code_window_v2\":\"$WSHA\",\"bench_client_sha\":\"$BSHA\",\"launch_rec_sha\":\"$LSHA\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
ledger rec_boot "START launcher_sha=$LSHA"
env_clear() { unset $(env | grep -oE "^(OEPLB|EA)_[A-Z0-9_]+" | tr "\n" " ") 2>/dev/null; }
kill_server_hard() {
  pkill -9 -f "sglang[.]launch[_]server" 2>/dev/null; pkill -9 -f "sglang[:][:]" 2>/dev/null
  for i in $(seq 1 90); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 1000 ] && return 0
    sleep 4
  done; return 1
}
env_clear; kill_server_hard; env_clear
ART=$ART setsid nohup bash $X/scripts/launch_rec_e3.sh > "$ART/server.log" 2>&1 &
SPID=$!; echo $SPID > "$ART/server.pid"
READY=no
for i in $(seq 1 225); do
  grep -q "ready to roll" "$ART/server.log" 2>/dev/null && { READY=yes; break; }
  grep -qE "Traceback|CUDA out of memory|RuntimeError" "$ART/server.log" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || break
  sleep 4
done
if [ $READY != yes ]; then
  ledger rec_boot "READY_FAIL after $((i*4))s"
  kill_server_hard; exit 1
fi
ledger rec_boot "READY after $((i*4))s"
sleep 5
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout 900 \
  python3 bench_client_v2.py rec_e3 "$DS" 256) > "$ART/bench.log" 2>&1
BRC=$?
ledger rec_boot "BENCH rc=$BRC"
sleep 15
# graceful SIGTERM so recorder saves; wait up to 180s for .pt then hard kill
kill $SPID 2>/dev/null
PT=no
for j in $(seq 1 45); do
  sleep 4
  N=$(ls "$ART" | grep -c "expert_distribution_recorder.*[.]pt" || true)
  [ "${N:-0}" -ge 1 ] && { PT=yes; break; }
done
kill_server_hard
PTFILES=$(ls "$ART" | grep "expert_distribution_recorder.*[.]pt" || true)
PTSHA=$(sha256sum $ART/expert_distribution_recorder*.pt 2>/dev/null | head -1 | cut -d" " -f1)
echo "{\"label\":\"rec_e3\",\"bench_rc\":$BRC,\"pt_found\":\"$PT\",\"pt_files\":\"$PTFILES\",\"pt_sha256\":\"$PTSHA\",\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
ledger rec_boot "DONE bench_rc=$BRC pt=$PT sha=$PTSHA"
