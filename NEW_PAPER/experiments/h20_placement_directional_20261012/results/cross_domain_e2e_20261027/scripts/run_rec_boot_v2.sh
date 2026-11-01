#!/bin/bash
# run_rec_boot_v2.sh - prep profile-recording boot driver v2.
# Flow: guards -> launch (stat+normal recipe) -> ready -> HTTP arm recording
# -> bench full E3 (8192, conc 256) -> HTTP stop+dump -> collect .pt (8 ranks)
# -> sha -> hard kill. Prep boot #2 (v1 zero-data recipe failure logged).
# AUTHORIZATION: author confirmation 2026-10-27 (prep <=3).
set -u
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
PBPKG=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
DS=$F/B_data/streams/e3_multidomain_switch.jsonl
ART=$X/prep/record_boot2
L=$X/LEDGER_xd.tsv
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$L"; }
mkdir -p "$ART"
ACTUAL_SHA=$(sha256sum "$DS" | cut -d" " -f1)
[ "$ACTUAL_SHA" = "d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879" ] || { ledger rec_boot_v2 "ABORT stream_sha_mismatch $ACTUAL_SHA"; exit 2; }
ASHA=$(sha256sum $PBPKG/async_swapper.py | cut -d" " -f1)
CSHA=$(sha256sum $PBPKG/controller.py | cut -d" " -f1)
SSHA=$(sha256sum $PBPKG/sp_state.py | cut -d" " -f1)
WSHA=$(sha256sum $PBPKG/window_v2.py | cut -d" " -f1)
BSHA=$(sha256sum $F/scripts/bench_client_v2.py | cut -d" " -f1)
LSHA=$(sha256sum $X/scripts/launch_rec_e3_v2.sh | cut -d" " -f1)
if [ "$ASHA" != "2e4652a0306cb3152354894ccf25c8c1858b6e0e366b56d8138bfa1a16c46e5f" ] || \
   [ "$CSHA" != "4964cfc8e8af397a57764453c4538a35a52abdf545d86ad432f7a35b5b850697" ] || \
   [ "$SSHA" != "525e88f07e9d0cb54633b71e51b3a477f18af086ab9d625787b3d5e9ebe3b466" ] || \
   [ "$WSHA" != "810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74" ] || \
   [ "$BSHA" != "066682433d6a83ba641c34cb6ccb6a328bf306dfedf7a6263263539be733b18c" ]; then
  ledger rec_boot_v2 "ABORT code_sha_mismatch A=$ASHA C=$CSHA S=$SSHA W=$WSHA B=$BSHA"; exit 2
fi
echo "{\"label\":\"rec_e3_v2\",\"purpose\":\"profile_recording\",\"recipe\":\"stat+normal+graph128+manual_arm (launch_profile.sh proven)\",\"ds_sha256\":\"$ACTUAL_SHA\",\"code_async_swapper\":\"$ASHA\",\"code_controller\":\"$CSHA\",\"code_sp_state\":\"$SSHA\",\"code_window_v2\":\"$WSHA\",\"bench_client_sha\":\"$BSHA\",\"launch_rec_v2_sha\":\"$LSHA\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
ledger rec_boot_v2 "START launcher_sha=$LSHA"
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
ART=$ART setsid nohup bash $X/scripts/launch_rec_e3_v2.sh > "$ART/server.log" 2>&1 &
SPID=$!; echo $SPID > "$ART/server.pid"
READY=no
for i in $(seq 1 270); do
  grep -q "ready to roll" "$ART/server.log" 2>/dev/null && { READY=yes; break; }
  grep -qE "Traceback|CUDA out of memory|RuntimeError" "$ART/server.log" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || break
  sleep 4
done
if [ $READY != yes ]; then
  ledger rec_boot_v2 "READY_FAIL after $((i*4))s"
  kill_server_hard; exit 1
fi
ledger rec_boot_v2 "READY after $((i*4))s"
sleep 5
curl -s -m 60 -X POST http://localhost:30000/start_expert_distribution_record > "$ART/rec_arm_start.json" 2>&1
ledger rec_boot_v2 "ARMED resp=$(head -c 120 $ART/rec_arm_start.json)"
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout 900 \
  python3 bench_client_v2.py rec_e3_v2 "$DS" 256) > "$ART/bench.log" 2>&1
BRC=$?
ledger rec_boot_v2 "BENCH rc=$BRC"
sleep 10
curl -s -m 60 -X POST http://localhost:30000/stop_expert_distribution_record > "$ART/rec_arm_stop.json" 2>&1
curl -s -m 300 -X POST http://localhost:30000/dump_expert_distribution_record > "$ART/rec_arm_dump.json" 2>&1
ledger rec_boot_v2 "DUMP_CALLED stop=$(head -c 60 $ART/rec_arm_stop.json) dump=$(head -c 60 $ART/rec_arm_dump.json)"
PT=no; PREV=-1; STABLE=0
for j in $(seq 1 60); do
  sleep 4
  N=$(ls "$ART" | grep -c "expert_distribution_recorder.*[.]pt" || true)
  if [ "${N:-0}" -ge 1 ] && [ "$N" = "$PREV" ]; then STABLE=$((STABLE+1)); else STABLE=0; fi
  PREV=$N
  [ "$STABLE" -ge 3 ] && { PT=yes; break; }
done
kill_server_hard
NPT=$(ls "$ART" | grep -c "expert_distribution_recorder.*[.]pt" || true)
sha256sum "$ART"/expert_distribution_recorder*.pt > "$ART/pt_sha256.txt" 2>/dev/null || true
PT0SHA=$(head -1 "$ART/pt_sha256.txt" 2>/dev/null | cut -d" " -f1)
echo "{\"label\":\"rec_e3_v2\",\"bench_rc\":$BRC,\"pt_found\":\"$PT\",\"pt_count\":$NPT,\"pt0_sha256\":\"$PT0SHA\",\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
ledger rec_boot_v2 "DONE bench_rc=$BRC pt=$PT count=$NPT pt0_sha=$PT0SHA"
