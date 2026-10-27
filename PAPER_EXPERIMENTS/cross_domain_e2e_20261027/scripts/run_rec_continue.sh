#!/bin/bash
# run_rec_continue.sh - continue prep2 recording boot after curl-missing incident.
# Server alive+healthy (identity layout, recorder unarmed, radix off => the
# aborted first bench pass left zero profile contamination). Steps: health
# gate -> ARM recorder via python urllib (no curl in this env) -> fresh full
# E3 bench (rec_e3_v2b) -> stop+dump -> collect .pt -> sha -> hard kill.
# Same boot remains prep #2. AUTHORIZATION: author confirmation 2026-10-27.
set -u
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
DS=$F/B_data/streams/e3_multidomain_switch.jsonl
ART=$X/prep/record_boot2
L=$X/LEDGER_xd.tsv
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$L"; }
ACTUAL_SHA=$(sha256sum "$DS" | cut -d" " -f1)
[ "$ACTUAL_SHA" = "d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879" ] || { ledger rec_cont "ABORT stream_sha_mismatch"; exit 2; }
python3 -c "import urllib.request as u; r=u.urlopen(u.Request(\"http://localhost:30000/health\"), timeout=10); print(r.status)" > $ART/cont_health.txt 2>&1 || { ledger rec_cont "ABORT server_not_healthy"; exit 1; }
ledger rec_cont "HEALTH ok"
python3 -c "import urllib.request as u; r=u.urlopen(u.Request(\"http://localhost:30000/start_expert_distribution_record\", method=\"POST\"), timeout=60); print(r.status, r.read()[:200])" > $ART/rec_arm_start.json 2>&1
ARC=$?
ledger rec_cont "ARM rc=$ARC resp=$(head -c 100 $ART/rec_arm_start.json)"
if [ $ARC != 0 ]; then ledger rec_cont "ABORT arm_failed see rec_arm_start.json"; exit 1; fi
sleep 3
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout 900 \
  python3 bench_client_v2.py rec_e3_v2b "$DS" 256) > "$ART/bench.log" 2>&1
BRC=$?
ledger rec_cont "BENCH rc=$BRC"
sleep 10
python3 -c "import urllib.request as u; r=u.urlopen(u.Request(\"http://localhost:30000/stop_expert_distribution_record\", method=\"POST\"), timeout=60); print(r.status, r.read()[:200])" > $ART/rec_arm_stop.json 2>&1
SRC=$?
python3 -c "import urllib.request as u; r=u.urlopen(u.Request(\"http://localhost:30000/dump_expert_distribution_record\", method=\"POST\"), timeout=300); print(r.status, r.read()[:200])" > $ART/rec_arm_dump.json 2>&1
DRC=$?
ledger rec_cont "STOP_DUMP stop_rc=$SRC dump_rc=$DRC"
kill_server_hard() {
  pkill -9 -f "sglang[.]launch[_]server" 2>/dev/null; pkill -9 -f "sglang[:][:]" 2>/dev/null
  for i in $(seq 1 90); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 1000 ] && return 0
    sleep 4
  done; return 1
}
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
echo "{\"label\":\"rec_e3_v2b\",\"bench_rc\":$BRC,\"arm_rc\":$ARC,\"stop_rc\":$SRC,\"dump_rc\":$DRC,\"pt_found\":\"$PT\",\"pt_count\":$NPT,\"pt0_sha256\":\"$PT0SHA\",\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
ledger rec_cont "DONE bench_rc=$BRC pt=$PT count=$NPT pt0_sha=$PT0SHA"
