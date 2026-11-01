#!/bin/bash
# v3 automation chain: wait for ILP map -> boot validation -> freeze v3 -> campaign.
# Every step's artifacts are retained; on any failure the chain STOPS (no retry).
set -u
EXP=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012
SETUP=$EXP/results/fullservice_v2/setup
OUT=$EXP/results/fullservice_v3
CLOG=$OUT/chain.log
mkdir -p "$OUT"
log(){ echo "[$(date -Is)] CHAIN $*" | tee -a "$CLOG"; }

log "waiting for ILP map (pid 788318)"
for i in $(seq 1 540); do
  if [ -f "$SETUP/frozen_map_ilp.json" ] && grep -q "Solved" "$SETUP/ilp_solve_calib.log" 2>/dev/null; then break; fi
  if ! ps -p 788318 > /dev/null 2>&1 && [ ! -f "$SETUP/frozen_map_ilp.json" ]; then
    log "FATAL ILP process gone without producing map"; exit 1
  fi
  sleep 10
done
if [ ! -f "$SETUP/frozen_map_ilp.json" ]; then log "FATAL map never appeared"; exit 1; fi
log "MAP_READY"

# structural validation: 94x128 bijection
python3 - "$SETUP/frozen_map_ilp.json" <<'PYMAP'
import json, sys
d = json.load(open(sys.argv[1]))["physical_to_logical_map"]
assert len(d) == 94, len(d)
for row in d:
    assert len(row) == 128 and sorted(row) == list(range(128))
print("MAP_STRUCT_OK")
PYMAP
if [ $? -ne 0 ]; then log "FATAL map structural check failed"; exit 1; fi

log "STEP bootval"
bash "$EXP/scripts/bootval_ilp_map.sh" >> "$CLOG" 2>&1
brc=$?
if [ $brc -ne 0 ]; then log "FATAL bootval failed rc=$brc (retained; fallback decision needed)"; exit $brc; fi
log "BOOTVAL_PASS"

log "STEP freeze v3"
python3 "$EXP/scripts/freeze_fullservice_protocol_v3.py" >> "$CLOG" 2>&1
frc=$?
if [ $frc -ne 0 ]; then log "FATAL freeze failed rc=$frc"; exit $frc; fi
log "FREEZE_OK"

log "STEP campaign (9 runs)"
bash "$EXP/scripts/run_fullservice_campaign_v3.sh" >> "$OUT/driver_stdout.log" 2>&1
crc=$?
log "CAMPAIGN_DRIVER_EXIT rc=$crc"
exit $crc
