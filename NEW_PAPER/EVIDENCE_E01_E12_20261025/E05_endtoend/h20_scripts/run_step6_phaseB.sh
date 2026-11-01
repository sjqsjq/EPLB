#!/bin/bash
# Step-6 Phase B: formal comparison, 2 starts/cell, round-robin 2 cycles (protocol §4).
# F-local(c) = argmin total_time_s over Phase-A grid on c; tie -> smaller W (pre-registered).
# If argmin=16 -> FL merges with FG for that condition (protocol §3): only one fixed-16 cell.
# Mechanical fallback (not result-driven): if a cond has no usable grid cell, FL:=16 merged, noted.
source "$(dirname "$0")/step6_common.sh"
CLOG=$S6/phaseB.log
verify_hashes || { echo "HASH ABORT" | tee -a "$CLOG"; exit 1; }
declare -A FLW
for cond in W0 W1 W2; do
  best=$(python3 - "$S6" "$cond" <<'PY'
import json,sys,os
S6,cond=sys.argv[1],sys.argv[2]
cands=[]
for W in (8,16,32,64):
    p=f"{S6}/phaseA/{cond}_f{W:02d}__s1/bench.json"
    if os.path.exists(p):
        d=json.load(open(p))
        if d.get("ok")=={"W0":8037,"W1":4000,"W2":8000}[cond] and d.get("errors")==0:
            cands.append((d["total_time_s"],W))
print(min(cands)[1] if cands else 16)  # tie: min() picks first inserted = smaller W when equal? no-
PY
)
  # enforce tie->smaller W explicitly (min on (time,W) tuple does exactly that)
  FLW[$cond]=$best
  echo "[$(date -Is)] F-local($cond) = W$best" | tee -a "$CLOG"
done
cells=()
for cond in W0 W1 W2; do
  cells+=("$cond:V2:16")
  cells+=("$cond:FG:16")
  [ "${FLW[$cond]}" != "16" ] && cells+=("$cond:FL:${FLW[$cond]}")
done
echo "[$(date -Is)] PHASE_B cells: ${cells[*]}" | tee -a "$CLOG"
for cycle in 1 2; do
  for cell in "${cells[@]}"; do
    IFS=: read -r cond arm W <<< "$cell"
    mode=fixed; [ "$arm" = "V2" ] && mode=v2
    run_start "phaseB/${cond}_${arm}__s${cycle}" "s6b_${cond}_${arm,,}_s${cycle}" "$cond" "$mode" "$W" "$CLOG"
  done
done
echo "[$(date -Is)] PHASE_B_DONE" | tee -a "$CLOG"
