#!/bin/bash
# Control that driver19 skipped: idA/idB used launch57b_base_g8.sh while bal/r135
# used launch57b_fixed_g8.sh, so "idA vs bal" varied the placement AND whether
# --init-expert-location was used at all.  Pass an IDENTITY map through the flag:
# same placement as idA, same code path as bal/r135.  If this is bit-identical to
# idA, the flag is inert and driver19's perturbation is attributable to placement.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d21] $2 FAILED TO BOOT"; return 1; }
  echo "[d21] $2 ready"
}
boot "$LOGD/launch57b_fixed_g8.sh $LOGD/plc57b_ident.json" st_flagid \
  && (cd $LOGD && python3 corpus_probe.py $LOGD/cp_stat_flagid.json /data/models/Qwen2-57B-A14B-Instruct 200)
pkill -9 -f "$PAT"; echo D21_DONE
