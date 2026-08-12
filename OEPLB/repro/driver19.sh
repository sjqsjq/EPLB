#!/bin/bash
# Does the swap MECHANISM perturb outputs, or does merely holding a different
# (permuted) placement do it?  --init-expert-location gives a static layout with
# zero swaps ever executed, so:
#   identity(bootA) vs identity(bootB)  -> cross-boot numerical floor
#   identity        vs r135 (static)    -> pure-placement perturbation, no swaps
#   [driver15 gave]                    -> OEPLB with 256 swaps
# If placement alone reproduces driver15's mean|d|~0.1, swaps add nothing and the
# perturbation is numerics.  If it stays ~0, the swap path is corrupting state.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen2-57B-A14B-Instruct
boot () {
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup bash $1 > $LOGD/server57b_$2.log 2>&1 &
  for i in $(seq 1 260); do grep -q "ready to roll" $LOGD/server57b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server57b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server57b_$2.log || { echo "[d19] $2 FAILED TO BOOT"; return 1; }
  echo "[d19] $2 ready"
}
probe () { cd $LOGD && python3 corpus_probe.py $LOGD/cp_stat_$1.json /data/models/Qwen2-57B-A14B-Instruct 200; echo "[d19] probe $1 done"; }

boot "$LOGD/launch57b_base_g8.sh"                          st_idA  && probe idA
boot "$LOGD/launch57b_base_g8.sh"                          st_idB  && probe idB
boot "$LOGD/launch57b_fixed_g8.sh $LOGD/plc57b_bal.json"  st_bal  && probe bal
boot "$LOGD/launch57b_fixed_g8.sh $LOGD/plc57b_r135.json" st_r135 && probe r135
pkill -9 -f "$PAT"
echo D19_DONE
