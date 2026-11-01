#!/bin/bash
LOGD=/workspace/logs; RG=/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
SD=/workspace/data/sd_prover512_O32.jsonl; XD=/workspace/data/xd_freq6_O10_grid.jsonl
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
declare -A L=([identity]=$LOGD/launch_identity.sh [oeplb]=$LOGD/launch_oeplb.sh
 [eplb_dyn]=$LOGD/launch_eplb.sh [oeplb_noswap]=$LOGD/launch_oeplb_noswap.sh)
ts(){ date '+%H:%M:%S'; }
runu(){ # label dataset conc utilfile
  ( while true; do echo "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits|tr '\n' ' ')"; sleep 1; done > $4 ) &
  local sp=$!
  python3 $RG $1 $2 $3 > $LOGD/metric_$1.log 2>&1
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
}
boot(){ bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3; rm -f $LOGD/srv_m_$1.log
  setsid bash $2 > $LOGD/srv_m_$1.log 2>&1 </dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_m_$1.log && return 0; grep -qi "Scheduler hit an exception" $LOGD/srv_m_$1.log && return 1; sleep 8; done; return 1; }
for tag in identity oeplb eplb_dyn oeplb_noswap; do
  echo "[$(ts)] ===== $tag ====="
  boot $tag ${L[$tag]} || { echo "[$(ts)] $tag BOOTFAIL"; continue; }
  echo "[$(ts)] $tag ready"; sleep 4; cd /workspace/EPLB/OEPLB/scripts
  case $tag in oeplb|eplb_dyn) python3 $RG mw_$tag $SD 256 >/dev/null 2>&1;; esac  # 预热收敛
  for r in 1 2; do runu m_${tag}_sd_r$r $SD 256 $LOGD/util_${tag}_sd_r$r.csv; done
  echo "[$(ts)] $tag 同域O32 done"
  for r in 1 2; do runu m_${tag}_xd_r$r $XD 32 $LOGD/util_${tag}_xd_r$r.csv; done
  echo "[$(ts)] $tag 跨域O10 done"
done
bash $LOGD/killall.sh >/dev/null 2>&1; echo "[$(ts)] METRICS_DONE"
