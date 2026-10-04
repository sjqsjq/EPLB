#!/bin/bash
LOGD=/workspace/logs; DS1=/workspace/data/prover_256tok_out1.jsonl
BD=/workspace/EPLB/OEPLB/baselines/moetuner
ts(){ date '+%H:%M:%S'; }
declare -A L=( [identity]=$LOGD/launch_identity.sh [datafore]=$LOGD/launch_datafore.sh
  [eplb_static]=$LOGD/launch_eplb_static.sh [eplb_dyn]=$LOGD/launch_eplb.sh [oeplb]=$LOGD/launch_oeplb.sh )
boot(){ bash $LOGD/killall.sh >/dev/null 2>&1; sleep 3; rm -f $LOGD/srv_$1.log
  setsid bash ${L[$1]} > $LOGD/srv_$1.log 2>&1 < /dev/null &
  for i in $(seq 1 45); do grep -q "ready to roll" $LOGD/srv_$1.log && return 0
    grep -qi "Scheduler hit an exception" $LOGD/srv_$1.log && return 1; sleep 8; done; return 1; }
for tag in identity datafore eplb_static eplb_dyn oeplb; do
  echo "[$(ts)] === $tag boot ==="
  if ! boot $tag; then echo "[$(ts)] $tag BOOT_FAIL"; continue; fi
  sleep 5; echo "[$(ts)] $tag ready"
  cd $BD
  for r in w1 w2; do python3 bench_0914.py $DS1 256 t1_${tag}_$r >/dev/null 2>&1; done
  for r in 1 2 3 4 5; do python3 bench_0914.py $DS1 256 t1_${tag}_r$r >/dev/null 2>&1; done
  echo "[$(ts)] $tag table1 done"
  for r in 1 2; do python3 $LOGD/freq6_local.py 32 1800 t2_${tag}_r$r > $LOGD/freq6_${tag}_r$r.log 2>&1; done
  echo "[$(ts)] $tag table2 done"
done
bash $LOGD/killall.sh >/dev/null 2>&1
echo "[$(ts)] ALL_TABLES_DONE"
