#!/bin/bash
set -uo pipefail
T=/workspace/EPLB/OEPLB/theory_a100
DS=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_256tok_out1.jsonl
M57=/workspace/models/Qwen2-57B-A14B-Instruct

echo "[step4] waiting step3b + 57B download..."
while ! grep -q STEP3B_DONE /workspace/logs/step3b.log 2>/dev/null; do sleep 60; done
while pgrep -f "modelscope download" >/dev/null 2>&1; do sleep 60; done
while [ ! -f $M57/config.json ]; do sleep 30; done
sleep 60
echo "[step4] 57B ready: $(du -sh $M57 | cut -f1)"

cat > $T/launch_scan57.sh <<'LEOF'
#!/bin/sh
set -e
. /workspace/EPLB/OEPLB/baselines/a100/env_a100.sh
EXTRA=""
[ -n "${SCAN_PLC:-}" ] && EXTRA="--init-expert-location $SCAN_PLC"
exec python3 -m sglang.launch_server \
  --model-path /workspace/models/Qwen2-57B-A14B-Instruct \
  --dtype bfloat16 --tp ${SCAN_EP:?} --ep-size ${SCAN_EP:?} \
  --moe-runner-backend triton --attention-backend flashinfer \
  --mem-fraction-static 0.85 --context-length 4096 --max-running-requests 256 \
  --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache \
  --watchdog-timeout 600 --port 30000 --host 0.0.0.0 --trust-remote-code $EXTRA
LEOF
chmod +x $T/launch_scan57.sh

run_arm57() { # tag plc ep prefix
  local TAG=$1 PLC=$2 EP=$3 PFX=$4
  local SLOG=/workspace/logs/server_a100_57b_${PFX}${TAG}.log; : > "$SLOG"
  for i in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; sleep 5; done
  export SCAN_PLC="$PLC" SCAN_EP=$EP
  setsid nohup bash $T/launch_scan57.sh > "$SLOG" 2>&1 < /dev/null &
  local PID=$! ready=0 dead=0
  for i in $(seq 1 400); do
    grep -q "The server is fired up" "$SLOG" 2>/dev/null && { ready=1; break; }
    if ! kill -0 $PID 2>/dev/null; then dead=$((dead+1)); [ $dead -ge 4 ] && break; else dead=0; fi
    sleep 3
  done
  if [ $ready -ne 1 ]; then echo "BOOT_FAIL $PFX$TAG"; tail -25 "$SLOG"; kill -9 $PID 2>/dev/null; return 1; fi
  sleep 5
  for r in 1 2 3 4 5; do
    SAVE=1; [ $r -eq 1 ] && SAVE=0
    python3 $T/bench_scan.py "$DS" 256 "$TAG" $r $SAVE "$PFX" 2>&1 | tee $T/logs/${PFX}${TAG}_r${r}.log
    sleep 2
  done
  kill -TERM -$PID 2>/dev/null; sleep 6; kill -KILL -$PID 2>/dev/null
  echo "[$PFX$TAG] DONE"
}

scan_ep() { # ep targets...
  local EP=$1; shift
  echo "[step4] ===== EP=$EP gen_placement: $* ====="
  cd /workspace/EPLB/OEPLB
  python3 repro/gen_placement.py repro/counts57b.json $EP $T/placements/plc57E$EP "$@" | tee /workspace/logs/plc57E$EP.txt
  local RB=$(grep -E "identity" /workspace/logs/plc57E$EP.txt | grep -oE "r_avg=[0-9.]+" | head -1 | cut -d= -f2)
  echo "[step4] EP=$EP identity r_avg=$RB"
  run_arm57 id "" $EP _scan57E${EP}_
  run_arm57 bal $T/placements/plc57E${EP}_bal.json $EP _scan57E${EP}_
  for t in "$@"; do
    tag=$(printf "r%.2f" $t | tr -d '.')
    run_arm57 $tag $T/placements/plc57E${EP}_${tag}.json $EP _scan57E${EP}_
  done
  local LAST=${!#}
  local HOLDT=$(printf "r%.2f" $LAST | tr -d '.')
  python3 repro/fit_f3.py $RB 1.02 id,$HOLDT plc57E$EP.txt _scan57E${EP}_ | tee $T/logs/fit_f3_57E$EP.txt
}

echo "[step4] EP8 skipped on A100: 57B has 28 attn heads (28%8!=0), H20's EP8 relied on dp-attention (excluded on A100 by design)"
scan_ep 4 1.01 1.02 1.04 1.06 1.08
scan_ep 2 1.01 1.02 1.03 1.05 1.07

echo "[step4] ===== EP 幂律汇总 ====="
for EP in 4 2; do
  echo "-- EP$EP:"; grep -E "^\[hinge" $T/logs/fit_f3_57E$EP.txt
done
echo STEP4_DONE
