#!/bin/bash
# Usage: run_f6_boot.sh LABEL LAUNCHER CODE MIN_SWAPS
# LAUNCHER: paper_plain | paper_pb | camp_I | camp_P
# Paper-era freq6 cross-domain protocol (NEW_PAPER §5.3.1): 6seg book<->prover, N=600, conc=32,
# O=10, run1=warmup run2=steady. Model path substituted (old /data/models gone): same model
# /root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 (disclosed).
LABEL=$1; LAUNCHER=$2; CODE=$3; MIN_SWAPS=${4:-0}
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
B=/data/minghua/sjq/OEPLBdata/experiment_logs/baseline_comparison_20260914
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
BOOK=/data/minghua/sjq/OEPLBdata/datasets/prefill_decode_correlation/book_4438tok_O10.jsonl
PROV=/data/minghua/sjq/OEPLBdata/datasets/single_domain/prover_2048tok_out1.jsonl
ART=$G/author_diag_20261021/f6_boots/$LABEL
mkdir -p "$ART"
DEP=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
for f in "$BOOK" "$PROV" "$B/freq6_bench.py"; do
  [ -f "$f" ] || { echo "{\"error\":\"missing_input:$f\"}" > $ART/BOOT_ABORT.json; exit 2; }
done
ASHA=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); CSHA=$(sha256sum $DEP/controller.py | cut -d' ' -f1); FSHA=$(sha256sum $DEP/config.py | cut -d' ' -f1)
echo "{\"label\":\"$LABEL\",\"launcher\":\"$LAUNCHER\",\"code\":\"$CODE\",\"async\":\"$ASHA\",\"controller\":\"$CSHA\",\"config\":\"$FSHA\",\"book_sha\":\"$(sha256sum $BOOK | cut -d' ' -f1)\",\"prover_sha\":\"$(sha256sum $PROV | cut -d' ' -f1)\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
env_clear() { unset $(env | grep -oE '^(OEPLB|EA)_[A-Z0-9_]+' | tr '\n' ' ') 2>/dev/null; }
kill_server_quiet() {
  pkill -9 -f 'sglang.launch_server' 2>/dev/null; pkill -9 -f 'sglang::' 2>/dev/null
  for i in $(seq 1 90); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 1000 ] && return 0
    sleep 4
  done; return 1
}
# paper-era launch scripts (verbatim flags from baseline_comparison_20260914, model path substituted)
cat > "$ART/launch_paper.sh" << LEOF
. /workspace/logs/env_235b.sh
exec python3 -m sglang.launch_server \\
  --model-path $MODEL --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \\
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \\
  --dtype bfloat16 --quantization fp8 --mem-fraction-static 0.78 \\
  --disable-cuda-graph --port 30000 --host 0.0.0.0 --trust-remote-code \\
  --disable-radix-cache --watchdog-timeout 600$( [ "$LAUNCHER" = "paper_pb" ] && echo ' \
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02 --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 16 --pb-oeplb-max-total-swap-layers 94 --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-min-swap-ops 8 --pb-oeplb-max-total-ops 300 --pb-oeplb-decay-factor 0.5 \
  --pb-oeplb-adaptive-window --pb-oeplb-adaptive-decay' )
LEOF
env_clear; kill_server_quiet; env_clear
case $LAUNCHER in
  paper_plain|paper_pb) setsid nohup bash "$ART/launch_paper.sh" > "$ART/server.log" 2>&1 & SPID=$! ;;
  camp_I) ART=$ART setsid nohup bash $F/scripts/launch_arm.sh I "" > "$ART/server.log" 2>&1 & SPID=$! ;;
  camp_P) ART=$ART setsid nohup bash $F/scripts/launch_arm.sh P "" > "$ART/server.log" 2>&1 & SPID=$! ;;
  *) echo "{\"error\":\"bad_launcher\"}" > $ART/BOOT_ABORT.json; exit 2 ;;
esac
echo $SPID > "$ART/server.pid"
READY=no
for i in $(seq 1 225); do
  grep -q 'ready to roll' "$ART/server.log" 2>/dev/null && { READY=yes; break; }
  grep -qE 'Traceback|CUDA out of memory|RuntimeError' "$ART/server.log" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || break
  sleep 4
done
echo "[$LABEL] ready=$READY after $((i*4))s $(date -u +%FT%TZ)"
if [ $READY != yes ]; then
  kill_server_quiet
  echo "{\"label\":\"$LABEL\",\"outcome\":\"BOOT_FAIL\",\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
  exit 1
fi
sleep 5
RC1=1; RC2=1
timeout 900 python3 $B/freq6_bench.py 32 600 > "$ART/bench_run1.log" 2>&1; RC1=$?
echo "[$LABEL] run1 rc=$RC1 $(date -u +%FT%TZ)"
timeout 900 python3 $B/freq6_bench.py 32 600 > "$ART/bench_run2.log" 2>&1; RC2=$?
echo "[$LABEL] run2 rc=$RC2 $(date -u +%FT%TZ)"
sleep 10
kill_server_quiet
python3 - "$ART" "$LABEL" "$LAUNCHER" "$CODE" "$RC1" "$RC2" "$MIN_SWAPS" << 'PEOF'
import json, re, sys
art, label, launcher, code, rc1, rc2, minsw = sys.argv[1:8]
rc1, rc2, minsw = int(rc1), int(rc2), int(minsw)
def parse(p):
    try:
        m = re.search(r'ok=(\d+) time=([\d.]+)s tps=([\d.]+)', open(p).read())
        return (int(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (None, None, None)
    except Exception:
        return (None, None, None)
ok1, t1, tps1 = parse(art + '/bench_run1.log'); ok2, t2, tps2 = parse(art + '/bench_run2.log')
slog = open(art + '/server.log', errors='ignore').read()
issued = re.findall(r'window#(\d+): issued (\d+) swap', slog)
faults = len(re.findall(r'Watchdog|COALESCED.*Timeout|Traceback', slog))
res = dict(label=label, launcher=launcher, code=code, bench_rc1=rc1, bench_rc2=rc2,
           ok_run1=ok1, tps_run1=tps1, ok_run2=ok2, tps_run2=tps2,
           issued_windows=len(set(w for w,_ in issued)),
           total_swaps_issued=sum(int(s) for _,s in issued)//8 if issued else 0,
           faults=faults, min_swaps_required=minsw,
           pass_= rc1==0 and rc2==0 and ok1==600 and ok2==600 and faults==0 and len(set(w for w,_ in issued))>=minsw)
json.dump(res, open(art + '/f6_result.json','w'), indent=1)
print(json.dumps(res))
sys.exit(0 if res['pass_'] else 1)
PEOF
ARC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc1\":$RC1,\"bench_rc2\":$RC2,\"assert_rc\":$ARC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
exit $ARC
