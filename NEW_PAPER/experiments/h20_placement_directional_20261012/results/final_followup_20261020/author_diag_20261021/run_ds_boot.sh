#!/bin/bash
# Usage: run_ds_boot.sh LABEL LAUNCHER DS CODE
# LAUNCHER: paper_plain | paper_pb ; DS: E3 | E4
# Campaign streams (bench_client_v2, conc 256) under PAPER-ERA server flags. Author_diag pool.
LABEL=$1; LAUNCHER=$2; DS=$3; CODE=$4
G=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_followup_20261020
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
case $DS in
  E3) DSP=$F/B_data/streams/e3_multidomain_switch.jsonl; DSSHA=d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879; BTO=1500 ;;
  E4) DSP=$F/B_data/streams/e4_fullduplex.jsonl;      DSSHA=a024c5e61d0dbf321dee6245ef3d5f270da7e4374304d910e943cdd0657bc0a6; BTO=2400 ;;
  *) echo bad_ds; exit 2 ;;
esac
ART=$G/author_diag_20261021/ds_boots/$LABEL
mkdir -p "$ART"
[ "$(sha256sum $DSP | cut -d' ' -f1)" = "$DSSHA" ] || { echo "{\"error\":\"stream_sha\"}" > $ART/BOOT_ABORT.json; exit 2; }
DEP=/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb
ASHA=$(sha256sum $DEP/async_swapper.py | cut -d' ' -f1); CSHA=$(sha256sum $DEP/controller.py | cut -d' ' -f1)
echo "{\"label\":\"$LABEL\",\"launcher\":\"$LAUNCHER\",\"ds\":\"$DS\",\"code\":\"$CODE\",\"async\":\"$ASHA\",\"controller\":\"$CSHA\",\"start\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json"
env_clear() { unset $(env | grep -oE '^(OEPLB|EA)_[A-Z0-9_]+' | tr '\n' ' ') 2>/dev/null; }
kill_server_quiet() {
  pkill -9 -f 'sglang.launch_server' 2>/dev/null; pkill -9 -f 'sglang::' 2>/dev/null
  for i in $(seq 1 90); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${M:-99999}" -lt 1000 ] && return 0
    sleep 4
  done; return 1
}
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
setsid nohup bash "$ART/launch_paper.sh" > "$ART/server.log" 2>&1 &
SPID=$!; echo $SPID > "$ART/server.pid"
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
(cd $F/scripts && OEPLB_BENCH_OUT=$ART OEPLB_MODEL=$MODEL timeout $BTO \
  python3 bench_client_v2.py $LABEL $DSP 256) > "$ART/bench.log" 2>&1
BRC=$?
echo "[$LABEL] bench rc=$BRC $(date -u +%FT%TZ)"
sleep 15
kill_server_quiet
python3 - "$ART" "$LABEL" "$LAUNCHER" "$DS" "$CODE" "$BRC" << 'PEOF'
import json, re, sys
art, label, launcher, ds, code, brc = sys.argv[1:7]; brc = int(brc)
slog = open(art + '/server.log', errors='ignore').read()
issued = re.findall(r'window#(\d+): issued (\d+) swap', slog)
res = dict(label=label, launcher=launcher, ds=ds, code=code, bench_rc=brc,
           tps=None, ttft_p95=None, errors=None,
           issued_windows=len(set(w for w,_ in issued)),
           total_swaps=sum(int(s) for _,s in issued)//8 if issued else 0,
           resets=len(re.findall(r'PB-OEPLB-RESET', slog))//8,
           faults=len(re.findall(r'Watchdog|COALESCED.*Timeout', slog)))
try:
    d = json.load(open(art + '/' + label + '.json'))
    res['tps'] = round(d['tps'], 2); res['ttft_p95'] = d.get('ttft_p95')
    res['errors'] = d.get('n_errors', d.get('errors'))
except Exception: pass
res['pass_'] = brc == 0 and res['tps'] is not None and res['faults'] == 0
json.dump(res, open(art + '/ds_result.json', 'w'), indent=1)
print(json.dumps(res))
sys.exit(0 if res['pass_'] else 1)
PEOF
ARC=$?
echo "{\"label\":\"$LABEL\",\"bench_rc\":$BRC,\"assert_rc\":$ARC,\"end\":\"$(date -u +%FT%TZ)\"}" > "$ART/boot_meta.json.end"
exit $ARC
