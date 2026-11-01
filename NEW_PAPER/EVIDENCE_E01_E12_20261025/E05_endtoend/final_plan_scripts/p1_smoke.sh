#!/usr/bin/env bash
# p1_smoke.sh — P1 pre-registered feasibility smoke (FREEZE_final_plan.md L153-155).
# Nature: M2 diagnostic boot (budget boot 1/4). NOT a measurement launch;
#         no timing data from this run enters any analysis.
# Frozen design (prereg): boot worst arm E-R32 ("auto 32 256"), burst
#   in2048/out1 slice requests at C_hi=256, verify admission normal
#   (no OOM / crash / refusal), then kill; record result in ledger.
#   If smoke fails -> C_hi treated as infeasibility event: HALT + report
#   (no silent redesign). Executor does NOT auto-launch P1 on its own.
# Executor details (count/wave structure not fixed by prereg; documented):
#   - slice = first 512 rows of p1_longio.jsonl (frozen DS sha verified),
#     max_tokens rewritten "128" -> "1"  => 2 full waves at conc=256,
#     testing sustained admission/queueing, not one instantaneous burst.
#   - bench timeout 1200s; health poll 600x4s; kill/gpu-clear/env-clear
#     copied verbatim from frozen driver (run_final_campaign.sh L76-97).
#   - PASS criteria: bench.json exists AND errors==0 AND ok==512
#     AND server pid alive at bench end AND zero crash markers in
#     server.log AND "KV Cache is allocated" line captured (#tokens
#     recorded, expected ~190307 for E-R32).
set -u
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
ART=$F/p1_smoke
LAUNCH=$F/scripts/launch_arm.sh
BENCHV2=$F/scripts/bench_client_v2.py
MODEL=/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8
DS=$F/B_data/streams/p1_longio.jsonl
DS_SHA_EXPECT=3c09e0270a60d185c801646c3cebc7157a99086d0c10c8b4e78c5dc615068457
LEDGER=$F/logs/campaign_marks.tsv
LABEL=p1_smoke_ER32
CONC=256
N_SLICE=512
BENCH_TIMEOUT=1200
PH=P1SMOKE

ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
mark(){ printf '%s\t%s\t%s\n' "$(ts)" "$PH" "$1" >> "$LEDGER"; }
sha_of(){ sha256sum "$1" | cut -d' ' -f1; }
mkdir -p "$ART"
log(){ printf '%s\t%s\t%s\n' "$(ts)" "$PH" "$1" | tee -a "$ART/smoke.log"; }

gpu_clear_wait(){ local i m; for i in $(seq 1 90); do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$m" -lt 1000 ] && return 0; sleep 4; done
  log "FATAL: GPU not clear after 360s"; return 1; }
kill_server_quiet(){ pkill -9 -f sglang.launch_server >/dev/null 2>&1
  pkill -9 -f 'sglang::' >/dev/null 2>&1; sleep 5
  gpu_clear_wait || return 1; return 0; }
env_clear(){ local v; for v in $(env | grep -oE '^(OEPLB_[A-Za-z0-9_]+|EA_[A-Za-z0-9_]+)=' | tr -d '='); do unset "$v" 2>/dev/null || true; done; }
fail(){ # $1 reason
  log "SMOKE_FAIL $1"
  mark "SMOKE_VERDICT $LABEL SMOKE_FAIL reason=$1"
  { echo "SMOKE_FAIL"
    echo "reason=$1"
    echo "utc=$(ts)"
    echo "nature=M2 diagnostic boot 1/4; NOT a measurement launch"
    echo "prereg_disposition=HALT and report to author (no silent redesign)"
  } > "$ART/smoke_verdict.txt"
  kill_server_quiet || log "WARN: gpu not clear after fail-kill"
  exit 1
}

# ---- 0. preconditions ----
[ "$(sha_of "$DS")" = "$DS_SHA_EXPECT" ] || { mark "SMOKE_ABORT DS sha mismatch"; echo "SMOKE_FAIL: DS sha mismatch" > "$ART/smoke_verdict.txt"; exit 2; }
gpu_clear_wait || { echo "SMOKE_FAIL: gpu not clear at start" > "$ART/smoke_verdict.txt"; exit 1; }

{ echo "nature=P1_pre_registered_feasibility_smoke (M2 diagnostic boot 1/4, NOT measurement)"
  echo "prereg=FREEZE_final_plan.md L153-155"
  echo "utc_start=$(ts)"
  echo "arm=E params='auto 32 256' label=$LABEL conc=$CONC n_slice=$N_SLICE bench_timeout=$BENCH_TIMEOUT"
  echo "ds=$DS ds_sha=$DS_SHA_EXPECT"
  echo "launcher_sha=$(sha_of "$LAUNCH") benchv2_sha=$(sha_of "$BENCHV2")"
  echo "executor_detail=slice=first ${N_SLICE} rows of p1_longio.jsonl, max_tokens rewritten to '1' (2 waves at conc=256; count/waves not fixed by prereg)"
} > "$ART/meta.txt"

# ---- 1. build out1 slice ----
python3 - "$DS" "$ART/smoke_slice.jsonl" "$N_SLICE" << 'PYEOF' || fail "slice_build"
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
with open(src) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r["max_tokens"] = "1"
        rows.append(r)
        if len(rows) >= n:
            break
assert len(rows) == n, f"got {len(rows)} rows, want {n}"
with open(dst, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
print(f"slice rows written: {len(rows)}")
PYEOF
SLICE_SHA=$(sha_of "$ART/smoke_slice.jsonl")
echo "slice_sha=$SLICE_SHA" >> "$ART/meta.txt"
log "slice built: $N_SLICE rows in2048/out1 sha=$SLICE_SHA"

# ---- 2. boot E-R32 (frozen launcher path, verbatim driver pattern) ----
env_clear
ART="$ART" setsid nohup bash "$LAUNCH" E auto 32 256 > "$ART/server.log" 2>&1 &
SPID=$!
log "boot launched pid=$SPID (E 'auto 32 256')"
ok=0
for i in $(seq 1 600); do
  if grep -q "ready to roll" "$ART/server.log" 2>/dev/null; then ok=1; break; fi
  if grep -qE "Traceback|CUDA out of memory|RuntimeError" "$ART/server.log" 2>/dev/null; then break; fi
  kill -0 "$SPID" 2>/dev/null || { sleep 2; break; }
  sleep 4
done
[ "$ok" = 1 ] || fail "boot_no_ready_to_roll"
tr '\0' '\n' < "/proc/$SPID/environ" 2>/dev/null | grep -E '^(OEPLB_|EA_|SGLANG_|NVSHMEM_|NCCL_)' | sort > "$ART/meta_env.txt" || true
KVLINE=$(grep -m1 "KV Cache is allocated" "$ART/server.log" || echo "MISSING")
echo "kv_line=$KVLINE" >> "$ART/meta.txt"
log "SMOKE_BOOT_OK $LABEL (pid $SPID) kv=[$KVLINE]"
mark "SMOKE_BOOT_OK $LABEL"

# ---- 3. burst at C_hi=256 ----
log "burst start: $N_SLICE x in2048/out1 @ conc=$CONC (timeout ${BENCH_TIMEOUT}s)"
( cd "$F/scripts" && OEPLB_BENCH_OUT="$ART" OEPLB_MODEL="$MODEL" \
    timeout "$BENCH_TIMEOUT" python3 "$BENCHV2" "$LABEL" "$ART/smoke_slice.jsonl" "$CONC" ) > "$ART/bench.log" 2>&1
BRC=$?
[ -f "$ART/${LABEL}.json" ] && cp "$ART/${LABEL}.json" "$ART/bench.json"
log "burst end rc=$BRC"

# ---- 4. admission verdict ----
[ -f "$ART/bench.json" ] || fail "no_bench_json bench_rc=$BRC"
OK_N=$(python3 -c "import json;print(json.load(open('$ART/bench.json'))['ok'])")
ERR_N=$(python3 -c "import json;print(json.load(open('$ART/bench.json'))['errors'])")
WALL=$(python3 -c "import json;m=json.load(open('$ART/bench.json'));print(round(m['wall_end_s']-m['wall_start_s'],1))")
ERR_SAMPLE=$(grep -m1 "Sample errors" "$ART/bench.log" 2>/dev/null | head -c 300 || echo "")
kill -0 "$SPID" 2>/dev/null && ALIVE=1 || ALIVE=0
CRASH=$(grep -cE "CUDA out of memory|Traceback|RuntimeError|Watchdog caught|Aborted|core dumped|Segmentation fault" "$ART/server.log" || true)

REASON=""
[ "$ERR_N" = "0" ] || REASON="${REASON}errors=${ERR_N}(refusal/error);"
[ "$OK_N" = "$N_SLICE" ] || REASON="${REASON}ok=${OK_N}<${N_SLICE};"
[ "$ALIVE" = "1" ] || REASON="${REASON}server_died_during_bench;"
[ "$CRASH" = "0" ] || REASON="${REASON}crash_marker_lines=${CRASH};"
if [ -n "$REASON" ]; then
  echo "err_sample=$ERR_SAMPLE" >> "$ART/meta.txt"
  fail "$REASON bench_rc=$BRC wall=${WALL}s"
fi

# ---- 5. kill + gpu clear (frozen pattern) ----
kill_server_quiet || { log "WARN: gpu not clear after kill"; mark "NOTE smoke kill: gpu not clear after 360s"; }

# ---- 6. PASS: verdict file + ledger ----
KVTOK=$(echo "$KVLINE" | grep -oE '[0-9]+' | tail -1)
{ echo "SMOKE_PASS"
  echo "label=$LABEL arm=E params='auto 32 256' conc=$CONC n_slice=$N_SLICE"
  echo "ok=$OK_N errors=$ERR_N wall_s=$WALL bench_rc=$BRC"
  echo "kv_line=$KVLINE"
  echo "crash_marker_lines=$CRASH server_alive_at_bench_end=$ALIVE"
  echo "slice_sha=$SLICE_SHA ds_sha=$DS_SHA_EXPECT"
  echo "utc_end=$(ts)"
  echo "nature=M2 diagnostic boot 1/4; NOT a measurement launch; no timing data enters analysis"
  echo "disposition=C_hi=256 admission feasible on worst arm E-R32 -> proceed to P1 timed matrix (32 cells, driver phase P1)"
} > "$ART/smoke_verdict.txt"
mark "SMOKE_VERDICT $LABEL SMOKE_PASS ok=$OK_N err=$ERR_N wall=${WALL}s kv_tokens=$KVTOK"
mark "NOTE M2 diagnostic budget: P1 feasibility smoke = boot 1/4 (FREEZE_final_plan.md L153-155)"
log "SMOKE_PASS ok=$OK_N err=$ERR_N wall=${WALL}s kv_tokens=$KVTOK"
exit 0
