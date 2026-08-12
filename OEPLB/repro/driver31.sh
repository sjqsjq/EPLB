#!/bin/bash
# Rigorous redo of the adaptive-window experiment (d29 had a design flaw).
#
# d29 used seg_L200 = 2.8 memory-lengths per segment, so NO arm could converge
# within a segment: both were 11-72x slower than the static optimum, i.e. in a
# pathological "forever chasing changepoints" regime outside the theory's domain.
# One arm accumulated 77k swaps and hung on NCCL sequence-number divergence.
#
# Fixes: (1) seg_L1000 = 13.9 memory-lengths, so segments are convergeable;
# (2) a uniform swap budget 0.10 on every arm so no arm can hang and pollute the
#     comparison (arms that hit it are reported);
# (3) decision counts read from DP0 only (d29 counted 8 ranks = 8x inflation);
# (4) a no-OEPLB baseline arm to detect if we are pathological again.
# Pre-registered hypotheses P1'/P2/P3/P4 are in NOTES.md, written before running.
LOGD=/workspace/logs
SFX=server; PAT="sglang.launch_$SFX"
export OEPLB_MODEL=/data/models/Qwen3-235B-A22B-FP8
DS=$LOGD/segp_L1000.jsonl  # pure-prefill (max_tokens=1) so the changepoint is ONLY a routing-distribution change; the earlier seg_L1000 mixed max_tokens 1 vs 2048, making each switch also a prefill/decode-ratio switch and every arm 10x slower
BUDGET=0.10
boot () {  # $1=launch $2=tag $3=envstr
  pkill -9 -f "$PAT"; for i in $(seq 1 40); do pgrep -f "$PAT" >/dev/null || break; sleep 3; done; sleep 8
  setsid nohup env $3 bash $1 > $LOGD/server235b_$2.log 2>&1 &
  for i in $(seq 1 320); do grep -q "ready to roll" $LOGD/server235b_$2.log && break; grep -q "Traceback (most recent call last)" $LOGD/server235b_$2.log && break; sleep 4; done
  grep -q "ready to roll" $LOGD/server235b_$2.log || { echo "[d31] $2 FAILED"; return 1; }
  echo "[d31] $2 ready"
}
run () {  # $1=tag
  cd /workspace/EPLB/OEPLB/scripts && timeout 900 python3 run_grid_bench.py _d31_$1 $DS 256 > /dev/null 2>&1
  L=$LOGD/server235b_$1.log
  D=$(grep "DIAG" $L 2>/dev/null|grep -c "DP0")
  S=$(grep "swap(s) done" $L 2>/dev/null|grep -c "DP0")
  B=$(grep -c "swap budget exhausted" $L 2>/dev/null)
  echo "[d31] $1 done | decisions=$D issued=$S budget_hit=$B"
}
# baseline reference (no OEPLB) -- detects a pathological regime
boot "$LOGD/launch235b_fixed.sh $LOGD/plc235b_bal.json" b_base "" && run b_base
# (W,alpha) grid, groups sharing M
for spec in "M16_W16a0 16 0" "M16_W8a50 8 0.5" \
            "M32_W32a0 32 0" "M32_W16a50 16 0.5" "M32_W8a75 8 0.75" \
            "M64_W64a0 64 0" "M64_W32a50 32 0.5" "M64_W16a75 16 0.75"; do
  set -- $spec; TAG=$1; W=$2; A=$3
  boot "$LOGD/launch235b_oeplb.sh" g_$TAG "OE_SW=$W OE_DECAY=$A OEPLB_SWAP_BUDGET_FRAC=$BUDGET" || continue
  run g_$TAG
done
# adaptive arms (P3)
boot "$LOGD/launch235b_oeplb.sh" a_adw "OE_SW=16 OE_DECAY=0.5 OE_ADW=--pb-oeplb-adaptive-window OEPLB_SWAP_BUDGET_FRAC=$BUDGET" && run a_adw
boot "$LOGD/launch235b_oeplb.sh" a_adwd "OE_SW=16 OE_DECAY=0.5 OE_ADW=--pb-oeplb-adaptive-window OEPLB_ADAPTIVE_DECAY=1 OEPLB_SWAP_BUDGET_FRAC=$BUDGET" && run a_adwd
pkill -9 -f "$PAT"; echo D31_DONE
