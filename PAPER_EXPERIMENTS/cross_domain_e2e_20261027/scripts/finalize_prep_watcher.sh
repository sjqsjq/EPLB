#!/bin/bash
# finalize_prep_watcher.sh - cross_domain_e2e_20261027 prep auto-chain.
# Steps: wait ILP done -> MT placement sanity (per-layer permutation) ->
# freeze placement shas (one-time read-only env lock) -> append AMENDMENT-1
# to PROTOCOL -> ES_val prep boot (#3) -> on PASS (rc=0 + init_by_eplb
# injection evidence) launch formal 24-boot queue detached.
# Any gate failure: STOP + LEDGER row for operator triage (no silent skip,
# no scope widening). AUTHORIZATION: author confirmation 2026-10-27.
set -u
X=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027
F=/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019
DS=$F/B_data/streams/e3_multidomain_switch.jsonl
DSSHA=d55fe2f73376e2fe8c16b3c5e91a2462b0889a4aff60e253b73f5029f7a5d879
L=$X/LEDGER_xd.tsv
ledger(){ printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" >> "$L"; }
MTJ=$X/placements/placement_e3_fair.json
MTM=$X/placements/placement_e3_fair.json.meta.json
ILOG=$X/prep/profile_build/ilp_solve.log
ledger prep_chain "WATCHER_START waiting_ilp"
for i in $(seq 1 270); do
  if [ -f "$MTM" ] && grep -q "Solved 94 layers" "$ILOG" 2>/dev/null; then break; fi
  sleep 20
done
if [ ! -f "$MTM" ] || ! grep -q "Solved 94 layers" "$ILOG" 2>/dev/null; then
  ledger prep_chain "STOP ilp_timeout_or_incomplete log_tail=$(tail -1 $ILOG 2>/dev/null | head -c 100)"; exit 1
fi
python3 - > $X/prep/profile_build/mt_sanity.txt 2>&1 << "PYEOF"
import json
base = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027/placements/placement_e3_fair.json"
d = json.load(open(base))
m = json.load(open(base + ".meta.json"))
mm = d["physical_to_logical_map"]
assert len(mm) == 94, "layers %d" % len(mm)
for l in range(94):
    assert sorted(mm[l]) == list(range(128)), "layer %d not a permutation" % l
st = m["per_layer_stats"]
assert len(st) == 94, "stats %d" % len(st)
imb = [s["imbalance_ratio"] for s in st]
tot = sum(s["solve_s"] for s in st)
mean_imb = sum(imb) / len(imb)
max_imb = max(imb)
print("MT_SANITY_OK layers=94 perms_ok mean_imb=%.4f max_imb=%.4f total_solve_s=%.1f" % (mean_imb, max_imb, tot))
with open("/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027/prep/profile_build/mt_sanity_env.txt", "w") as f:
    f.write("MT_MEAN_IMB=%.4f\nMT_MAX_IMB=%.4f\nMT_TOTAL_SOLVE_S=%.1f\n" % (mean_imb, max_imb, tot))
PYEOF
if ! grep -q MT_SANITY_OK $X/prep/profile_build/mt_sanity.txt 2>/dev/null; then
  ledger prep_chain "STOP mt_sanity_failed detail=$(head -3 $X/prep/profile_build/mt_sanity.txt 2>/dev/null | tr "\n" " " | head -c 150)"; exit 1
fi
ledger prep_chain "$(cat $X/prep/profile_build/mt_sanity.txt)"
. $X/prep/profile_build/mt_sanity_env.txt
LOCK=$X/code_lock/placement_shas.env
if [ -f "$LOCK" ]; then ledger prep_chain "STOP lock_already_exists"; exit 1; fi
PL_ES=$(sha256sum $X/placements/datafore_e3_placement.json | cut -d" " -f1)
PL_DF=$(sha256sum $X/placements/datafore_e3_remap_placement.json | cut -d" " -f1)
PL_MT=$(sha256sum $MTJ | cut -d" " -f1)
PL_MTM=$(sha256sum $MTM | cut -d" " -f1)
printf "PL_ES=%s\nPL_DF=%s\nPL_MT=%s\nPL_MT_META=%s\n" "$PL_ES" "$PL_DF" "$PL_MT" "$PL_MTM" > "$LOCK"
chmod 444 "$LOCK"
ledger prep_chain "PLACEMENTS_FROZEN es=$PL_ES df=$PL_DF mt=$PL_MT mt_meta=$PL_MTM"
cat >> $X/PROTOCOL.md << AMEOF

## AMENDMENT-1 (appended by finalize_prep_watcher at profile freeze; resolves sec.2 placeholders)
- Prep #1 (rec_e3 v1, per_token+auto): READY_FAIL zero-data. Root cause:
  _DetailSinglePassGatherer crashes on single-node deepep
  (num_tokens_per_rdma_rank=None, expert_distribution.py:406). INCIDENT logged.
- Prep #2 (rec_e3 v2b, proven launch_profile.sh recipe stat+normal+graph128
  +manual HTTP arm): SUCCESS. Arm 200 at 05:25:00Z; bench ok=8192 err=0,
  190.86s, 42.9 tps; stop+dump rc=0/0; recorder .pt rank0 shape (1000,94,128)
  sum=3154479072 sha=fa04965525ecb44ccd89e67be525e815a5ae5474ccbb44c6c7c9f9c13dc6df4a
  (48130049 B). Mid-boot incident: no curl in env; first bench pass ran
  unarmed => zero profile contamination (radix off, artifacts renamed
  INVALID_partial); re-armed via python urllib, fresh full bench in same
  server session. Profile covers the FULL E3 stream (all four domains).
- Pipeline: pt2npz.py -> P_e3.npz (per-layer sum 33558288 = 4194260 prompt
  tokens x 8 topk, exact; layer0 imbalance 2.379 vs prover-only 4.093).
  ES/DF shared json logical_count 94x128, byte-identical files (author key
  point 2: shared pre-frozen mixed-domain calibration profile; DF greedy
  least-loaded executes inside sglang rebalance_experts at init without
  redundancy; ES runs eplb algorithm with R16).
- MT solve cost (single-listed): Gurobi ILP1 G=8 K=16 time_limit=30
  mip_gap=0.005; mean_imbalance=$MT_MEAN_IMB max_imbalance=$MT_MAX_IMB
  total_solve_s=$MT_TOTAL_SOLVE_S; profile capture cost = rec boot bench
  190.86s + pt 48.1MB; per-layer solve_s in placement_e3_fair.json.meta.json.
- Frozen shas (code_lock/placement_shas.env, read-only 444):
  PL_ES=$PL_ES
  PL_DF=$PL_DF
  PL_MT=$PL_MT
  PL_MT_META=$PL_MTM
- Prep #3 = ES_val (injection validation). Formal 24-boot queue auto-launches
  only on ES_val PASS (runner rc=0 AND init_by_eplb injection evidence in
  server.log). Otherwise STOP + triage.
AMEOF
PSHA=$(sha256sum $X/PROTOCOL.md | cut -d" " -f1)
ledger prep_chain "AMENDMENT_1_APPENDED protocol_sha=$PSHA"
ledger prep_chain "ES_VAL_START prep3"
bash $X/scripts/run_xd_boot.sh ES_val ES "" "$DS" "$DSSHA" 8192 256 900
ERC=$?
INJ=$(grep -c "init_by_eplb" $X/boots/ES_val/server.log 2>/dev/null)
INJ=${INJ:-0}
ledger prep_chain "ES_VAL_DONE rc=$ERC injection_lines=$INJ"
if [ $ERC != 0 ] || [ "$INJ" -lt 1 ]; then
  ledger prep_chain "STOP es_val_failed rc=$ERC inj=$INJ triage_needed"; exit 1
fi
cd $X
setsid nohup bash $X/scripts/queue_formal_xd.sh > $X/queue_formal.log 2>&1 &
QPID=$!
ledger prep_chain "FORMAL_QUEUE_LAUNCHED pid=$QPID boots=24 order=ORDER_TABLE_locked"
