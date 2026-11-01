#!/usr/bin/env python3
"""Freeze the E-A calibration-grid protocol (writes ea_v1_calibration_grid_protocol.json).
Refuses to overwrite. Hashes every locked artifact at freeze time."""
import json, hashlib, os, sys, datetime
EXP="/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CONF=EXP+"/configs/followup_v1"
SITE="/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
OUT=CONF+"/ea_v1_calibration_grid_protocol.json"
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<22),b''): h.update(c)
    return h.hexdigest()
if os.path.exists(OUT): sys.exit("REFUSE overwrite "+OUT)
paths={
 "calibration_workload":EXP+"/inputs/fullservice_v2/workload/calibration.jsonl",
 "deployed_controller":SITE+"/controller.py",
 "deployed_config":SITE+"/config.py",
 "repo_controller":"/workspace/EPLB/OEPLB/src/controller.py",
 "repo_config":"/workspace/EPLB/OEPLB/src/config.py",
 "launcher_template":CONF+"/ea_launchers/launch235b_eagrid.sh",
 "driver":EXP+"/scripts/run_ea_calibration_grid.sh",
 "bench_tool":"/workspace/EPLB/OEPLB/scripts/run_grid_bench.py",
 "code_lock":CONF+"/ea_v1_code_lock.json",
 "static_verification":EXP+"/results/ea_v1/switch_independence_static.json",
 "v3_online_launcher_reference":"/workspace/logs/launch235b_oeplb.sh",
 "v3_protocol_reference":CONF+"/fullservice_protocol_v3.json",
}
locked={k:{"path":p,"sha256":sha(p)} for k,p in paths.items()}
prot={
 "spec_id":"h20-placement-ea-v1-calibration-grid",
 "status":"frozen_before_execution",
 "frozen_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "purpose":("Pre-freeze calibration phase of experiment E-A (four-arm W x D ablation). "
   "Selects the fixed control parameters (W*, alpha*) for the fixed arms using ONLY the "
   "8037 real calibration requests, and runs the C5-dynamic log-signature smokes for the "
   "A10/A01/A11 arm configurations on the decoupled controller (ea_v1_code_lock). "
   "No evaluation-request access; no performance claims from this phase; smoke data never "
   "enters selection."),
 "why_existing_scans_insufficient":("All retained fixed-vs-adaptive scans (driver35/38/39, "
   "oeplb_fixed_adaptive_results.json, oeplb_adaptive_both_constructions.json) ran on synthetic "
   "grid/crossdomain workloads (L512_O1_realprover, crossdomain_freq6_O10, universal_16k) outside "
   "the paper's real-request scope, and none swept (sync_window, decay_factor) jointly. Per the "
   "user rule this counts as insufficient -> small grid on calibration requests, chosen before "
   "evaluation and frozen."),
 "grid":{
   "W_candidates":[8,16,32,64],
   "W16_justification":"v3-lineage value (sync_window=16) included as candidate so the adaptive arms' starting window is in the fixed candidate set; avoids weak-control strawman per user rule",
   "alpha_candidates":[0.0,0.5,0.9],
   "starts_per_combo":2,
   "adaptive_flags":"adaptive_window=False, adaptive_decay=False for ALL grid combos (A00-style)",
   "order":"2 rounds; within round W ascending x alpha ascending; deterministic, fixed before execution",
   "n_starts":24},
 "smokes":[
   {"label":"easmoke_A10_s1","arm":"A10 (adaptive W + fixed D)","W":16,"alpha":0.5,"adaptive_window":True,"adaptive_decay":False,
    "expected_signature":{"adw_shrink_or_grow":">=0 (mechanism enabled; firing workload-dependent)","window_shift":">=0","reset":"==0 (D off -> no history zeroing anywhere)","warmstart":"==0","calib_tier":"==0","arg_adw_true":">=1","arg_add_true":"==0"}},
   {"label":"easmoke_A01_s1","arm":"A01 (fixed W + adaptive D)","W":16,"alpha":0.5,"adaptive_window":False,"adaptive_decay":True,
    "expected_signature":{"adw_shrink":"==0 (window frozen)","adw_grow":"==0","window_shift":"==0","window_stable":"==0","adw_decay0_window_fixed":">=0 (fires only at changepoints)","reset":">=0 (enabled; firing workload-dependent)","arg_adw_true":"==0","arg_add_true":">=1"}},
   {"label":"easmoke_A11_s1","arm":"A11 (both adaptive, v3-equivalent)","W":16,"alpha":0.5,"adaptive_window":True,"adaptive_decay":True,
    "expected_signature":{"note":"v3-style signatures (ADW shrink/grow, WINDOW shift, RESET events); exact counts workload-dependent; arg flags both True"}}],
 "fixed_shared_flags":("threshold-ratio 1.02; min-prefill-tokens 256; window-floor 8; "
   "window-shift-confirm 2; max-total-swap-layers 94; max-swaps-per-layer 64; min-swap-ops 8; "
   "max-total-ops 300; dead_zone_ratio default 0.0 -> controller auto-sets 1+0.00408*EP^1.52=1.096 "
   "IDENTICALLY in every arm (C6, audit F4: no CLI value disables it; documented, not varied); "
   "calibrate_adaptive_sensitivity default False -> warm-start/tier paths unreachable in every arm (C7); "
   "model/parallelism/backend/mem-fraction/cuda-graph identical to v3 online launcher."),
 "arm_parameter_mapping_after_selection":{
   "A00":"sync_window=W*, decay_factor=alpha*, both adaptive flags OFF",
   "A10":"sync_window=16 (adaptive start, v3 lineage), decay_factor=alpha*, adaptive_window ON, adaptive_decay OFF",
   "A01":"sync_window=W*, decay_factor=0.5 (v3 lineage base alpha; modulated by D), adaptive_window OFF, adaptive_decay ON",
   "A11":"sync_window=16, decay_factor=0.5, both adaptive flags ON -> byte-identical launch configuration to v3 online arm (contemporaneous re-run required by user rule because controller code changed)"},
 "selection_rule":{
   "metric":"total_time_s of the full calibration pass (8037 requests, concurrency 256, identical completed workload per start)",
   "validity":"bench_rc==0 AND errors==0 AND ok==8037; invalid starts are retained+retired, never retried",
   "combo_score":"mean total_time_s over valid starts of the combo; combo with 1 valid start uses n=1 mean (flagged); 0 valid -> ineligible",
   "choice":"(W*,alpha*) = argmin combo_score",
   "tie_break":["smaller W","smaller alpha","(16,0.5) lineage"],
   "freeze_point":"written to configs/followup_v1/ea_v1_fixed_params.json BEFORE any evaluation or campaign start; never revised afterwards",
   "prohibited":["best-start selection within combos","using smoke runs","using evaluation requests","revisiting after seeing campaign data"]},
 "locked_artifacts":locked,
 "recording_per_start":["full server log","bench json (ok/errors/total_time_s/tps/tokens/latency)","campaign_state entry with mechanism line counts (ADW/WINDOW/RESET/WARMSTART/CALIB/REFINE/arg flags/tracebacks)","wall-clock timestamps"],
 "prohibitions":["evaluation_requests_in_setup_or_selection","retry_failed_start_same_campaign","best_start_or_repeat_selection","post_freeze_flag_workload_or_launcher_change","artificial_delay_floor","differential_delay","sleep_or_spin","mid_layer_collective_or_gate","push_without_separate_instruction","performance_claims_from_smokes"],
 "estimated_runtime":"27 starts x (boot ~8-13 min + calibration pass ~7 min + teardown ~1-2 min) ~ 7.5-9.5 h",
}
tmp=OUT+".tmp"
json.dump(prot,open(tmp,"w"),indent=1)
os.rename(tmp,OUT)
print("WROTE",OUT); print("sha256:",sha(OUT))
