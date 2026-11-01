#!/usr/bin/env python3
"""Freeze the E-P (P11 planner control) protocol. Refuses to overwrite."""
import json, hashlib, os, sys, datetime
EXP="/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CONF=EXP+"/configs/followup_v1"
REPO="/workspace/EPLB/OEPLB/src"
OUT=CONF+"/ep_v1_planner_protocol.json"
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<22),b''): h.update(c)
    return h.hexdigest()
if os.path.exists(OUT): sys.exit("REFUSE overwrite "+OUT)
paths={
 "repo_rebalancer":REPO+"/rebalancer.py",
 "repo_config":REPO+"/config.py",
 "repo_controller":REPO+"/controller.py",
 "counts_calibration_npz":EXP+"/results/fullservice_v3/setup/counts_calibration.npz",
 "counts_calibration_json":EXP+"/results/fullservice_v3/setup/counts_calibration.json",
 "offline_comparison_script":EXP+"/scripts/run_ep_offline_planner_comparison.py",
 "code_lock_reference":CONF+"/ea_v1_code_lock.json",
 "static_verification_reference":EXP+"/results/ea_v1/switch_independence_static.json",
 "ea_grid_protocol_reference":CONF+"/ea_v1_calibration_grid_protocol.json",
}
prot={
 "spec_id":"h20-placement-ep-v1-planner",
 "status":"frozen_before_execution",
 "frozen_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "purpose":("Experiment E-P: fifth arm P11 = the A11 controller (adaptive W + adaptive D, same "
   "thresholds, dead-zone auto 1.096, swap budgets, acceptance margin, executor, recording, model, "
   "parallelism, backend, requests, service protocol) with the planner's pair-selection rule "
   "replaced by a fully specified best-improvement greedy. Two-level evaluation: (1) frozen "
   "layout quality offline on retained REAL calibration counts (deterministic, no timing); "
   "(2) online net system effect in the five-arm campaign (pre-registered contrast A11-P11)."),
 "planner_rule_definition":{
   "name":"best_improvement_rank_pair",
   "implementation":"OEPLB/src/rebalancer.py :: try_build_swap_plan_best_improvement (repo sha locked below; deployed AFTER the E-A grid completes, under a new code lock)",
   "activation":"cfg.planner_best_improvement via env OEPLB_PLANNER_BEST_IMPROVEMENT=1 (same _b() env-fallback mechanism as adaptive_decay; no server_args patch)",
   "search_scope":("per global iteration: worst-ratio non-exhausted layer (same layer ordering as default); "
     "hot ranks descending (break below threshold), cold ranks ascending (break when not lighter) -- same "
     "order and breaks as the default planner; within each untried (hot rank, cold rank) pair: enumerate ALL "
     "16x16 slot pairs, exact O(1) rank-sum simulation, keep the pair minimising the resulting max rank load "
     "(ties: larger delta, then smaller slot_a, then smaller slot_b)"),
   "acceptance":"identical margin: new_ratio < ratio - 0.0005",
   "exhaustion":"rank pair failing the margin is marked tried for the layer; layer exhausted when no untried rank pair remains",
   "budgets":"identical: max_total_ops=300 per build, max_total_swap_layers=94, min_swap_ops=8 controller-side skip",
   "unchanged":"SwapOp construction, DIAG log fields, executor, controller state machine, W/D mechanisms",
   "comparison_statement":("the paper compares these two CONCRETE rules: 'heuristic_first_improvement' "
     "(two-mode pair construction: max-delta if no overshoot else delta closest to gap/2, first pair clearing "
     "the margin wins) vs 'best_improvement_rank_pair' (full enumeration within rank pairs, best wins). "
     "NEITHER is written as 'our method vs the greedy category' -- both are greedy local search.")},
 "offline_frozen_layout_quality":{
   "script":"scripts/run_ep_offline_planner_comparison.py (sha-locked)",
   "inputs":"retained REAL calibration counts counts_calibration.npz (P int64 94x128, logical counts; identity p2l => physical==logical indexing), identity initial placement (trivial map, as every service start)",
   "budget_modes":{"single_build":"one plan build, max_total_ops=300",
     "iterated":"rebuild from updated state until empty plan or plan<min_swap_ops=8, cap 10 builds x 300 ops"},
   "metrics":"per-layer padded W_max (w_e=128*ceil(n_e/128), 0 if n_e==0), r_raw, W_sum, LB tokens, unsplittability price W_max-LB; aggregates mean/sum/max; total ops",
   "determinism":"both planners run twice on identical inputs; plan equality asserted",
   "no_timing":"offline comparison performs no timing measurement; planning-cost exposure is quantified online (PROF planbuild_ms, issued ops, migration wall upper bound)",
   "output":"results/ep_v1/planner_offline_quality.json"},
 "online_evaluation":{
   "arm":"P11 in the five-arm 25-start campaign (task: five rounds round-robin, 5 independent starts per arm)",
   "primary_contrast":"A11 - P11 (pre-registered together with A11 - A00)",
   "explanatory":["planbuild ms per window (PROF lines)","issued swap ops count","migration bytes/wall upper bound (reported alongside, never subtracted)","W/D mechanism counts (must match A11 signatures: P11 shares W/D config)"],
   "distinction_rule":("if P11 offline layout quality is better but online A11-P11 < 0 (A11 faster), the default "
     "rule's advantage is planning/migration cost, not layout; if offline quality is equal and online differs, "
     "the difference is cost-side (planbuild exposure, op counts); if offline P11 worse and online worse, rule quality")},
 "deployment_plan":("repo code is complete and dev-unit-tested (synthetic tensors, labelled DEV-TEST, not "
   "experimental data). Deployed site-packages are NOT touched while the E-A calibration grid runs (no "
   "post-freeze code change mid-campaign). After the grid completes: copy rebalancer.py/config.py/controller.py "
   "to site-packages, py_compile, re-run static switch verification, write campaign code lock (controller + "
   "rebalancer + config shas, deployed==repo except known inert diffs), then freeze the five-arm campaign protocol."),
 "locked_artifacts":{k:{"path":p,"sha256":sha(p)} for k,p in paths.items()},
 "prohibitions":["deployed_code_change_during_frozen_grid","evaluation_requests_in_offline_or_setup","timing_measurements_offline","retry_failed_start_same_campaign","best_start_or_repeat_selection","post_freeze_flag_workload_or_launcher_change","push_without_separate_instruction","greedy_category_framing"],
}
tmp=OUT+".tmp"
json.dump(prot,open(tmp,"w"),indent=1)
os.rename(tmp,OUT)
print("WROTE",OUT); print("sha256:",sha(OUT))
