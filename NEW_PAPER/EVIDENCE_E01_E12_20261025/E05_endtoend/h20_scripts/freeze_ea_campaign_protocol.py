#!/usr/bin/env python3
"""Freeze the five-arm campaign protocol BEFORE any evaluation start.
Run only after: (a) analyze_ea_calibration_grid.py froze W*/alpha* into
ea_v1_fixed_params.json, and (b) deploy_p11_and_lock.py wrote
eacampaign_code_lock.json. Refuses to overwrite an existing protocol.
"""
import json, hashlib, os, sys, datetime

EXP="/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CONF=os.path.join(EXP,"configs/followup_v1")
SITE="/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
REPO="/workspace/EPLB/OEPLB/src"
OUT=os.path.join(CONF,"ea_v1_campaign_protocol.json")
if os.path.exists(OUT): sys.exit("REFUSE to overwrite "+OUT)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''): h.update(b)
    return h.hexdigest()

FP=os.path.join(CONF,"ea_v1_fixed_params.json")
GS=os.path.join(EXP,"results/ea_v1/grid_selection.json")
CL=os.path.join(CONF,"eacampaign_code_lock.json")
for req in (FP,GS,CL):
    if not os.path.exists(req): sys.exit("MISSING prerequisite: "+req)
fp=json.load(open(FP)); gs=json.load(open(GS))
Wstar=fp["W_star"]; Astar=fp["alpha_star"]
assert fp.get("status")=="frozen_before_evaluation"
sel=gs.get("selection",{})
assert sel.get("W_star")==Wstar and sel.get("alpha_star")==Astar, "fixed_params and grid_selection disagree"
print("W*=%s alpha*=%s (selection rule: %s)"%(Wstar,Astar,sel.get("rule","frozen")))

# grid protocol arm mapping (must match)
gp=json.load(open(os.path.join(CONF,"ea_v1_calibration_grid_protocol.json")))
print("grid mapping:",json.dumps(gp["arm_parameter_mapping_after_selection"],ensure_ascii=False))

arms={
 "A00":{"W":Wstar,"alpha":Astar,"adw":0,"add":0,"planner_bi":0,
        "meaning":"fixed W* + fixed alpha* (both adaptive flags OFF); control fixed at the calibration-selected point, not a known-weak strawman"},
 "A10":{"W":16,"alpha":Astar,"adw":1,"add":0,"planner_bi":0,
        "meaning":"adaptive W (start 16, v3 lineage) + fixed alpha*; adaptive_decay OFF"},
 "A01":{"W":Wstar,"alpha":0.5,"adw":0,"add":1,"planner_bi":0,
        "meaning":"fixed W* + adaptive D (base decay 0.5, v3 lineage); adaptive_window OFF"},
 "A11":{"W":16,"alpha":0.5,"adw":1,"add":1,"planner_bi":0,
        "meaning":"both adaptive; byte-identical launch configuration to v3 online arm; contemporaneous re-run required (controller code changed after v3)"},
 "P11":{"W":16,"alpha":0.5,"adw":1,"add":1,"planner_bi":1,
        "meaning":"A11 controller with planner rule best_improvement_rank_pair (E-P fifth arm); OEPLB_PLANNER_BEST_IMPROVEMENT=1"},
}
ARM_ORDER=["A00","A10","A01","A11","P11"]
schedule=[]
for r in range(1,6):
    for pos in range(5):
        a=ARM_ORDER[(r-1+pos)%5]
        schedule.append({"label":f"eacR{r}_{a}","round":r,"position":pos+1,"arm":a})
assert len(schedule)==25
# each arm at each position exactly once
pairs=sorted((e["arm"],e["position"]) for e in schedule)
assert pairs==sorted((a,p) for a in ARM_ORDER for p in range(1,6)), "rotation not a Latin square"

lock=lambda p:{"path":p,"sha256":sha(p)}
prot={
 "spec_id":"h20-placement-ea-v1-fivearm-campaign",
 "status":"frozen_before_execution",
 "frozen_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "purpose":"E-A four-arm W x D ablation + E-P fifth-arm planner-rule control, on 7963 real evaluation requests (v3 order/concurrency 256/service protocol). Answers: A11-A00 net benefit; A11-A10 increment of D history strategy; A11-A01 increment of adaptive W; WxD interaction; A11-P11 net effect of planner rule (same controller).",
 "user_rules_honored":[
  "A11 re-run in this new campaign as contemporaneous control (controller code changed for switch fixes); identity and frozen-ILP NOT re-run (v3 references retained)",
  "five rounds do not guarantee resolution of all small effects; budget frozen; uncertainty reported as uncertainty; no significance-driven additions",
  "migration wall upper bound reported alongside; complete service time already contains real exposed overhead; never subtracted again",
  "7963 requests are NOT 7963 independent replicates; independent starts are replicates; rounds are paired blocks",
  "failed starts retained and retired, never retried under this frozen campaign; no best-start/repeat selection"],
 "fixed_params_source":{"path":FP,"sha256":sha(FP),"W_star":Wstar,"alpha_star":Astar,
  "selection":"argmin mean total_time_s over valid calibration-grid starts; tie-break smaller W, smaller alpha, then (16,0.5); frozen BEFORE any evaluation start; calibration requests only"},
 "arms":arms,
 "schedule":schedule,
 "schedule_design":"5 rounds x 5 arms; round-robin rotation (round r order = ARM_ORDER rotated by r-1); each arm occupies each within-round position exactly once (verified Latin square)",
 "workload":{"path":os.path.join(EXP,"inputs/fullservice_v2/workload/evaluation.jsonl"),
   "sha256":sha(os.path.join(EXP,"inputs/fullservice_v2/workload/evaluation.jsonl")),
   "n_requests":7963,"concurrency":256,"order":"original v3 evaluation order, unmodified",
   "service_protocol":"identical to v3 (run_grid_bench.py, same flags); calibration.jsonl NEVER touched by this campaign"},
 "primary_endpoint":{
   "total_time_s":"complete service time to serve all 7963 evaluation requests (same completed token workload across arms; total_comp_tokens recorded per start)",
   "tps_fixed_caliber":"total_comp_tokens / total_time_s as recorded by run_grid_bench.py"},
 "recording_per_start":{
   "primary":["bench.total_time_s","bench.tps","bench.ok","bench.errors","bench.total_comp_tokens","bench.lat_mean_ms"],
   "W_explanatory":["mechanisms.adw_ratio_jump/adw_shrink/adw_grow/window_shift/window_stable (check counts)","diag.window_events_dp0 (window trajectory with timestamps)","diag.prof_dp0 (per-window record/allreduce/total ms = check-and-act delay)","diag.issued_windows_dp0"],
   "D_explanatory":["server_args_pb_oeplb.decay_factor + adaptive_decay (alpha)","diag.reset_events_dp0 (zeroing moments + converged/prev/current ratios)","mechanisms.adw_decay0_window_fixed","diag.ready_after_forwards (history recovery)","diag.swap_stats.n_reversals/n_repeats (reverse or repeated swaps)"],
   "planner_explanatory":["diag.prof_dp0.planbuild_ms (planning time)","diag.diag_dp0 layers_touched/total_ops/ratios (candidate and effective swaps)","diag.swap_stats.n_ops (executed swaps; migration bytes derivable offline from ops x expert-slot size at frozen geometry)","diag.wall_events + timing_begin_max_ms_per_event (migration wall per event)","mechanisms.planner_bi_log/planner_default_log (rule actually active)"],
   "system":["config_verify CONFIG_OK (actual configuration asserted from server_args dump + planner rule log, every start)","diag.server_args_pb_oeplb (actual config)","mechanisms.tracebacks","boot_rc/bench_rc/status"],
   "raw":["server log (logs/<label>_server.log)","bench json (bench/<label>.json)","diag json (diag/<label>_diag.json, sha256 recorded in state)"]},
 "statistics_plan":{
   "replicate_unit":"independent formal start (5 per arm)","blocks":"rounds (paired)",
   "primary_contrasts_prefixed":["A11-A00 (net adaptive benefit)","A11-P11 (planner rule: heuristic_first_improvement vs best_improvement_rank_pair, same controller)"],
   "secondary_contrasts":["A11-A10 (increment of D history strategy)","A11-A01 (increment of adaptive W)","WxD interaction = (A11-A10)-(A01-A00) per round"],
   "method":"paired per-round differences of total_time_s; mean +/- t(0.975, df=n_pairs-1) 95% CI; negative effects preserved exactly; CI crossing zero reported as 'compatible with zero' (not proof of no effect, not equivalence); no imputation; rounds with an invalid start excluded from that contrast only (n reported)",
   "underpowered_rule":"if fewer than 3 valid paired rounds, contrast reported as underpowered/indeterminate; NO additional runs (budget frozen)",
   "validity_per_start":"status=completed AND boot_rc=0 AND config_verify=CONFIG_OK AND bench_rc=0 AND errors=0 AND ok=7963 AND tracebacks=0",
   "analyzer":{"path":os.path.join(EXP,"scripts/analyze_fivearm_campaign.py"),"sha256":sha(os.path.join(EXP,"scripts/analyze_fivearm_campaign.py"))}},
 "ep_two_level_evaluation":"level 1 frozen-layout quality: from ep_v1 offline (planner_offline_quality.json, sha 5b460ee0...) + per-start diag layout-quality proxies (avg/max ratio_after); level 2 online net effect: A11-P11 paired contrast incl. planbuild and migration-wall descriptors; distinction_rule per ep_v1_planner_protocol.json (better layout found vs less planning/migration time)",
 "identity_ilp_policy":"v3 identity (400.85s) and frozen-ILP (389.11s) references NOT re-run per user rule; A11 in this campaign serves as the contemporaneous online control",
 "locked_artifacts":{
   "evaluation_workload":lock(os.path.join(EXP,"inputs/fullservice_v2/workload/evaluation.jsonl")),
   "deployed_controller":lock(os.path.join(SITE,"controller.py")),
   "deployed_rebalancer":lock(os.path.join(SITE,"rebalancer.py")),
   "deployed_config":lock(os.path.join(SITE,"config.py")),
   "repo_controller":lock(os.path.join(REPO,"controller.py")),
   "repo_rebalancer":lock(os.path.join(REPO,"rebalancer.py")),
   "repo_config":lock(os.path.join(REPO,"config.py")),
   "campaign_launcher":lock(os.path.join(CONF,"ea_launchers/launch235b_eacampaign.sh")),
   "campaign_driver":lock(os.path.join(EXP,"scripts/run_fivearm_campaign.sh")),
   "campaign_analyzer":lock(os.path.join(EXP,"scripts/analyze_fivearm_campaign.py")),
   "bench_tool":lock("/workspace/EPLB/OEPLB/scripts/run_grid_bench.py"),
   "campaign_code_lock":lock(CL),
   "fixed_params":lock(FP),
   "grid_selection":lock(GS),
   "grid_protocol":lock(os.path.join(CONF,"ea_v1_calibration_grid_protocol.json")),
   "ea_code_lock_superseded":lock(os.path.join(CONF,"ea_v1_code_lock.json")),
   "c5_dynamic_verification":lock(os.path.join(EXP,"results/ea_v1/c5_dynamic_verification.json")),
   "static_verification_postp11":lock(os.path.join(EXP,"results/ea_v1/switch_independence_static_postp11.json")),
   "ep_planner_protocol":lock(os.path.join(CONF,"ep_v1_planner_protocol.json")),
   "ep_offline_quality":lock(os.path.join(EXP,"results/ep_v1/planner_offline_quality.json")),
   "v3_protocol_reference":lock(os.path.join(CONF,"fullservice_protocol_v3.json"))},
 "prohibitions":["push_without_separate_instruction","best_start_or_repeat_selection","overwrite_existing_raw_result","retry_failed_start_same_campaign","post_freeze_flag_workload_or_launcher_change","deployed_code_change_during_campaign","evaluation_requests_in_setup_or_map_generation","subtract_overhead_from_measured_contrast","artificial_delay_floor","differential_delay","sleep_or_spin","mid_layer_collective_or_gate","significance_driven_extra_runs"],
 "estimated_runtime":"25 starts x ~8 min (boot ~3 min + bench ~400 s + teardown) ~= 3.5-4 h",
}
tmp=OUT+".tmp"
json.dump(prot,open(tmp,"w"),indent=1,ensure_ascii=False)
os.rename(tmp,OUT)
print("WROTE",OUT)
print("sha256",sha(OUT))
print("FROZEN")
