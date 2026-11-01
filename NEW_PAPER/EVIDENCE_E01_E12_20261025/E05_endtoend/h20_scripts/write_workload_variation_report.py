#!/usr/bin/env python3
"""Assemble the offline workload-variation analysis artifacts (read-only over raw data)."""
import json, os, hashlib, statistics
from datetime import datetime, timezone

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUT = f"{EXP}/results/ea_v1/workload_variation"
tl = json.load(open(f"{OUT}/timeline_events.json"))
ct = json.load(open(f"{EXP}/results/crosstab_v1/analysis.json"))
tk = json.load(open("/tmp/eval_tokens.json"))

def sha(p):
    h=hashlib.sha256()
    for c in open(p,'rb'): h.update(c)
    return h.hexdigest()

# ---- part2 computation (recomputed here for the artifact) ----
LAYERS=['layer-024','layer-028','layer-044','layer-071']
COLS=['identity','base-of-evaluation-01','base-of-evaluation-02','base-of-evaluation-03']
per_layer={}; tot_s=tot_w=0.0
for lname in LAYERS:
    L=ct['layers'][lname]
    sums={c:0.0 for c in COLS}; rowmin=0.0; rowmin_col={}
    for rn,row in L['rows'].items():
        best=min(COLS,key=lambda c:row['cells'][c]['mean_us'])
        rowmin+=row['cells'][best]['mean_us']; rowmin_col[rn]=best
        for c in COLS: sums[c]+=row['cells'][c]['mean_us']
    sc=min(COLS,key=lambda c:sums[c]); gap=sums[sc]-rowmin
    per_layer[lname]={"T_static_cand_us":round(sums[sc],1),"T_static_col":sc,
                      "T_switch_cand_us":round(rowmin,1),"oracle_gap_us":round(gap,1),
                      "oracle_gap_pct":round(100*gap/sums[sc],3),"row_min_cols":rowmin_col,
                      "verdict":L['verdict'],"tracking_room_supported":L['tracking_room_supported']}
    tot_s+=sums[sc]; tot_w+=rowmin
pooled={"T_static_cand_us":round(tot_s,1),"T_switch_cand_us":round(tot_w,1),
        "oracle_gap_us":round(tot_s-tot_w,1),"oracle_gap_pct":round(100*(tot_s-tot_w)/tot_s,3),
        "K_note":"rows are 3 same-domain (realprover) batches of ~4050 hidden rows each -> equal K_i"}

# ---- representative timeline: A11 R1 ----
s=tl['starts']['eacR1_A11']
rep={"run":"eacR1_A11","dur_s":s['dur_s'],"bounds_frac":tl['boundaries_cum_token_frac'],
     "resets":[{"frac":e['frac'],"delta":round(e['current']-e['prev'],3),"zone":e['zone']} for e in s['events']['reset']],
     "cos":[{"w":e['w'],"frac":e['frac'],"cos":e['cos'],"zone":e['zone']} for e in s['events']['stability']],
     "adw":[{"frac":e['frac'],"kind":e['kind'],"jump":e['jump'],"zone":e['zone']} for e in s['events']['adw']],
     "walls":[{"frac":w['frac'],"max_ms":w['max_ms']} for w in s['walls']]}

analysis={
 "analysis":"workload_variation_and_multidomain_tracking_space_offline",
 "created_at_utc":datetime.now(timezone.utc).isoformat(),
 "mode":"offline_only_no_new_experiments",
 "inputs":{"timeline_events":{"path":f"{OUT}/timeline_events.json","sha256":sha(f"{OUT}/timeline_events.json")},
           "crosstab_analysis":{"path":f"{EXP}/results/crosstab_v1/analysis.json","sha256":sha(f"{EXP}/results/crosstab_v1/analysis.json")},
           "fivearm_analysis":{"path":f"{EXP}/results/ea_v1/fivearm_campaign_analysis.json","sha256":sha(f"{EXP}/results/ea_v1/fivearm_campaign_analysis.json")},
           "eval_token_counts_source":"/tmp/eval_tokens.json (tokenizer: Qwen3-235B-A22B-FP8 tokenizer.json, add_special_tokens=False)"},
 "part1_workload_structure":{
   "source_file":"/data/minghua/sjq/OEPLBdata/datasets/multi_domain/prefill_heavy_universal.jsonl (16000 req = 4 x 4000 contiguous per-domain blocks)",
   "evaluation_order":"NOT a random mix: 4 contiguous domain segments sharegpt[0..1924] prover[1925..3947] book[3948..5931] code[5932..7962]; adjacent-order domain switches = 3",
   "calibration_order":"same 4-block structure (boundaries 2075/4052/6068)",
   "per_domain":{"sharegpt":{"n":1925,"tokens":1344654,"token_frac":0.1570,"mean_tok_per_req":698},
                 "prover":{"n":2023,"tokens":1028280,"token_frac":0.1201,"mean_tok_per_req":508},
                 "book":{"n":1984,"tokens":2031281,"token_frac":0.2372,"mean_tok_per_req":1024},
                 "code":{"n":2031,"tokens":4159488,"token_frac":0.4857,"mean_tok_per_req":2048}},
   "total_tokens":8563703,
   "boundaries_cum_token_frac":{"sharegpt|prover":0.1570,"prover|book":0.2771,"book|code":0.5143},
   "boundaries_wall_est_s_at_370s_run":[58.1,102.5,190.3],
   "gpu_batch_mixing":{"direct_record":"NONE - per-batch domain composition was never logged; explicitly unanswerable at batch granularity",
     "analytic_inference":"concurrency=256 in-flight requests -> each boundary produces one ~256-request (~10-12 s) two-domain transition zone; within segments forwards pack same-domain requests (chunked prefill: ~16 prover reqs or ~4 code reqs per 8192-token forward). Client-side domain switches (3) therefore equal GPU-workload transitions (3 short zones), NOT high-frequency switching."},
   "distribution_probes":{
     "cos_sim_semantics":"cosine between consecutive-window RAW all-reduced logical load vectors (per-layer, averaged); placement-independent; alpha-independent; logged by DP0 every 4th window only",
     "A00_clean_probe":"all cos >= 0.9551 across 5 starts (segment means 0.9648-0.9946) -> within-segment distribution near-stationary at 16-forward granularity",
     "sampling_caveat":"every-4th-window logging can miss a one-window step at a boundary (A00 boundary windows ~w#10.5/18.6/34.4 were not logged); boundary steps are instead evidenced by the D arms' ratio detectors which see every window",
     "boundary_evidence":"A01: 4 resets/start in frac 0.17-0.23 right after boundary1 in 5/5 starts (deltas 0.15/0.465/0.057/0.051); A11: resets at all three boundaries in most starts (frac 0.157-0.22, 0.28-0.30, 0.51-0.55) plus mid-segment extra; P11 similar to A11",
     "per_domain_histograms":"NEVER recorded anywhere (grep over results: none); all 18 prediction_v1 real captures come from the SINGLE-domain dataset L512_O1_realprover_n8192.jsonl -> per-domain normalized distributions cannot be shown"},
   "w_reset_attribution":{
     "exogenous_part":"A01 first reset per start locks to boundary1 across all 5 starts with large delta (0.15-0.48) - genuine domain-shift detection; A11/P11 boundary2/boundary3 resets likewise",
     "noise_part":"A01 subsequent resets delta only 0.05-0.07 (just above the 0.03 threshold) clustered in the prover segment = short-prompt sampling noise re-fires; A10 (W-only): ~16 shrink + 11 grow events per start spread over ALL segments, largest ratio jumps (mean 0.27, max 0.49) inside prover, not at boundaries; A10 cos drops to 0.58-0.88 in prover while A00 (same alpha=0, fixed W=16) stays >=0.958 -> the drop is self-induced by window shrink (8-forward windows halve tokens/window), i.e. endogenous churn, not exogenous change",
     "conclusion":"W changes mainly reflect noise amplification + window-length self-feedback in the short-prompt segment; resets partially track the 3 real domain boundaries but re-fire on noise; A11 carries 3-5 extra mid-segment resets per start vs A01"},
   "unanswerable_with_current_records":[
     "per-batch (GPU forward) domain composition - not logged",
     "per-domain normalized logical expert distributions - no routing dumps from the multi-domain runs; all real captures are single-domain realprover",
     "exact wall-clock times of domain boundaries - inferred under uniform token throughput (supported: A00 window#-vs-wall r2=0.9999) but per-domain rate differences not directly measured; +-few % uncertainty",
     "cross-arm absolute cos_sim comparison when window lengths differ (window size changes histogram sampling noise)"],
   "representative_timeline_A11_R1":rep},
 "part2_tracking_space":{
   "four_level_discrimination":{
     "L1_domain_labels_differ":"YES - 4 contiguous segments, 3 switches/run, segments 58-190 s",
     "L2_expert_distributions_differ":"INDIRECT YES - boundary ratio jumps +0.15..+0.48 and reproducible reset firing; magnitudes per domain NOT quantifiable (no histograms recorded)",
     "L3_greedy_layouts_differ_per_domain":"UNMEASURED - no per-domain layouts were ever built (building them requires per-domain routing counts that do not exist)",
     "L4_execution_time_superiority_per_domain":"WITHIN-DOMAIN MEASURED ~ ZERO; CROSS-DOMAIN UNMEASURED (no sharegpt/book/code captures exist)"},
   "crosstab_scope_warning":"crosstab_v1 rows are 3 batches of the SAME domain (realprover): common_static_sufficient there certifies absence of tracking room for within-domain batch-to-batch variation only; it neither proves nor disproves cross-domain room (both directions explicitly avoided)",
   "oracle_bound_within_domain":per_layer,"pooled":pooled,
   "cost_side_measured":{
     "adaptivity_net_service_cost_A11_minus_A00":"+10.40 s CI [+5.00,+15.80] (excludes zero)",
     "module_costs":"A10-A00=+4.79 s CI crosses zero; A01-A00=+6.45 s CI crosses zero",
     "planbuild_ms_per_window":{"A11_mean":39.1,"A11_max":64.6,"P11_mean":98.6,"P11_max":207.9,"A00_mean":58.2,"A00_max":95.3},
     "migration_wall_s_per_event":"1.3-2.7 (max per start 1.56-2.68); initial rebalance window#1 234 swaps ~2.3 s",
     "per_window_controller_overhead_s":"record+allreduce+finalize ~0.9-1.9 (present in A00 too; marginal adaptivity cost is the +10.4 s net)"},
   "break_even_estimate":{
     "assumptions":"MoE layer replay ~1450 us/layer/forward x 94 layers ~ 136 ms/forward; full service ~350 ms/forward (370 s / ~1050 forwards) -> expert-execution share ~39%; oracle gap g = fraction of layer-forward time that per-domain switching recovers over the best common static layout (K-weighted)",
     "formula":"ideal service-level gain ~ 0.39 * g * 370 s, before capture-rate discount and transition costs",
     "scenarios":{"g=0.17% (measured within-domain)":"~0.24 s << +10.4 s measured cost",
                   "g=2%":"~2.9 s - still below measured adaptivity cost",
                   "g=5%":"~7.2 s - break-even only if the controller captures most of it AND churn is tamed to near the 3 real switches",
                   "g=10%":"~14.4 s - net win plausible"},
     "conclusion":"cross-domain g is UNKNOWN and is the single missing fact; with only 3 switches/run and measured within-domain g=0.17%, the prior is unfavorable but not decisive -> measure g offline-cheap first (Phase A), do NOT jump to service-level experiments"},
   "code_domain_note":"code = 48.6% of tokens: a code-weighted STATIC layout (or the existing mixed-calibration static) already captures most of any achievable gain without adaptation"},
 "part3_minimal_design_status":"PROTOCOL_AND_BUDGET_ONLY_NOT_RUN - see report md sections Phase A / Phase B",
 "deliverables_summary":{
   "a_trackable_variation":"structurally yes but sparse: 3 exogenous domain switches per run (long 58-190 s segments), real load-pattern jumps at boundaries; within-segment near-stationary; the most frequent 'variation' detected by the controller is prover-segment sampling noise amplified by adaptive window shrink (self-induced)",
   "b_multidomain_pro_con":"see report md table (4 pro / 6 con); decisive unknown = cross-domain level-4 execution-time superiority",
   "c_minimal_matrix":"Phase A cross-domain capture-replay crosstab (16 capture units, 160 replay starts, ~1 GPU-day, pre-registered gates) -> conditional Phase B paired-order service experiment (2 arms x 3 orders x 5 rounds = 30 starts, ~6 GPU-h, pre-registered success/failure)",
   "d_not_needed_now":"five-arm reruns, gating-G implementation, BI-planner follow-ups, W/alpha re-grids, identity/ILP reruns, delay/floor studies, synthetic-route studies, capture expansion before gates pass"},
 "invariants_reaffirmed":["offline only; no experiment launched","no raw artifact overwritten; outputs in new directory only","no push without separate instruction","evaluation requests never used for layout construction in any proposed design","negative results preserved exactly; CI crossing zero worded as compatible-with-zero"],
}
ap=f"{OUT}/workload_variation_analysis.json"
assert not os.path.exists(ap), "refuse overwrite"
json.dump(analysis,open(ap,'w'),indent=1)
print("WROTE",ap,sha(ap))
