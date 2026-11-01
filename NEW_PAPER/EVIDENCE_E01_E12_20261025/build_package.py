#!/usr/bin/env python3
# E01-E12 evidence audit packaging script (READ-ONLY on sources; writes ONLY under PKG).
# Policy: copy files <= copy_max into package; larger files -> path+sha256 in manifest;
# core dumps -> path+size only (no hash, retained in place); missing sources -> SOURCE_MISSING entry.
import os, sys, json, hashlib, shutil, time
from datetime import datetime, timezone

PKG = "/workspace/EPLB/NEW_PAPER/EVIDENCE_E01_E12_20261025"
HASH_CAP = 2 * 1024**3
COPYALL_GUARD = 200 * 1024**2
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
MB = 1024**2

H20 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
RES = H20 + "/results"
FF  = RES + "/final_followup_20261020"
FP  = RES + "/final_plan_20261019"
DZ  = "/workspace/EPLB/NEW_PAPER/experiments/deadzone_combine_20261011"
MBD = "/workspace/EPLB/NEW_PAPER/experiments/microbench_deepgemm"
GB  = "/workspace/EPLB/NEW_PAPER/experiments/gb200_ep4_repro"
A100B = "/workspace/EPLB/NEW_PAPER/experiments/a100_baselines_repro_20260926"
A100BF = "/workspace/EPLB/NEW_PAPER/experiments/a100_bf16_comparison"
D38 = "/workspace/EPLB/NEW_PAPER/experiments/d38_L512_O1_comparison"
THA = "/workspace/EPLB/NEW_PAPER/experiments/a100_theory_validation_20260929"
THO = "/workspace/EPLB/OEPLB/theory_a100"
OE  = "/workspace/EPLB/OEPLB"
BM  = "/workspace/EPLB/benchmark"
SJQ = "/data/minghua/sjq"
NP  = "/workspace/EPLB/NEW_PAPER"
PPT = SJQ + "/paperpicturetrace"

manifest = []
stats = {}
t_start = time.time()

def log(msg):
    print("[%7.1fs] %s" % (time.time() - t_start, msg), flush=True)

def is_core(name):
    return name.startswith("core-") or name.startswith("core.")

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def rec(e, tag, src, rel, copy_to=None, do_hash=True, note="", hash_max=HASH_CAP):
    try:
        st = os.stat(src)
    except OSError as ex:
        manifest.append({"e": e, "tag": tag, "src": src, "rel": rel, "bytes": None,
                         "mtime": None, "sha256": None, "packaged": None,
                         "note": (note + "; " if note else "") + "stat_error:%s" % ex})
        return
    entry = {"e": e, "tag": tag, "src": src, "rel": rel, "bytes": st.st_size,
             "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
             "sha256": None, "packaged": None, "note": note}
    bn = os.path.basename(src)
    if is_core(bn):
        entry["note"] = (note + "; " if note else "") + "CORE_DUMP: hash skipped, retained in place as forensic evidence"
    elif os.path.islink(src):
        entry["note"] = (note + "; " if note else "") + "symlink->%s" % os.path.realpath(src)
        if do_hash and st.st_size <= hash_max:
            try: entry["sha256"] = sha256_of(src)
            except Exception as ex: entry["note"] += "; hash_error:%s" % ex
    elif do_hash and st.st_size <= hash_max:
        try: entry["sha256"] = sha256_of(src)
        except Exception as ex: entry["note"] += "; hash_error:%s" % ex
    elif do_hash:
        entry["note"] = (entry["note"] + "; " if entry["note"] else "") + "hash skipped: size>%dMB cap" % (hash_max // MB)
    if copy_to:
        if st.st_size > COPYALL_GUARD:
            entry["note"] = (entry["note"] + "; " if entry["note"] else "") + "copy skipped: size>COPYALL_GUARD(200MB), path+sha only"
        else:
            try:
                os.makedirs(os.path.dirname(copy_to), exist_ok=True)
                shutil.copy2(src, copy_to)
                entry["packaged"] = os.path.relpath(copy_to, PKG)
            except Exception as ex:
                entry["note"] = (entry["note"] + "; " if entry["note"] else "") + "copy_error:%s" % ex
    manifest.append(entry)
    s = stats.setdefault(e, {"files": 0, "bytes": 0, "copied": 0, "copied_bytes": 0, "hashed": 0, "missing": 0})
    s["files"] += 1; s["bytes"] += st.st_size
    if entry["packaged"]: s["copied"] += 1; s["copied_bytes"] += st.st_size
    if entry["sha256"]: s["hashed"] += 1

def miss(e, tag, path, rel, note="SOURCE_MISSING"):
    manifest.append({"e": e, "tag": tag, "src": path, "rel": rel, "bytes": None, "mtime": None,
                     "sha256": None, "packaged": None, "note": note})
    s = stats.setdefault(e, {"files": 0, "bytes": 0, "copied": 0, "copied_bytes": 0, "hashed": 0, "missing": 0})
    s["missing"] += 1

def walk_root(r):
    e, tag, root, mode = r["e"], r["tag"], r["path"], r["mode"]
    copy_max = r.get("copy_max", 0)
    hash_max = r.get("hash_max", HASH_CAP)
    t0 = time.time()
    if not os.path.exists(root):
        miss(e, tag, root, "", "SOURCE_MISSING (whole root)")
        log("ROOT %s/%s MISSING" % (e, tag)); return
    if os.path.isfile(root):
        rec(e, tag, root, os.path.basename(root),
            copy_to=os.path.join(PKG, e, tag, os.path.basename(root)) if mode != "hash_only" and mode != "list_only" else None,
            do_hash=mode != "list_only", hash_max=hash_max)
        log("ROOT %s/%s single-file done" % (e, tag)); return
    if mode == "depth0":
        for fn in sorted(os.listdir(root)):
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                rec(e, tag, p, fn, copy_to=os.path.join(PKG, e, tag, fn), do_hash=True, hash_max=hash_max)
        log("ROOT %s/%s depth0 done (%.1fs)" % (e, tag, time.time() - t0)); return
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, root)
            try: size = os.stat(src).st_size
            except OSError: size = -1
            do_copy = False
            if mode == "copy_all" and not is_core(fn): do_copy = True
            elif mode in ("copy_small",) and 0 <= size <= copy_max and not is_core(fn): do_copy = True
            do_hash = mode != "list_only"
            dest = os.path.join(PKG, e, tag, rel) if do_copy else None
            rec(e, tag, src, rel, copy_to=dest, do_hash=do_hash, hash_max=hash_max)
            n += 1
            if n % 500 == 0: log("  ...%s/%s %d files (%.0fs)" % (e, tag, n, time.time() - t0))
    s = stats.get(e, {})
    log("ROOT %s/%s done: %d files (%.1fs)" % (e, tag, n, time.time() - t0))

def files_root(r):
    e, tag, root, rels = r["e"], r["tag"], r["path"], r["rels"]
    hash_max = r.get("hash_max", HASH_CAP)
    for rel in rels:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            rec(e, tag, p, rel, copy_to=os.path.join(PKG, e, tag, rel), do_hash=True, hash_max=hash_max)
        else:
            miss(e, tag, p, rel)
    log("ROOT %s/%s files done (%d rels)" % (e, tag, len(rels)))

def R(e, tag, path, mode, copy_max=0, hash_max=HASH_CAP, rels=None, note=""):
    return {"e": e, "tag": tag, "path": path, "mode": mode, "copy_max": copy_max, "hash_max": hash_max, "rels": rels, "note": note}

C = "_COMMON_docs"
ROOTS = [
 # ---------------- _COMMON ----------------
 R(C, "h20_top_docs", H20, "depth0"),
 R(C, "ff_top_docs", FF, "depth0"),
 R(C, "oeplb_top_docs", OE, "files", rels=["README.md","outline.md","MOTIVATION_DATA.md","DATA_CATALOG.md","ADAPTIVE_DESIGN.md","SYSTEM.md","QUICKSTART.md.bak_before_repro_align"]),
 R(C, "newpaper_top_docs", NP, "files", rels=["EXPERIMENT_TRACE_INDEX.md","PB_OEPLB_FINAL_EXPERIMENT_PLAN_20261019.md","paper.md","paper_GB200.md","pictureguide/PICTUREGUIDE.md"]),
 # ---------------- E01 ----------------
 R("E01_count_padding", "microbench_deepgemm", MBD, "copy_all"),
 R("E01_count_padding", "h20_e0e1_files", RES, "files", rels=["e0_pilot.log","e0_smoke.log","e1_pilot.log","e1_layout_effects.csv","e1_summary.json","raw_index.json","evidence_index_final.json"]),
 R("E01_count_padding", "h20_raw", RES + "/raw", "copy_all"),
 R("E01_count_padding", "confirmatory_v1", RES + "/confirmatory_v1", "copy_all"),
 R("E01_count_padding", "microbench_hooks", OE + "/scripts", "files", rels=["microbench_hook.py","microbench_hook2.py","gemm_profiler_hook.py","layer_imbalance_analysis.py","analyze_expert_heatmap.py"]),
 R("E01_count_padding", "deepgemm_precompile_log", "/workspace/logs", "files", rels=["deepgemm_precompile.log"]),
 R("E01_count_padding", "fig1_motivation_crossdomain_npz_LIST", PPT + "/fig1_2_3_4_7_9_13_motivation_crossdomain", "list_only"),
 # ---------------- E02 ----------------
 R("E02_readiness_delay", "deadzone_top_docs", DZ, "depth0"),
 R("E02_readiness_delay", "deadzone_scripts", DZ + "/scripts", "copy_all"),
 R("E02_readiness_delay", "deadzone_logs", DZ + "/logs", "copy_all"),
 R("E02_readiness_delay", "deadzone_build", DZ + "/build", "copy_all"),
 R("E02_readiness_delay", "deadzone_results", DZ + "/results", "copy_small", copy_max=4*MB),
 R("E02_readiness_delay", "deadzone_traces", DZ + "/traces", "hash_only"),
 # ---------------- E03 ----------------
 R("E03_bound_crosshw", "a100_theory_validation_newpaper", THA, "copy_all"),
 R("E03_bound_crosshw", "theory_a100_oeplb", THO, "copy_all"),
 R("E03_bound_crosshw", "oeplb_repro", OE + "/repro", "copy_all"),
 R("E03_bound_crosshw", "a100_baselines_repro", A100B, "copy_all"),
 R("E03_bound_crosshw", "a100_bf16_comparison", A100BF, "copy_all"),
 R("E03_bound_crosshw", "gb200_top_docs", GB, "depth0"),
 R("E03_bound_crosshw", "gb200_results", GB + "/results", "copy_all"),
 R("E03_bound_crosshw", "gb200_profiles", GB + "/profiles", "copy_all"),
 R("E03_bound_crosshw", "gb200_placements", GB + "/placements", "copy_all"),
 R("E03_bound_crosshw", "gb200_scripts", GB + "/scripts", "copy_all"),
 R("E03_bound_crosshw", "gb200_logs", GB + "/logs", "copy_small", copy_max=2*MB),
 R("E03_bound_crosshw", "prediction_v1", RES + "/prediction_v1", "copy_small", copy_max=2*MB),
 R("E03_bound_crosshw", "prediction_capture_failed_attempt01", RES + "/prediction_capture_failed_preprofile_attempt01_20261012T052543Z", "copy_small", copy_max=2*MB),
 R("E03_bound_crosshw", "oeplb_fig_bound", OE, "files", rels=["fig_bound.png"]),
]
# sjq cross-HW bench grids (small, copy all)
for d in ["A100_result","A100_fresult","A100_presult","A100_lamma_result","A100_lama70B_result","A100_qwen32B_result",
          "H20_result","H20_fresult","H20_presult","H20_lamma_result","H20_lama70B_result",
          "H800_result","H800_presult","H800_lamma_result","H800_lama70B_result","H800_qwen32B_result"]:
    ROOTS.append(R("E03_bound_crosshw", "sjq_" + d, SJQ + "/" + d, "copy_all"))
# sjq profiler trees (hash-only)
for d in ["A100profiler_result","A100lama70B_profiler_result","H20profiler_result","H20lama70B_profiler_result",
          "H800profiler_result","H800lama70B_profiler_result"]:
    ROOTS.append(R("E03_bound_crosshw", "sjq_" + d + "_HASHONLY", SJQ + "/" + d, "hash_only"))
ROOTS += [
 R("E03_bound_crosshw", "sjq_H800profiler_tar_HASHONLY", SJQ, "files", rels=["H800profiler_result.tar.gz"]),
 R("E03_bound_crosshw", "figG_L_bound_theory_npz_LIST", PPT + "/figG_H_I_J_K_L_bound_theory", "list_only"),
 # ---------------- E04 ----------------
 R("E04_pd_representativeness", "oeplb_pd_docs", OE, "files", rels=["PREFILL_DECODE_CORRELATION_STUDY.md","PREFILLBOUNDARY_DATA.md","short_pd_9dataset_samesession.json","short_pd_all7.json","short_pd_conc256_O10.json"]),
 R("E04_pd_representativeness", "capture_and_pd_scripts", BM, "files", rels=["run_routing_record.sh","run_expert_dist.sh","bench_expert_swap.py","run_pd_benchmark.sh","analysis/compute_pd_rho.py"]),
 R("E04_pd_representativeness", "freeze_segmented_scripts", OE + "/scripts", "files", rels=["freeze_requests.py"]),
 R("E04_pd_representativeness", "gb200_nine_task_datasets", GB + "/datasets", "copy_all"),
 R("E04_pd_representativeness", "h20_placements_fixed_map", H20 + "/placements", "copy_all"),
 R("E04_pd_representativeness", "rt2_nine_task_npz_HASHONLY", PPT + "/fig5_6_8_14_PD_correlation", "hash_only"),
 R("E04_pd_representativeness", "ppt_readmes", PPT, "files", rels=["README.md","MASTER_README.md"]),
 R("E04_pd_representativeness", "pdcorr_3task", SJQ + "/OEPLBdata/datasets/prefill_decode_correlation", "copy_small", copy_max=1*MB, hash_max=100*MB),
 # ---------------- E05 ----------------
 R("E05_endtoend", "phase4_v2_fixed", FF + "/phase4_core", "copy_all"),
 R("E05_endtoend", "code_lock_final", FF + "/code_lock", "copy_all"),
 R("E05_endtoend", "final_plan_top", FP, "depth0"),
 R("E05_endtoend", "final_plan_scripts", FP + "/scripts", "copy_all"),
 R("E05_endtoend", "final_plan_analysis", FP + "/analysis", "copy_all"),
 R("E05_endtoend", "final_plan_c_configs", FP + "/C_configs", "copy_all"),
 R("E05_endtoend", "final_plan_d_matrix", FP + "/D_matrix", "copy_all"),
 R("E05_endtoend", "final_plan_calib", FP + "/calib", "copy_all"),
 R("E05_endtoend", "final_plan_smoke_f0", FP + "/smoke_f0", "copy_all"),
 R("E05_endtoend", "final_plan_p1_smoke", FP + "/p1_smoke", "copy_all"),
 R("E05_endtoend", "final_plan_main", FP + "/main", "copy_small", copy_max=128*1024),
 R("E05_endtoend", "final_plan_p1", FP + "/p1", "copy_small", copy_max=128*1024),
 R("E05_endtoend", "final_plan_b_data", FP + "/B_data", "copy_small", copy_max=1*MB, hash_max=200*MB),
 R("E05_endtoend", "final_plan_logs", FP + "/logs", "copy_small", copy_max=1*MB),
 R("E05_endtoend", "h20_configs_followup_v1", H20 + "/configs", "copy_small", copy_max=2*MB, hash_max=200*MB),
 R("E05_endtoend", "h20_scripts", H20 + "/scripts", "copy_small", copy_max=512*1024, hash_max=50*MB),
 R("E05_endtoend", "d38_L512_O1_comparison", D38, "copy_all"),
 # ---------------- E06 ----------------
 R("E06_slo_goodput", "phase3_prep", FF + "/phase3_prep", "copy_all"),
 R("E06_slo_goodput", "bench_client_v2_copy", FP + "/scripts", "files", rels=["bench_client_v2.py"]),
 # ---------------- E07 ----------------
 R("E07_phase_aware", "sp_coupling_v1", RES + "/sp_coupling_v1", "copy_small", copy_max=2*MB),
 R("E07_phase_aware", "cdx0_v1", RES + "/cdx0_v1", "copy_small", copy_max=2*MB),
 R("E07_phase_aware", "crosstab_v1", RES + "/crosstab_v1", "copy_small", copy_max=2*MB),
 R("E07_phase_aware", "fullservice_v1", RES + "/fullservice_v1", "copy_all"),
 R("E07_phase_aware", "fullservice_v2", RES + "/fullservice_v2", "copy_small", copy_max=2*MB),
 R("E07_phase_aware", "fullservice_v3", RES + "/fullservice_v3", "copy_small", copy_max=2*MB),
 R("E07_phase_aware", "et_v1", RES + "/et_v1", "copy_all"),
 R("E07_phase_aware", "tracer_src", OE + "/src", "files", rels=["per_request_tracer.py","routing_tracer.py"]),
 R("E07_phase_aware", "perds_gain", OE, "files", rels=["perds_gain.json"]),
 R("E07_phase_aware", "figA_E_overhead_npz_LIST", PPT + "/figA_B_C_D_E_placement_overhead_eplb", "list_only"),
 # ---------------- E08 ----------------
 R("E08_planner_gap", "ea_v1", RES + "/ea_v1", "copy_small", copy_max=2*MB),
 R("E08_planner_gap", "ep_v1", RES + "/ep_v1", "copy_all"),
 R("E08_planner_gap", "nearscan_v1", RES + "/nearscan_v1", "copy_small", copy_max=2*MB),
 R("E08_planner_gap", "rebalancer_src", OE + "/src", "files", rels=["rebalancer.py","probe_admission.py"]),
 R("E08_planner_gap", "predict_gain_tool", OE + "/tools", "files", rels=["predict_gain.py"]),
 # ---------------- E09 ----------------
 R("E09_adaptive_ablation", "adw_v2", RES + "/adw_v2", "copy_small", copy_max=2*MB),
 R("E09_adaptive_ablation", "adw_v2p1", RES + "/adw_v2p1", "copy_small", copy_max=2*MB),
 R("E09_adaptive_ablation", "step6", RES + "/step6", "copy_small", copy_max=2*MB),
 R("E09_adaptive_ablation", "stopping_v1", RES + "/stopping_v1", "copy_small", copy_max=2*MB),
 R("E09_adaptive_ablation", "oeplb_adaptive_results", OE, "files", rels=["oeplb_adaptive_both_constructions.json","oeplb_fixed_adaptive_results.json","reproduce_d38_adaptive.json"]),
 R("E09_adaptive_ablation", "h20_inputs_fullservice", H20 + "/inputs", "copy_small", copy_max=1*MB, hash_max=200*MB),
 R("E09_adaptive_ablation", "figF_ablation_npz_LIST", PPT + "/figF_F2_M_N_ablation", "list_only"),
 R("E09_adaptive_ablation", "fig15_17_online_npz_LIST", PPT + "/fig15_15b_16_17_17b_oeplb_online", "list_only"),
 # ---------------- E10 ----------------
 R("E10_reliability_cost", "correctness", FF + "/correctness", "copy_small", copy_max=8*MB),
 R("E10_reliability_cost", "forensics", FF + "/forensics", "copy_all"),
 R("E10_reliability_cost", "freeze_code_fixed", FF + "/freeze", "copy_all"),
 R("E10_reliability_cost", "cpu_audit", FF + "/cpu_audit", "copy_all"),
 R("E10_reliability_cost", "locked_v2_frozen", OE + "/src/locked_v2_frozen_20261016", "copy_all"),
 R("E10_reliability_cost", "correctness_logs_h20", RES, "files", rels=["correctness_all_layouts_pilot.log","correctness_identity_smoke.log","correctness_tolerance_validation.log"]),
 R("E10_reliability_cost", "cost_originals_scripts", BM, "files", rels=["bench_nvl72_p2p.py","bench_ce_bandwidth.py","bench_cross_node_ce.py","bench_cross_node_ce_v2.py","bench_cross_node_ce_v3.py"]),
 R("E10_reliability_cost", "src_snapshot_current", OE + "/src", "copy_small", copy_max=1*MB),
 # ---------------- E11 ----------------
 R("E11_baselines_redundancy", "moetuner", OE + "/baselines/moetuner", "copy_small", copy_max=4*MB),
 R("E11_baselines_redundancy", "baselines_a100", OE + "/baselines/a100", "copy_small", copy_max=4*MB, hash_max=100*MB),
 R("E11_baselines_redundancy", "baselines_top_docs", OE + "/baselines", "depth0"),
 R("E11_baselines_redundancy", "datafore_repro", OE + "/datafore_repro", "copy_all"),
 R("E11_baselines_redundancy", "quickstart_a100", OE + "/quickstart_a100", "copy_all"),
 R("E11_baselines_redundancy", "quickstart_h20", OE + "/quickstart_h20", "copy_all"),
 R("E11_baselines_redundancy", "bm_bookcorpus_grid", OE + "/benchmarks/bookcorpus_grid", "copy_small", copy_max=1*MB, hash_max=100*MB),
 R("E11_baselines_redundancy", "bm_comprehensive_grid", OE + "/benchmarks/comprehensive_grid", "copy_small", copy_max=1*MB, hash_max=100*MB),
 R("E11_baselines_redundancy", "bm_final_grid", OE + "/benchmarks/final_grid", "copy_small", copy_max=1*MB, hash_max=100*MB),
 R("E11_baselines_redundancy", "bm_multidomain", OE + "/benchmarks/multidomain", "copy_all"),
 R("E11_baselines_redundancy", "bm_prompts", OE + "/benchmarks/prompts", "copy_all"),
 R("E11_baselines_redundancy", "sjq_multi_domain", SJQ + "/OEPLBdata/datasets/multi_domain", "copy_small", copy_max=10*MB, hash_max=800*MB),
 R("E11_baselines_redundancy", "sjq_single_domain", SJQ + "/OEPLBdata/datasets/single_domain", "copy_small", copy_max=10*MB, hash_max=200*MB),
 # ---------------- E12 ----------------
 R("E12_generalization", "oeplb_benchmarks_results", OE + "/benchmarks/results", "copy_all"),
 R("E12_generalization", "studypaper", OE + "/studypaper", "copy_all"),
 R("E12_generalization", "oeplb_scripts", OE + "/scripts", "copy_all"),
 R("E12_generalization", "author_diag", FF + "/author_diag_20261021", "copy_small", copy_max=1*MB, hash_max=100*MB),
 R("E12_generalization", "fig12_hotgpu_npz_LIST", PPT + "/fig12_12b_12c_12d_hotgpu_pinned_volatile", "list_only"),
]

# known-empty dirs registered explicitly (verified 2026-10-25)
EMPTY_DIRS = [
 (FP + "/A_gap_table", "E05_endtoend", "EMPTY at audit time (planned gap-table dir unused; gap evidence lives in ea_v1/nearscan_v1/A-cells)"),
 (FP + "/E_p1_protocol", "E05_endtoend", "EMPTY at audit time"),
 (FP + "/F_package", "E05_endtoend", "EMPTY at audit time"),
 (FP + "/freeze", "E05_endtoend", "EMPTY at audit time (freeze evidence: FF/freeze + FP/C_configs)"),
 (H20 + "/traces", "E05_endtoend", "EMPTY at audit time"),
 (H20 + "/logs", "E05_endtoend", "EMPTY at audit time"),
 (SJQ + "/OEPLBdata/results", "E11_baselines_redundancy", "EMPTY at audit time"),
 (PPT + "/fig12_12b_12c_12d_hotgpu_pinned_volatile/rt2_book_rank0", "E12_generalization", "EMPTY per 2026-10-24 audit (fig12 book rank0 trace missing)"),
]

def main():
    log("BUILD START; roots=%d" % len(ROOTS))
    for i, r in enumerate(ROOTS):
        log("ROOT %d/%d %s/%s mode=%s" % (i + 1, len(ROOTS), r["e"], r["tag"], r["mode"]))
        try:
            if r["mode"] == "files": files_root(r)
            else: walk_root(r)
        except Exception as ex:
            log("ROOT ERROR %s/%s: %s" % (r["e"], r["tag"], ex))
            manifest.append({"e": r["e"], "tag": r["tag"], "src": r["path"], "rel": "", "bytes": None,
                             "mtime": None, "sha256": None, "packaged": None, "note": "ROOT_WALK_ERROR:%s" % ex})
    for path, e, note in EMPTY_DIRS:
        manifest.append({"e": e, "tag": "EMPTY_DIR_REGISTRY", "src": path, "rel": "", "bytes": 0,
                         "mtime": None, "sha256": None, "packaged": None, "note": note})
    # write manifest
    with open(os.path.join(PKG, "manifest.json"), "w") as f:
        json.dump({"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "policy": {"copy": "files <= copy_max per root copied into package",
                              "hash": "sha256 for all non-core files <= hash cap (default 2GB)",
                              "cores": "core dumps: path+size registered, NOT hashed, retained in place",
                              "missing": "SOURCE_MISSING entries register absent referenced material"},
                   "stats_per_e": stats, "entries": manifest}, f, indent=1)
    with open(os.path.join(PKG, "manifest.tsv"), "w") as f:
        f.write("e\ttag\trel\tbytes\tsha256\tpackaged\tsrc\tnote\n")
        for m in manifest:
            f.write("\t".join(str(x) if x is not None else "" for x in
                    [m["e"], m["tag"], m.get("rel",""), m.get("bytes"), m.get("sha256"),
                     m.get("packaged"), m["src"], (m.get("note") or "").replace("\t"," ")]) + "\n")
    with open(os.path.join(PKG, "BUILD_STATS.json"), "w") as f:
        json.dump(stats, f, indent=1)
    tot_c = sum(s["copied"] for s in stats.values()); tot_cb = sum(s["copied_bytes"] for s in stats.values())
    tot_f = sum(s["files"] for s in stats.values()); tot_h = sum(s["hashed"] for s in stats.values())
    tot_m = sum(s["missing"] for s in stats.values())
    log("BUILD DONE files=%d hashed=%d copied=%d copied_bytes=%.1fMB missing=%d elapsed=%.0fs"
        % (tot_f, tot_h, tot_c, tot_cb / MB, tot_m, time.time() - t_start))

if __name__ == "__main__":
    main()
