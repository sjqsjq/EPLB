# analyze_xd.py -- cross_domain_e2e_20261027 aggregation + stats
# Usage: python3 analyze_xd.py [--final]
# --final asserts all 24 formal boots status ok (used once at campaign end).
# Facts-first: records every boot raw; stats per PROTOCOL sec.7:
#   primary total_time_s (full-service wall clock), co-report tps,
#   round-paired diffs, mean + 95% CI (paired t, df=3, tcrit=3.182),
#   two-sided paired t-test, P0 vs five arms Holm-corrected (5 comparisons).
# Without --final writes to analysis/prerun (partial-data smoke), never clobbers final outputs.
import json, os, sys, math
X = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027"
FINAL = "--final" in sys.argv
ARMS = ["I", "ES", "ED", "MT", "DF", "P0"]
ROUNDS = [1, 2, 3, 4]
TCRIT_DF3 = 3.182445  # t_{0.975, 3}
try:
    from scipy import stats as SCS
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

boots = {}
for r in ROUNDS:
    for arm in ARMS:
        label = arm + "_r" + str(r)
        d = os.path.join(X, "boots", label)
        rec = {"label": label, "arm": arm, "round": r, "status": "missing"}
        if os.path.isdir(d):
            endf = os.path.join(d, "boot_meta.json.end")
            if os.path.exists(endf):
                rec["end"] = json.load(open(endf))
            bj = os.path.join(d, label + ".json")
            if os.path.exists(bj):
                try:
                    b = json.load(open(bj))
                    rec["bench"] = {k: b.get(k) for k in ["ok", "errors", "total_time_s", "tps",
                        "total_comp_tokens", "total_prompt_tokens", "ttft_mean_ms", "lat_mean_ms",
                        "wall_start_s", "wall_end_s", "client_version"]}
                except Exception as e:
                    rec["bench_error"] = str(e)
            mj = os.path.join(d, "MECH_CHECK.json")
            if os.path.exists(mj):
                try:
                    m = json.load(open(mj))
                    rec["completeness_pass"] = m.get("completeness_pass")
                    rec["baseline_facts"] = m.get("baseline_facts")
                    mf = m.get("mechanism_facts") or {}
                    if arm == "P0":
                        keep = ["n_rows", "detection_cycles_committed", "event_counts", "residency_final",
                                "rg_annotation_rows", "rg_policy_tagged_transitions", "state_transitions",
                                "recording", "planning", "migration", "integrity", "exercise_summary",
                                "shw_trajectory_rle", "z_range", "h_rate_range", "alert_k_max", "stable_s_max",
                                "v2p1_gate"]
                        rec["mech"] = {k: mf.get(k) for k in keep}
                        sp = os.path.join(d, "sp_run_end.json")
                        if os.path.exists(sp):
                            try:
                                sre = json.load(open(sp))
                                rec["sp_run_end"] = {k: sre.get(k) for k in ["transitions",
                                    "decode_record_batches", "nocommit_planned_ops_total",
                                    "swaps_started_total", "recorded_steps", "window_count",
                                    "residency_committed_h", "sample_opportunities", "h_cycles",
                                    "true_steps_total", "state_final", "S_final", "H_final", "W_final"]}
                            except Exception as e:
                                rec["sp_run_end_error"] = str(e)
                    else:
                        rec["mech"] = {k: mf.get(k) for k in sorted(mf.keys()) if k in
                            ("eplb_rebalance_starts", "injection_evidence", "placement_lock_match",
                             "layout_frozen", "banner") or k.startswith("placement")}
                except Exception as e:
                    rec["mech_error"] = str(e)
            brec = rec.get("bench") or {}
            if brec.get("total_time_s"):
                rec["status"] = "ok" if brec.get("ok") == 8192 and (brec.get("errors") in (0, None)) else "bench_incomplete"
            elif rec.get("end"):
                rec["status"] = "no_bench_json"
        boots[label] = rec

n_ok = sum(1 for v in boots.values() if v["status"] == "ok")
def series(arm, key):
    out = []
    for r in ROUNDS:
        b = (boots[arm + "_r" + str(r)].get("bench") or {})
        out.append(b.get(key) if boots[arm + "_r" + str(r)]["status"] == "ok" else None)
    return out

def paired(p0, x):
    ds, rr = [], []
    for i in range(4):
        if p0[i] is not None and x[i] is not None:
            ds.append(p0[i] - x[i]); rr.append(ROUNDS[i])
    n = len(ds)
    res = {"n_pairs": n, "rounds": rr, "diffs": ds}
    if n >= 2:
        mean = sum(ds) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in ds) / (n - 1))
        se = sd / math.sqrt(n)
        res["mean_diff"] = mean; res["sd"] = sd; res["se"] = se
        if n == 4:
            res["ci95"] = [mean - TCRIT_DF3 * se, mean + TCRIT_DF3 * se]
        else:
            res["ci95_note"] = "insufficient pairs for df=3 tcrit; n=" + str(n)
        if se > 0:
            t = mean / se
            res["t_stat"] = t; res["df"] = n - 1
            if HAVE_SCIPY:
                res["p_two_sided"] = float(2 * SCS.t.sf(abs(t), n - 1))
        else:
            res["t_stat"] = None; res["p_two_sided"] = None
    return res

p0_tt = series("P0", "total_time_s")
stats_out = {"primary_metric": "total_time_s", "co_report": "tps", "tcrit_df3": TCRIT_DF3,
             "reference_arm": "P0",
             "test_method": "two-sided paired t-test (round-paired, df=3) + Holm across 5 comparisons",
             "scipy": HAVE_SCIPY, "comparisons": {}}
for arm in ["I", "ES", "ED", "MT", "DF"]:
    stats_out["comparisons"]["P0_vs_" + arm] = paired(p0_tt, series(arm, "total_time_s"))
pv = [(k, v.get("p_two_sided")) for k, v in stats_out["comparisons"].items() if v.get("p_two_sided") is not None]
if pv:
    pv.sort(key=lambda x: x[1])
    m = len(pv); holm = {}; runmax = 0.0
    for j, (k, p) in enumerate(pv):
        runmax = max(runmax, (m - j) * p); holm[k] = min(1.0, runmax)
    stats_out["holm_adjusted_p"] = holm
    stats_out["holm_note"] = "Holm step-down over available p-values (n=" + str(m) + "); missing arms excluded only if boot absent"

tables = {"per_boot_raw": [{ "label": v["label"], "arm": v["arm"], "round": v["round"], "status": v["status"],
             "ok": (v.get("bench") or {}).get("ok"), "errors": (v.get("bench") or {}).get("errors"),
             "total_time_s": (v.get("bench") or {}).get("total_time_s"), "tps": (v.get("bench") or {}).get("tps"),
             "completeness_pass": v.get("completeness_pass"),
             "bench_rc": (v.get("end") or {}).get("bench_rc"), "mech_rc": (v.get("end") or {}).get("mech_check_rc")}
             for v in boots.values()],
          "per_arm_series": {a: {"total_time_s": series(a, "total_time_s"), "tps": series(a, "tps")} for a in ARMS}}

outdir = os.path.join(X, "analysis") if FINAL else os.path.join(X, "analysis", "prerun")
os.makedirs(outdir, exist_ok=True)
json.dump({"campaign": "cross_domain_e2e_20261027", "final": FINAL,
           "boots_ok": str(n_ok) + "/24", "stats": stats_out,
           "tables": tables, "boots": boots},
          open(os.path.join(outdir, "summary.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(outdir, "per_boot_raw.tsv"), "w") as f:
    f.write("label\tarm\tround\tstatus\tok\terrors\ttotal_time_s\ttps\tcompleteness_pass\tbench_rc\tmech_rc\n")
    for row in tables["per_boot_raw"]:
        f.write("\t".join(str(row[k]) for k in ["label", "arm", "round", "status", "ok", "errors",
                  "total_time_s", "tps", "completeness_pass", "bench_rc", "mech_rc"]) + "\n")
print("ANALYZE_OK boots_ok=" + str(n_ok) + "/24 scipy=" + str(HAVE_SCIPY) + " outdir=" + outdir)
if FINAL and n_ok != 24:
    missing = [v["label"] for v in boots.values() if v["status"] != "ok"]
    print("FINAL_BUT_MISSING: " + ", ".join(missing)); sys.exit(1)
for k, v in stats_out["comparisons"].items():
    print(k, "n=" + str(v["n_pairs"]), "mean_diff=" + str(round(v["mean_diff"], 2) if "mean_diff" in v else None),
          "p=" + str(round(v["p_two_sided"], 4) if v.get("p_two_sided") is not None else None))
print("holm:", {k: round(v, 4) for k, v in (stats_out.get("holm_adjusted_p") or {}).items()})
