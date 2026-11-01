#!/usr/bin/env python3
"""analyze_calibration_m1m2m3.py -- calibration report generator (author point 6).

DESCRIPTIVE STATISTICS ONLY (PRE_REG §E): no inferential claims, no endpoint-based
judgments, never a gate. Reads all 15 run artifact dirs + gate outputs and writes:
  $CAL/analysis_calibration.json   (machine-readable dump)
  $CAL/REPORT_calibration_m1m2m3.md (author deliverable)

Reporting rules (frozen):
  - cal_m2_w1_abort is a MECHANISM configuration (FORCE_ABORT_AT=1, eps_risk=1ms):
    reported in the mechanism section, NEVER pooled into performance/spread stats.
  - Cross-rank times: MAX or per-rank only; NEVER SUM across the 8 GPUs.
  - Blanks stay blank (coverage_guarantee=None etc.) -- never filled with assumptions.
  - All abort/contamination/anomaly events retained verbatim.
  - If Delta_net is not reliably identifiable, the report states the limitation
    plainly; nothing is tuned to pass.
  - M3 data = development calibration only: no back-filling sealed conclusions,
    no pre-picking future campaign parameters, no deployment-gain claims.
Missing artifacts are marked MISSING; the script always exits 0.
"""
import csv
import json
import os
import re
import statistics
import sys

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
CAL = os.path.join(EXP, "results/adw_v2p1/calibration_m1m2m3")
FE = os.path.join(EXP, "results/adw_v2p1/formal_eval")
RUNLIST = os.path.join(CAL, "run_list_calibration.csv")

BLOCKS = [("b1", "W1", "cal_m3_b1_aaa", "cal_m3_b1_aba"),
          ("b2", "W1", "cal_m3_b2_aaa", "cal_m3_b2_aba"),
          ("b3", "W2", "cal_m3_b3_aaa", "cal_m3_b3_aba"),
          ("b4", "W2", "cal_m3_b4_aaa", "cal_m3_b4_aba")]

BLANKS = [  # PRE_REG §E -- stay BLANK, never filled
    "coverage guarantee of Delta_net (no confidence/interval claim -- withdrawn)",
    "production admission thresholds (probe_commit=0 everywhere this campaign)",
    "T-b production trigger specification (calibration uses deterministic T-c)",
    "TTFT delta endpoint (not measured against a sealed TTFT budget here)",
    "formal eps_prac justification (0.01 x rate_ref is provisional)",
    "W64 behavior (out of v1 scope by author directive)",
    "MDE / power analysis (no inferential claim made)",
]

RESIDUAL_ASSUMPTIONS = ("A1 drift near-linear over the span; A2 no regime event "
                        "inside the probe; A3 stream-slice heterogeneity slow; "
                        "A4 layout stable in segment or flagged")


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "window_id" in r:
                    rows.append(r)
    except FileNotFoundError:
        pass
    return rows


def final_prof_block(server_log):
    """8-rank stats over the LAST [PB-OEPLB-PROF] window block (symmetric method)."""
    vals, probes, lastwin = [], [], None
    try:
        with open(server_log, errors="replace") as f:
            for line in f:
                if "[PB-OEPLB-PROF]" not in line or "avg_record_us=" not in line:
                    continue
                try:
                    w = int(line.split("window#")[1].split()[0])
                    v = float(line.split("avg_record_us=")[1].split()[0])
                    pm = re.search(r"avg_probe_us=([0-9.]+)", line)
                    lb = re.search(r"last_bcost_us=(\d+)", line)
                except Exception:
                    continue
                ent = (v, float(pm.group(1)) if pm else None,
                       int(lb.group(1)) if lb else None)
                if lastwin is None or w == lastwin:
                    vals.append(ent); lastwin = w
                else:
                    vals = [ent]; lastwin = w
    except FileNotFoundError:
        return None
    if not vals:
        return None
    recs = [v[0] for v in vals]
    return {"window": lastwin, "n": len(vals),
            "record_us_median": statistics.median(recs),
            "record_us_min": min(recs), "record_us_max": max(recs),
            "record_us_vals": [round(r, 2) for r in recs],
            "avg_probe_us_max": max([v[1] for v in vals if v[1] is not None], default=None),
            "last_bcost_us_max": max([v[2] for v in vals if v[2] is not None], default=None)}


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def fmt(x, nd=2, unit=""):
    if x is None:
        return "MISSING"
    if isinstance(x, float):
        return f"{x:.{nd}f}{unit}"
    return f"{x}{unit}"


def parse_meta(path):
    d = {}
    try:
        for line in open(path):
            if "=" in line:
                k, _, v = line.strip().partition("=")
                d[k] = v
    except FileNotFoundError:
        pass
    return d


def main():
    rows_csv = list(csv.DictReader(open(RUNLIST)))
    analysis = {"campaign": "calibration_m1m2m3", "spec_rev": "v1.1-calib",
                "runs": {}, "blocks": [], "gates": {}, "budget": {},
                "sealed_record_us_ref": {}}

    # sealed record_us reference (same runtime-derived anchor as the M1 gate)
    for cond in ("W1", "W2"):
        analysis["sealed_record_us_ref"][cond] = final_prof_block(
            os.path.join(FE, f"{cond}_A__r1", "server.log"))

    # budget/events from state tsv
    counts = {"BOOT_OK": 0, "VERDICT_VALID": 0, "RESULT_RETIRE": 0,
              "INFRA_MISSING": 0, "MECH_RETIRE": 0, "INTEGRITY_FAIL": 0,
              "STAGE_END": [], "HALT": []}
    try:
        for line in open(os.path.join(CAL, "calib_state.tsv")):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            ev = parts[1]
            if ev == "STAGE_END":
                counts["STAGE_END"].append(parts[2] if len(parts) > 2 else "?")
            elif ev == "HALT":
                counts["HALT"].append(parts)
            elif ev == "BOOT_OK":
                counts["BOOT_OK"] += 1
            elif ev == "VERDICT" and len(parts) >= 7 and parts[6] == "VALID":
                counts["VERDICT_VALID"] += 1
            elif ev in counts and not isinstance(counts[ev], list):
                counts[ev] += 1
    except FileNotFoundError:
        pass
    analysis["budget"] = counts

    # per-run collection
    for rc_ in rows_csv:
        label = rc_["bench_label"]
        art = os.path.join(CAL, label)
        run = {"seq": int(rc_["seq"]), "stage": rc_["stage"], "cond": rc_["cond"],
               "mode": rc_["mode"], "eps_risk_us": int(rc_["eps_risk_us"]),
               "force_abort_at": int(rc_["force_abort_at"]), "label": label}
        vj = jload(os.path.join(art, "verdict.json"))
        run["verdict"] = vj.get("verdict") if vj else "MISSING"
        run["verdict_lists"] = ({k: vj.get(k, []) for k in ("mech", "result", "integrity")}
                                if vj else None)
        bj = jload(os.path.join(art, "bench.json"))
        if bj:
            run["bench"] = {k: bj.get(k) for k in
                            ("total_time_s", "tps", "ok", "errors", "num_requests",
                             "ttft_p99_ms", "tpot_p99_ms", "lat_p99_ms",
                             "total_comp_tokens", "mono_start_ns", "mono_end_ns",
                             "wall_start_s", "wall_end_s")}
        else:
            run["bench"] = None
        run["meta"] = parse_meta(os.path.join(art, "meta.txt"))
        jsonl = load_jsonl(os.path.join(art, "v2p1_rank0.jsonl"))
        run["n_windows"] = len(jsonl)
        run["sum_steps"] = sum(r.get("steps_actual", 0) for r in jsonl) if jsonl else None
        run["c_window_ms_median"] = med([r.get("c_window_ms") for r in jsonl])
        run["bcost_partial_us_max_median"] = med([r.get("bcost_partial_us_max") for r in jsonl])
        run["prev_bcost_us_median"] = med([r.get("prev_bcost_us") for r in jsonl])
        run["t_confirm_us_median"] = med([r.get("t_confirm_us") for r in jsonl])
        run["lockstep_all_ok"] = (all(r.get("lockstep_ok") for r in jsonl)
                                  if jsonl else None)
        run["prof"] = final_prof_block(os.path.join(art, "server.log"))
        pe = jload(os.path.join(art, "probe_run_end.json"))
        run["probe_run_end"] = pe
        if pe:
            ev = pe.get("eval")
            run["eval"] = ev
            run["probe_events"] = pe.get("events")
            run["probe_flags_seg"] = pe.get("seg_flags")
            run["probe_final_state"] = pe.get("state")
            run["license_used"] = pe.get("license_used")
            run["trigger_window"] = pe.get("trigger_window")
            run["excess_cum_us"] = pe.get("excess_cum_us")
            run["overshoot_us"] = pe.get("overshoot_us")
            run["abort_reason"] = pe.get("abort_reason")
            run["contaminated"] = pe.get("contaminated")
            run["admission_applied"] = pe.get("admission_applied")
            seglens = {s: len(pe.get("segments", {}).get(s, [])) for s in ("REF", "CAND", "FB")}
            run["segment_lens"] = seglens
            lag = [(s, e.get("window_id")) for s in ("REF", "CAND", "FB")
                   for e in pe.get("segments", {}).get(s, []) if e.get("lag_window")]
            run["lag_windows"] = lag
        analysis["runs"][label] = run

    # gate outputs
    analysis["gates"]["m1"] = jload(os.path.join(CAL, "m1_gate.json"))
    analysis["gates"]["m2"] = jload(os.path.join(CAL, "m2_gate.json"))

    # M1-off pooled baselines for the self-cost ledger
    off = [analysis["runs"][l] for l in analysis["runs"]
           if analysis["runs"][l]["mode"] == "off" and analysis["runs"][l]["verdict"] == "VALID"]
    base_cwin = med([r["c_window_ms_median"] for r in off])
    base_bcost = med([r["bcost_partial_us_max_median"] for r in off])
    base_tconf = med([r["t_confirm_us_median"] for r in off])
    analysis["m1_off_baseline"] = {"c_window_ms_median": base_cwin,
                                   "bcost_partial_us_max_median": base_bcost,
                                   "t_confirm_us_median": base_tconf,
                                   "n_runs": len(off)}

    # paired blocks (M3)
    for bid, cond, aaa_l, aba_l in BLOCKS:
        aaa, aba = analysis["runs"].get(aaa_l, {}), analysis["runs"].get(aba_l, {})
        bt = aaa.get("bench") or {}
        bb = aba.get("bench") or {}
        pair = {"block": bid, "cond": cond, "aaa": aaa_l, "aba": aba_l,
                "aaa_total_time_s": bt.get("total_time_s"),
                "aba_total_time_s": bb.get("total_time_s")}
        if pair["aaa_total_time_s"] is not None and pair["aba_total_time_s"] is not None:
            pair["diff_s"] = round(pair["aba_total_time_s"] - pair["aaa_total_time_s"], 2)
            pair["diff_pct"] = round(100.0 * pair["diff_s"] / pair["aaa_total_time_s"], 3)
        else:
            pair["diff_s"] = pair["diff_pct"] = None
        pair["aaa_eval_cls"] = (aaa.get("eval") or {}).get("classification")
        pair["aba_eval_cls"] = (aba.get("eval") or {}).get("classification")
        pair["aaa_delta_net"] = (aaa.get("eval") or {}).get("delta_net_us_per_step")
        pair["aba_delta_net"] = (aba.get("eval") or {}).get("delta_net_us_per_step")
        analysis["blocks"].append(pair)

    write_json = os.path.join(CAL, "analysis_calibration.json")
    with open(write_json, "w") as f:
        json.dump(analysis, f, indent=1, default=str)

    render_report(analysis, base_cwin, base_bcost, base_tconf)
    print(f"analysis written: {write_json}")
    print(f"report written:   {os.path.join(CAL, 'REPORT_calibration_m1m2m3.md')}")
    return 0


def render_report(A, base_cwin, base_bcost, base_tconf):
    runs = A["runs"]
    L = []
    ap = L.append
    n_done = sum(1 for r in runs.values() if r["verdict"] != "MISSING")
    ap("# REPORT — calibration M1→M2→M3 (route-B v1.1, development calibration)")
    ap("")
    ap(f"- STATUS: {'COMPLETE (15/15)' if n_done >= 15 else f'PARTIAL ({n_done}/15 runs have verdicts)'}")
    ap("- spec_rev: v1.1-calib；治理：LOCAL ONLY、描述性统计（PRE_REG §E）、留空项保持留空、"
       "MECHANISM 运行（cal_m2_w1_abort）永不混入性能统计、跨 rank 时间只用 MAX 或 per-rank（永不求和）、"
       "不扣除任何实测成本、不挑选有利重复。")
    ap("- 估计器：`observational_ABA_diff_trend_corrected`；`causal_isolation_claimed=False`；"
       f"残留假设 {RESIDUAL_ASSUMPTIONS}（随结果报告，不称已隔离因果）。")
    ap("")
    ap("## 1. 全部启动结果（作者第 6 点：全部 15 次）")
    ap("")
    ap("| seq | stage | cond | mode | label | verdict | boot_s | total_time_s | tps | ok/req | errors | ttft_p99_ms | lockstep |")
    ap("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    boot_by_seq = {}
    try:
        for line in open(os.path.join(CAL, "driver.log")):
            m = re.search(r"seq (\d+) boot OK in (\d+)s", line)
            if m:
                boot_by_seq[int(m.group(1))] = int(m.group(2))
    except FileNotFoundError:
        pass
    for label in sorted(runs, key=lambda l: runs[l]["seq"]):
        r = runs[label]
        b = r["bench"] or {}
        mech = " **(MECHANISM)**" if label == "cal_m2_w1_abort" else ""
        ap(f"| {r['seq']} | {r['stage']} | {r['cond']} | {r['mode']} | {label}{mech} | "
           f"{r['verdict']} | {boot_by_seq.get(r['seq'], 'MISSING')} | "
           f"{fmt(b.get('total_time_s'))} | {fmt(b.get('tps'))} | "
           f"{b.get('ok', 'MISSING')}/{b.get('num_requests', 'MISSING')} | {b.get('errors', 'MISSING')} | "
           f"{fmt(b.get('ttft_p99_ms'), 1)} | {r['lockstep_all_ok']} |")
    bc = A["budget"]
    ap("")
    ap(f"预算：BOOT_OK={bc['BOOT_OK']}（计时启动，上限 15）；VERDICT VALID={bc['VERDICT_VALID']}；"
       f"RESULT_RETIRE={bc['RESULT_RETIRE']}；INFRA_MISSING={bc['INFRA_MISSING']}（上限 3，不占计时预算）；"
       f"MECH_RETIRE={bc['MECH_RETIRE']}；INTEGRITY_FAIL={bc['INTEGRITY_FAIL']}；"
       f"STAGE_END={bc['STAGE_END']}；HALT={bc['HALT']}")
    ap("")
    ap("## 2. 配置校验（meta.txt vs 冻结 run list）")
    ap("")
    ap("| label | MODE | EPS_RISK_US | FORCE_ABORT_AT | COMMIT | CALIB_TRIGGER | R | NREF/NCAND/NFB | EPS_PRAC_FRAC | EA_W | knobs |")
    ap("|---|---|---|---|---|---|---|---|---|---|---|")
    exp_knobs = {"OEPLB_V21_KAPPA": "3875.1768798828125",
                 "OEPLB_V21_Z_LO": "22.643858727592757",
                 "OEPLB_V21_Z_HI": "314.14010272527486",
                 "OEPLB_V21_Z_URGENT": "942.4203081758246"}
    for label in sorted(runs, key=lambda l: runs[l]["seq"]):
        r = runs[label]
        m = r["meta"] or {}
        knobs_ok = all(m.get(k) == v for k, v in exp_knobs.items() if k in m or True)
        ap(f"| {label} | {m.get('OEPLB_PROBE_MODE', 'MISSING')} | {m.get('OEPLB_PROBE_EPS_RISK_US', 'MISSING')} | "
           f"{m.get('OEPLB_PROBE_FORCE_ABORT_AT', 'MISSING')} | {m.get('OEPLB_PROBE_COMMIT', 'MISSING')} | "
           f"{m.get('OEPLB_PROBE_CALIB_TRIGGER', 'MISSING')} | {m.get('OEPLB_PROBE_R', 'MISSING')} | "
           f"{m.get('OEPLB_PROBE_NREF', '?')}/{m.get('OEPLB_PROBE_NCAND', '?')}/{m.get('OEPLB_PROBE_NFB', '?')} | "
           f"{m.get('OEPLB_PROBE_EPS_PRAC_FRAC', 'MISSING')} | {m.get('EA_W', 'MISSING')} | "
           f"{'FROZEN-OK' if knobs_ok else 'MISMATCH'} |")
    ap("")
    ap("## 3. M1 gate（决定论复原，m1_gate.json）")
    ap("")
    g1 = A["gates"]["m1"]
    if g1 is None:
        ap("MISSING（M1 gate 尚未运行）")
    else:
        ap(f"**PASS = {g1.get('pass')}**")
        for lbl, rr in g1.get("runs", {}).items():
            fails = [k for k, v in rr.get("checks", {}).items() if not v.get("pass")]
            ap(f"- {lbl}: checks={len(rr.get('checks', {}))} failed={fails if fails else '无'} "
               f"z_max|Δz|={fmt(rr.get('z_max_abs_delta'), 4)} z_min_margin={fmt(rr.get('z_min_threshold_margin'), 3)} "
               f"c_window_Δ={fmt(rr.get('c_window_delta_ms'), 4)}ms "
               f"record_us={((rr.get('record_us_final_block') or {}).get('median'))} cap={rr.get('record_us_cap')}")
        for cond, sr in g1.get("sealed_ref", {}).items():
            ru = sr.get("record_us") or {}
            ap(f"- 封存参照 {cond}: windows={sr.get('windows')} sum_steps={sr.get('sum_steps')} "
               f"swaps={sr.get('swaps')} record_us_median={ru.get('median')} (n={ru.get('n')})")
    ap("")
    ap("## 4. M2 gate（机制正确性，m2_gate.json）")
    ap("")
    g2 = A["gates"]["m2"]
    if g2 is None:
        ap("MISSING（M2 gate 尚未运行）")
    else:
        ap(f"**PASS = {g2.get('pass')}**")
        for lbl, rr in g2.get("runs", {}).items():
            fails = [k for k, v in rr.get("checks", {}).items() if not v.get("pass")]
            ap(f"- {lbl}: failed={fails if fails else '无'}")
        if g2.get("summary"):
            ap(f"- summary: {json.dumps(g2['summary'], ensure_ascii=False)}")
    ap("")
    # probe run timeline table
    ap("## 5. 探测运行时间线与估计（M2 机制 + M3 配对）")
    ap("")
    ap("| label | stage | mode | final_state | trigger | REF/CAND/FB 窗数 | classification | Δ_net µs/step | Δ_net rel | ε_prac | flags |")
    ap("|---|---|---|---|---|---|---|---|---|---|---|")
    probe_labels = [l for l in sorted(runs, key=lambda x: runs[x]["seq"])
                    if runs[l]["mode"] in ("gate", "aaa")]
    for label in probe_labels:
        r = runs[label]
        ev = r.get("eval") or {}
        sl = r.get("segment_lens") or {}
        mech = " **(MECHANISM，不入性能统计)**" if label == "cal_m2_w1_abort" else ""
        ap(f"| {label}{mech} | {r['stage']} | {r['mode']} | {r.get('probe_final_state', 'MISSING')} | "
           f"{r.get('trigger_window', 'MISSING')} | {sl.get('REF', '-')}/{sl.get('CAND', '-')}/{sl.get('FB', '-')} | "
           f"{ev.get('classification', 'MISSING')} | {fmt(ev.get('delta_net_us_per_step'), 2)} | "
           f"{fmt(ev.get('delta_net_rel'), 5)} | {fmt(ev.get('eps_prac_us_per_step'), 2)} | "
           f"{','.join(ev.get('flags') or []) or '—'} |")
    ap("")
    # AAA false positives
    ap("## 6. AAA 误判表现（零假设组：真 Δ=0，任何 favorable/unfavorable 即误判）")
    ap("")
    aaa_labels = [l for l in probe_labels if runs[l]["mode"] == "aaa"]
    fp = []
    ap("| label | cond | Δ_net µs/step | Δ_net rel | ε_prac | classification | 误判? |")
    ap("|---|---|---|---|---|---|---|")
    for label in aaa_labels:
        r = runs[label]
        ev = r.get("eval") or {}
        cls = ev.get("classification")
        is_fp = cls in ("favorable", "unfavorable")
        if is_fp:
            fp.append((label, ev.get("delta_net_us_per_step"), cls))
        ap(f"| {label} | {r['cond']} | {fmt(ev.get('delta_net_us_per_step'), 2)} | "
           f"{fmt(ev.get('delta_net_rel'), 5)} | {fmt(ev.get('eps_prac_us_per_step'), 2)} | "
           f"{cls if cls else 'MISSING'} | {'**是**' if is_fp else ('否' if cls else 'MISSING')} |")
    ap("")
    ap(f"AAA 误判数 = {len(fp)} / {len(aaa_labels)}" + (f"：{fp}" if fp else "（无误判）"))
    ap("")
    # ABA stability
    ap("## 7. ABA 估计稳定性（M3 正式配对组；M2 观测运行单列不合并）")
    ap("")
    for cond in ("W1", "W2"):
        m3_aba = [p for p in A["blocks"] if p["cond"] == cond]
        deltas = [p["aba_delta_net"] for p in m3_aba if p["aba_delta_net"] is not None]
        aaa_d = [p["aaa_delta_net"] for p in m3_aba if p["aaa_delta_net"] is not None]
        ap(f"**{cond}**（M3 ABA：{', '.join(p['aba'] for p in m3_aba)}）")
        for p in m3_aba:
            r = runs.get(p["aba"], {})
            ev = r.get("eval") or {}
            ap(f"- {p['aba']}: rate_ref={fmt(ev.get('rate_ref_us_per_step'), 2)} "
               f"rate_cand={fmt(ev.get('rate_cand_us_per_step'), 2)} rate_fb={fmt(ev.get('rate_fb_us_per_step'), 2)} "
               f"g={fmt(ev.get('g_us_per_step2'), 6) if ev.get('g_us_per_step2') is not None else 'MISSING'} "
               f"pred={fmt(ev.get('predicted_cand_us_per_step'), 2)} Δ_net={fmt(ev.get('delta_net_us_per_step'), 2)} "
               f"rel={fmt(ev.get('delta_net_rel'), 5)} ε_prac={fmt(ev.get('eps_prac_us_per_step'), 2)} "
               f"cls={ev.get('classification')} midpoints={ev.get('midpoints_steps')} n_steps={ev.get('n_steps')}")
        if len(deltas) >= 2:
            ap(f"- 描述：ABA Δ_net 范围 [{min(deltas):.2f}, {max(deltas):.2f}]，极差 {max(deltas)-min(deltas):.2f} µs/step"
               f"{'；符号一致' if all(d < 0 for d in deltas) or all(d > 0 for d in deltas) else '；符号不一致'}")
        if aaa_d:
            ap(f"- AAA（零假设组）Δ_net 范围 [{min(aaa_d):.2f}, {max(aaa_d):.2f}]，极差 {max(aaa_d)-min(aaa_d):.2f} µs/step")
        ap("")
    m2_obs = [l for l in probe_labels if runs[l]["stage"] == "M2" and runs[l]["mode"] == "gate"]
    ap("M2 gate 模式运行（机制阶段，单列，不与 M3 合并）：")
    for label in m2_obs:
        ev = runs[label].get("eval") or {}
        mech = " **(MECHANISM)**" if label == "cal_m2_w1_abort" else ""
        ap(f"- {label}{mech}: Δ_net={fmt(ev.get('delta_net_us_per_step'), 2)} cls={ev.get('classification')} "
           f"excess_cum={runs[label].get('excess_cum_us')} overshoot={runs[label].get('overshoot_us')} "
           f"abort_reason={runs[label].get('abort_reason')}")
    ap("")
    # paired complete-service diffs
    ap("## 8. 完整服务差（每 block：ABA − AAA，total_time_s；探测成本不扣除、全部保留在内）")
    ap("")
    ap("| block | cond | AAA total_time_s | ABA total_time_s | 差 (s) | 差 (%) |")
    ap("|---|---|---|---|---|---|")
    for p in A["blocks"]:
        ap(f"| {p['block']} | {p['cond']} | {fmt(p['aaa_total_time_s'])} | {fmt(p['aba_total_time_s'])} | "
           f"{fmt(p['diff_s'])} | {fmt(p['diff_pct'], 3, '%')} |")
    ap("")
    ap("注：差值包含探测窗自身的全部成本（W32 CAND 段 + 边界开销），不做任何扣除；这是完整服务口径。")
    ap("")
    # retained events
    ap("## 9. 中止/污染/异常事件全记录（保留，不筛选）")
    ap("")
    any_ev = False
    for label in probe_labels:
        r = runs[label]
        pe = r.get("probe_run_end")
        if not pe:
            ap(f"- {label}: probe_run_end MISSING")
            any_ev = True
            continue
        evs = pe.get("events") or []
        segflags = pe.get("seg_flags") or {}
        nonzero = {k: v for k, v in segflags.items() if v}
        lag = r.get("lag_windows") or []
        interesting = [e for e in evs if e and len(e) >= 2 and
                       e[1] not in ("trigger", "cand_start", "fb_start", "eval")]
        if interesting or nonzero or lag or pe.get("contaminated") or pe.get("abort_reason"):
            any_ev = True
            ap(f"- {label}: state={pe.get('state')} contaminated={pe.get('contaminated')} "
               f"abort_reason={pe.get('abort_reason')} excess_cum={pe.get('excess_cum_us')} "
               f"overshoot={pe.get('overshoot_us')}")
            if interesting:
                ap(f"  - 事件：{interesting}")
            if nonzero:
                ap(f"  - 段 flags：{nonzero}")
            if lag:
                ap(f"  - lag 窗：{lag}")
    if not any_ev:
        ap("（无中止/污染/异常事件——所有探测运行 clean）")
    n_probe = len(probe_labels)
    n_cont = sum(1 for l in probe_labels if (runs[l].get("probe_run_end") or {}).get("contaminated"))
    n_abort = sum(1 for l in probe_labels if (runs[l].get("probe_run_end") or {}).get("abort_reason"))
    n_aborted_state = sum(1 for l in probe_labels if runs[l].get("probe_final_state") == "ABORTED")
    ap("")
    ap(f"率（分母 = 全部 {n_probe} 次探测运行，含 MECHANISM）：污染 {n_cont}/{n_probe}；"
       f"中止（abort_reason 非空）{n_abort}/{n_probe}；终态 ABORTED {n_aborted_state}/{n_probe}。"
       "MECHANISM 运行（cal_m2_w1_abort）的中止是设计行为，单列说明，不解读为环境风险。")
    ap("")
    # self-cost ledger
    ap("## 10. 新增成本（日志自成本账本 vs ε_log：边界新增中位 ≤ 1 ms/窗；每 forward 对封存参照 ≤ +10%）")
    ap("")
    ap(f"M1-off 基线（{A['m1_off_baseline']['n_runs']} runs）：c_window_ms 中位={fmt(base_cwin, 4)}；"
       f"bcost_partial_us_max 中位={fmt(base_bcost, 1)}；t_confirm_us 中位={fmt(base_tconf, 1)}")
    ap("")
    ap("| label | mode | c_window_ms 中位 (Δ vs off) | bcost_partial 中位 | t_confirm 中位 | record_us 末块中位 (封存参照, Δ%) | prof_probe_cum_ms 末值 | avg_probe_us max | last_bcost_us max |")
    ap("|---|---|---|---|---|---|---|---|---|")
    for label in sorted(runs, key=lambda l: runs[l]["seq"]):
        r = runs[label]
        cond = r["cond"]
        sref = (A["sealed_record_us_ref"].get(cond) or {}).get("record_us_median")
        prof = r.get("prof") or {}
        rec = prof.get("record_us_median")
        dpct = (100.0 * (rec - sref) / sref) if (rec is not None and sref) else None
        pe = r.get("probe_run_end") or {}
        cum = pe.get("prof_probe_cum_ms")
        if cum is None:
            jl = None
            cum = "—" if r["mode"] == "off" else "MISSING"
        dcwin = (r["c_window_ms_median"] - base_cwin) if (r["c_window_ms_median"] is not None and base_cwin is not None) else None
        ap(f"| {label} | {r['mode']} | {fmt(r['c_window_ms_median'], 4)} ({fmt(dcwin, 4, '') if dcwin is not None else 'MISSING'}) | "
           f"{fmt(r['bcost_partial_us_max_median'], 1)} | {fmt(r['t_confirm_us_median'], 1)} | "
           f"{fmt(rec, 2)} ({fmt(sref, 2)}, {fmt(dpct, 2, '%') if dpct is not None else 'MISSING'}) | "
           f"{cum} | {fmt(prof.get('avg_probe_us_max'), 2)} | {prof.get('last_bcost_us_max', 'MISSING')} |")
    ap("")
    ap("ε_log 判定（描述性）：边界项 = c_window_ms 中位相对 M1-off 基线之差 ≤ +1.0 ms/窗；"
       "每 forward 项 = avg_record_us 末块 8-rank 中位对封存 <cond>_A__r1 参照变化 ≤ +10%（运行时导出参照，"
       "PRE_REG §F #5 更正）。M1 gate G4 为正式判定；此处为全部 15 run 的同法描述。"
       "prof_probe_cum_ms = 新增仪表（p2l 指纹）累计账目，如实报告不扣除。")
    ap("")
    # blanks
    ap("## 11. 留空项（不支持的统计保证——明确留空，不用假设值补齐）")
    ap("")
    for b in BLANKS:
        ap(f"- **BLANK**：{b}")
    ap("")
    ap("全部 eval dict 携带 `causal_isolation_claimed=False`、`coverage_guarantee=None`（撤回“完成探测自动获得条件置信”表述）。")
    ap("")
    # identifiability
    ap("## 12. 可辨识性陈述（规则化，不调参）")
    ap("")
    for cond in ("W1", "W2"):
        cb = [p for p in A["blocks"] if p["cond"] == cond]
        aba_d = [p["aba_delta_net"] for p in cb if p["aba_delta_net"] is not None]
        aaa_d = [p["aaa_delta_net"] for p in cb if p["aaa_delta_net"] is not None]
        aba_cls = [p["aba_eval_cls"] for p in cb if p["aba_eval_cls"]]
        if not aba_d or not aaa_d:
            ap(f"- {cond}: 数据不全（MISSING），无法陈述。")
            continue
        overlap = (min(aaa_d) <= max(aba_d) and min(aba_d) <= max(aaa_d))
        consistent = all(c == aba_cls[0] for c in aba_cls) and aba_cls[0] in ("favorable", "unfavorable")
        if consistent and not overlap:
            ap(f"- {cond}: ABA 分类一致（{aba_cls[0]}）且 Δ_net 区间与 AAA 零假设组不重叠——"
               "估计器在本样本量下给出一致可检验的分类。仍为观察性估计（A1–A4 残留假设），不称因果隔离。")
        else:
            reasons = []
            if not consistent:
                reasons.append(f"ABA 分类不一致（{aba_cls}）")
            if overlap:
                reasons.append(f"ABA Δ_net 区间 [{min(aba_d):.2f},{max(aba_d):.2f}] 与 AAA 零假设组 "
                               f"[{min(aaa_d):.2f},{max(aaa_d):.2f}] 重叠")
            ap(f"- {cond}: **在本样本量下无法把 ABA 估计与 AAA 零假设分布可靠区分**（{'; '.join(reasons)}）。"
               "如实报告该限制：现有校准数据不足以支持净效果方向判定；不反复调到通过。")
    ap("")
    ap("本轮成功标准 = 测量可信 + 机制正确 + 估计器能够接受检验；**不以是否出现正收益判定实验成功**。"
       "正式部署收益另行评价。M3 数据仅作开发校准：不回填封存结论、不预挑正式战役参数。")
    ap("")
    ap("## 13. 工件索引")
    ap("")
    ap("- analysis_calibration.json（本报告机器可读源）；m1_gate.json；m2_gate.json；"
       "各 run 目录：verdict.json / bench.json / v2p1_rank0.jsonl / probe_run_end.json / server.log / meta.txt；"
       "calib_state.tsv（预算与事件账本）；driver.log")
    with open(os.path.join(CAL, "REPORT_calibration_m1m2m3.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
