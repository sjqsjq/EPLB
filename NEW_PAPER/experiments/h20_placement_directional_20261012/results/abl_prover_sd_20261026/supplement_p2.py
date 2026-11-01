#!/usr/bin/env python3
"""READ-ONLY supplement for author-requested items (2026-10-26): true-forward dwell, transition reasons,
first-migration timing, plan/layout fingerprints, exact wall clocks, difference intervals.
No GPU, no writes outside supplement_p2.json / SUPPLEMENT_P2.md. Preregistered results untouched;
paired-by-block intervals are DESCRIPTIVE supplements, not replacements for the Holm tests."""
import json, glob, os, re, math, time
from collections import defaultdict
from scipy import stats

A = os.path.dirname(os.path.abspath(__file__))
ARMS = [("P", "apsd2_P_b"), ("F_1_16_16", "apsd2_F_1_16_16_b"), ("F_4_16_64", "apsd2_F_4_16_64_b")]
out = {"generated_utc": time.strftime("%FT%TZ", time.gmtime()), "mode": "read_only_supplement", "per_boot": {}, "per_arm": {}}

for arm, pref in ARMS:
    agg = {"dwell_fwd_by_cfg": defaultdict(int), "dwell_fwd_by_state": defaultdict(int),
           "reasons": defaultdict(set), "first_swap_windows": [], "first_swap_fwd": [],
           "planhash": defaultdict(set), "planhash_nops": defaultdict(set), "p2l_fp8_final": set(), "p2l_version_final": set(),
           "det_map_traj": set(), "wall": [], "tps": [], "boot_wall_s": [], "ctrl_ms": [], "bcost_us": [],
           "swap_offset_s": [], "trans_seq": set()}
    for d in sorted(glob.glob(os.path.join(A, "boots_p2", pref + "*"))):
        lab = os.path.basename(d)
        rows = [json.loads(l) for l in open(os.path.join(d, "v2p1_rank0.jsonl"))]
        bench = json.load(open(os.path.join(d, lab + ".json")))
        meta = json.load(open(os.path.join(d, "boot_meta.json")))
        meta_end = json.load(open(os.path.join(d, "boot_meta.json.end"))) if os.path.exists(os.path.join(d, "boot_meta.json.end")) else {}
        # 1) true-forward dwell per config/state
        prev = 0; dwell_cfg = defaultdict(int); dwell_state = defaultdict(int)
        for r in rows:
            t = r["true_steps_total"]; inc = t - prev; prev = t
            cfg = (r["sp_s"], r["sp_h"], r["sp_w"])
            dwell_cfg[str(cfg)] += inc
            dwell_state[r.get("sp_state_after") or r.get("sp_state_before")] += inc
            if r.get("sp_transition"):
                agg["reasons"][str((r["window_id"], r["sp_state_before"], r["sp_state_after"]))].add(tuple(r.get("sp_reasons") or []))
        agg["trans_seq"].add(tuple((r["window_id"], r["sp_state_before"], r["sp_state_after"]) for r in rows if r.get("sp_transition")))
        for k, v in dwell_cfg.items(): agg["dwell_fwd_by_cfg"][k] += v
        for k, v in dwell_state.items(): agg["dwell_fwd_by_state"][k] += v
        # 3) first migration timing (forward-based)
        sw = [r for r in rows if (r.get("sp_swaps_started") or 0) > 0]
        if sw:
            agg["first_swap_windows"].append(sw[0]["window_id"]); agg["first_swap_fwd"].append(sw[0]["true_steps_total"])
        # 4) plan/layout fingerprints from server.log PLANHASH
        ph = defaultdict(set); nops = defaultdict(set); first_ph_ts = None
        for line in open(os.path.join(d, "server.log"), errors="replace"):
            m = re.search(r"^\[([\d\- :]+) DP\d+.*PLANHASH\] window#(\d+) rank=(\d+) n_ops=(\d+) plan_hash=(\w+)", line)
            if m:
                ts, w, rk, no, h = m.groups()
                ph[int(w)].add(h); nops[int(w)].add(int(no))
                if first_ph_ts is None: first_ph_ts = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        agg["planhash"].update({k: v for k, v in ph.items()}); agg["planhash_nops"].update(nops)
        agg["p2l_fp8_final"].add(rows[-1].get("p2l_fp8")); agg["p2l_version_final"].add(rows[-1].get("p2l_version"))
        agg["det_map_traj"].add(tuple(sorted(set(r["det_map_version"] for r in rows))))
        if first_ph_ts is not None: agg["swap_offset_s"].append(round(first_ph_ts - bench["wall_start_s"], 2))
        # 5) exact wall clocks
        agg["wall"].append(bench["total_time_s"]); agg["tps"].append(bench["tps"])
        try:
            t0 = time.mktime(time.strptime(meta["start"], "%Y-%m-%dT%H:%M:%SZ")); t1 = time.mktime(time.strptime(meta_end["end"], "%Y-%m-%dT%H:%M:%SZ"))
            agg["boot_wall_s"].append(round(t1 - t0, 1))
        except Exception: pass
        # controller overhead
        agg["ctrl_ms"].append(round(sum(r.get("c_window_ms") or 0 for r in rows), 1))
        agg["bcost_us"].append(round(sum(r.get("bcost_partial_us_max") or 0 for r in rows), 0))
        out["per_boot"][lab] = {"arm": arm, "dwell_fwd_by_cfg": dict(dwell_cfg), "dwell_fwd_by_state": dict(dwell_state),
                                "wall_s": bench["total_time_s"], "tps": bench["tps"],
                                "wall_start_s": bench["wall_start_s"], "wall_end_s": bench["wall_end_s"],
                                "first_swap_window": sw[0]["window_id"] if sw else None,
                                "first_swap_fwd": sw[0]["true_steps_total"] if sw else None,
                                "planhash": {str(k): sorted(v) for k, v in ph.items()}, "planhash_nops": {str(k): sorted(v) for k, v in nops.items()},
                                "p2l_fp8_final": rows[-1].get("p2l_fp8"), "p2l_version_final": rows[-1].get("p2l_version"),
                                "det_map_traj": sorted(set(r["det_map_version"] for r in rows)),
                                "first_planhash_minus_benchstart_s": agg["swap_offset_s"][-1] if agg["swap_offset_s"] else None,
                                "ctrl_ms_sum": agg["ctrl_ms"][-1], "bcost_us_sum": agg["bcost_us"][-1]}
    def msd(v): 
        m = sum(v)/len(v); s = math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1)) if len(v) > 1 else 0.0
        return round(m, 3), round(s, 3)
    out["per_arm"][arm] = {
        "n_boots": len(agg["wall"]),
        "dwell_fwd_by_cfg_total2048": dict(agg["dwell_fwd_by_cfg"]),
        "dwell_fwd_pct": {k: round(100.0*v/sum(agg["dwell_fwd_by_cfg"].values()), 1) for k, v in agg["dwell_fwd_by_cfg"].items()},
        "dwell_fwd_by_state_total": dict(agg["dwell_fwd_by_state"]),
        "transition_sequences_distinct": [ [ [w, b, a] for w, b, a in seq ] for seq in agg["trans_seq"] ],
        "reasons": {k: [list(x) for x in sorted(v)] for k, v in agg["reasons"].items()},
        "first_swap_window_set": sorted(set(agg["first_swap_windows"])), "first_swap_fwd_set": sorted(set(agg["first_swap_fwd"])),
        "first_planhash_minus_benchstart_s": agg["swap_offset_s"],
        "planhash_by_window": {str(k): sorted(v) for k, v in agg["planhash"].items()},
        "planhash_nops_by_window": {str(k): sorted(v) for k, v in agg["planhash_nops"].items()},
        "planhash_8rank_unique_per_window": {str(k): len(v) for k, v in agg["planhash"].items()},
        "p2l_fp8_final_set": sorted(str(x) for x in agg["p2l_fp8_final"]), "p2l_version_final_set": sorted(str(x) for x in agg["p2l_version_final"]),
        "det_map_traj_set": [list(t) for t in agg["det_map_traj"]],
        "wall_s_mean_sd": msd(agg["wall"]), "tps_mean_sd": msd(agg["tps"]), "boot_wall_s_mean_sd": msd(agg["boot_wall_s"]) if agg["boot_wall_s"] else None,
        "ctrl_ms_mean_sd": msd(agg["ctrl_ms"]), "bcost_us_mean_sd": msd(agg["bcost_us"]),
        "tps_list": agg["tps"], "wall_list": [round(x, 2) for x in agg["wall"]]}

# 6) difference intervals
def welch_ci(x, y):
    nx, ny = len(x), len(y); mx, my = sum(x)/nx, sum(y)/ny
    vx = sum((v-mx)**2 for v in x)/(nx-1); vy = sum((v-my)**2 for v in y)/(ny-1)
    se = math.sqrt(vx/nx + vy/ny); df = se**4/((vx/nx)**2/(nx-1) + (vy/ny)**2/(ny-1))
    tc = stats.t.ppf(0.975, df); d = mx - my
    return {"diff_mean": round(d, 3), "ci95": [round(d-tc*se, 3), round(d+tc*se, 3)], "df": round(df, 2), "welch_p": round(float(2*stats.t.sf(abs(d/se), df)), 5)}
def paired_ci(x, y):
    dd = [a-b for a, b in zip(x, y)]; n = len(dd); m = sum(dd)/n
    sd = math.sqrt(sum((v-m)**2 for v in dd)/(n-1)); se = sd/math.sqrt(n); tc = stats.t.ppf(0.975, n-1)
    return {"paired_diffs": [round(v, 2) for v in dd], "mean": round(m, 3), "ci95": [round(m-tc*se, 3), round(m+tc*se, 3)], "paired_p": round(float(2*stats.t.sf(abs(m/se), n-1)), 5), "note": "DESCRIPTIVE supplement (block-paired), not part of preregistered Holm set"}
P = out["per_arm"]["P"]["tps_list"]; F1 = out["per_arm"]["F_1_16_16"]["tps_list"]; F4 = out["per_arm"]["F_4_16_64"]["tps_list"]
out["difference_intervals"] = {
    "welch_P_minus_F1": welch_ci(P, F1), "welch_P_minus_F4": welch_ci(P, F4), "welch_F1_minus_F4": welch_ci(F1, F4),
    "paired_by_block_P_minus_F1": paired_ci(P, F1), "paired_by_block_P_minus_F4": paired_ci(P, F4), "paired_by_block_F1_minus_F4": paired_ci(F1, F4)}
json.dump(out, open(os.path.join(A, "supplement_p2.json"), "w"), indent=1)

# compact md
L = ["# SUPPLEMENT_P2 — 作者只读补充项(判定 6.5% 的机制解释方向)", "",
     "只读提取自 boots_p2/ 已落盘工件;预注册结果与冻结件未动;块内配对区间为描述性补充,不替代 Holm 主检验。", "",
     "## 1. 真实 forward 驻留(每臂 8 boot 合计)", ""]
for arm, _ in ARMS:
    a = out["per_arm"][arm]
    pct = json.dumps(a["dwell_fwd_pct"]); st = json.dumps(a["dwell_fwd_by_state_total"])
    L.append(f"- {arm}: 按配置 {pct} %;按状态 {st}")
L += ["", "## 2. 转换原因(sp_reasons 原文,跨 boot 去重)", ""]
for arm, _ in ARMS:
    r = json.dumps(out["per_arm"][arm]["reasons"], ensure_ascii=False)
    L.append(f"- {arm}: {r}")
L += ["", "## 3. 首次迁移时机", ""]
for arm, _ in ARMS:
    a = out["per_arm"][arm]
    w = json.dumps(a["first_swap_window_set"]); fw = json.dumps(a["first_swap_fwd_set"]); off = json.dumps(a["first_planhash_minus_benchstart_s"])
    L.append(f"- {arm}: 首 swap 窗 {w} @true_steps {fw};首 PLANHASH 时间 − bench 开始 = {off} s")
L += ["", "## 4. 计划与布局指纹(跨臂比较)", ""]
for arm, _ in ARMS:
    a = out["per_arm"][arm]
    ph = json.dumps(a["planhash_by_window"]); no = json.dumps(a["planhash_nops_by_window"]); fp = json.dumps(a["p2l_fp8_final_set"]); pv = json.dumps(a["p2l_version_final_set"]); dm = json.dumps(a["det_map_traj_set"])
    L.append(f"- {arm}: plan_hash 按窗 {ph}; n_ops {no}; p2l_fp8 终值集 {fp}; p2l_version {pv}; det_map 轨迹 {dm}")
L += ["", "## 5. 精确墙钟", ""]
for arm, _ in ARMS:
    a = out["per_arm"][arm]
    wl = json.dumps(a["wall_s_mean_sd"]); wli = json.dumps(a["wall_list"]); bw = json.dumps(a["boot_wall_s_mean_sd"]); cm = json.dumps(a["ctrl_ms_mean_sd"]); bc = json.dumps(a["bcost_us_mean_sd"])
    L.append(f"- {arm}: bench wall_s mean±sd {wl} (list {wli});整 boot 墙钟 mean±sd {bw};控制器 Σc_window_ms mean±sd {cm};Σbcost_us mean±sd {bc}")
L += ["", "## 6. 差值区间(tps)", ""]
for k, v in out["difference_intervals"].items():
    vv = json.dumps(v, ensure_ascii=False)
    L.append(f"- {k}: {vv}")
open(os.path.join(A, "SUPPLEMENT_P2.md"), "w").write("\n".join(L) + "\n")
print("SUPP_OK")
for arm, _ in ARMS:
    a = out["per_arm"][arm]
    print(arm, "dwell_pct", json.dumps(a["dwell_fwd_pct"]), "states", json.dumps(a["dwell_fwd_by_state_total"]), "ph_windows", json.dumps(sorted(a["planhash_by_window"].keys())), "p2l_fp8", json.dumps(a["p2l_fp8_final_set"]), "p2l_ver", json.dumps(a["p2l_version_final_set"]), "first_swap", json.dumps(a["first_swap_window_set"]), json.dumps(a["first_swap_fwd_set"]))
di = out["difference_intervals"]
print("welch_P_F4", json.dumps(di["welch_P_minus_F4"]))
print("welch_F1_F4", json.dumps(di["welch_F1_minus_F4"]))
print("paired_P_F4", json.dumps(di["paired_by_block_P_minus_F4"]))
print("paired_P_F1", json.dumps({k: di["paired_by_block_P_minus_F1"][k] for k in ("mean", "ci95", "paired_p")}))
print("reasons_P", json.dumps(out["per_arm"]["P"]["reasons"], ensure_ascii=False)[:600])
print("ctrl", {arm: (out["per_arm"][arm]["ctrl_ms_mean_sd"], out["per_arm"][arm]["bcost_us_mean_sd"], out["per_arm"][arm]["wall_s_mean_sd"]) for arm in out["per_arm"]})
print("swap_offset_P", out["per_arm"]["P"]["first_planhash_minus_benchstart_s"], "F4", out["per_arm"]["F_4_16_64"]["first_planhash_minus_benchstart_s"])
