#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_formal_eval.py — ONE-SHOT analysis of the ADW v2.1 formal evaluation
(48 timed starts). Protocol: PROTOCOL_formal_eval_v2p1 (§7 statistics, §9
narration, §13 deliverables T1–T5). RUN ONCE after CAMPAIGN_END; no interim use.

Rules hard-wired per protocol + author clarifications:
 - Never deduct any cost from service times; primary endpoint = total_time_s.
 - Paired per condition only; never pool W1/W2.
 - Paired t 95% CI primary; exact sign-flip permutation companion (complete
   enumeration, two-sided p = #{|T_S| >= |T_obs|}/2^n, min p = 2/2^n);
   zero-crossing disagreement -> "方法间不一致，判为不确定".
 - 8 marginal CIs, no alpha correction; no cherry-picking; all reported.
 - n>=4 missing pairs -> CI with flag; n<=3 -> descriptive "insufficient pairs".
 - All failures reported per arm; success-pair intervals describe successful
   runs only (survivorship statement printed with every contrast).
 - Wilcoxon signed-rank descriptive (exact enumeration).
 - No best-start selection; retired starts appear in T1 with their verdicts.
"""
import csv, json, math, os, sys, itertools
from collections import defaultdict

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
FE = os.path.join(EXP, "results/adw_v2p1/formal_eval")
CSV = os.path.join(FE, "run_list_formal_eval_v2p1.csv")
STATE = os.path.join(FE, "driver_state.tsv")
OUT_MD = os.path.join(FE, "REPORT_formal_eval_v2p1.md")
OUT_JSON = os.path.join(FE, "report_formal_eval_v2p1.json")

Z_HI = 314.14010272527486          # frozen detection threshold (A arm)
BND_ROW_W2 = 4000                  # true boundary row in w2 dataset (§9)
NROWS = {"W1": 4000, "W2": 8000}
ARMS = ["A", "F", "L", "M"]
CONTRASTS = [("A", "F", "primary"), ("A", "L", "secondary"),
             ("A", "M", "secondary"), ("M", "F", "auxiliary")]
CONDS = ["W1", "W2"]
# two-sided t critical values, 95%, df=1..30 (no scipy dependency)
TCRIT = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
         7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788,
         13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009,
         19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639,
         25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423}

def tcrit(df):
    if df in TCRIT: return TCRIT[df]
    if df < 1: return float("nan")
    return 1.96 + 2.4 / df  # coarse fallback, df>30 not expected (n<=6)

# ---------------- load ----------------
def load_state():
    verdict = {}   # seq -> (verdict, t, extra)
    infra = defaultdict(list)
    for ln in open(STATE, encoding="utf-8"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 3: continue
        if f[1] == "VERDICT":
            verdict[f[2]] = {"label": f[3], "cond": f[4], "arm": f[5],
                             "verdict": f[6], "extra": f[7:]}
        elif f[1] == "INFRA_REBOOT":
            infra[f[2]].append(f)
    return verdict, infra

def load_runs():
    runs = []
    st, infra = load_state()
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        seq = r["seq"]
        art = os.path.join(EXP, r["artifact_dir"])
        rec = dict(r); rec["artifact_dir"] = art
        s = st.get(seq)
        rec["verdict"] = s["verdict"] if s else "NOT_RUN"
        rec["infra_rebooted"] = seq in infra
        t = None; b = None
        try:
            b = json.load(open(os.path.join(art, "bench.json")))
            t = b.get("total_time_s")
        except Exception:
            pass
        rec["bench"] = b
        rec["t"] = t if (s and s["verdict"] == "VALID") else None
        # meta boot_s / attempt
        rec["boot_s"] = None; rec["attempt"] = None
        try:
            for ln in open(os.path.join(art, "meta.txt"), encoding="utf-8"):
                if ln.startswith("boot_s="):
                    parts = ln.split()
                    rec["boot_s"] = parts[0].split("=")[1]
                    for p in parts:
                        if p.startswith("attempt="): rec["attempt"] = p.split("=")[1]
        except Exception:
            pass
        runs.append(rec)
    return runs

def load_jsonl(art):
    p = os.path.join(art, "v2p1_rank0.jsonl")
    rows = []
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try: rows.append(json.loads(ln))
                except Exception: pass
    return rows

# ---------------- statistics (§7) ----------------
def signflip_perm(diffs):
    """Complete enumeration. Returns (p_two_sided, perm_ci_lo, perm_ci_hi, n_flips).
    p = #{|T_S| >= |T_obs|}/2^n (min = 2/2^n). CI = exact acceptance region of
    theta under the same test (breakpoints at (T_S+T_obs)/2 midpoints)."""
    n = len(diffs); m = 1 << n
    T = []
    for mask in range(m):
        s = sum(diffs[i] if (mask >> i) & 1 else -diffs[i] for i in range(n)) / n
        T.append(s)
    tobs = sum(diffs) / n
    cnt = sum(1 for x in T if abs(x) >= abs(tobs) - 1e-12)
    p = cnt / m
    # exact CI by midpoint scan
    mids = sorted({(x + tobs) / 2.0 for x in T})
    def p_at(theta):
        c = 0
        for x in T:
            if abs(x - theta) >= abs(tobs - theta) - 1e-12: c += 1
        return c / m
    # walk outward from tobs; acceptance region is contiguous
    lo = None
    for i in range(len(mids) - 1, -1, -1):
        if mids[i] <= tobs:
            if p_at(mids[i]) > 0.05 + 1e-12:
                lo = mids[i]
            else:
                break
    hi = None
    for i in range(len(mids)):
        if mids[i] >= tobs:
            if p_at(mids[i]) > 0.05 + 1e-12:
                hi = mids[i]
            else:
                break
    if lo is None: lo = mids[0] if mids else float("-inf")
    if hi is None: hi = mids[-1] if mids else float("inf")
    return p, lo, hi, m

def wilcoxon_exact(diffs):
    dz = [d for d in diffs if abs(d) > 1e-12]
    n = len(dz)
    if n == 0: return None, None
    order = sorted(range(n), key=lambda i: abs(dz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(dz[order[j + 1]]) == abs(dz[order[i]]): j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1): ranks[order[k]] = avg
        i = j + 1
    wpos = sum(ranks[i] for i in range(n) if dz[i] > 0)
    m = 1 << n; cnt = 0; tot = 0
    for mask in range(m):
        s = sum(ranks[i] if (mask >> i) & 1 else 0 for i in range(n))
        tot += 1
        if abs(s - n * (n + 1) / 4.0) >= abs(wpos - n * (n + 1) / 4.0) - 1e-9: cnt += 1
    return wpos, min(1.0, cnt / tot)

def paired_stats(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    if n >= 2:
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
        hw = tcrit(n - 1) * sd / math.sqrt(n)
    else:
        sd = float("nan"); hw = float("inf")
    p, plo, phi, nflips = signflip_perm(diffs)
    w, wp = wilcoxon_exact(diffs)
    return {"n": n, "mean": mean, "sd": sd, "t_ci": (mean - hw, mean + hw),
            "perm_p": p, "perm_ci": (plo, phi), "n_flips": nflips,
            "wilcoxon_W+": w, "wilcoxon_p": wp}

def classify(st):
    t_excl = st["t_ci"][0] > 0 or st["t_ci"][1] < 0
    p_excl = st["perm_p"] <= 0.05
    if t_excl and p_excl and ((st["t_ci"][0] > 0) == (st["perm_ci"][0] > 0)):
        return ("排除零（t 与置换一致）", st["mean"])
    if t_excl != p_excl:
        return ("方法间不一致，判为不确定", None)
    return ("compatible with zero（跨零）", None)

# ---------------- report ----------------
def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

def main():
    runs = load_runs()
    by = {(r["condition"], r["arm"]): r for r in runs}
    L = []
    J = {"generated_by": "analyze_formal_eval.py", "protocol": "PROTOCOL_formal_eval_v2p1 (FINAL LOCKED §15)"}
    A = L.append
    A("# ADW v2.1 正式评价 — 一次性完整报告（T1–T5）")
    A("")
    A("协议：PROTOCOL_formal_eval_v2p1（FINAL LOCKED，§15 作者确认）。本报告为")
    A("终点后一次性交付；未做任何中期分析。方向约定 Δ = time(前者) − time(后者)，")
    A("Δ<0 = 前者更快。主终点 = 完整服务墙钟 total_time_s（含全部真实开销，无任何扣除）。")
    A("")
    # ---- T1 ----
    A("## T1 全部 48 次启动原始记录（含退役/缺失，按臂失败计数）")
    A("")
    A("| seq | 轮 | 条件 | 臂 | label | 判定 | total_time_s | tps | boot_s | attempt | INFRA-R |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    fail = defaultdict(lambda: defaultdict(int))
    for r in runs:
        b = r["bench"] or {}
        t = r["t"]
        A(f"| {r['seq']} | r{r['round']} | {r['condition']} | {r['arm']} | {r['bench_label']} | "
          f"{r['verdict']} | {fmt(t)} | {fmt(b.get('tps'),1)} | {r['boot_s'] or '—'} | "
          f"{r['attempt'] or '—'} | {'是' if r['infra_rebooted'] else ''} |")
        if r["verdict"] != "VALID":
            fail[r["arm"]][r["verdict"]] += 1
    A("")
    A("**逐臂失败计数（全部报告，不删除）**：")
    for arm in ARMS:
        tot_bad = sum(fail[arm].values())
        det = ", ".join(f"{k}×{v}" for k, v in sorted(fail[arm].items())) or "无"
        A(f"- {arm} 臂：非 VALID {tot_bad} 次（{det}）")
    J["T1_failures_per_arm"] = {a: dict(fail[a]) for a in ARMS}
    A("")
    A("幸存者偏差声明：以下 T2 各区间**只描述成功运行中的性能**；A/M 臂的失败计数")
    A("（上表）是部署代价的一部分，与速度区间并排解读，不得删除失败后仅凭成功样本")
    A("宣布部署获益。缺失配对规则：n≥4 → 照常给区间并加旗标；n≤3 → 仅描述性")
    A("（\"insufficient pairs\"）。")
    A("")
    # ---- T2 ----
    A("## T2 配对差与区间（每条件独立；八个边际 95% CI，不校正，全部报告）")
    A("")
    A("八个区间为**边际 95%**，无整族 95% 覆盖保证；整体结论须同时权衡全部八个")
    A("区间、逐臂失败计数与 T5 范围声明，不得事后挑选排除零的一项。")
    A("")
    tmap = {(r["condition"], r["arm"], r["round"]): r["t"] for r in runs}
    j2 = {}
    for cond in CONDS:
        A(f"### 条件 {cond}")
        A("")
        # F-arm round means for % denominator
        fvals = [tmap.get((cond, "F", str(rr))) for rr in range(1, 7)]
        fmean_valid = [v for v in fvals if v is not None]
        fden = sum(fmean_valid) / len(fmean_valid) if fmean_valid else None
        A(f"F 臂 {cond} 有效轮均值（%描述的分母）：{fmt(fden)} s")
        A("")
        A("| 比较 | n对 | 缺失轮 | Δ均值(s) | sd | t 95% CI | 置换p(最小2/2ⁿ) | 置换95%CI | Wilcoxon p | 判定 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for x, y, kind in CONTRASTS:
            pairs = []; missing = []
            for rr in range(1, 7):
                tx = tmap.get((cond, x, str(rr))); ty = tmap.get((cond, y, str(rr)))
                if tx is not None and ty is not None: pairs.append(tx - ty)
                else: missing.append(rr)
            tag = f"{x}−{y} ({kind})"
            if len(pairs) <= 3:
                A(f"| {tag} | {len(pairs)} | {missing or '—'} | {fmt(sum(pairs)/len(pairs)) if pairs else '—'} | — | insufficient pairs（仅描述性） | — | — | — | 数据不足 |")
                j2[f"{cond}:{x}-{y}"] = {"n": len(pairs), "verdict": "insufficient pairs", "diffs": pairs}
                continue
            st = paired_stats(pairs)
            verdict_txt, _ = classify(st)
            flag = " ⚑缺对" if missing else ""
            pct = f"（{st['mean']/fden*100:+.2f}% of F）" if fden else ""
            A(f"| {tag} | {st['n']}{flag} | {missing or '—'} | {fmt(st['mean'])}{pct} | {fmt(st['sd'])} | "
              f"[{fmt(st['t_ci'][0])}, {fmt(st['t_ci'][1])}] | {st['perm_p']:.4f} | "
              f"[{fmt(st['perm_ci'][0])}, {fmt(st['perm_ci'][1])}] | {fmt(st['wilcoxon_p'],4)} | {verdict_txt} |")
            j2[f"{cond}:{x}-{y}"] = {"n": st["n"], "missing_rounds": missing, "mean": st["mean"],
                "sd": st["sd"], "t_ci": st["t_ci"], "perm_p": st["perm_p"],
                "perm_ci": st["perm_ci"], "wilcoxon_p": st["wilcoxon_p"],
                "verdict": verdict_txt, "diffs": pairs}
        A("")
    J["T2_contrasts"] = j2
    # ---- T3 ----
    A("## T3 A 臂行为链（检测/收缩分开定义，同一动作不重复归因）")
    A("")
    A("窗口→行号映射：累计 steps_actual 占比 × 条件总行数（§9）。W2 真实边界 = 行 4000。")
    A("检测 = 边界后首个 committed 窗口 z ≥ z_hi(314.14010272527486)；收缩 = 首个 committed")
    A("confirmed_shrink/urgent_shrink。两延迟分开报告，不相加。亚阈值 alert 仅作上下文。")
    A("")
    j3 = {}
    for cond in CONDS:
        A(f"### {cond}")
        for rr in range(1, 7):
            r = by.get((cond, "A"))
            r = next((x for x in runs if x["condition"] == cond and x["arm"] == "A" and x["round"] == str(rr)), None)
            if r is None or r["verdict"] != "VALID":
                A(f"- r{rr}: 非 VALID（{r['verdict'] if r else 'NOT_RUN'}），无行为链。")
                continue
            rows = load_jsonl(r["artifact_dir"])
            committed = [x for x in rows if x.get("committed")]
            total_steps = sum(x.get("steps_actual", 0) for x in rows) or None
            cum = 0; bnd_idx = None
            for i, x in enumerate(committed):
                cum += x.get("steps_actual", 0)
                if cond == "W2" and bnd_idx is None and total_steps and cum / total_steps * NROWS["W2"] >= BND_ROW_W2:
                    bnd_idx = i
            wseq = [x.get("W_after") for x in committed]
            wchanges = sum(1 for i in range(1, len(wseq)) if wseq[i] != wseq[i - 1])
            events = defaultdict(int)
            for x in committed: events[x.get("event")] += 1
            det_w = det_z = None; shr_w = None
            alerts = sum(1 for x in committed if x.get("event") in ("hold_alert",))
            if cond == "W2" and bnd_idx is not None:
                for i in range(bnd_idx, len(committed)):
                    z = committed[i].get("z")
                    if det_w is None and z is not None and z >= Z_HI:
                        det_w = i; det_z = z
                    if committed[i].get("event") in ("confirmed_shrink", "urgent_shrink"):
                        shr_w = i; break
            A(f"- r{rr}: 窗口 {len(committed)}（W 变化 {wchanges} 次，轨迹 {'→'.join(str(w) for w in dict.fromkeys(wseq))}）")
            if cond == "W2":
                A(f"  - 边界窗 idx={bnd_idx}；检测窗 idx={det_w}（z={fmt(det_z,1)}，延迟 {det_w - bnd_idx if det_w is not None and bnd_idx is not None else '—'} 窗）；"
                  f"首收缩窗 idx={shr_w}（距边界 {shr_w - bnd_idx if shr_w is not None and bnd_idx is not None else '—'} 窗）")
            A(f"  - committed 事件：{dict(events)}；亚阈值 alert(上下文)：{alerts}")
            j3[f"{cond}:r{rr}"] = {"windows": len(committed), "w_changes": wchanges,
                                   "w_traj": list(dict.fromkeys(wseq)),
                                   "bnd_idx": bnd_idx, "detect_idx": det_w, "detect_z": det_z,
                                   "shrink_idx": shr_w, "events": dict(events), "alerts": alerts}
        A("")
    J["T3_behavior"] = j3
    # ---- T4 ----
    A("## T4 M 臂完整性与代价记账 + M−F")
    A("")
    for rr in range(1, 7):
        for cond in CONDS:
            r = next((x for x in runs if x["condition"] == cond and x["arm"] == "M" and x["round"] == str(rr)), None)
            if r is None or r["verdict"] != "VALID":
                A(f"- {cond} r{rr}: 非 VALID（{r['verdict'] if r else 'NOT_RUN'}）")
                continue
            rows = load_jsonl(r["artifact_dir"])
            committed = [x for x in rows if x.get("committed")]
            wset = sorted({x.get("W_after") for x in committed})
            cw = [x.get("c_window_ms") for x in rows if isinstance(x.get("c_window_ms"), (int, float))]
            med = sorted(cw)[len(cw) // 2] if cw else None
            tot_ms = sum(cw) if cw else None
            A(f"- {cond} r{rr}: committed {len(committed)} 窗，W_after 集合={wset}（必须=[16]），"
              f"控制器 c_window_ms 中位={fmt(med,2)}，累计={fmt(tot_ms,0)} ms（rank0 并行记账，**不从服务时间扣除**）")
    A("")
    A("M−F 为辅助描述性比较（T2 表内）：同期估计监测净效应，不预设大小或可分辨性；")
    A("c_window_ms 是 PROF 式累计时间，不等于暴露在关键路径上的代价，只作记账披露。")
    A("")
    # ---- T5 ----
    A("## T5 范围声明与限制（原文重申）")
    A("")
    A("- 本报告只回答：A−F（部署净价值）、A−L（相对场景固定参考）、A−M（窗口策略作用）、")
    A("  M−F（监测代价辅助）。正、负或不确定都是合法结论；\"compatible with zero\" 按原样保留。")
    A("- 限制（协议 §11，全部有效）：prompt 重复/重合（125/250 unique×32，与校准集共享")
    A("  1383 条 unique prompt，L9）；O1 工作流（max_tokens=1，非真实生成分布）；单节点")
    A("  8×H20；W0-only 校准外推（W1/W2 未见；未演练路径存在）；A/M 与 F/L 机制不完全")
    A("  对称（仪表化差异已披露，M 臂即为其对照）；σ 估计来自 df=3 历史数据；L 臂窗口为")
    A("  单启动 argmin 事前选择（选择噪声已披露）；z_lo 边界口径敏感性已存档不回改。")
    A("- 公平性：全部臂同启动器、同模型、同并发 256、同数据集 sha；每启动前部署代码")
    A("  sha256sum -c 校验；A/M banner 8-rank + knob 唯一性门；F/L 负门（无 ADW banner）。")
    A("- 统计：n=6 固定；不校正多重性（边际区间）；置换伴随完整枚举（最小 p=2/2ⁿ，")
    A("  n=4→0.125，n=5→0.0625，n=6→0.03125，事前接受其保守性）；方法间不一致→不确定。")
    A("- 本轮结束后形成论文结论，不再进入无期限校准（作者指令，§15）。")
    A("")
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(J, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("report written:", OUT_MD)
    print("json written:", OUT_JSON)

if __name__ == "__main__":
    main()
