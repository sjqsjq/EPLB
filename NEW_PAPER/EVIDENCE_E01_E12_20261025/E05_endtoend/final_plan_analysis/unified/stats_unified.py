#!/usr/bin/env python3
"""stats_unified.py — unified analysis step 2: frozen statistics per plan §8.2 /
FREEZE §6. Confirmatory family = {P-vs-I, P-vs-E} ONLY (Holm). Secondary family
(declared separately, unadjusted, all reported) = {P-vs-F*, A3@E1, A2@E2, A2@E3,
A4@E3, A1@E4}. P1 = descriptive range estimates (§10.3: no confirmatory claim,
no ad-hoc SLO). Completion report includes RESULT_RETIRE cells (§9.9)."""
import json, math, csv
from scipy import stats

F = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019"
OUT = f"{F}/analysis/unified"
main = json.load(open(f"{OUT}/main_table.json"))
p1t = json.load(open(f"{OUT}/p1_table.json"))
CONDS = ["E1", "E2", "E3", "E4"]
BLOCKS = list(range(1, 9))
rep = []          # human-readable lines
def P(s=""):
    rep.append(s); print(s)

valid = {(r["cond"], r["block"], r["arm"]): r for r in main if r["verdict"] == "VALID"}
def cell(c, b, a): return valid.get((c, b, a))

def summarize(vals):
    n = len(vals); mean = sum(vals)/n
    if n < 2: return {"n": n, "mean": mean, "ci_lo": None, "ci_hi": None, "sd": None, "p_two_sided": None}
    sd = math.sqrt(sum((x-mean)**2 for x in vals)/(n-1))
    tc = stats.t.ppf(0.975, n-1); half = tc*sd/math.sqrt(n) if sd > 0 else 0.0
    p = float(stats.ttest_1samp(vals, 0.0).pvalue) if sd > 0 else 0.0
    return {"n": n, "mean": mean, "ci_lo": mean-half, "ci_hi": mean+half, "sd": sd, "p_two_sided": p}

def gfmt(s):
    if s["ci_lo"] is None: return f"n={s['n']} mean={s['mean']:.4f}"
    return (f"n={s['n']} mean_logratio={s['mean']:.4f} [{s['ci_lo']:.4f},{s['ci_hi']:.4f}] "
            f"G=exp(mean)-1={(math.exp(s['mean'])-1)*100:+.2f}% [{(math.exp(s['ci_lo'])-1)*100:+.2f}%,{(math.exp(s['ci_hi'])-1)*100:+.2f}%] p={s['p_two_sided']:.2e}")

# ============ 1. CONFIRMATORY FAMILY (frozen: exactly 2 tests, Holm) ============
P("="*100); P("1. CONFIRMATORY FAMILY — P-vs-I and P-vs-E (equal-weight 4-workflow log-time-ratio, complete blocks, Holm)"); P("="*100)
conf = {}
gb_detail = {}
for B in ["I", "E"]:
    gbs = []
    for b in BLOCKS:
        terms, ok = [], True
        for j in CONDS:
            p, q = cell(j, b, "P"), cell(j, b, B)
            if not p or not q: ok = False; break
            terms.append(math.log(q["T_s"]/p["T_s"]))
        if ok:
            gbs.append({"block": b, "g_b": sum(terms)/4.0, "terms": {j: round(t, 5) for j, t in zip(CONDS, terms)}})
    vals = [g["g_b"] for g in gbs]
    s = summarize(vals)
    conf[B] = s; gb_detail[B] = gbs
    lost = [b for b in BLOCKS if b not in [g["block"] for g in gbs]]
    P(f"\nP-vs-{B}: complete blocks {[g['block'] for g in gbs]} (lost: {lost})")
    for g in gbs: P(f"  g_b(b{g['block']}) = {g['g_b']:+.4f}  terms {g['terms']}")
    P(f"  {gfmt(s)}")

# Holm on the two confirmatory p-values
pvals = {"I": conf["I"]["p_two_sided"], "E": conf["E"]["p_two_sided"]}
order = sorted(pvals, key=lambda k: pvals[k]); m = 2; holm = {}; prev = 0.0
for i, k in enumerate(order):
    adj = max(prev, (m-i)*pvals[k]); adj = min(adj, 1.0); holm[k] = adj; prev = adj
P(f"\nHolm-adjusted p-values: P-vs-I={holm['I']:.3e}, P-vs-E={holm['E']:.3e} (alpha=0.05)")
for k in ["I", "E"]:
    sig = "SIGNIFICANT" if holm[k] < 0.05 else "not significant"
    P(f"  P-vs-{k}: G={(math.exp(conf[k]['mean'])-1)*100:+.2f}%, Holm p={holm[k]:.3e} -> {sig}")

# ============ 2. PER-WORKFLOW EFFECTS (reporting obligation, unadjusted) ============
P(); P("="*100); P("2. PER-WORKFLOW EFFECTS (unadjusted; raw times, paired diffs, time-reduction, throughput-gain named separately)"); P("="*100)
per_wf = {}
for B in ["I", "E", "F"]:
    for j in CONDS:
        pairs = []
        for b in BLOCKS:
            p, q = cell(j, b, "P"), cell(j, b, B)
            if p and q: pairs.append({"block": b, "T_P": p["T_s"], f"T_{B}": q["T_s"],
                                       "logratio": math.log(q["T_s"]/p["T_s"]),
                                       "paired_diff_s": q["T_s"]-p["T_s"],
                                       "time_reduction_P": 1-p["T_s"]/q["T_s"],
                                       "throughput_gain_P": q["T_s"]/p["T_s"]-1})
        lr = summarize([x["logratio"] for x in pairs])
        per_wf[(B, j)] = {"pairs": pairs, "stats": lr}
        mt = lambda key: (min(x[key] for x in pairs), max(x[key] for x in pairs)) if pairs else (None, None)
        tp_lo, tp_hi = mt("T_P"); tb_lo, tb_hi = mt(f"T_{B}")
        P(f"\nP-vs-{B} @{j}: n={lr['n']} blocks {[x['block'] for x in pairs]}")
        P(f"  T_P range [{tp_lo:.1f},{tp_hi:.1f}]s  T_{B} range [{tb_lo:.1f},{tb_hi:.1f}]s" if pairs else "  no pairs")
        P(f"  {gfmt(lr)}")
        if pairs:
            tr = [x["time_reduction_P"] for x in pairs]; tg = [x["throughput_gain_P"] for x in pairs]
            P(f"  time_reduction(1-T_P/T_{B}): mean {sum(tr)/len(tr)*100:+.2f}% [{min(tr)*100:+.2f},{max(tr)*100:+.2f}]")
            P(f"  throughput_gain(T_{B}/T_P-1): mean {sum(tg)/len(tg)*100:+.2f}% [{min(tg)*100:+.2f},{max(tg)*100:+.2f}]")

# ============ 3. SECONDARY FAMILY (separately declared; unadjusted; all reported) ============
P(); P("="*100); P("3. SECONDARY FAMILY — P-vs-F* (complete blocks) + 5 condition×ablation contrasts (each vs same-cond P)"); P("="*100)
sec = {}
gbsF = []
for b in BLOCKS:
    terms, ok = [], True
    for j in CONDS:
        p, q = cell(j, b, "P"), cell(j, b, "F")
        if not p or not q: ok = False; break
        terms.append(math.log(q["T_s"]/p["T_s"]))
    if ok: gbsF.append({"block": b, "g_b": sum(terms)/4.0})
sF = summarize([g["g_b"] for g in gbsF]); sec["P_vs_F_star"] = {"blocks": gbsF, "stats": sF}
P(f"\nP-vs-F* (complete blocks {[g['block'] for g in gbsF]}, lost {[b for b in BLOCKS if b not in [g['block'] for g in gbsF]]}):")
for g in gbsF: P(f"  g_b(b{g['block']}) = {g['g_b']:+.4f}")
P(f"  {gfmt(sF)}  [small-n caveat: n={sF['n']}]")

ABL = [("A3", "E1"), ("A2", "E2"), ("A2", "E3"), ("A4", "E3"), ("A1", "E4")]
for A, j in ABL:
    pairs = []
    for b in BLOCKS:
        p, q = cell(j, b, "P"), cell(j, b, A)
        if p and q: pairs.append({"block": b, "T_P": p["T_s"], "T_A": q["T_s"], "logratio": math.log(q["T_s"]/p["T_s"])})
    s = summarize([x["logratio"] for x in pairs])
    sec[f"{A}@{j}"] = {"pairs": pairs, "stats": s}
    P(f"\n{A}@{j} vs P@{j}: n={s['n']} blocks {[x['block'] for x in pairs]}")
    for x in pairs: P(f"  b{x['block']}: T_P={x['T_P']:.1f}s T_A={x['T_A']:.1f}s logratio={x['logratio']:+.4f}")
    P(f"  {gfmt(s)}")

# ============ 4. P1 (descriptive range estimates; §10.3 boundaries) ============
P(); P("="*100); P("4. P1 — redundancy R vs pressure (descriptive range estimates; NOT confirmatory; no ad-hoc SLO)"); P("="*100)
p1v = {(r["params"] if r["arm"] == "E" else "P", r["pressure"], r["block"]): r for r in p1t if r["verdict"] == "VALID"}
ARMS = [("P", "P_final"), ("auto 0 256", "E-R0"), ("auto 16 256", "E-R16"), ("auto 32 256", "E-R32")]
p1_stats = {}
for pres in ["PL", "PH"]:
    P(f"\n--- pressure {pres} ({'conc=64' if pres=='PL' else 'conc=256 (C_hi)'}) ---")
    for key, name in ARMS:
        cells = [p1v[(key, pres, b)] for b in range(1, 5) if (key, pres, b) in p1v]
        Ts = [c["T_s"] for c in cells]
        P(f"  {name:7s}: T mean {sum(Ts)/len(Ts):7.1f}s range [{min(Ts):.1f},{max(Ts):.1f}]  tps_out mean {sum(c['tps_out'] for c in cells)/len(cells):6.1f}"
          f"  ttft_p99 mean {sum(c['ttft_p99_ms'] for c in cells)/len(cells)/1000:6.2f}s  lat_p99 mean {sum(c['lat_p99_ms'] for c in cells)/len(cells)/1000:6.2f}s"
          f"  ok={sum(c['ok'] for c in cells)}/{sum(c['num_requests'] for c in cells)} err={sum(c['errors'] for c in cells)}  KV_min {min(c['kv_tokens_min'] for c in cells)}")
        p1_stats[(name, pres)] = {"T": Ts, "tps": [c["tps_out"] for c in cells],
                                   "ttft_p99_ms": [c["ttft_p99_ms"] for c in cells],
                                   "lat_p99_ms": [c["lat_p99_ms"] for c in cells],
                                   "kv_min": [c["kv_tokens_min"] for c in cells]}
    for key, name in ARMS[1:]:
        lr = []
        for b in range(1, 5):
            p, q = p1v.get(("P", pres, b)), p1v.get((key, pres, b))
            if p and q: lr.append(math.log(q["T_s"]/p["T_s"]))
        if lr:
            s = summarize(lr)
            p1_stats[(f"{name}_vs_P", pres)] = s
            P(f"  paired log(T_{name}/T_P): n={s['n']} mean {s['mean']:+.4f} -> P advantage G-form {(math.exp(s['mean'])-1)*100:+.2f}%"
              + (f" [{(math.exp(s['ci_lo'])-1)*100:+.2f},{(math.exp(s['ci_hi'])-1)*100:+.2f}%] (t-CI, df={s['n']-1}, range-estimate only)" if s["ci_lo"] is not None else ""))

# ============ 5. COMPLETION REPORT (includes RESULT_RETIRE; §9.9) ============
P(); P("="*100); P("5. COMPLETION REPORT — all planned boots; RESULT_RETIRE counted, never deleted"); P("="*100)
comp = {}
for r in main:
    k = (r["cond"], r["arm"])
    c = comp.setdefault(k, {"boots": 0, "valid": 0, "retired": 0, "ok": 0, "planned": 0})
    c["boots"] += 1
    c["planned"] += r["planned_nreq"]
    if r["verdict"] == "VALID": c["valid"] += 1; c["ok"] += r["ok"]
    else:
        c["retired"] += 1
        c["ok"] += (r["ok"] or 0)   # partial completions from completion-rate path; crash cells count 0 (client killed; not measured)
comp_rows = []
P(f"\n{'cond':5s} {'arm':4s} boots valid retired  completed/planned  rate")
for (c_, a_) in sorted(comp):
    v = comp[(c_, a_)]
    rate = v["ok"]/v["planned"]
    comp_rows.append({"cond": c_, "arm": a_, **v, "rate": round(rate, 5)})
    flag = "  <-- retired" if v["retired"] else ""
    P(f"{c_:5s} {a_:4s} {v['boots']:5d} {v['valid']:5d} {v['retired']:7d}  {v['ok']:8d}/{v['planned']:<8d} {rate*100:6.2f}%{flag}")
tot_ok = sum(v["ok"] for v in comp.values()); tot_pl = sum(v["planned"] for v in comp.values())
P(f"\nMAIN aggregate request completion (retired crash cells count 0 completed — client killed after watchdog; partial-completion cells keep measured ok): {tot_ok}/{tot_pl} = {tot_ok/tot_pl*100:.2f}%")
p1_ok = sum(r["ok"] for r in p1t); p1_pl = sum(r["num_requests"] for r in p1t)
P(f"P1 aggregate: {p1_ok}/{p1_pl} = {p1_ok/p1_pl*100:.2f}% (32/32 boots VALID, zero errors)")
P("Retired cell detail: m_E4_F_b4 ok=1281/2048 err=767 (completion-rate path); m_E4_F_b7 ok=1024/2048 err=1024 (completion-rate path); m_E3_{P_b6,F_b8,F_b4,F_b3,P_b3} no bench (NCCL hang -> watchdog abort -> client killed; completed count not measured, counted 0 conservatively)")

# ============ 6. MECHANISM COUNTS PER BOOT (aggregated per launch; no pseudo-replication tests) ============
P(); P("="*100); P("6. MECHANISM/COUNTS PER BOOT + ablation manipulation checks (§7)"); P("="*100)
beh = {}
for r in main:
    if r["verdict"] != "VALID": continue
    a = r["arm"]
    if not any(k.startswith("behav_") and r.get(k) is not None for k in r): continue
    d = beh.setdefault((a, r["cond"]), {"swaps": [], "transitions": [], "sp_fwd": [], "recorded_steps": [], "nocommit": [], "decode_rec": [], "boots": 0})
    d["boots"] += 1
    if r.get("behav_swaps_started_total") is not None: d["swaps"].append(r["behav_swaps_started_total"])
    if r.get("behav_transitions") is not None: d["transitions"].append(r["behav_transitions"])
    if r.get("behav_sp_fwd_total") is not None: d["sp_fwd"].append(r["behav_sp_fwd_total"])
    if r.get("behav_recorded_steps") is not None: d["recorded_steps"].append(r["behav_recorded_steps"])
    if r.get("behav_nocommit_planned_ops_total") is not None: d["nocommit"].append(r["behav_nocommit_planned_ops_total"])
    if r.get("behav_decode_record_batches") is not None: d["decode_rec"].append(r["behav_decode_record_batches"])
for k in sorted(beh):
    d = beh[k]
    rng = lambda v: f"[{min(v)},{max(v)}]" if v else "n/a"
    P(f"  {k[0]}@{k[1]} boots={d['boots']}: swaps_started {rng(d['swaps'])}  transitions {rng(d['transitions'])}  sp_fwd {rng(d['sp_fwd'])}  recorded_steps {rng(d['recorded_steps'])}"
      + (f"  nocommit_planned {rng(d['nocommit'])}" if d["nocommit"] else "") + (f"  decode_record_batches {rng(d['decode_rec'])}" if d["decode_rec"] else ""))
P("\nManipulation checks:")
a4 = beh.get(("A4", "E3")); a1 = beh.get(("A1", "E4")); a3 = beh.get(("A3", "E1"))
if a4: P(f"  A4 (noCommit) @E3: nocommit_planned_ops {min(a4['nocommit']) if a4['nocommit'] else '?'}>0 & swaps_started max {max(a4['swaps']) if a4['swaps'] else '?'} -> planning continued, commits blocked" if a4["nocommit"] else "  A4: no nocommit counter found — CHECK")
if a1: P(f"  A1 (allphase) @E4: decode_record_batches range [{min(a1['decode_rec'])},{max(a1['decode_rec'])}] -> decode recording actually exercised" if a1["decode_rec"] else "  A1: decode_record_batches missing — CHECK")
if a3: P(f"  A3 (denseSample) @E1: recorded_steps range [{min(a3['recorded_steps'])},{max(a3['recorded_steps'])}] vs P@E1 [{min(beh[('P','E1')]['recorded_steps'])},{max(beh[('P','E1')]['recorded_steps'])}]")

# ============ save ============
def clean(o):
    if isinstance(o, dict): return {("|".join(map(str, k)) if isinstance(k, tuple) else k): clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(x) for x in o]
    return o
json.dump(clean({"confirmatory": conf, "confirmatory_g_blocks": gb_detail, "holm": holm,
                 "per_workflow": per_wf, "secondary": sec, "p1": p1_stats,
                 "completion_by_cond_arm": comp_rows,
                 "totals": {"main_ok": tot_ok, "main_planned": tot_pl, "p1_ok": p1_ok, "p1_planned": p1_pl}}),
          open(f"{OUT}/unified_stats.json", "w"), indent=1)
open(f"{OUT}/stats_report.txt", "w").write("\n".join(rep) + "\n")
P(f"\nsaved -> {OUT}/unified_stats.json, stats_report.txt")
