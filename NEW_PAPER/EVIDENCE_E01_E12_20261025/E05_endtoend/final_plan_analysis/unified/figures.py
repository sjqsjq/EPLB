#!/usr/bin/env python3
"""F6/F7/F8 per plan section 13 (frozen unified analysis).
Rules honored: per-boot points shown; retired cells annotated (never hidden,
never plotted as zero); identical statistics for every method; no dual axes;
CVD-validated palette (OKLab dE, Vienot sim) + marker-shape secondary encoding;
effects recomputed here from sealed tables with the frozen section 8.2 formulas.
"""
import json, math, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats as sps

MAIN = json.load(open("main_table.json"))
P1   = json.load(open("p1_table.json"))

C_MAIN = {"I":"#000000","E":"#E69F00","P":"#0072B2","F":"#009E73"}
MK     = {"I":"o","E":"D","P":"s","F":"^"}
C_P1   = {"P":"#0072B2","E-R0":"#E69F00","E-R16":"#D55E00","E-R32":"#7A2E00"}
MK_P1  = {"P":"s","E-R0":"o","E-R16":"D","E-R32":"^"}
ARMS_MAIN = ["I","E","P","F"]
CONDS = ["E1","E2","E3","E4"]
COND_TITLE = {
 "E1":"E1 stable-code  (N=3221, in2048/O1, conc=256)",
 "E2":"E2 prover  (N=4082, in512/O1, conc=256)",
 "E3":"E3 segmented 4-domain  (N=8192, in512/O1, conc=256)",
 "E4":"E4 short-in long-out  (N=2048, in512/out128, conc=256)"}
RNG = np.random.default_rng(20261019)

def style_ax(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)

def tci(x):
    x = np.asarray(x, float); n = len(x)
    if n == 0: return np.nan, (np.nan, np.nan), np.nan
    m = x.mean()
    if n < 2: return m, (np.nan, np.nan), np.nan
    se = x.std(ddof=1)/math.sqrt(n)
    if se == 0: return m, (m, m), 0.0
    tcrit = sps.t.ppf(0.975, n-1)
    return m, (m-tcrit*se, m+tcrit*se), 2*sps.t.sf(abs(m/se), n-1)

def stars(p):
    if not np.isfinite(p): return ""
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else " ns"))

def Tv(cond, arm, verdict="VALID"):
    out = {}
    for r in MAIN:
        if r["cond"]==cond and r["arm"]==arm and r["verdict"]==verdict and r.get("T_s"):
            out[r["block"]] = r["T_s"]
    return out

def paired_effect(tb, tp):
    """paired per-block log-ratio -> G=exp(mean)-1, 95% t-CI, p (two-sided)."""
    lr = [math.log(tb[b]/tp[b]) for b in sorted(set(tb)&set(tp))]
    if not lr: return None
    m, ci, p = tci(lr)
    return dict(n=len(lr), G=math.expm1(m), lo=math.expm1(ci[0]), hi=math.expm1(ci[1]), p=p)

# ---------------- F6a: main matrix per-workflow ----------------
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
for ax, cond in zip(axes.flat, CONDS):
    style_ax(ax)
    nret = {}
    for arm in ARMS_MAIN:
        nret[arm] = sum(1 for r in MAIN if r["cond"]==cond and r["arm"]==arm and r["verdict"]=="RESULT_RETIRE")
    tp = Tv(cond, "P")
    ymax = 0
    for i, arm in enumerate(ARMS_MAIN):
        tv = Tv(cond, arm)
        blocks = sorted(tv); vals = [tv[b] for b in blocks]
        xs = i + RNG.uniform(-0.10, 0.10, len(vals))
        ax.scatter(xs, vals, s=16, color=C_MAIN[arm], marker=MK[arm], alpha=0.9,
                   edgecolors="white", linewidths=0.5, zorder=3)
        if len(vals) >= 2:
            m, (lo, hi), _ = tci(vals)
            ax.errorbar([i], [m], yerr=[[m-lo], [hi-lo if np.isfinite(hi) else 0]],
                        fmt="none", ecolor=C_MAIN[arm], elinewidth=1.4, capsize=4, zorder=4)
            ax.scatter([i], [m], s=90, color=C_MAIN[arm], marker=MK[arm],
                       edgecolors="white", linewidths=1.2, zorder=5)
        if vals: ymax = max(ymax, max(vals))
    # effect annotations: P vs each baseline (unadjusted, per frozen spec)
    lines = []
    for arm in ("E","I","F"):
        eff = paired_effect(Tv(cond, arm), tp)
        if eff:
            lines.append(f"P vs {arm}: {eff['G']*100:+.1f}% [{eff['lo']*100:+.1f},{eff['hi']*100:+.1f}]{stars(eff['p'])}")
    ax.text(0.02, 0.985, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", lw=0.6, alpha=0.9))
    ax.set_title(COND_TITLE[cond], fontsize=9.5, pad=6)
    ax.set_xticks(range(4))
    labels = []
    for arm in ARMS_MAIN:
        labels.append(f"{arm}\n({nret[arm]}R)" if nret[arm] else arm)
    ax.set_xticklabels(labels, fontsize=9)
    for tick, arm in zip(ax.get_xticklabels(), ARMS_MAIN):
        tick.set_color(C_MAIN[arm] if arm != "I" else "#333333")
    ax.set_xlim(-0.55, 3.55); ax.set_ylim(0, ymax*1.30)
    if cond in ("E1","E3"): ax.set_ylabel("total time T (s)", fontsize=9)
fig.suptitle("F6 · Main matrix — per-workflow total time by arm (8 blocks/boot each; dots = per-boot T, diamond+whisker = mean ± 95% t-CI)\n"
             "168-cell frozen matrix; RESULT_RETIRE cells excluded from T but counted in ticks as (nR); stars: unadjusted per-workflow p (only the 2 confirmatory tests are Holm-adjusted)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("fig_f6_main_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- F6b: confirmatory forest (g_b per complete block) ----------------
def gb_series(base_arm):
    tb = {c: Tv(c, base_arm) for c in CONDS}
    tp = {c: Tv(c, "P") for c in CONDS}
    out = {}
    for b in range(1, 9):
        lrs = [math.log(tb[c][b]/tp[c][b]) for c in CONDS if b in tb[c] and b in tp[c]]
        if len(lrs) == 4: out[b] = float(np.mean(lrs))
    return out

gE = gb_series("E"); gI = gb_series("I")
raw = {}
for name, g in (("P vs E", gE), ("P vs I", gI)):
    vals = [g[b] for b in sorted(g)]
    m, ci, p = tci(vals)
    raw[name] = dict(g=g, G=math.expm1(m), lo=math.expm1(ci[0]), hi=math.expm1(ci[1]), p=p)
ks = sorted(raw, key=lambda k: raw[k]["p"])
holm = {}
prev = 0.0
for i, k in enumerate(ks):
    q = min(1.0, max(prev, (len(ks)-i)*raw[k]["p"])); holm[k] = q; prev = q

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax, name in zip(axes, ("P vs E", "P vs I")):
    style_ax(ax); d = raw[name]
    blocks = sorted(d["g"]); vals = [d["g"][b]*100 for b in blocks]
    ax.axvline(0, color="#999999", lw=1)
    ax.scatter(vals, [f"block {b}" for b in blocks], s=42, color="#0072B2", zorder=3,
               edgecolors="white", linewidths=0.8)
    m, ci, _ = tci(vals)
    ax.scatter([m], ["pooled G"], s=130, color="#D55E00", marker="D", zorder=4,
               edgecolors="white", linewidths=1.2)
    ax.errorbar([m], ["pooled G"], xerr=[[m-ci[0]], [ci[1]-m]], fmt="none",
                ecolor="#D55E00", elinewidth=1.8, capsize=5, zorder=4)
    ax.set_xlabel("g_b = mean_j log(T_arm / T_P) per block  →  %", fontsize=8.5)
    sig = "SIGNIFICANT" if holm[name] < 0.05 else "not significant"
    ax.set_title(f"{name} (confirmatory)\nG = {d['G']*100:+.2f}% [{d['lo']*100:+.2f},{d['hi']*100:+.2f}] · Holm p = {holm[name]:.3g} → {sig}",
                 fontsize=9)
    ax.invert_yaxis()
fig.suptitle("F6b · Confirmatory family (frozen: exactly 2 tests, Holm α=0.05, complete blocks only, n=%d blocks each)" % len(gE), fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig("fig_f6_confirmatory.png", dpi=150, bbox_inches="tight")
plt.close(fig)
with open("f6_forest_values.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["test","block","g_b_pct"])
    for name in raw:
        for b in sorted(raw[name]["g"]): w.writerow([name, b, f"{raw[name]['g'][b]*100:.4f}"])
        w.writerow([name, "pooled_G_pct", f"{raw[name]['G']*100:.4f}"])
        w.writerow([name, "holm_p", f"{holm[name]:.6g}"])

# ---------------- F7: ablations (service forest + behavior facet) ----------------
CONTRASTS = [("A3","E1","A3 denseSample @E1"), ("A2","E2","A2 maxdelta @E2"),
             ("A2","E3","A2 maxdelta @E3"), ("A4","E3","A4 noCommit @E3"),
             ("A1","E4","A1 allphase @E4")]
NOT_EXERCISED = {("A3","E1")}
fig, axes = plt.subplots(2, 5, figsize=(17.5, 7.0))
f7rows = []
for col, (arm, cond, title) in enumerate(CONTRASTS):
    ta = Tv(cond, arm); tp = Tv(cond, "P")
    lr = {b: math.log(ta[b]/tp[b]) for b in sorted(set(ta)&set(tp))}
    blocks = sorted(lr); vals = [lr[b]*100 for b in blocks]
    m, ci, p = tci(vals)
    ax = axes[0][col]; style_ax(ax)
    ne = (arm, cond) in NOT_EXERCISED
    ccol = "#8C8C8C" if ne else "#0072B2"
    ax.axvline(0, color="#999999", lw=1)
    ax.scatter(vals, [str(b) for b in blocks], s=40, color=ccol, zorder=3,
               edgecolors="white", linewidths=0.8)
    ax.scatter([m], ["mean"], s=120, color="#D55E00" if not ne else "#5A5A5A",
               marker="D", zorder=4, edgecolors="white", linewidths=1.2)
    if np.isfinite(ci[0]):
        ax.errorbar([m], ["mean"], xerr=[[m-ci[0]], [ci[1]-m]], fmt="none",
                    ecolor="#D55E00" if not ne else "#5A5A5A", elinewidth=1.6, capsize=4, zorder=4)
    ax.invert_yaxis()
    Gv, Glo, Ghi = math.expm1(m/100.0), math.expm1(ci[0]/100.0), math.expm1(ci[1]/100.0)
    head = f"{title}\nG = {Gv*100:+.2f}% [{Glo*100:+.2f},{Ghi*100:+.2f}]  p={p:.3g} (unadj.)"
    if ne: head += "\nNOT EXERCISED (identical behavior counts vs P)"
    ax.set_title(head, fontsize=8.6, color=("#B00020" if ne else "black"))
    ax.set_xlabel("block   →   log(T_A/T_P) %", fontsize=8)
    # behavior facet: swaps_started_total per boot
    ax2 = axes[1][col]; style_ax(ax2)
    for xi, a2 in enumerate(("P", arm)):
        sw = sorted(r["behav_swaps_started_total"] for r in MAIN
                    if r["cond"]==cond and r["arm"]==a2 and r["verdict"]=="VALID"
                    and r.get("behav_swaps_started_total") is not None)
        xs = xi + RNG.uniform(-0.09, 0.09, len(sw))
        ax2.scatter(xs, sw, s=30, color=C_MAIN["P"] if a2=="P" else "#D55E00",
                    marker="s" if a2=="P" else "D", alpha=0.9,
                    edgecolors="white", linewidths=0.6, zorder=3)
        if sw: ax2.scatter([xi], [np.mean(sw)], s=95, color=C_MAIN["P"] if a2=="P" else "#D55E00",
                           marker="_", linewidths=2.4, zorder=4)
    ax2.set_xticks([0,1]); ax2.set_xticklabels(["P", arm], fontsize=9)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylabel("swaps_started / boot", fontsize=8.5)
    ax2.set_title("behavior facet (manipulation check)", fontsize=8.2, color="#555555")
    f7rows.append([title, Gv*100, Glo*100, Ghi*100, p, m, ci[0], ci[1],
                   "not_exercised" if ne else "exercised"])
fig.suptitle("F7 · Ablations (secondary family, unadjusted p, all 5 reported) — top: per-block service effect log(T_A/T_P); bottom: per-boot behavior counts\n"
             "A4@E3 verified (nocommit_planned_ops≈8.9k>0, swaps=0 → planning continued, commits blocked); A1@E4 verified (decode_record_batches 1020–1054 vs P=0).\nHeadline G = exp(mean log-ratio)−1 per frozen §8.2; pooled diamond sits at the log-mean on the per-block log-ratio axis.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig("fig_f7_ablations.png", dpi=150, bbox_inches="tight")
plt.close(fig)
with open("f7_forest_values.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["contrast","G_pct","G_ci_lo","G_ci_hi","p_unadj","logmean_pct","log_ci_lo","log_ci_hi","status"])
    for row in f7rows:
        w.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.6g}",
                    f"{row[5]:.4f}", f"{row[6]:.4f}", f"{row[7]:.4f}", row[8]])

# ---------------- F8: P1 ----------------
def armkey(r):
    return "P" if r["arm"]=="P" else "E-R"+r["params"].split()[1]
ORDER = ["P","E-R0","E-R16","E-R32"]
PRES = [("PL","PL  conc=64"),("PH","PH  conc=256")]

fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
panels = [("total time T (s)", lambda r: r["T_s"]),
          ("output throughput (tok/s)", lambda r: r["tps_out"]),
          ("TTFT p99 (s)", lambda r: r["ttft_p99_ms"]/1000.0),
          ("e2e latency p99 (s)", lambda r: r["lat_p99_ms"]/1000.0)]
for pi, (ylab, getter) in enumerate(panels):
    ax = axes.flat[pi]; style_ax(ax)
    for gi, (pk, plab) in enumerate(PRES):
        for ai, ak in enumerate(ORDER):
            x0 = gi + (ai-1.5)*0.19
            rows = [r for r in P1 if r["pressure"]==pk and armkey(r)==ak and r["verdict"]=="VALID"]
            vals = [getter(r) for r in rows if getter(r) is not None]
            if not vals: continue
            xs = x0 + RNG.uniform(-0.05, 0.05, len(vals))
            ax.scatter(xs, vals, s=15, color=C_P1[ak], marker=MK_P1[ak], alpha=0.9,
                       edgecolors="white", linewidths=0.5, zorder=3)
            m, (lo, hi), _ = tci(vals)
            ax.errorbar([x0], [m], yerr=[[m-lo], [hi-lo if np.isfinite(hi) else 0]],
                        fmt="none", ecolor=C_P1[ak], elinewidth=1.3, capsize=3.5, zorder=4)
            ax.scatter([x0], [m], s=80, color=C_P1[ak], marker=MK_P1[ak],
                       edgecolors="white", linewidths=1.1, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([p[1] for p in PRES], fontsize=9)
    ax.set_ylabel(ylab, fontsize=9); ax.set_title(ylab, fontsize=9.5)
# KV capacity panel
ax = axes.flat[4]; style_ax(ax)
for ai, ak in enumerate(ORDER):
    rows = [r for r in P1 if armkey(r)==ak and r["verdict"]=="VALID" and r.get("kv_tokens_min")]
    vals = [r["kv_tokens_min"]/1000.0 for r in rows]
    xs = ai + RNG.uniform(-0.10, 0.10, len(vals))
    ax.scatter(xs, vals, s=16, color=C_P1[ak], marker=MK_P1[ak], alpha=0.9,
               edgecolors="white", linewidths=0.5, zorder=3)
    m, (lo, hi), _ = tci(vals)
    ax.errorbar([ai], [m], yerr=[[m-lo], [hi-lo if np.isfinite(hi) else 0]],
                fmt="none", ecolor=C_P1[ak], elinewidth=1.3, capsize=4, zorder=4)
    ax.scatter([ai], [m], s=90, color=C_P1[ak], marker=MK_P1[ak],
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.annotate(f"{m:.1f}k", (ai, m), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(4)); ax.set_xticklabels(ORDER, fontsize=9)
ax.set_ylabel("KV capacity (k tokens, min over 8 ranks)", fontsize=8.5)
ax.set_title("KV cache capacity vs redundancy R (PL+PH pooled)", fontsize=9.5)
# annotation panel
ax = axes.flat[5]; ax.axis("off")
ax.text(0.03, 0.95,
        "P1 pressure-extension block (32 cells)\n\n"
        "· 32/32 VALID — zero incidents, zero errors\n"
        "· completion 65536/65536 = 100% (all cells)\n"
        "· stream: in2048/out128, N=2048 per cell\n"
        "· PL = conc 64 (low pressure), PH = conc 256 (high)\n"
        "· E arms: official EPLB, R ∈ {0,16,32}, period 256,\n"
        "   deepep normal (CUDA graph disabled — frozen §4)\n"
        "· P arm: PB adaptive (ADW21 frozen knobs)\n\n"
        "Status: DESCRIPTIVE RANGE ESTIMATES ONLY\n"
        "(frozen §10.3 — no ad-hoc SLO, no confirmatory test)",
        transform=ax.transAxes, fontsize=8.8, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F7F7F7", ec="#BBBBBB", lw=0.8))
handles = [Line2D([], [], marker=MK_P1[ak], color=C_P1[ak], ls="", markersize=8,
                  markeredgecolor="white", label=ak) for ak in ORDER]
ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, title="arm", title_fontsize=9)
fig.suptitle("F8 · P1 — redundancy R vs capacity / throughput / tail latency (per-boot dots + mean ± 95% t-CI; 4 boots per arm×pressure)", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("fig_f8_p1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("FIGURES_DONE")
