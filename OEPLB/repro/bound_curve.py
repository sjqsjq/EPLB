"""Perfect-balance throughput ceiling as a drawable line.

Delta_ceiling(r) = beta * max(0, r - r_k),   beta = B / T_flat

beta and r_k belong to the (model, GPU-count) configuration; r belongs to the
dataset.  So one config is one hinge line, and one dataset is one x position on
it -- computed offline from recorded routing counts by r_avg.py, no timed run.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, T_flat, B, r_k) straight from the three T(r) sweeps
CFG = [
    ("Qwen2-57B  EP=8", 82.86, 23.60, 1.099, "tab:blue"),
    ("Qwen2-57B  EP=4", 135.48, 46.35, 1.032, "tab:green"),
    ("Qwen3-235B EP=8", 167.07, 58.78, 1.093, "tab:red"),
]
# dataset r_before, offline from r_avg.py (identity placement)
DS = {
    "Qwen2-57B  EP=8": {"L256": 1.2177, "L512": 1.2288, "multi": 1.2085, "ShareGPT": 1.2040},
    "Qwen2-57B  EP=4": {"L256": 1.1071, "L512": 1.1125, "multi": 1.0980, "ShareGPT": 1.0965},
    "Qwen3-235B EP=8": {"L512": 1.7370},
}
# measured PB-OEPLB gains, same dataset & config (%)
MEAS = {
    ("Qwen2-57B  EP=8", "L256"): 1.0,
    ("Qwen2-57B  EP=4", "L256"): 2.70,
    ("Qwen2-57B  EP=4", "L512"): 2.39,
    ("Qwen3-235B EP=8", "L512"): 17.5,
}

print(f"{'config':18s} {'beta':>7} {'r_k':>6} | {'dataset':9s} {'r_before':>8} "
      f"{'ceiling':>8} {'measured':>9} {'eta':>6}")
for lab, T, B, rk, _ in CFG:
    beta = B / T
    for d, r in sorted(DS[lab].items(), key=lambda kv: -kv[1]):
        ceil = 100 * beta * max(0.0, r - rk)
        m = MEAS.get((lab, d))
        eta = f"{100*m/ceil:5.0f}%" if m and ceil > 0 else "     -"
        ms = f"{m:8.2f}%" if m else "        -"
        print(f"{lab:18s} {beta:7.4f} {rk:6.3f} | {d:9s} {r:8.4f} "
              f"{ceil:7.2f}% {ms} {eta}")

# --- placement ceiling vs routing ceiling -------------------------------
# Placement moves whole experts, so it can only reach the LPT floor; perfect
# routing reaches r=1.  But T(r) is flat below r_k, so the two coincide as long
# as the LPT floor lands inside the dead zone.
LPT = {"Qwen2-57B  EP=8": 1.0100, "Qwen2-57B  EP=4": 1.0039, "Qwen3-235B EP=8": 1.0003}
print(f"\n{'config':18s} {'r_LPT':>7} {'r_k':>6} {'placement':>10} {'routing':>8} "
      f"{'extra from routing':>19}")
for lab, T, B, rk, _ in CFG:
    beta, rl = B / T, LPT[lab]
    for d, r in sorted(DS[lab].items(), key=lambda kv: -kv[1])[:1]:
        cp = 100 * beta * max(0.0, r - max(rl, rk))
        cr = 100 * beta * max(0.0, r - max(1.0, rk))
        print(f"{lab:18s} {rl:7.4f} {rk:6.3f} {cp:9.2f}% {cr:7.2f}% "
              f"{cr-cp:18.2f}%")

fig, ax = plt.subplots(figsize=(7.2, 4.6))
xs = [1.0 + 0.005 * i for i in range(161)]
for lab, T, B, rk, col in CFG:
    beta = B / T
    ax.plot(xs, [100 * beta * max(0.0, x - rk) for x in xs], color=col, lw=2,
            label=f"{lab}  ($\\beta$={beta:.3f}, $r_k$={rk:.3f})")
    for d, r in DS[lab].items():
        c = 100 * beta * max(0.0, r - rk)
        ax.plot([r], [c], "o", color=col, ms=5, mfc="white", mew=1.6)
        ax.annotate(d, (r, c), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color=col)
        m = MEAS.get((lab, d))
        if m:
            ax.plot([r], [m], "^", color=col, ms=6)
            ax.plot([r, r], [m, c], color=col, lw=0.9, ls=":", alpha=0.7)
for lab, T, B, rk, col in CFG:
    ax.axvline(rk, color=col, lw=0.7, ls="--", alpha=0.35)
ax.set_xlabel(r"dataset imbalance $r_{\rm before}$  (offline from routing counts)")
ax.set_ylabel("throughput headroom (%)")
ax.set_title(r"Perfect-balance ceiling  $\Delta=\beta\max(0,\,r-r_k)$"
             "\n(circle = ceiling, triangle = measured PB-OEPLB, dotted = unrealized)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
ax.set_xlim(1.0, 1.8); ax.set_ylim(0, 26)
fig.tight_layout()
out = "/workspace/EPLB/OEPLB/fig_bound.png"
fig.savefig(out, dpi=170)
print("\nsaved", out)
