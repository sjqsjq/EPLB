"""Draw the calibrated same-path ratio proxy for throughput headroom.

Delta_proxy(r) = beta * max(0, r - r_k),   beta = B / T_flat

Both beta and r_k must be fitted for the same execution path and workload
family.  Raw route ratio r only supplies an x coordinate: it is not a causal
conversion to compute time.  Padding-aware work and all-rank readiness remain
the mechanism-level quantities.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, T_flat, B, r_k) from same-path empirical T(r) sweeps. This curve is
# a descriptive engineering proxy, not a causal conversion from raw route r to
# compute time; padding/aligned work and all-rank readiness remain path-specific.
# Empirical cross-section notes (paper.md §5.11; not transfer laws):
#   * Existing GEMM-dominated points are descriptively close to the fitted
#     r_k-1 = c*EP^1.52 curve, but new paths still require directional
#     re-calibration.  The independently measured H800 threshold (1.7~2.3)
#     demonstrates non-transfer to a spin-absorbing communication path.
#   * The fitted threshold also changes with workload/protocol (A100 L256
#     1.053~1.070 vs L494 1.133); prompt length is a covariate, not a mechanism.
#   * beta is a same-path slope proxy for f_sens, not a hardware invariant:
#     GB200 f_sens=0.184 (nsys), while H800's empirical slope is about 0.03.
#   * The hinge's linear segment has only been checked through r ~= 1.8.
CFG = [
    ("Qwen2-57B  EP=8", 82.86, 23.60, 1.099, "tab:blue"),
    ("Qwen2-57B  EP=4", 135.48, 46.35, 1.032, "tab:green"),
    ("Qwen3-235B EP=8", 167.07, 58.78, 1.093, "tab:red"),
    # A100 measured (T(r) scan, 256tok burst, robust fit ex-conc):
    ("Qwen3-235B EP=8 A100-L256", 5.46, 1.80, 1.053, "tab:orange"),
    # 30B EP=4: beta reverse-fit from the path-specific 6.36% headroom estimate.
    ("Qwen3-30B  EP=4", 135.48, 28.09, 1.032, "tab:purple"),
]
# dataset r_before, offline from r_avg.py (identity placement)
DS = {
    "Qwen2-57B  EP=8": {"L256": 1.2177, "L512": 1.2288, "multi": 1.2085, "ShareGPT": 1.2040},
    "Qwen2-57B  EP=4": {"L256": 1.1071, "L512": 1.1125, "multi": 1.0980, "ShareGPT": 1.0965},
    "Qwen3-235B EP=8": {"L512": 1.7370},
    "Qwen3-235B EP=8 A100-L256": {"L256": 1.7163},  # A100-native recorder counts
    "Qwen3-30B  EP=4": {"L512": 1.3376},
}
# measured PB-OEPLB gains, same dataset & config (%)
MEAS = {
    ("Qwen2-57B  EP=8", "L256"): 1.05,  # conservative paired estimate; see archived repeats
    ("Qwen2-57B  EP=4", "L256"): 1.83,  # mean/mean of two repeats; prior 2.70 was one pairing
    ("Qwen2-57B  EP=4", "L512"): 2.39,
    ("Qwen3-235B EP=8", "L512"): 17.46,  # mean/mean of two repeats: +16.7%, +18.2%
    ("Qwen3-235B EP=8 A100-L256", "L256"): 21.8,  # DataForest measured, A100 scan session
    ("Qwen3-30B  EP=4", "L512"): -2.16,  # four-repeat mean; stdev 3.20%
}

print(f"{'config':18s} {'beta':>7} {'r_k':>6} | {'dataset':9s} {'r_before':>8} "
      f"{'proxy':>8} {'measured':>9} {'eta':>6}")
for lab, T, B, rk, _ in CFG:
    beta = B / T
    for d, r in sorted(DS[lab].items(), key=lambda kv: -kv[1]):
        ceil = 100 * beta * max(0.0, r - rk)
        m = MEAS.get((lab, d))
        eta = f"{100*m/ceil:5.0f}%" if m and ceil > 0 else "     -"
        ms = f"{m:8.2f}%" if m else "        -"
        print(f"{lab:18s} {beta:7.4f} {rk:6.3f} | {d:9s} {r:8.4f} "
              f"{ceil:7.2f}% {ms} {eta}")

# --- placement proxy vs routing proxy -----------------------------------
# Placement moves whole experts, so its ratio reaches only the LPT floor;
# perfect routing reaches r=1.  Within this fitted proxy, both map to the same
# value when the LPT floor lies below the same-path empirical threshold.
LPT = {"Qwen2-57B  EP=8": 1.0100, "Qwen2-57B  EP=4": 1.0039,
       "Qwen3-235B EP=8": 1.0003, "Qwen3-235B EP=8 A100-L256": 1.0003,
       "Qwen3-30B  EP=4": 1.0001}
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
            label=f"{lab}  ($\\beta$={beta:.3f}, $\\hat r_k$={rk:.3f})")
    for d, r in DS[lab].items():
        c = 100 * beta * max(0.0, r - rk)
        ax.plot([r], [c], "o", color=col, ms=5, mfc="white", mew=1.6)
        ax.annotate(d, (r, c), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color=col)
        m = MEAS.get((lab, d))
        if m is not None:
            ax.plot([r], [m], "^", color=col, ms=6)
            ax.plot([r, r], [m, c], color=col, lw=0.9, ls=":", alpha=0.7)
for lab, T, B, rk, col in CFG:
    ax.axvline(rk, color=col, lw=0.7, ls="--", alpha=0.35)
ax.set_xlabel(r"dataset imbalance $r_{\rm before}$  (offline from routing counts)")
ax.set_ylabel("same-path headroom proxy (%)")
ax.set_title(r"Calibrated ratio proxy  $\widehat{\Delta}^{(r)}=\beta\max(0,\,r-\widehat r_k)$"
             "\n(circle = proxy, triangle = measured result, dotted = residual)")
ax.legend(fontsize=8, loc="upper left")
ax.axhline(0.0, color="0.35", lw=0.8, alpha=0.7)
ax.grid(alpha=0.25)
ax.set_xlim(1.0, 1.8); ax.set_ylim(-4, 26)
fig.tight_layout()
out = "/workspace/EPLB/OEPLB/fig_bound.png"
fig.savefig(out, dpi=170)
print("\nsaved", out)
