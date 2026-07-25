#!/usr/bin/env python3
"""Analyze the sync_window × input_length grid experiment results.

Produces:
1. Full req/s matrix table (markdown)
2. vs-baseline % table
3. Best window per input length
4. Hypothesis A: avg_ratio_before from DIAG logs
5. Hypothesis B: overhead-per-token from PROF logs
6. Matplotlib trend plots (saved as PNG)
"""
import json, glob, re, os, sys
import numpy as np

RESULT_DIR = "/workspace/EPLB/OEPLB/benchmarks/results"
LOG_DIR = "/tmp"
OUT_DIR = "/workspace/EPLB/OEPLB/benchmarks"

LENGTHS = [256, 512, 1024, 2048, 4096]
WINDOWS = [16, 32, 64, 128]
REPS = [1, 2]

# ============================================================
# Part 1: Load benchmark results
# ============================================================
def load_result(label):
    path = f"{RESULT_DIR}/{label}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

print("=" * 70)
print("  SYNC_WINDOW × INPUT_LENGTH GRID ANALYSIS")
print("=" * 70)

# Baselines
bl = {}
for L in LENGTHS:
    vals = []
    for r in REPS:
        res = load_result(f"gridL{L}_bl_r{r}")
        if res:
            vals.append(res["tps"])
    if vals:
        bl[L] = {"mean": np.mean(vals), "std": np.std(vals), "vals": vals}

print("\n### Baseline req/s")
print("| Length | R1 | R2 | Mean | Std |")
print("|--------|------|------|------|-----|")
for L in LENGTHS:
    if L in bl:
        v = bl[L]
        r1 = f"{v['vals'][0]:.1f}" if len(v['vals']) > 0 else "-"
        r2 = f"{v['vals'][1]:.1f}" if len(v['vals']) > 1 else "-"
        print(f"| {L} | {r1} | {r2} | {v['mean']:.2f} | {v['std']:.2f} |")

# OEPLB results
oe = {}
for L in LENGTHS:
    for SW in WINDOWS:
        vals = []
        for r in REPS:
            res = load_result(f"gridL{L}_sw{SW}_r{r}")
            if res:
                vals.append(res["tps"])
        if vals:
            oe[(L, SW)] = {"mean": np.mean(vals), "std": np.std(vals), "vals": vals}

# Full matrix: req/s
print("\n### OEPLB req/s (mean of 2 runs)")
header = "| Length |" + "|".join(f" sw={sw} " for sw in WINDOWS) + "| Best sw |"
sep = "|--------|" + "|".join("-------" for _ in WINDOWS) + "|---------|"
print(header)
print(sep)
best_windows = {}
for L in LENGTHS:
    row = f"| {L} |"
    best_sw, best_tps = None, -1
    for SW in WINDOWS:
        key = (L, SW)
        if key in oe:
            v = oe[key]["mean"]
            row += f" {v:.2f} |"
            if v > best_tps:
                best_tps = v
                best_sw = SW
        else:
            row += " - |"
    row += f" **{best_sw}** |" if best_sw else " - |"
    best_windows[L] = best_sw
    print(row)

# vs baseline %
print("\n### vs Baseline %")
header = "| Length | BL |" + "|".join(f" sw={sw} " for sw in WINDOWS) + "|"
sep = "|--------|------|" + "|".join("-------" for _ in WINDOWS) + "|"
print(header)
print(sep)
for L in LENGTHS:
    if L not in bl:
        continue
    b = bl[L]["mean"]
    row = f"| {L} | {b:.1f} |"
    for SW in WINDOWS:
        key = (L, SW)
        if key in oe:
            pct = (oe[key]["mean"] - b) / b * 100
            row += f" {pct:+.1f}% |"
        else:
            row += " - |"
    print(row)

# Detailed per-run table
print("\n### Detailed per-run req/s")
header = "| Length | SW | R1 | R2 | Mean | Std | vs BL |"
sep = "|--------|-----|------|------|------|-----|-------|"
print(header)
print(sep)
for L in LENGTHS:
    for SW in WINDOWS:
        key = (L, SW)
        if key not in oe:
            continue
        v = oe[key]
        b = bl.get(L, {}).get("mean", 0)
        pct = (v["mean"] - b) / b * 100 if b > 0 else 0
        r1 = f"{v['vals'][0]:.1f}" if len(v['vals']) > 0 else "-"
        r2 = f"{v['vals'][1]:.1f}" if len(v['vals']) > 1 else "-"
        print(f"| {L} | {SW} | {r1} | {r2} | {v['mean']:.2f} | {v['std']:.2f} | {pct:+.1f}% |")

# Best window summary
print("\n### Best sync_window per input length")
print("| Length | Best SW | req/s | vs BL |")
print("|--------|---------|-------|-------|")
for L in LENGTHS:
    sw = best_windows.get(L)
    if sw and (L, sw) in oe and L in bl:
        v = oe[(L, sw)]["mean"]
        b = bl[L]["mean"]
        pct = (v - b) / b * 100
        print(f"| {L} | {sw} | {v:.2f} | {pct:+.1f}% |")

# ============================================================
# Part 2: Hypothesis A — avg_ratio_before from DIAG logs
# ============================================================
print("\n" + "=" * 70)
print("  HYPOTHESIS A: Imbalance magnitude (avg_ratio_before)")
print("=" * 70)

def extract_diag(label):
    logpath = f"{LOG_DIR}/{label}.log"
    if not os.path.exists(logpath):
        return []
    ratios = []
    with open(logpath, errors="ignore") as f:
        for line in f:
            if "PB-OEPLB-DIAG" not in line or "DP0 " not in line:
                continue
            m = re.search(r"avg_ratio_before=([\d.]+)", line)
            if m:
                ratios.append(float(m.group(1)))
    return ratios

print("\n| Length | SW | Window1 ratio | Steady-state ratio (w2-end avg) |")
print("|--------|-----|---------------|--------------------------------|")
for L in LENGTHS:
    for SW in WINDOWS:
        label = f"gridL{L}_sw{SW}_r1"
        ratios = extract_diag(label)
        if not ratios:
            # Try old naming
            if SW in [32, 64] and L in [256, 2048]:
                label = f"tok_oeplb{SW}_r1_tok{L}"
                ratios = extract_diag(label)
        if ratios:
            w1 = ratios[0] if ratios else "-"
            steady = np.mean(ratios[1:]) if len(ratios) > 1 else "-"
            print(f"| {L} | {SW} | {w1:.3f} | {steady:.3f} |" if isinstance(steady, float) else
                  f"| {L} | {SW} | {w1:.3f} | {steady} |")

# ============================================================
# Part 3: Hypothesis B — overhead per token from PROF logs
# ============================================================
print("\n" + "=" * 70)
print("  HYPOTHESIS B: Fixed overhead / tokens_protected ratio")
print("=" * 70)

def extract_prof(label):
    logpath = f"{LOG_DIR}/{label}.log"
    if not os.path.exists(logpath):
        return []
    entries = []
    with open(logpath, errors="ignore") as f:
        for line in f:
            if "PB-OEPLB-PROF" not in line or "DP0 " not in line:
                continue
            m = re.search(
                r"window#(\d+)\s+calls=(\d+)\s+"
                r"record=([\d.]+)ms\s+allreduce=([\d.]+)ms\s+"
                r"planbuild=([\d.]+)ms\s+finalize=([\d.]+)ms",
                line
            )
            if m:
                entries.append({
                    "window": int(m.group(1)),
                    "calls": int(m.group(2)),
                    "record_ms": float(m.group(3)),
                    "allreduce_ms": float(m.group(4)),
                    "planbuild_ms": float(m.group(5)),
                    "finalize_ms": float(m.group(6)),
                })
    return entries

def extract_tok_global(label):
    logpath = f"{LOG_DIR}/{label}.log"
    if not os.path.exists(logpath):
        return []
    tokens = []
    with open(logpath, errors="ignore") as f:
        for line in f:
            m = re.search(r"window#(\d+):.*\((\d+) tok global", line)
            if m and "DP0 " in line:
                tokens.append({"window": int(m.group(1)), "tok": int(m.group(2))})
    return tokens

print("\n| Length | SW | allreduce_ms(total) | windows | ar_per_window | tok_per_window(est) | ar_ms/Mtok |")
print("|--------|-----|--------------------|---------|--------------|--------------------|-----------|")
for L in LENGTHS:
    for SW in WINDOWS:
        label = f"gridL{L}_sw{SW}_r1"
        profs = extract_prof(label)
        if not profs:
            if SW in [32, 64] and L in [256, 2048]:
                label = f"tok_oeplb{SW}_r1_tok{L}"
                profs = extract_prof(label)
        if not profs:
            continue
        last = profs[-1]
        n_win = last["window"]
        ar_total = last["allreduce_ms"]
        ar_per = ar_total / max(n_win, 1)

        toks = extract_tok_global(label)
        if toks:
            avg_tok = np.mean([t["tok"] for t in toks])
        else:
            avg_tok = L * 2048 * 8  # rough estimate: L tokens * N_requests/windows * dp_size

        ar_per_mtok = ar_per / (avg_tok / 1e6) if avg_tok > 0 else 0
        print(f"| {L} | {SW} | {ar_total:.1f} | {n_win} | {ar_per:.2f} | {avg_tok:.0f} | {ar_per_mtok:.2f} |")

# ============================================================
# Part 4: Plot
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: req/s vs input length, one line per window
    ax = axes[0]
    for SW in WINDOWS:
        xs, ys = [], []
        for L in LENGTHS:
            if (L, SW) in oe:
                xs.append(L)
                ys.append(oe[(L, SW)]["mean"])
        if xs:
            ax.plot(xs, ys, "o-", label=f"sw={SW}")
    bx = [L for L in LENGTHS if L in bl]
    by = [bl[L]["mean"] for L in bx]
    ax.plot(bx, by, "ks--", label="Baseline")
    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("req/s")
    ax.set_title("Throughput vs Input Length")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: vs baseline % — one line per window
    ax = axes[1]
    for SW in WINDOWS:
        xs, ys = [], []
        for L in LENGTHS:
            if (L, SW) in oe and L in bl:
                xs.append(L)
                pct = (oe[(L, SW)]["mean"] - bl[L]["mean"]) / bl[L]["mean"] * 100
                ys.append(pct)
        if xs:
            ax.plot(xs, ys, "o-", label=f"sw={SW}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("vs Baseline (%)")
    ax.set_title("OEPLB Gain vs Input Length")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Best window per input length
    ax = axes[2]
    bw_x = [L for L in LENGTHS if L in best_windows and best_windows[L] is not None]
    bw_y = [best_windows[L] for L in bw_x]
    ax.plot(bw_x, bw_y, "ro-", markersize=10, linewidth=2)
    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("Best sync_window")
    ax.set_title("Optimal Window vs Input Length")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_yticks(WINDOWS)
    ax.set_yticklabels([str(w) for w in WINDOWS])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpng = f"{OUT_DIR}/length_grid_analysis.png"
    plt.savefig(outpng, dpi=150)
    print(f"\nPlot saved: {outpng}")
except ImportError:
    print("\nmatplotlib not available, skipping plots")
except Exception as e:
    print(f"\nPlot error: {e}")

print("\nDONE")
