#!/usr/bin/env python3
"""Aggregate A100 §5.3.1 baseline reproduction per REPRODUCE_BASELINES.md §7.3.
- same-domain (0914): 3 runs, DROP r1, median of r2/r3
- cross-domain (freq6): 3 runs, median of 3
- gain normalized to SAME-SESSION identity
"""
import json, glob, os, statistics as st

RES = "/workspace/EPLB/OEPLB/benchmarks/results"
METHODS = ["identity", "datafore", "moetuner", "eplb_static", "eplb_dyn", "oeplb"]
NAMES = {"identity": "identity", "datafore": "DataForest-Remap", "moetuner": "MoETuner",
         "eplb_static": "EPLB静态", "eplb_dyn": "EPLB动态", "oeplb": "PB-OEPLB(本文)"}

def load(prefix, m):
    runs = []
    for r in (1, 2, 3):
        f = os.path.join(RES, f"_{prefix}_a100_{m}_r{r}.json")
        if os.path.exists(f):
            runs.append(json.load(open(f)))
    return runs

def agg_0914(m):
    runs = load("0914", m)
    if len(runs) < 2: return None, runs
    keep = [x["tps"] for x in runs[1:]]   # drop r1
    return st.median(keep), runs

def agg_freq6(m):
    runs = load("freq6", m)
    if not runs: return None, runs
    return st.median([x["tps"] for x in runs]), runs

print("=" * 72)
print("同域表 (prover_256tok_out1, N=256, 无上限并发, 丢r1取r2/r3中位) [A100]")
print("=" * 72)
sd = {}
for m in METHODS:
    med, runs = agg_0914(m)
    sd[m] = med
    rs = " ".join(f"r{i+1}={x['tps']:.1f}" for i, x in enumerate(runs))
    ok = all(x["ok"] == x["N"] for x in runs) if runs else False
    print(f"{NAMES[m]:<18} median={med if med is None else round(med,2):>8} req/s   [{rs}]  all_ok={ok}")
if sd.get("identity"):
    print(f"\n-- gain vs same-session identity ({sd['identity']:.2f}) --")
    for m in METHODS:
        if sd.get(m): print(f"{NAMES[m]:<18} {(sd[m]/sd['identity']-1)*100:+.1f}%")

print()
print("=" * 72)
print("跨域表 (freq6: book↔prover 6段, N=1800, conc=32, 3-run 中位) [A100]")
print("=" * 72)
fd = {}
for m in METHODS:
    med, runs = agg_freq6(m)
    fd[m] = med
    rs = " ".join(f"r{i+1}={x['tps']:.2f}" for i, x in enumerate(runs))
    ok = all(x["ok"] == x["N"] for x in runs) if runs else False
    print(f"{NAMES[m]:<18} median={med if med is None else round(med,3):>8} req/s   [{rs}]  all_ok={ok}")
if fd.get("identity"):
    print(f"\n-- gain vs same-session identity ({fd['identity']:.3f}) --")
    for m in METHODS:
        if fd.get(m): print(f"{NAMES[m]:<18} {(fd[m]/fd['identity']-1)*100:+.1f}%")

print()
print("=" * 72)
print("相对排序核对 (§10): 同域期望 DataForest ≳ MoETuner > EPLB静态; 跨域期望离线放置 ≤ identity")
print("=" * 72)
for tbl, d in (("同域", sd), ("跨域", fd)):
    vals = sorted(((v, m) for m, v in d.items() if v), reverse=True)
    print(f"{tbl}: " + " > ".join(f"{NAMES[m]}({v:.2f})" for v, m in vals))
