"""Placement ceiling vs routing ceiling.

Placement can only move whole experts, so the hottest single expert sets a floor
    r_place >= mean_l [ max_e c_e / (sum_e c_e / G) ]
Perfect routing has no such floor: r = 1.

But T(r) is flat for r <= r_k, so if the LPT floor already sits inside the dead
zone the two ceilings COINCIDE -- placement is provably enough and redundant
experts buy nothing.  This script finds where that stops being true.
"""
import json, sys

def load(p):
    d = json.load(open(p)); return d["counts"], d["num_layers"], d["num_experts"]

def lpt(c, ep, E):
    per = E // ep; load = [0.0]*ep; cnt = [0]*ep
    for e in sorted(range(E), key=lambda e: -c[e]):
        g = min((i for i in range(ep) if cnt[i] < per), key=lambda i: load[i])
        load[g] += c[e]; cnt[g] += 1
    return max(load)/(sum(c)/ep) if sum(c) > 0 else 1.0

def hot(c, ep):
    return max(c)/(sum(c)/ep) if sum(c) > 0 else 1.0

for path, name, RK in [("counts57b.json", "Qwen2-57B", {8:1.099, 4:1.032}),
                       ("counts235b.json", "Qwen3-235B", {8:1.093})]:
    try: C, L, E = load(path)
    except Exception as e: print(f"{path}: {e}"); continue
    print(f"\n=== {name}  ({L} layers, {E} experts) ===")
    print(f"{'EP':>4} {'exp/GPU':>8} {'r_ident':>8} {'r_LPT':>7} {'r_hot_floor':>12} "
          f"{'r_k':>7} {'LPT in dead zone?':>18}")
    ep = 2
    while ep <= E:
        if E % ep: ep *= 2; continue
        ri = sum(max(sum(C[l][e] for e in range(E) if e//(E//ep) == g) for g in range(ep))
                 / (sum(C[l])/ep) for l in range(L)) / L
        rl = sum(lpt(C[l], ep, E) for l in range(L)) / L
        rh = sum(hot(C[l], ep) for l in range(L)) / L
        rk = RK.get(ep)
        verdict = ("yes -> placement == routing" if rk and rl <= rk else
                   "NO -> routing/redundancy strictly better" if rk else
                   "r_k unmeasured")
        print(f"{ep:>4} {E//ep:>8} {ri:8.4f} {rl:7.4f} {rh:12.4f} "
              f"{(f'{rk:.3f}' if rk else '  -'):>7} {verdict:>18}")
        ep *= 2
