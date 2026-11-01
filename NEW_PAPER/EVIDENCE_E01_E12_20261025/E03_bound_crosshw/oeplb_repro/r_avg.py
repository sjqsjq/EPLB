"""r_avg for the identity placement, and the per-layer LPT-balanced floor.

Usage: python3 r_avg.py counts.json [ep ...]

r_avg = mean_l ( max_g L[l,g] / mean_g L[l,g] ), the same quantity PB-OEPLB-DIAG
reports as avg_ratio_before/after.  The LPT floor is what a per-layer balancer
could reach at best, so x = (r_id - r_lpt)/r_id is the dataset's own headroom
instead of a value borrowed from another workload.
"""
import json, sys

d = json.load(open(sys.argv[1]))
C = d["counts"]                      # [num_layers][num_experts]
L, E = d["num_layers"], d["num_experts"]
eps = [int(a) for a in sys.argv[2:]] or [8, 4]


def ratio(c, assign, ep):
    g = [0.0] * ep
    for e in range(E):
        g[assign[e]] += c[e]
    tot = sum(g)
    return max(g) / (tot / ep) if tot > 0 else 1.0


def lpt(c, ep):
    per = E // ep
    load = [0.0] * ep
    cnt = [0] * ep
    a = [0] * E
    for e in sorted(range(E), key=lambda e: -c[e]):
        g = min((i for i in range(ep) if cnt[i] < per), key=lambda i: load[i])
        a[e] = g
        load[g] += c[e]
        cnt[g] += 1
    return a


print(f"{sys.argv[1]}  L={L} E={E}  total_tokens={sum(sum(r) for r in C):,.0f}")
for ep in eps:
    per = E // ep
    ident = [e // per for e in range(E)]
    ri = sum(ratio(C[l], ident, ep) for l in range(L)) / L
    rl = sum(ratio(C[l], lpt(C[l], ep), ep) for l in range(L)) / L
    print(f"  ep={ep}: r_identity={ri:.4f}  r_lpt_floor={rl:.4f}  "
          f"x=(r_id-r_lpt)/r_id={(ri - rl) / ri:.4f}")
