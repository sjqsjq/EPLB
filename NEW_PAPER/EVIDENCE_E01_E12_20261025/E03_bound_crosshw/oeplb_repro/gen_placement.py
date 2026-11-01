"""Generate physical_to_logical_map JSONs whose *predicted* per-layer average
imbalance ratio r_avg = mean_l ( max_g L[l,g] / mean_g L[l,g] ) hits a set of
targets, using recorded per-layer per-expert token counts.

Why the per-layer average and not the aggregate ratio: every MoE layer is its
own dispatch/GEMM/combine barrier, so the stall accumulates layer by layer.
This is also exactly the quantity PB-OEPLB-DIAG reports as avg_ratio_before/
after, so the swept x-axis is directly comparable to the paper's measured r.

Why the map must be per layer: one permutation shared by all layers can neither
create nor remove per-layer imbalance -- each layer routes to different logical
experts, so a single placement averages out to r_agg ~ 1.05 however it is
arranged.  Each layer therefore gets its own permutation, all of them piling
their hot experts onto GPU 0 so the effect does not cancel across layers.

No redundant experts: every layer's map is a permutation of the logical ids.

Usage: gen_placement.py counts.json ep_size out_prefix r1 r2 ...
"""
import json, sys

counts_f, ep, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
targets = [float(x) for x in sys.argv[4:]]
D = json.load(open(counts_f))
C = D["counts"]; L = D["num_layers"]; E = D["num_experts"]
per = E // ep
assert per * ep == E


def ratio(assign, c):
    """assign[logical] = gpu  ->  (r, per-GPU load) for one layer"""
    load = [0.0] * ep
    for e, g in enumerate(assign):
        load[g] += c[e]
    m = sum(load) / ep
    return (max(load) / m if m > 0 else 1.0), load


def lpt(cands, c, gpus):
    """LPT greedy: pack cands onto the given gpus, `per` experts each."""
    load = {g: 0.0 for g in gpus}
    cnt = {g: 0 for g in gpus}
    out = {}
    for e in sorted(cands, key=lambda e: -c[e]):
        g = min((g for g in gpus if cnt[g] < per), key=lambda g: load[g])
        out[e] = g; load[g] += c[e]; cnt[g] += 1
    return out


def balanced(c):
    a = lpt(range(E), c, list(range(ep)))
    return [a[e] for e in range(E)]


def concentrated(c):
    """Max r: the `per` hottest experts all land on GPU 0."""
    order = sorted(range(E), key=lambda e: -c[e])
    assign = [0] * E
    for e in order[:per]:
        assign[e] = 0
    for e, g in lpt(order[per:], c, list(range(1, ep))).items():
        assign[e] = g
    return assign


def toward(c, target):
    """Deficit-greedy: aim for the load vector GPU0 = target*mean and every
    other GPU = (ep-target)/(ep-1)*mean (so the max is GPU0 and the target is
    hit exactly if the expert granularity allows).  Experts are placed hot to
    cold, each onto the slotted GPU with the largest remaining deficit."""
    total = sum(c)
    if total <= 0:
        return balanced(c)
    mean = total / ep
    want = [target * mean] + [(ep - target) / (ep - 1) * mean] * (ep - 1)
    load = [0.0] * ep
    cnt = [0] * ep
    assign = [0] * E
    for e in sorted(range(E), key=lambda e: -c[e]):
        g = max((i for i in range(ep) if cnt[i] < per),
                key=lambda i: want[i] - load[i])
        assign[e] = g; load[g] += c[e]; cnt[g] += 1
    return assign


def to_p2l(assign):
    """assign[logical]=gpu  ->  physical_to_logical_map[phys_slot]=logical"""
    buckets = [[] for _ in range(ep)]
    for e, g in enumerate(assign):
        buckets[g].append(e)
    out = []
    for g in range(ep):
        assert len(buckets[g]) == per, (g, len(buckets[g]))
        out += buckets[g]
    return out


def report(name, pla, write=True):
    rs, agg = [], [0.0] * ep
    for l in range(L):
        r, load = ratio(pla[l], C[l])
        rs.append(r)
        for g, v in enumerate(load):
            agg[g] += v
    r_avg = sum(rs) / L
    r_agg = max(agg) / (sum(agg) / ep)
    fn = "(not written)"
    if write:
        fn = f"{prefix}_{name}.json"
        json.dump({"physical_to_logical_map": [to_p2l(pla[l]) for l in range(L)]},
                  open(fn, "w"))
    print(f"{name:>8}  r_avg={r_avg:.3f}  r_agg={r_agg:.3f}  "
          f"per-layer max={max(rs):.3f} min={min(rs):.3f}  -> {fn}")
    return r_avg


print(f"layers={L} experts={E} ep={ep} per_gpu={per}")
report("identity", [[e // per for e in range(E)] for _ in range(L)], write=False)
report("bal", [balanced(C[l]) for l in range(L)])
for t in targets:
    report(("r%.2f" % t).replace(".", ""), [toward(C[l], t) for l in range(L)])
report("conc", [concentrated(C[l]) for l in range(L)])
