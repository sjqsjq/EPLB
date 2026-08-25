"""
DataFore (ISCA 2026) Case Study 2 -- Prefill-Guided Expert Placement.
Faithful implementation of Algorithm 2 from the paper (Sec VI).

  remap_based_placement(D, G): sort experts by prefill freq, greedily assign
    to least-loaded GPU; keeps E/G experts per GPU (no extra slots). A permutation.
  dup_based_placement(D, G, R): default contiguous + duplicate R hottest experts
    onto least-loaded GPUs (R extra slots/GPU -> E+R*G total physical).

Input  D: prefill expert frequency, shape (num_layers, num_experts)
Output: per-layer physical_to_logical_map consumable by SGLang's
        --init-expert-location (identity EP layout: physical slot -> logical expert).

Reference: DataFore Fig.16 + Algorithm 2.
  Default=contiguous  Remap(Ours)=greedy least-loaded  Dup(Ours)=default+dup
  Best/Worst=oracle from DECODE-stage selections
"""
import json
import numpy as np


def gpu_loads_for_layer(freq_layer, placement, G, num_local):
    """placement: list mapping physical_slot -> logical_expert. Physical slot s
    lives on GPU (s // num_local). GPU load = sum freq[logical] over its slots."""
    loads = np.zeros(G)
    for slot, logical in enumerate(placement):
        loads[slot // num_local] += freq_layer[logical]
    return loads


def imbalance_ratio(loads):
    mn = loads.min()
    return float(loads.max() / mn) if mn > 0 else float("inf")


def remap_based_placement(D, G):
    """Per-layer greedy: sort experts by prefill freq desc, assign each to the
    least-loaded GPU that still has room (E/G per GPU). Returns per-layer map."""
    num_layers, E = D.shape
    num_local = E // G
    maps = []
    for l in range(num_layers):
        f = D[l]
        order = np.argsort(-f)
        gpu_load = np.zeros(G)
        gpu_count = np.zeros(G, dtype=int)
        exp_to_gpu = np.full(E, -1)
        for e in order:
            cand = [g for g in range(G) if gpu_count[g] < num_local]
            g_star = min(cand, key=lambda g: (gpu_load[g], g))
            exp_to_gpu[e] = g_star
            gpu_load[g_star] += f[e]
            gpu_count[g_star] += 1
        placement = [0] * E
        for g in range(G):
            slots = list(range(g * num_local, (g + 1) * num_local))
            experts_here = [e for e in range(E) if exp_to_gpu[e] == g]
            for s, e in zip(slots, experts_here):
                placement[s] = int(e)
        maps.append(placement)
    return maps


def dup_based_placement(D, G, R=1):
    """Default contiguous + duplicate R hottest experts per GPU onto least-loaded
    GPUs. total slots = E + R*G. Returns (per-layer map, redundant_count=R*G)."""
    num_layers, E = D.shape
    num_local = E // G
    maps = []
    for l in range(num_layers):
        f = D[l]
        placement = list(range(E)) + [-1] * (R * G)
        gpu_load = gpu_loads_for_layer(f, list(range(E)), G, num_local).copy()
        gpu_remaining = np.full(G, R, dtype=int)
        order = np.argsort(-f)
        for _ in range(R * G):
            best = None
            for e in order:
                if f[e] == 0:
                    continue
                for g in range(G):
                    if gpu_remaining[g] <= 0:
                        continue
                    host_gpus = set(s // num_local for s in range(E) if placement[s] == e)
                    if g in host_gpus:
                        continue
                    delta = -gpu_load[g]   # prefer least-loaded (paper argmin cost)
                    if best is None or delta < best[0]:
                        best = (delta, int(e), g)
                break
            if best is None:
                break
            _, e, g = best
            placement[placement.index(-1)] = e
            gpu_load[g] += f[e] / 2.0
            gpu_remaining[g] -= 1
        maps.append(placement)
    return maps, R * G


def oracle_best_placement(D_decode, G):
    return remap_based_placement(D_decode, G)


def oracle_worst_placement(D_decode, G):
    num_layers, E = D_decode.shape
    num_local = E // G
    maps = []
    for l in range(num_layers):
        order = np.argsort(-D_decode[l])
        placement = [0] * E
        oi = 0
        for g in range(G):
            for s in range(num_local):
                placement[g * num_local + s] = int(order[oi]); oi += 1
        maps.append(placement)
    return maps


def to_sglang_json(per_layer_maps, num_layers, num_experts_per_layer, G):
    arr = np.array(per_layer_maps, dtype=np.int64)
    return {
        "physical_to_logical_map": arr.tolist(),
        "num_layers": int(num_layers),
        "num_experts_per_layer": int(num_experts_per_layer),
        "ep_size": int(G),
    }


def report(D_prefill, D_decode, G, label=""):
    num_layers, E = D_prefill.shape
    num_local = E // G
    print(f"\n=== Placement report {label} (E={E}, G={G}, {num_local}/GPU) ===")
    placements = {
        "Default(trivial)": [list(range(E)) for _ in range(num_layers)],
        "Remap(prefill)":   remap_based_placement(D_prefill, G),
        "Best(decode-ora)": oracle_best_placement(D_decode, G),
        "Worst(decode)":    oracle_worst_placement(D_decode, G),
    }
    print(f"{'Placement':<18} {'avg max/min':<12} {'med max/min':<12} {'max layer':<10}")
    res = {}
    for name, maps in placements.items():
        ratios = np.array([imbalance_ratio(gpu_loads_for_layer(D_decode[l], maps[l], G, num_local))
                            for l in range(num_layers)])
        print(f"{name:<18} {ratios.mean():<12.3f} {np.median(ratios):<12.3f} {ratios.max():<10.3f}")
        res[name] = maps
    return res


if __name__ == "__main__":
    d = np.load("/tmp/routing_trace_datafore/datafore_ob3_final.npz")
    pf = d["prefill_freq"].astype(np.int64)
    dc = d["decode_freq"].astype(np.int64)
    G = 8
    placements = report(pf, dc, G, label="(ShareGPT trace, measured DECODE imbalance)")

    import os
    out_dir = "/workspace/EPLB/OEPLB/datafore_repro/placements"
    os.makedirs(out_dir, exist_ok=True)
    for name, maps in placements.items():
        fname = name.split("(")[0].strip().lower().replace(" ", "_")
        j = to_sglang_json(maps, 94, 128, G)
        path = f"{out_dir}/{fname}.json"
        with open(path, "w") as f:
            json.dump(j, f)
        print(f"  wrote {path}")
