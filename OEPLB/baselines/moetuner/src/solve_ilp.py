#!/usr/bin/env python3
"""MoETuner ILP1 (per-layer balanced clustering) + greedy cluster->GPU.

Input:  profile .npz with P[L,E] (and R[L-1,E,E], unused for single-node).
Output: placement.json in SGLang init_expert_location format.

Usage:
  python solve_ilp.py --profile artifacts/profile.npz \
                      --num-clusters 8 \
                      --experts-per-cluster 16 \
                      --out artifacts/placement.json
"""
import argparse, json, os, sys, time, logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("moetuner-ilp1")

def solve_layer(P_l: np.ndarray, G: int, capacity: int, time_limit: float, mip_gap: float):
    """Solve ILP1 for one layer.
    Variables: x[c,e] in {0,1}, cluster assignment.
    Constraints: (1) each expert to exactly one cluster; (2) cluster size = capacity.
    Objective: minimize sum_c |T_c - T_bar|, linearized as sum_c d_c where d_c >= |T_c - T_bar|.
    Returns: assignment array E-long -> cluster id.
    """
    E = P_l.shape[0]
    T_bar = P_l.sum() / G
    m = gp.Model()
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)
    x = m.addVars(G, E, vtype=GRB.BINARY, name="x")
    d = m.addVars(G, lb=0, name="d")
    for e in range(E):
        m.addConstr(gp.quicksum(x[c, e] for c in range(G)) == 1, name=f"one_c_e{e}")
    for c in range(G):
        m.addConstr(gp.quicksum(x[c, e] for e in range(E)) == capacity, name=f"cap_c{c}")
        T_c = gp.quicksum(P_l[e] * x[c, e] for e in range(E))
        m.addConstr(d[c] >= T_c - T_bar, name=f"abs_pos_c{c}")
        m.addConstr(d[c] >= T_bar - T_c, name=f"abs_neg_c{c}")
    m.setObjective(gp.quicksum(d[c] for c in range(G)), GRB.MINIMIZE)
    t0 = time.time()
    m.optimize()
    dt = time.time() - t0
    if m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"solver status={m.status}")
    assignment = np.full(E, -1, dtype=np.int32)
    for c in range(G):
        for e in range(E):
            if x[c, e].X > 0.5:
                assignment[e] = c
    assert (assignment != -1).all(), "expert unassigned"
    return assignment, m.objVal, dt

def greedy_cluster_to_gpu(assignment: np.ndarray, P_l: np.ndarray, G: int):
    """Assign clusters to GPUs: largest total token count to lowest-loaded GPU.
    Simple offline balanced permutation of {0..G-1}.
    Returns: cluster_to_gpu[G].
    """
    cluster_tokens = np.zeros(G, dtype=np.int64)
    for e in range(len(assignment)):
        cluster_tokens[assignment[e]] += P_l[e]
    order = np.argsort(-cluster_tokens)  # heaviest first
    gpu_load = np.zeros(G, dtype=np.int64)
    cluster_to_gpu = np.full(G, -1, dtype=np.int32)
    for c in order:
        g = int(np.argmin(gpu_load))
        cluster_to_gpu[c] = g
        gpu_load[g] += cluster_tokens[c]
    return cluster_to_gpu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--num-clusters", "-G", type=int, default=8)
    ap.add_argument("--experts-per-cluster", "-K", type=int, default=None,
                    help="Default = E / G")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--mip-gap", type=float, default=0.005)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer ids")
    args = ap.parse_args()

    d = np.load(args.profile)
    P = d["P"]
    if "num_forwards" in d.files:
        log.info(f"profile: fwds={int(d['num_forwards'])}, P.shape={P.shape}, total_tokens_per_layer_mean={P.sum(axis=1).mean():.0f}")
    L, E = P.shape
    G = args.num_clusters
    K = args.experts_per_cluster or (E // G)
    assert G * K == E, f"G*K={G*K} != E={E}"

    if args.layers == "all":
        layer_ids = list(range(L))
    else:
        layer_ids = [int(x) for x in args.layers.split(",")]

    # Physical layout: G * K positions.  On EP=8 with 128 experts, 16 experts/gpu.
    # SGLang init_expert_location: dict per layer, mapping physical_id -> logical_id.
    # A simpler form used by SGLang eplb: 'physical_to_logical_map' [L, num_physical_experts].
    physical_to_logical = np.zeros((L, E), dtype=np.int32)
    per_layer_stats = []
    total_t = 0.0
    for l in layer_ids:
        Pl = P[l].astype(np.float64)
        if Pl.sum() == 0:
            log.warning(f"layer {l}: no tokens observed, defaulting to identity")
            physical_to_logical[l] = np.arange(E, dtype=np.int32)
            continue
        assignment, obj, dt = solve_layer(Pl, G, K, args.time_limit, args.mip_gap)
        cluster_to_gpu = greedy_cluster_to_gpu(assignment, Pl, G)
        # Build physical->logical mapping.
        # Physical slots: GPU g holds K contiguous positions g*K .. g*K + K - 1.
        # Fill each GPU's slots with the experts assigned to its cluster,
        # ordered by descending token load.
        for c in range(G):
            g = int(cluster_to_gpu[c])
            experts_in_c = np.where(assignment == c)[0]
            experts_in_c = experts_in_c[np.argsort(-Pl[experts_in_c])]
            for k, e in enumerate(experts_in_c):
                physical_to_logical[l, g * K + k] = int(e)
        # Layer stats.
        gpu_loads = np.zeros(G)
        for e in range(E):
            gpu_loads[cluster_to_gpu[assignment[e]]] += Pl[e]
        imbalance = gpu_loads.max() / (gpu_loads.sum() / G)
        per_layer_stats.append({"layer": l, "obj": float(obj), "gpu_max_load": float(gpu_loads.max()), "gpu_mean_load": float(gpu_loads.mean()), "imbalance_ratio": float(imbalance), "solve_s": dt})
        total_t += dt
        if l % 10 == 0:
            log.info(f"layer {l}: obj={obj:.0f} imbalance={imbalance:.3f} solve={dt:.2f}s")

    log.info(f"Solved {len(layer_ids)} layers in total {total_t:.1f}s. mean_imbalance={np.mean([s['imbalance_ratio'] for s in per_layer_stats]):.3f}")

    # SGLang init_by_mapping ONLY accepts physical_to_logical_map kwarg
    # Sidecar meta goes to <out>.meta.json
    out = {"physical_to_logical_map": physical_to_logical.tolist()}
    meta = {
        "num_layers": L,
        "num_physical_experts": E,
        "num_logical_experts": E,
        "num_gpus": G,
        "experts_per_gpu": K,
        "solver": "moetuner_ilp1_greedy",
        "profile_source": os.path.abspath(args.profile),
        "per_layer_stats": per_layer_stats,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    meta_path = args.out.replace(".json", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"wrote {args.out} + {meta_path}")

if __name__ == "__main__":
    main()
