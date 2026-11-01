import json, os, sys, hashlib
import numpy as np
import torch
HP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results"
G2 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/planner_matrix_c1c4_20261031"
T1 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/replay_routing_t1_20261101"
sys.path.insert(0, os.path.join(G2, "scripts"))
import driver_g2 as D

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

assert sha(D.FROZEN).startswith("c250026b"), sha(D.FROZEN)[:16]
CAP = os.path.join(HP, "cross_domain_e2e_20261027", "prep", "record_boot2", "expert_distribution_recorder_1793078907.6631618.pt")
obj = torch.load(CAP, map_location="cpu", weights_only=False)
lc = obj["logical_count"]
assert tuple(lc.shape) == (1000, 94, 128), tuple(lc.shape)
agg = lc.to(torch.int64).sum(dim=0).numpy().astype(np.int64)
assert agg.shape == (94, 128)
PE3 = os.path.join(HP, "cross_domain_e2e_20261027", "prep", "profile_build", "P_e3.npz")
P = np.load(PE3)["P"].astype(np.int64)
assert np.array_equal(agg, P), "CAPTURE_AGG_NOT_EQUAL_P_E3"
identity_ok = "capture_aggregate_equals_P_e3_elementwise"

conds = [("ORIG", None),
         ("GAP32", "C2_gap_32_prod1096"),
         ("GAP128", "C2_gap_128_prod1096"),
         ("GAP300", "C2_gap_300_prod1096"),
         ("MD300", "C2_maxdelta_300_prod1096"),
         ("BI300", "C2_gap_300_prod1096_BI")]
init_raw = D.ratio_stats(agg, pad=False)
init_pad = D.ratio_stats(agg, pad=True)
expert_edges = 0
for l in range(94):
    expert_edges += int(np.count_nonzero(agg[l]))
rows = []
out = {"identity_fact": identity_ok, "capture_sum": int(agg.sum()), "init_ratio_raw": init_raw, "init_ratio_padded": init_pad, "expert_edges_layout_invariant": expert_edges, "conditions": {}}
for name, cell in conds:
    if cell is None:
        plan = []
    else:
        plan = json.load(open(os.path.join(G2, "analysis", "plans", cell + ".json")))
    fin = D.apply_plan(agg, [tuple(x) for x in plan])
    p2l = np.arange(128, dtype=np.int64).repeat(94).reshape(94, 128).copy()
    for op in plan:
        layer, a, b = int(op[0]), int(op[1]), int(op[2])
        p2l[layer, a], p2l[layer, b] = p2l[layer, b], p2l[layer, a]
    fin_raw = D.ratio_stats(fin, pad=False)
    fin_pad = D.ratio_stats(fin, pad=True)
    padtok = 0
    occupied = 0
    for l in range(94):
        raw_l = D.rank_loads_np(fin[l], pad=False)
        pad_l = D.rank_loads_np(fin[l], pad=True)
        padtok += int((pad_l - raw_l).sum())
        occupied += len({p // 16 for p in range(128) if fin[l][p] > 0})
    plan_len = len(plan)
    rec = {"source_cell": cell, "source_plan_sha256": (sha(os.path.join(G2, "analysis", "plans", cell + ".json")) if cell else None),
           "plan_len": plan_len,
           "final_ratio_raw": fin_raw, "final_ratio_padded": fin_pad,
           "imbalance_reduction_raw_max": round(init_raw["max"] - fin_raw["max"], 10),
           "padding_tokens_total": padtok,
           "occupied_rank_slots_sum_layers": occupied,
           "rank_pair_edges_x8": 8 * occupied,
           "expert_edges_layout_invariant": expert_edges,
           "moved_mb_estimate_only": round(plan_len * D.MOVED_MB_PER_OP, 1),
           "online_migration_wall_s": None, "online_exec_rep1_s": None, "online_exec_rep2_s": None,
           "benefit_model_form": "conditional paper-analysis model: imbalance reduction conditioned on measured migration cost (ops x 37.74MB estimate + online wall); NOT an online algorithm; NO payback solver; h_rate/h_w NOT headroom"}
    out["conditions"][name] = rec
    json.dump({"condition": name, "physical_to_logical_map": p2l.tolist()}, open(os.path.join(T1, "analysis", "layouts", "layout_" + name + ".json"), "w"))
    rows.append([name, cell or "identity", plan_len, round(init_raw["max"], 6), round(fin_raw["max"], 6), round(init_pad["max"], 6), round(fin_pad["max"], 6), padtok, occupied, 8 * occupied, expert_edges, rec["moved_mb_estimate_only"]])
json.dump(out, open(os.path.join(T1, "analysis", "OFFLINE_t1.json"), "w"), indent=1)
hdr = ["condition", "source_cell", "plan_len", "init_max_raw", "fin_max_raw", "init_max_pad", "fin_max_pad", "padding_tokens", "occupied_rank_slots", "rank_pair_edges_x8", "expert_edges", "moved_mb_est"]
with open(os.path.join(T1, "analysis", "EXPORT_t1_offline_table.tsv"), "w") as f:
    f.write("\t".join(hdr) + "\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print("OFFLINE_T1_OK", identity_ok)
for r in rows:
    print(r)
