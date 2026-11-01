from sglang.srt.eplb import eplb_algorithms

CB = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/book_prover_static_20261030"
import json, os, hashlib, datetime
import torch

def load_counts(path):
    lc = json.load(open(path))["logical_count"]
    t = torch.tensor(lc, dtype=torch.int64)
    assert tuple(t.shape) == (94, 128), t.shape
    return t

def expected_p2l(counts2d):
    lc = counts2d.unsqueeze(0)
    algo = eplb_algorithms.compute_algorithm(raw_algorithm="auto", num_groups=1, num_nodes=1)
    p2l, l2p, ec = eplb_algorithms.rebalance_experts(
        tokens_per_expert=lc,
        num_physical_experts=128,
        num_local_physical_experts=16,
        num_groups=1,
        num_nodes=1,
        algorithm=algo)
    return algo, p2l, l2p, ec

def rank_ratio(flat, counts128):
    loads = []
    for r in range(8):
        s = 0
        for slot in range(r * 16, (r + 1) * 16):
            s += int(counts128[int(flat[slot])])
        loads.append(s)
    avg = sum(loads) / 8.0
    return (max(loads) / avg if avg > 0 else 0.0), loads

old_c = load_counts(CB + "/placements/book_profile_placement.json")
ref_c = load_counts(CB + "/placements/datafore_prover_placement.json")
prover_w = ref_c.sum(dim=0)
book_w = old_c.sum(dim=0)
print("workload count sums: prover=%d book=%d" % (int(prover_w.sum()), int(book_w.sum())))
summary = {}
for name, c2d in (("OLD", old_c), ("REF", ref_c)):
    pname = "book_profile_placement.json" if name == "OLD" else "datafore_prover_placement.json"
    ppath = CB + "/placements/" + pname
    algo, p2l, l2p, ec = expected_p2l(c2d)
    algo2, p2l2, _, _ = expected_p2l(c2d)
    flat = p2l.flatten().tolist()
    flat2 = p2l2.flatten().tolist()
    assert flat == flat2, "replay NOT deterministic"
    canon = json.dumps(flat, separators=(",", ":"))
    psha = hashlib.sha256(canon.encode()).hexdigest()
    r_own, loads_own = rank_ratio(flat, c2d.sum(dim=0))
    r_prov, loads_prov = rank_ratio(flat, prover_w)
    r_book, loads_book = rank_ratio(flat, book_w)
    summary[name] = {"algo": str(algo), "p2l_sha256": psha, "ratio_under_own_counts": round(r_own, 6), "ratio_under_prover_counts": round(r_prov, 6), "ratio_under_book_counts": round(r_book, 6)}
    ev = "EXPECTED-MAPPING EVIDENCE (author fix2 2026-10-30): offline deterministic replay of the frozen loader entry point init_by_eplb -> compute_algorithm(auto, num_groups=1, num_nodes=1) -> rebalance_experts(num_physical=128, num_local=16, groups=1, nodes=1). Runtime p2l export NOT available (0 hits in sglang/srt/entrypoints); this is NOT an actual runtime fingerprint."
    out = {"generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "arm": name, "profile_path": ppath, "profile_sha256": hashlib.sha256(open(ppath, "rb").read()).hexdigest(), "evidence_class": ev, "algorithm": str(algo), "determinism_check": "second replay identical PASS", "p2l": flat, "p2l_sha256": psha, "expert_count_shape": str(tuple(ec.shape)) if hasattr(ec, "shape") else str(type(ec)), "rank_loads_under_own_counts": loads_own, "rank_loads_under_prover_counts": loads_prov, "rank_loads_under_book_counts": loads_book, "ratios": {"own_counts": r_own, "prover_counts": r_prov, "book_counts": r_book}}
    op = CB + "/analysis/expected_mapping_%s.json" % name
    json.dump(out, open(op, "w"), indent=1)
    print("%s: algo=%s p2l_sha=%s ratios own=%.6f prover=%.6f book=%.6f" % (name, algo, psha[:16], r_own, r_prov, r_book))
print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
