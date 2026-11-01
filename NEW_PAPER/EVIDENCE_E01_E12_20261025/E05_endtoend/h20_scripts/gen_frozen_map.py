#!/usr/bin/env python3
"""Generate the frozen optimized placement map for h20-placement-fullservice-v1.

SETUP ONLY -- runs outside all timed blocks, before any measured campaign run.

Input : counts JSON {"counts": [[128 floats] x 94 layers], ...} produced by
        dump_counts.py from a full profiling pass of the frozen workload on the
        identity placement (SGLang expert-distribution recorder, mode "stat").
Method: per layer, greedy_base() imported verbatim from
        generate_stopping_layouts.py -- the exact frozen lexicographic-descent
        optimizer lineage used in experiments 1/2 of this campaign
        (score = (max raw rank load, max 128-aligned padded load,
                  sorted raw profile desc, map sha256); all 7168 legal
        cross-rank swaps scanned per step).
Output: frozen_map.json      {"physical_to_logical_map": 94 x 128 nested list}
                             (SGLang --init-expert-location init_by_mapping format)
        frozen_map_report.json  per-layer r_identity/r_greedy/steps/moves + hashes.

No timing, no performance information, no randomness: output is a deterministic
function of the input counts. Selection never uses performance timing.
"""
import hashlib
import json
import os
import sys
import types
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate_stopping_layouts import greedy_base  # frozen optimizer lineage
from prediction_common import placement_map_sha256

NUM_LAYERS_EXPECTED = 94
NUM_EXPERTS = 128
LOCAL_EXPERTS = 16


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def r_of(counts, p2l):
    loads = [0.0] * 8
    for slot, logical in enumerate(p2l):
        loads[slot // LOCAL_EXPERTS] += counts[logical]
    mean = sum(loads) / 8.0
    return (max(loads) / mean) if mean > 0 else float("nan")


def main():
    if len(sys.argv) != 3:
        print("usage: gen_frozen_map.py <counts.json> <out_dir>", file=sys.stderr)
        return 2
    counts_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    map_path = os.path.join(out_dir, "frozen_map.json")
    report_path = os.path.join(out_dir, "frozen_map_report.json")
    for p in (map_path, report_path):
        if os.path.exists(p):
            print(f"refusing to overwrite {p}", file=sys.stderr)
            return 3
    payload = json.load(open(counts_path))
    counts = payload["counts"]
    assert len(counts) == NUM_LAYERS_EXPECTED, f"layers {len(counts)}"
    assert all(len(row) == NUM_EXPERTS for row in counts)
    assert all(v >= 0 for row in counts for v in row)
    total = sum(sum(row) for row in counts)
    assert total > 0

    placement = []
    per_layer = []
    for layer in range(NUM_LAYERS_EXPECTED):
        meta = types.SimpleNamespace(expert_counts=list(counts[layer]))
        res = greedy_base(meta)
        p2l = list(res["p2l"])
        assert sorted(p2l) == list(range(NUM_EXPERTS)), f"layer {layer} not a permutation"
        moves = sum(1 for i, v in enumerate(p2l) if v != i)
        r_id = r_of(counts[layer], tuple(range(NUM_EXPERTS)))
        r_greedy = r_of(counts[layer], tuple(p2l))
        placement.append(p2l)
        per_layer.append({
            "layer": layer,
            "r_identity": r_id,
            "r_greedy": r_greedy,
            "step_count": res["step_count"],
            "scans": res["scans"],
            "moves_vs_identity": moves,
            "layer_total_assignments": sum(counts[layer]),
        })

    json.dump({"physical_to_logical_map": placement}, open(map_path, "w"))
    r_ids = [d["r_identity"] for d in per_layer]
    r_gre = [d["r_greedy"] for d in per_layer]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_counts_path": counts_path,
        "input_counts_sha256": sha256_file(counts_path),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "optimizer_lineage": {
            "function": "greedy_base",
            "module": "generate_stopping_layouts.py",
            "module_sha256": sha256_file(os.path.join(HERE, "generate_stopping_layouts.py")),
            "prediction_common_sha256": sha256_file(os.path.join(HERE, "prediction_common.py")),
        },
        "total_assignments": total,
        "map_format": "physical_to_logical_map 94x128 (SGLang init_by_mapping)",
        "frozen_map_sha256": sha256_file(map_path),
        "per_layer": per_layer,
        "aggregate": {
            "r_identity_mean": sum(r_ids) / len(r_ids),
            "r_identity_max": max(r_ids),
            "r_greedy_mean": sum(r_gre) / len(r_gre),
            "r_greedy_max": max(r_gre),
            "layers_with_moves": sum(1 for d in per_layer if d["moves_vs_identity"] > 0),
            "total_moves": sum(d["moves_vs_identity"] for d in per_layer),
            "total_steps": sum(d["step_count"] for d in per_layer),
        },
        "selection_uses_performance_timing": False,
    }
    json.dump(report, open(report_path, "w"), indent=1)
    print(json.dumps(report["aggregate"], indent=1))
    print("wrote", map_path)
    print("wrote", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
