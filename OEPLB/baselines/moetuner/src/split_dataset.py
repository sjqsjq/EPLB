#!/usr/bin/env python3
"""Deterministic head-2048 / tail-6144 split of L512_O1_realprover_n8192.jsonl.

Regenerates the two artifact files used by driver_moetuner.sh fair variant.
Idempotent.
"""
import os, sys
DEF_SRC = "/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl"
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEF_SRC
    out_dir = os.path.dirname(os.path.abspath(__file__)) + "/../artifacts"
    os.makedirs(out_dir, exist_ok=True)
    with open(src) as f:
        lines = f.readlines()
    n = len(lines)
    print(f"total: {n}")
    open(f"{out_dir}/L512_O1_realprover_head2048.jsonl", "w").writelines(lines[:2048])
    open(f"{out_dir}/L512_O1_realprover_tail6144.jsonl", "w").writelines(lines[2048:])
    print(f"wrote {out_dir}/{{head2048,tail6144}}.jsonl")
if __name__ == "__main__":
    main()
