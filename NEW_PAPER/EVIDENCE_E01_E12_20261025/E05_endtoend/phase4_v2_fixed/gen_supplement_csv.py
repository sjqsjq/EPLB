"""Phase-4 locked CSV generator: 64-boot core supplement.
E3/E4 x {I, E_native(E auto 16 256), fixed-P(P), fixed-F(F 1 16 16)} x 8 blocks.
Order: block-major round-robin over cells so temporal drift hits all arms
equally within each block. Labels s_<stream>_<arm>_b<block> (NEW pool prefix,
old m_* pool untouched). Run ONLY after phase-3 gate + CSV commit."""
import csv, hashlib, os
G = os.path.dirname(os.path.abspath(__file__))
F = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/final_plan_20261019"
CELLS = [
    ("E3", "I", "", "e3_multidomain_switch.jsonl", 8192),
    ("E3", "E", "auto 16 256", "e3_multidomain_switch.jsonl", 8192),
    ("E3", "P", "", "e3_multidomain_switch.jsonl", 8192),
    ("E3", "F", "1 16 16", "e3_multidomain_switch.jsonl", 8192),
    ("E4", "I", "", "e4_fullduplex.jsonl", 2048),
    ("E4", "E", "auto 16 256", "e4_fullduplex.jsonl", 2048),
    ("E4", "P", "", "e4_fullduplex.jsonl", 2048),
    ("E4", "F", "1 16 16", "e4_fullduplex.jsonl", 2048),
]
rows, seq = [], 0
for block in range(1, 9):
    for stream, arm, params, ds, nreq in CELLS:
        seq += 1
        rows.append({
            "seq": seq, "phase": "SUPP", "block": block, "stream": stream,
            "arm": arm, "params": params,
            "label": f"s_{stream}_{arm}_b{block}",
            "ds_path": f"{F}/B_data/streams/{ds}",
            "ds_sha256": hashlib.sha256(open(f"{F}/B_data/streams/{ds}", "rb").read()).hexdigest(),
            "nreq": nreq, "conc": 256, "bench_timeout": 3600,
            "code_freeze": "async_swapper=85a2dac882177667 controller=4964cfc8e8af397a",
        })
out = f"{G}/supplement_matrix_locked.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
print("csv_sha256:", hashlib.sha256(open(out, 'rb').read()).hexdigest())
