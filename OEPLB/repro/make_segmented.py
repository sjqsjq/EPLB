"""Build a piecewise-stationary workload by interleaving requests from datasets
with genuinely different routing distributions, switching every L_seg requests.

This is the controlled changepoint sequence the adaptive-window experiment needs:
existing 'multi-domain' data turned out to be near-homogeneous (r_before differs
<1.5% across datasets), so no real domain shift was ever exercised.

Usage: python3 make_segmented.py <L_seg> <total> <out.jsonl> <ds1> <ds2> [ds3...]
"""
import json, sys

L_seg = int(sys.argv[1]); total = int(sys.argv[2]); out = sys.argv[3]
datasets = sys.argv[4:]
pools = []
for d in datasets:
    reqs = [json.loads(l) for l in open(d)]
    pools.append(reqs)
    print(f"  loaded {len(reqs)} from {d.split('/')[-1]}")

result = []
seg = 0
idx = [0] * len(pools)
while len(result) < total:
    which = seg % len(pools)
    pool = pools[which]
    for _ in range(min(L_seg, total - len(result))):
        r = dict(pool[idx[which] % len(pool)])
        r["_seg"] = seg
        r["_domain"] = which
        result.append(r)
        idx[which] += 1
    seg += 1

with open(out, "w") as f:
    for r in result:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(result)} requests, {seg} segments of ~{L_seg}, {len(pools)} domains -> {out}")
