import json, os, subprocess, sys, hashlib
import numpy as np
X = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027"
ART = X + "/prep/record_boot2"
BUILD = X + "/prep/profile_build"
# 1) pt -> P.npz via proven converter
r = subprocess.run([sys.executable, "/workspace/EPLB/OEPLB/baselines/moetuner/src/pt2npz.py",
                    "--src-glob", ART + "/expert_distribution_recorder*.pt",
                    "--out", BUILD + "/P_e3.npz"], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-800:]); sys.exit(1)
# 2) ES/DF shared profile json from same P
d = np.load(BUILD + "/P_e3.npz")
P = d["P"]
print("P shape", P.shape, "dtype", P.dtype, "total sum", int(P.sum()), "num_forwards", int(d["num_forwards"]))
print("per-layer sum mean", float(P.sum(axis=1).mean()), "layer0 imb", float(P[0].max() / max(P[0].mean(), 1)))
assert P.shape == (94, 128)
lc = [[int(v) for v in row] for row in P]
es = {"logical_count": lc}
old = json.load(open("/data/minghua/sjq/OEPLBdata/experiment_logs/baseline_comparison_20260914/datafore_prover_placement.json"))
assert sorted(old.keys()) == sorted(es.keys()), "key structure differs from proven ES format"
p1 = X + "/placements/datafore_e3_placement.json"
p2 = X + "/placements/datafore_e3_remap_placement.json"
with open(p1, "w") as f: json.dump(es, f)
with open(p2, "w") as f: json.dump(es, f)
h1 = hashlib.sha256(open(p1, "rb").read()).hexdigest()
h2 = hashlib.sha256(open(p2, "rb").read()).hexdigest()
assert h1 == h2, "shared profile files must be byte-identical"
print("ES_DF_JSON_OK sha=" + h1)
