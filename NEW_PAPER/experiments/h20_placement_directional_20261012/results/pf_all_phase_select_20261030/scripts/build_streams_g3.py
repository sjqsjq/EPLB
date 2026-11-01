
import json, hashlib, os

SRC_ARC = "/data/minghua/sjq/shortdomain/crossdomain_3x4000_O10.jsonl"
SRC_PROVER = "/workspace/EPLB/NEW_PAPER/EVIDENCE_E01_E12_20261025/E11_baselines_redundancy/sjq_single_domain/prover_1024_out32.jsonl"
OUT_ARC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "B_data", "streams", "arcc_1024_out32.jsonl")
OUT_PROVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "B_data", "streams", "prover_1024_out32_ie.jsonl")

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

arc_rows = []
with open(SRC_ARC) as f:
    for line in f:
        r = json.loads(line)
        if str(r.get("id", "")).startswith("ARC_"):
            arc_rows.append(r)
        if len(arc_rows) == 1024:
            break
assert len(arc_rows) == 1024, len(arc_rows)
ids = [r["id"] for r in arc_rows]
assert ids[0] == "ARC_0" and ids[-1] == "ARC_1023", (ids[0], ids[-1])
delta_arc = {"max_tokens": [], "temperature": [], "ignore_eos": []}
out = []
for r in arc_rows:
    for k, want in (("max_tokens", 32), ("temperature", 0), ("ignore_eos", True)):
        if r.get(k) != want:
            delta_arc[k].append((r.get("id"), r.get(k), want))
        r[k] = want
    out.append(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
with open(OUT_ARC, "w") as f:
    f.write("\n".join(out) + "\n")

prov_rows = []
with open(SRC_PROVER) as f:
    for line in f:
        prov_rows.append(json.loads(line))
assert len(prov_rows) == 1024, len(prov_rows)
delta_prov = {"max_tokens": [], "temperature": [], "ignore_eos": []}
out2 = []
for r in prov_rows:
    for k, want in (("max_tokens", 32), ("temperature", 0), ("ignore_eos", True)):
        if r.get(k) != want:
            delta_prov[k].append((r.get("id"), r.get(k), want))
        r[k] = want
    out2.append(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
with open(OUT_PROVER, "w") as f:
    f.write("\n".join(out2) + "\n")

report = {
    "generated_from_builder_sha": None,
    "arc": {"src": SRC_ARC, "src_sha256": sha(SRC_ARC), "out": os.path.abspath(OUT_ARC), "out_sha256": sha(os.path.abspath(OUT_ARC)), "out_bytes": os.path.getsize(os.path.abspath(OUT_ARC)), "lines": 1024, "ids": "ARC_0..ARC_1023 contiguous", "delta_fields": {k: len(v) for k, v in delta_arc.items()}, "delta_examples": {k: v[:3] for k, v in delta_arc.items()}},
    "prover": {"src": SRC_PROVER, "src_sha256": sha(SRC_PROVER), "out": os.path.abspath(OUT_PROVER), "out_sha256": sha(os.path.abspath(OUT_PROVER)), "out_bytes": os.path.getsize(os.path.abspath(OUT_PROVER)), "lines": 1024, "delta_fields": {k: len(v) for k, v in delta_prov.items()}, "delta_examples": {k: v[:3] for k, v in delta_prov.items()}},
}
print("BUILD_REPORT=" + json.dumps(report, sort_keys=True))
