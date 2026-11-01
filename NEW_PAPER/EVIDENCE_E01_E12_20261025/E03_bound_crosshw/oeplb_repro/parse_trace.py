"""Bucket a torch-profiler trace into the components the beta model needs.

H1: beta = B/T_flat should equal the routed-expert GEMM share of wall time.
H2: the absorbed slack B*(r_k-1) should correspond to the overlappable part of
    dispatch/combine.
Both are one-profile quantities, so if they hold, beta and r_k stop needing a
14-run T(r) sweep.
"""
import gzip, json, sys, glob, collections, re

CAT = [
    ("gemm",     re.compile(r"gemm|fp8_gemm|grouped|cutlass|sm90|tensorop", re.I)),
    ("dispatch", re.compile(r"dispatch|notify|get_dispatch|moe_recv", re.I)),
    ("combine",  re.compile(r"combine", re.I)),
    ("attn",     re.compile(r"flash|attn|fmha", re.I)),
    ("nccl",     re.compile(r"nccl|allreduce|all_reduce|reduce_scatter|allgather", re.I)),
]

def bucket(name):
    for k, rx in CAT:
        if rx.search(name):
            return k
    return "other"

for path in sorted(sys.argv[1:]):
    op = gzip.open if path.endswith(".gz") else open
    try:
        ev = json.load(op(path, "rt"))["traceEvents"]
    except Exception as e:
        print(f"{path}: unreadable ({e})"); continue
    tot = collections.Counter(); n = collections.Counter()
    for e in ev:
        # GPU kernels only: ph=X with a duration, on a stream track
        if e.get("ph") != "X" or "dur" not in e: continue
        if e.get("cat") not in ("kernel", "gpu_op", "Kernel"): continue
        b = bucket(e.get("name", ""))
        tot[b] += e["dur"]; n[b] += 1
    s = sum(tot.values())
    if not s:
        print(f"{path}: no kernel events"); continue
    print(f"\n{path.split('/')[-1]}   total kernel time {s/1e6:.3f} s")
    for k in [c[0] for c in CAT] + ["other"]:
        if tot[k]:
            print(f"   {k:9s} {tot[k]/1e6:8.3f} s  {100*tot[k]/s:5.1f}%   ({n[k]} kernels)")
