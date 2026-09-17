"""
Aggregate per-domain hot-GPU distribution from identity rt2 traces.
Reproduce fig15b entropy: per-forward, aggregate expert tokens -> GPU load
(GPU_id = expert_id // 16 under identity map), find argmax -> that forward's
hot GPU. Entropy is over the 8-bin distribution of hot GPU across forwards.
"""
import numpy as np, glob, json, os
from pathlib import Path

TRACE_ROOT = "/data/minghua/sjq/paperpicturetrace/fig5_6_8_14_PD_correlation"
E, EP = 128, 8
EPG = E // EP
OUT = "/workspace/EPLB/NEW_PAPER/experiments/hotmap_20260917/hotgpu_distributions.json"

DOMAINS = {
    "MMLU":       "rt2_MMLU_25tok_QA",
    "GSM8K":      "rt2_GSM8K_60tok_math",
    "ARC":        "rt2_ARC_science_31tok",
    "ARC-E":      "rt2_ARC_easy_31tok",
    "CSQA":       "rt2_CSQA_20tok",
    "OBQA":       "rt2_OBQA_science_15tok",
    "prover":     "rt2_prover_107tok_math",
    "HumanEval":  "rt2_HumanEval_code",
    "CMMLU":      "rt2_CMMLU_zh_QA",
}

def entropy_bits(p):
    p = np.asarray(p, dtype=float); p = p[p > 0]
    if len(p) == 0: return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())

results = {}
for name, sub in DOMAINS.items():
    files = sorted((Path(TRACE_ROOT)/sub).glob("rank0_fwd_chunk*.npz"))
    if not files: continue
    hot_gpus = []
    for f in files:
        d = np.load(f)
        lh = d["layer_hists"]                                    # (N, 94, 128)
        gpu_load = lh.reshape(lh.shape[0], 94, EP, EPG).sum(axis=(1,3))  # (N, 8)
        hot = gpu_load.argmax(axis=1)                            # (N,)
        hot_gpus.extend(hot.tolist())
    arr = np.array(hot_gpus, dtype=int); n = len(arr)
    counts = np.bincount(arr, minlength=EP).astype(float)
    p = counts / counts.sum()
    H = entropy_bits(p)
    modal = int(counts.argmax())
    results[name] = dict(
        n_forwards=n, counts_per_gpu=counts.tolist(),
        prop_per_gpu=p.tolist(), entropy_bits=H,
        modal_gpu=modal, modal_share=float(p[modal]),
    )
    print(f"{name:10s} n={n:4d}  H={H:.3f}  modal=GPU{modal} ({p[modal]*100:.1f}%)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)
print("wrote:", OUT)
