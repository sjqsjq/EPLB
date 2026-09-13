#!/usr/bin/env python3
"""Duplication bench: replicate a hot expert to K cards, each gets M_hot/K tokens.
Measures real DeepGEMM FP8 kernel time at M = ceil(M_hot/K) for the straggler
replica (MoE all-to-all syncs => layer time = max replica GEMM time).
Proves: for decode M_hot (<=256), every K lands in flat floor => zero GEMM gain."""
import os, json, statistics, math
import torch
import deep_gemm as dg
from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8

K, N = 4096, 3072
REPS, ROUNDS, WARM = 500, 5, 30
torch.cuda.set_device(0)
print(f"[dup-bench] {torch.cuda.get_device_name(0)} sms={dg.get_num_sms()}")

b = torch.randn((N, K, ), device='cuda', dtype=torch.bfloat16)
b_fp8, sfb = per_block_cast_to_fp8(b, use_ue8m0=False)
_ = b_fp8.float()

_cache = {}
def gemm_us(m):
    m = max(1, m)
    if m in _cache:
        return _cache[m]
    a = torch.randn((m, K), device='cuda', dtype=torch.bfloat16)
    a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
    d = torch.empty((m, N), device='cuda', dtype=torch.bfloat16)
    fn = lambda: dg.fp8_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d, None,
                                recipe=(1, 128, 128), disable_ue8m0_cast=True)
    for _ in range(WARM): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ROUNDS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS): fn()
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / REPS)
    _cache[m] = statistics.median(ts)
    return _cache[m]

# M_hot spanning decode (8) -> prefill (1024); K replica counts
M_hots = [8, 16, 32, 64, 128, 256, 512, 768, 1024]
Ks = [1, 2, 4, 8]

results = []
print(f"{'M_hot':>6} {'K':>3} {'M_rep':>6} {'us':>7} {'gain_us':>8} {'gain_%':>7}")
for mhot in M_hots:
    base = gemm_us(mhot)  # K=1 baseline (no duplication)
    for k in Ks:
        m_rep = math.ceil(mhot / k)         # tokens per replica (padded up by GEMM tile)
        us = gemm_us(m_rep)
        gain = base - us                    # reduction in straggler GEMM time
        gpct = 100 * gain / base
        results.append({'M_hot': mhot, 'K': k, 'M_replica': m_rep,
                        'us': round(us, 2), 'gain_us': round(gain, 2),
                        'gain_pct': round(gpct, 1)})
        print(f"{mhot:>6} {k:>3} {m_rep:>6} {us:>7.2f} {gain:>+8.2f} {gpct:>+7.1f}")

out = {
    'kernel': 'deep_gemm.fp8_gemm_nt (per-token input + per-block weight)',
    'K_dim': K, 'N': N, 'recipe': [1, 128, 128], 'use_ue8m0': False,
    'timing': 'CUDA event, GPU-exec only',
    'desc': 'Duplication sweep: hot expert M_hot -> K replicas, each ceil(M_hot/K). '
            'straggler GEMM time = layer time (all-to-all sync).',
    'results': results,
}
op = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepgemm_duplication.json')
with open(op, 'w') as f: json.dump(out, f, indent=2)
print(f"[dup-bench] wrote {op}")
