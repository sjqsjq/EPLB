#!/usr/bin/env python3
"""Dense flat-region + staircase bench: REAL DeepGEMM FP8 kernel on H20.
Pure GPU-execution timing (CUDA events, cast OUTSIDE the timed loop) to isolate
the kernel tile-padding staircase from per-token-cast drift and launch variance.
Qwen3-235B w13 expert: K=4096, N=3072. per-token input / per-block weight, recipe=(1,128,128)."""
import os, json, statistics
import torch
import deep_gemm as dg
from deep_gemm.utils.math import per_token_cast_to_fp8, per_block_cast_to_fp8

K, N = 4096, 3072
REPS = 500
ROUNDS = 5
WARM = 30

torch.cuda.set_device(0)
print(f"[bench] {torch.cuda.get_device_name(0)}  sms={dg.get_num_sms()}  pure-GPU-event timing")

b = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
b_fp8, sfb = per_block_cast_to_fp8(b, use_ue8m0=False)
_ = b_fp8.float()  # prime caches

def bench_m(m):
    a = torch.randn((m, K), device='cuda', dtype=torch.bfloat16)
    a_fp8, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)   # cast ONCE, outside timing
    d = torch.empty((m, N), device='cuda', dtype=torch.bfloat16)
    fn = lambda: dg.fp8_gemm_nt((a_fp8, sfa), (b_fp8, sfb), d, None,
                                recipe=(1, 128, 128), disable_ue8m0_cast=True)
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ROUNDS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / REPS)  # us/call, GPU-only
    return statistics.median(times)

Ms = (list(range(1, 65))                                       # flat: every token 1..64
      + list(range(65, 131))                                   # first transitions 65..130
      + list(range(132, 257, 2))                               # 132..256 step 2
      + [257, 258, 260, 264, 272, 280, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 513,
         544, 576, 608, 640, 672, 704, 736, 768, 769, 800, 832, 864, 896, 928, 960, 992, 1024])

results = []
for i, m in enumerate(Ms):
    t = bench_m(m)
    results.append({'M': m, 'us': round(t, 2)})
    print(f"[{i+1}/{len(Ms)}] M={m:>4}  {t:7.2f} us")

out = {
    'kernel': 'deep_gemm.fp8_gemm_nt (REAL DeepGEMM FP8, per-token input + per-block weight)',
    'K': K, 'N': N, 'recipe': [1, 128, 128], 'use_ue8m0': False,
    'timing': 'CUDA event, GPU-execution only (cast outside loop)',
    'reps': REPS, 'rounds': ROUNDS, 'warmup': WARM,
    'desc': 'Dense flat (M=1..64 every token) + staircase to 1024, H20, BF16 output',
    'results': results,
}
op = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepgemm_flat_dense.json')
with open(op, 'w') as f:
    json.dump(out, f, indent=2)
print(f"[bench] wrote {op}  ({len(results)} pts)")
