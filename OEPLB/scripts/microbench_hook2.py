"""Hook DeepGEMM's actual kernel op (not the SGLang wrapper)."""
import torch, time, json, os, deep_gemm

LOG_FILE = os.environ.get("MICROBENCH_LOG", "/workspace/logs/microbench_gemm.log")
_results = []

_orig = deep_gemm.fp8_m_grouped_gemm_nt_masked

def _hook(*args, **kwargs):
    # args: (lhs, rhs, out, masked_m, expected_m, ...)
    masked_m = args[3] if len(args) > 3 else kwargs.get('masked_m')
    expected_m = args[4] if len(args) > 4 else kwargs.get('expected_m', 0)
    
    G = int((masked_m > 0).sum().item()) if masked_m is not None else 0
    M_total = int(masked_m.sum().item()) if masked_m is not None else 0
    M_avg = M_total / max(G, 1) if G > 0 else 0
    M_max = int(masked_m.max().item()) if masked_m is not None and G > 0 else 0
    
    lhs = args[0] if args else kwargs.get('lhs')
    rhs = args[1] if len(args) > 1 else kwargs.get('rhs')
    out = args[2] if len(args) > 2 else kwargs.get('out')
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = _orig(*args, **kwargs)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e6
    
    _results.append({
        'G': G, 'M_avg': round(M_avg, 1), 'M_max': M_max,
        'M_total': M_total, 'expected_m': expected_m,
        'time_us': round(dt, 1),
        'lhs_shape': list(lhs[0].shape) if lhs else [],
        'rhs_shape': list(rhs[0].shape) if rhs else [],
        'out_shape': list(out.shape) if out is not None else [],
    })
    
    if len(_results) % 50 == 0:
        _flush()
    
    return result

deep_gemm.fp8_m_grouped_gemm_nt_masked = _hook

def _flush():
    with open(LOG_FILE, 'w') as f:
        json.dump(_results, f)

import atexit
atexit.register(_flush)

print(f"[MICROBENCH2] Hook installed on deep_gemm.fp8_m_grouped_gemm_nt_masked, log={LOG_FILE}")
