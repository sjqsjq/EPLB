"""
Monkey-patch DeepGEMM masked GEMM to log (M, G, time).
Loaded as a sitecustomize or imported before serving.

Usage: add to server launch: PYTHONPATH=/workspace/EPLB/OEPLB/scripts python3 -m sglang.launch_server ...
"""
import torch, time, json, os

LOG_FILE = os.environ.get("MICROBENCH_LOG", "/workspace/logs/microbench_gemm.log")
_results = []

_original_fn = None

def _install_hook():
    global _original_fn
    try:
        from sglang.srt.layers.deep_gemm_wrapper import entrypoint
        _original_fn = entrypoint.grouped_gemm_nt_f8f8bf16_masked
        
        def _hooked_fn(lhs, rhs, out, masked_m, expected_m, overlap_args=None, max_block_n=256):
            # Extract M and G from masked_m
            G = (masked_m > 0).sum().item()
            M_total = masked_m.sum().item()
            M_avg = M_total / max(G, 1)
            M_max = masked_m.max().item() if G > 0 else 0
            
            # Time the actual kernel call
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = _original_fn(lhs, rhs, out, masked_m, expected_m, overlap_args, max_block_n)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e6
            
            # Log
            _results.append({
                'G': G, 'M_avg': round(M_avg, 1), 'M_max': M_max,
                'M_total': M_total, 'expected_m': expected_m,
                'time_us': round(dt, 1),
                'lhs_shape': list(lhs[0].shape),
                'rhs_shape': list(rhs[0].shape),
            })
            
            # Flush every 100 calls
            if len(_results) % 100 == 0:
                _flush()
            
            return result
        
        entrypoint.grouped_gemm_nt_f8f8bf16_masked = _hooked_fn
        
        # Also patch the reference in deep_gemm.py
        import sglang.srt.layers.moe.moe_runner.deep_gemm as dg
        dg.deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked = _hooked_fn
        
        print(f"[MICROBENCH] Hook installed on grouped_gemm_nt_f8f8bf16_masked, logging to {LOG_FILE}")
    except Exception as e:
        print(f"[MICROBENCH] Failed to install hook: {e}")

def _flush():
    with open(LOG_FILE, 'w') as f:
        json.dump(_results, f)

import atexit
atexit.register(_flush)

# Auto-install on import
_install_hook()
