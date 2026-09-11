"""
Monkey-patch DeepGEMM masked GEMM to log per-call (M, G, time_us).
Loaded via SGLANG_GEMM_PROFILER=1 env var.
"""
import torch, time, json, os

LOG_FILE = os.environ.get("GEMM_PROFILER_LOG", "/workspace/logs/gemm_profile.jsonl")
_prof_data = []

try:
    from sglang.srt.layers.deep_gemm_wrapper import entrypoint
    _orig = entrypoint.grouped_gemm_nt_f8f8bf16_masked

    def _patched(lhs, rhs, out, masked_m, expected_m, overlap_args=None, max_block_n=256):
        # Count active groups (non-zero masked_m)
        G_active = int((masked_m > 0).sum().item())
        G_total = masked_m.shape[0]
        # Sample first few masked_m values
        m_sample = masked_m[:4].tolist()
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = _orig(lhs, rhs, out, masked_m, expected_m, overlap_args, max_block_n)
        torch.cuda.synchronize()
        dt_us = (time.perf_counter() - t0) * 1e6
        
        entry = {
            "G_active": G_active, "G_total": G_total,
            "expected_m": expected_m, "m_sample": m_sample,
            "time_us": round(dt_us, 1),
            "N": rhs[0].shape[1], "K": lhs[0].shape[2]
        }
        _prof_data.append(entry)
        
        # Flush every 100 calls
        if len(_prof_data) >= 100:
            with open(LOG_FILE, "a") as f:
                for e in _prof_data:
                    f.write(json.dumps(e) + "\n")
            _prof_data.clear()
        
        return result

    entrypoint.grouped_gemm_nt_f8f8bf16_masked = _patched
    # Also patch the reference in deep_gemm.py if it imported directly
    import sglang.srt.layers.moe.moe_runner.deep_gemm as dg_runner
    if hasattr(dg_runner, 'deep_gemm_wrapper'):
        dg_runner.deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked = _patched
    print(f"[GEMM_PROFILER] patched, logging to {LOG_FILE}")
except Exception as e:
    print(f"[GEMM_PROFILER] FAILED to patch: {e}")
