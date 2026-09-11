#!/usr/bin/env python3
"""DeepGEMM operator-level microbenchmark.
Sweep tokens per expert (N) and expert count (G), measure grouped GEMM time.
No SGLang server needed — direct kernel call, zero serving noise.
"""
import torch
import deep_gemm
import time
import json
import os

# Qwen3-235B-A22B-FP8 expert shape
HIDDEN = 4096
MOE_INTERMEDIATE = 1536
NUM_EXPERTS_LOCAL = 16  # EP=8, 128/8
SCALE_BLOCK = 128
M_ALIGN = deep_gemm.get_m_alignment_for_contiguous_layout()  # 128
K_ALIGN = deep_gemm.get_k_alignment_for_contiguous_layout()  # 128

def make_fp8_tensor(shape, device="cuda"):
    """Create a random FP8 tensor with aligned scales."""
    t = torch.randn(shape, device=device).to(torch.float8_e4m3fn)
    # Scale: [num_groups, ceil(m/scale_block), ceil(k/scale_block)]
    return t

def make_scale(num_groups, m, k, device="cuda"):
    """Create per-token-group FP8 scale (all 1s for timing)."""
    sm = (m + SCALE_BLOCK - 1) // SCALE_BLOCK
    sk = (k + SCALE_BLOCK - 1) // SCALE_BLOCK
    # Align to 4 for UE8M0
    sm = (sm + 3) // 4 * 4
    sk = (sk + 3) // 4 * 4
    s = torch.ones((num_groups, sm, sk), device=device, dtype=torch.float32)
    # Convert to TMA-aligned layout
    s = deep_gemm.get_mn_major_tma_aligned_tensor(s)
    return s

def time_gemm(N_per_expert, G_active, num_layers=1, repeats=100, warmup=10):
    """Time masked grouped GEMM for G experts each with N tokens.
    
    This simulates ONE MoE layer's expert FFN (gate+up GEMM + silu + down GEMM).
    """
    device = "cuda"
    num_groups = NUM_EXPERTS_LOCAL  # always 16 slots, mask the rest
    
    # Pad m to alignment
    m = max(((N_per_expert + M_ALIGN - 1) // M_ALIGN) * M_ALIGN, M_ALIGN)
    
    # Create dummy weights (FP8)
    # w13: [16, 2*1536, 4096] (gate+up)
    # w2:  [16, 4096, 1536] (down)
    w13 = torch.randn(num_groups, 2*MOE_INTERMEDIATE, HIDDEN, device=device).to(torch.float8_e4m3fn)
    w13_scale = make_scale(num_groups, 2*MOE_INTERMEDIATE, HIDDEN, device)
    w2 = torch.randn(num_groups, HIDDEN, MOE_INTERMEDIATE, device=device).to(torch.float8_e4m3fn)
    w2_scale = make_scale(num_groups, HIDDEN, MOE_INTERMEDIATE, device)
    
    # Create dummy input (FP8): [16, m, 4096]
    x13 = torch.randn(num_groups, m, HIDDEN, device=device).to(torch.float8_e4m3fn)
    x13_scale = make_scale(num_groups, m, HIDDEN, device)
    
    # masked_m: how many tokens each expert actually has
    masked_m = torch.zeros(num_groups, dtype=torch.int32, device=device)
    masked_m[:G_active] = N_per_expert
    
    # Output buffers
    gateup_out = torch.empty(num_groups, m, 2*MOE_INTERMEDIATE, device=device, dtype=torch.bfloat16)
    down_out = torch.empty(num_groups, m, HIDDEN, device=device, dtype=torch.bfloat16)
    
    # For down GEMM, need intermediate after silu
    down_in = torch.randn(num_groups, m, MOE_INTERMEDIATE, device=device).to(torch.float8_e4m3fn)
    down_in_scale = make_scale(num_groups, m, MOE_INTERMEDIATE, device)
    
    expected_m = m
    
    # Warmup (triggers JIT compilation)
    for _ in range(warmup):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (x13, x13_scale), (w13, w13_scale), gateup_out, masked_m, expected_m)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (down_in, down_in_scale), (w2, w2_scale), down_out, masked_m, expected_m)
    torch.cuda.synchronize()
    
    # Time
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # GEMM 1: gate+up
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (x13, x13_scale), (w13, w13_scale), gateup_out, masked_m, expected_m)
        # GEMM 2: down
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (down_in, down_in_scale), (w2, w2_scale), down_out, masked_m, expected_m)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    
    times.sort()
    median = times[len(times)//2]
    p25 = times[len(times)//4]
    p75 = times[3*len(times)//4]
    
    return {
        "N": N_per_expert, "G": G_active, "m_padded": m,
        "median_us": round(median, 2),
        "p25_us": round(p25, 2),
        "p75_us": round(p75, 2),
        "per_expert_us": round(median / G_active, 2) if G_active > 0 else 0,
    }

print(f"DeepGEMM Microbenchmark: Qwen3-235B expert shape")
print(f"  hidden={HIDDEN}, moe_intermediate={MOE_INTERMEDIATE}, EP=8 (16 experts/GPU)")
print(f"  M alignment={M_ALIGN} (tile staircase step)")
print(f"  Expert weight: w13=12MB + w2=6MB = 18MB (FP8)")
print(f"  HBM floor (H20 ~3.35TB/s): {18/3.35:.1f} us")
print()

results = []

# Experiment 1: per-expert time vs N (G=1)
print("=== Exp 1: time vs N (G=1) ===")
for N in [1, 2, 4, 8, 16, 32, 64, 128, 129, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]:
    r = time_gemm(N, G_active=1)
    results.append(r)
    print(f"  N={N:5d} (m_pad={r['m_padded']:5d}): {r['median_us']:8.1f} us  per_expert={r['per_expert_us']:.1f} us")

# Experiment 2: per-GPU time vs G (N=1, decode-like)
print("\n=== Exp 2: time vs G (N=1, decode) ===")
for G in [1, 2, 4, 8, 12, 16]:
    r = time_gemm(N_per_expert=1, G_active=G)
    results.append(r)
    print(f"  G={G:2d}: {r['median_us']:8.1f} us  per_expert={r['per_expert_us']:.1f} us")

# Experiment 2b: N=64 vs G (should overlap with N=1 if flat regime)
print("\n=== Exp 2b: time vs G (N=64, should match N=1 if flat) ===")
for G in [1, 4, 8, 16]:
    r = time_gemm(N_per_expert=64, G_active=G)
    results.append(r)
    print(f"  G={G:2d} N=64: {r['median_us']:8.1f} us  per_expert={r['per_expert_us']:.1f} us")

# Experiment 3: layer-level (G=16, sweep total tokens = N*G)
print("\n=== Exp 3: layer-level (G=16, sweep N) ===")
for N in [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    r = time_gemm(N_per_expert=N, G_active=16)
    results.append(r)
    total_tokens = N * 16
    print(f"  N={N:5d} total={total_tokens:6d}: {r['median_us']:8.1f} us")

# Save
out_path = "/workspace/EPLB/NEW_PAPER/experiments/microbench_deepgemm/results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
