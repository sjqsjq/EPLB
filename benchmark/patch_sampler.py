"""Patch sampler.py to handle nan/inf in probability tensor.

This allows forward passes to complete even when MoE outputs contain nan
(e.g. with --ep-dispatch-algorithm fake or uninitialized redundant experts).
Output tokens will be garbage, but we only need forward pass counts to
trigger EPLB rebalance.
"""
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/sampler.py"

with open(TARGET, "r") as f:
    content = f.read()

OLD = """def sampling_from_probs_torch(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
):
    \"\"\"A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering.\"\"\"
    if sampling_seed is not None:
        sampled_index = multinomial_with_seed(probs, sampling_seed, positions)
    else:
        sampled_index = torch.multinomial(probs, num_samples=1)
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids"""

NEW = """def sampling_from_probs_torch(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
):
    \"\"\"A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering.\"\"\"
    # NaN/Inf guard: replace invalid values with uniform distribution
    if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)) or torch.any(probs < 0):
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp(min=0.0)
        row_sums = probs.sum(dim=-1, keepdim=True)
        zero_rows = (row_sums == 0)
        if zero_rows.any():
            probs[zero_rows.expand_as(probs)] = 1.0 / probs.shape[-1]
            row_sums = probs.sum(dim=-1, keepdim=True)
        probs = probs / row_sums
    if sampling_seed is not None:
        sampled_index = multinomial_with_seed(probs, sampling_seed, positions)
    else:
        sampled_index = torch.multinomial(probs, num_samples=1)
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids"""

if OLD not in content:
    if "NaN/Inf guard" in content:
        print("ALREADY_PATCHED")
        sys.exit(0)
    print("WARNING: exact match not found", file=sys.stderr)
    sys.exit(1)

content = content.replace(OLD, NEW)

with open(TARGET, "w") as f:
    f.write(content)

print("OK")
