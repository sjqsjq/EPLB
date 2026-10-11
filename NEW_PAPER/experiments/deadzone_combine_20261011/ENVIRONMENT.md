# Deadzone combine-progress experiment environment

## Frozen target path

- Hardware: 8× NVIDIA H20 96 GB, full 8-GPU NVLink topology (`NV18` between every pair).
- GPU: 78 SMs/device, compute capability 9.0, persistence mode enabled, 500 W power limit.
- Model: `Qwen3-235B-A22B-FP8`, layer count 94, hidden size 4096, MoE intermediate size 1536, 128 experts, top-k 8, FP8 block size 128×128.
- Parallel layout: TP=8, DP=8, EP=8 in the paper service; the isolated microbenchmark instantiates the same 8-rank EP group and one real-size MoE layer.
- Execution path: eager, DeepEP **normal**, DeepGEMM FP8, PB-OEPLB disabled, no expert exchange.
- DeepEP communication config: paper main runs use the package default **20 SMs** (not `/workspace/EPLB/deepep_config.json`, which requests 48 SMs and belongs to an older EP4 experiment).
- DeepEP normal dispatch expert alignment: 128 tokens when JIT DeepGEMM is enabled.
- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512` for parity with paper launch scripts.

## Software

| Component | Version / revision |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 5.10.134-010.ali5000.al8.x86_64 |
| NVIDIA driver | 580.105.08 (driver CUDA 13.0) |
| CUDA toolkit | 12.8 |
| PyTorch | 2.9.1+cu128 |
| SGLang | 0.5.6.post2 wheel |
| SGLang `qwen3_moe.py` SHA256 | `94179c997566e2a7926410691404e2d9b4cc2d64aa57f4e23a31627be9798a52` |
| SGLang `token_dispatcher/deepep.py` SHA256 | `a8577205fbb024d8022c1868672fe741542ac46287c384c8ed6b07eb9cfedd15` |
| DeepEP | 1.2.1+9af0e0d, git `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` |
| DeepGEMM | 2.3.0, git `06fbbd449e36b7e0d97580b1e913140457ba563a` |
| EPLB repository | git `1bd6e026340737b85ef28d7d524c3c1726cc2c90` at experiment start |

## Important local DeepEP modifications

The editable DeepEP checkout was **not clean** at experiment start:

1. `deep_ep/buffer.py`: enables IBGDA only when `runtime.get_num_rdma_ranks() > 1`; this avoids enabling IBGDA for the single-node NVLink-only case.
2. `csrc/kernels/internode_ll.cu`: comments out an internode low-latency RC-QP assertion.
3. Locally built `deep_ep_cpp.cpython-311-x86_64-linux-gnu.so` is untracked.

The present experiment is single-node DeepEP **normal** mode, so the internode low-latency assertion is not on its measured path. Preserve these modifications; do not silently reset the DeepEP checkout.

## Fixed test shapes

- EP-group total tokens: `N ∈ {256, 4096}`.
- top-k: 8.
- Mean assignments per rank: `N × top_k / EP = N`, i.e. 256 or 4096.
- For a balanced route with 16 local experts, mean assignments/expert before DeepEP alignment: 16 or 256.
- DeepEP normal mode aligns received expert counts to 128 for JIT DeepGEMM; therefore N=256 exercises small/padded expert shapes and N=4096 exercises 256-token/expert shapes.

## Timing rules

- CUDA events are only subtracted on the same GPU/stream.
- Cross-rank makespan is computed from per-rank elapsed intervals after a GPU collective gate; timestamps from different GPU clocks are never directly subtracted.
- Main timing and Nsight Systems capture are separate runs.
- Every measured condition is randomized by block and includes raw per-repetition records.
