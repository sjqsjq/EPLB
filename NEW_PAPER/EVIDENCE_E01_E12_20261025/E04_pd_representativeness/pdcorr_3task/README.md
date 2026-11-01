# Prefill→Decode Expert Routing Correlation Datasets & Traces

This directory releases the **3 clean datasets and their per-forward routing
traces** used to directly measure the prefill→decode expert-routing correlation
(PB-OEPLB paper §2.3 Observation 3, §3.6; follows DataFore ISCA 2026 Ob3
methodology). Model: **Qwen3-235B-A22B-FP8** (94 MoE layers, 128 experts, top-8).

## Setup (matches DataFore)
- Serving: SGLang, TP=8 DP=8 EP=8, DeepEP, no CUDA graph, `--skip-server-warmup`,
  mem-fraction-static 0.78. Routing tracer: `SGLANG_OEPLB_ROUTING_TRACE=1`.
- **Decode length O=10** (matches DataFore's `MAX_NEW_TOKENS=10`).
- Concurrency 32, aggregate-pooled across all requests + all 8 EP ranks.
- Metric: per-layer Spearman ρ between prefill-stage and decode-stage
  expert-frequency histograms (128 experts), averaged over 94 layers.

## The 3 datasets (all real, non-concatenated, single-passage)

| File | #req | prompt len | domain | O | mean ρ | layers ≥0.7 | top-5 ov |
|------|------|-----------|--------|---|--------|-------------|----------|
| `mmlu_25tok_QA_O10.jsonl` | 14042 | ~25 tok | MMLU multi-subject QA | 10 | **0.833** | **94/94** | 49% |
| `prover_1253tok_math_O10.jsonl` | 2048 | ~1253 tok | DeepSeek-Prover math | 10 | **0.980** | **94/94** | 90% |
| `book_4438tok_O10.jsonl` | 1000 | ~4438 tok | BookCorpus | 10 | **0.967** | **94/94** | 91% |

Each file is `{"id","prompt","max_tokens":10,"temperature":0,"ignore_eos":true}`
(one JSON per line).

## Traces (`traces/<name>/`)
Per-forward routing histograms (`rank{0..7}_fwd_chunk{N}.npz`), each containing:
- `is_prefill`: bool per forward (prefill vs decode)
- `layer_hists`: (94, 128) int expert-selection counts per layer
- `forward_ids`

## Reproduce
1. Launch the trace server (env_235b.sh for NVSHMEM/NCCL):
   ```
   SGLANG_OEPLB_ROUTING_TRACE=1 SGLANG_OEPLB_ROUTING_TRACE_DIR=./traces/<name> \
   python -m sglang.launch_server --model-path Qwen3-235B-A22B-FP8 --dtype bfloat16 \
     --tp 8 --dp 8 --ep-size 8 --enable-dp-attention --moe-a2a-backend deepep \
     --deepep-mode auto --moe-runner-backend deep_gemm --quantization fp8 \
     --mem-fraction-static 0.78 --disable-cuda-graph --skip-server-warmup \
     --port 30000 --trust-remote-code --disable-radix-cache --watchdog-timeout 600
   ```
2. Send the dataset (concurrency 32, O=10): each request `{"prompt":..., "max_tokens":10, "ignore_eos":true}`.
3. Analyze: pool all `is_prefill=True` histograms → prefill_freq (94×128);
   pool all `is_prefill=False` → decode_freq; per-layer Spearman(prefill_freq[l],
   decode_freq[l]); report mean + count(≥0.7).

## Notes
- O=10 decode is sparse per-request; aggregate pooling across all requests+8 ranks
  makes the pooled decode histogram well-sampled (DataFore uses the same pooling).
- MMLU (25 tok) gives the highest ρ among short prompts because task type
  (57 subjects) strongly determines expert selection (DataFore Ob4).
- Prover-math and book are long-prompt points (ρ 0.97–0.98) showing prefill is a
  near-perfect sufficient statistic when the prompt is long enough.
