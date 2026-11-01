# Plan B Feasibility Assessment — Adapting DeepSeek-V4-Flash to sglang 0.5.6 (frozen-deployment lineage)

- Date: 2026-10-22 (UTC)
- Author: assessment fork (read-only investigation)
- Method: pure filesystem inspection of /opt/conda (frozen, untouched), /opt/dsv4_env, /root/models/DeepSeek-V4-Flash, and this experiment dir. No server booted, no CUDA executed, no writes outside this file. `nvidia-smi` used once, memory-query only (all 8 GPUs 90–93 GB / 97.8 GB in use — live campaign confirmed).
- Trigger: author contingency — "if the 0.5.22 path cannot reach parity, adapt the NEW model to the OLD sglang (0.5.6)".

## VERDICT: **INFEASIBLE** (as a parity rescue within the campaign horizon)

Adapting DSV4 to a 0.5.6-lineage clone is not a model-file port: it is a **whole-runtime port of the 0.5.22 DSV4 stack (≥30–45k LOC across 45–70 files, incl. a 28 MB JIT-kernel package) onto a torch 2.9.1+cu128 / flashinfer 0.5.3 / sgl-kernel 0.3.19 base that lacks every substrate DSV4 needs** (compressed-KV memory pool, DSV4 attention backend, Engram/WKV, HashTopK, MHC kernels, breakable CUDA graphs, TBO). The kernel-ABI gates are rated CRITICAL and possibly unsolvable without swapping torch — which pulls the entire 0.5.22 dependency train back in. Decisively, the benefit **self-cancels**: the pb_oeplb "native integration" advantage lives in 0.5.6's `topk.py`/`model_runner.py` hot path — the exact files the DSV4 port must replace with 0.5.22-lineage code. The end state would be ≈ the already-validated 0.5.22 port, but on older, less-tested infrastructure.

The residual parity gap on 0.5.22 (−8.0% post-fix on E2) is **fully attributed** to two hot-path mechanisms — static-dispatch remap (+5.5s) and record instrumentation (+4.2s) — both of which exist in cheaper 0.5.6 forms that can be **micro-backported into the working 0.5.22 port in 2–3 files (days), not 45–70 files (weeks)**. Recommended: "Plan A+" (§9), not Plan B.

If Plan B is nonetheless mandated, the honest label is FEASIBLE-WITH-BACKPORT at 3–6 engineer-weeks + ~1 week machine validation, with an abort probability dominated by Risks R1/R2 below.

---

## 1. Old tree inventory (/opt/conda — Q1)

| Item | Value |
|---|---|
| sglang | **0.5.6.post2** (`sglang/version.py`; dist-info confirms), python 3.11.12 |
| torch | **2.9.1+cu128** (`torch/version.py`: cuda 12.8) — CUDA 12.8 runtime, ran on this host's driver 580.105.08 for the whole Qwen3 campaign |
| transformers | **4.57.1** — models/: deepseek_v2, deepseek_v3, deepseek_vl(_hybrid). **No deepseek_v4** (new env's 5.19.0 has it) |
| flashinfer | 0.5.3 (+cubin 0.5.3) |
| sgl-kernel | 0.3.19 (has SM100-oriented `cutlass_scaled_fp4_mm`/`cutlass_fp4_group_mm`/`scaled_fp4_quant`; **no `hash_topk` op**) |
| deep_ep / deep_gemm | **editable builds**: deep_ep 1.2.1+9af0e0d → /workspace/build/DeepEP, deep_gemm 2.3.0 → /workspace/build/DeepGEMM (both source dirs still present) |
| nvshmem | nvidia-nvshmem-cu12 **3.3.20** (the "tolerant transport" of the old campaign) |
| triton / torchao | 3.5.1 / 0.9.0 |
| env size | 11 GB (clone is trivially cheap disk-wise; / has 1.6 TB free) |

**DeepSeek architectures supported** (`EntryClass` scan; registry = dynamic `EntryClass` discovery in `srt/models/registry.py`):
`DeepseekV2ForCausalLM`, `DeepseekV3ForCausalLM`, **`DeepseekV32ForCausalLM`** (all in deepseek_v2.py:3997), `DeepseekV3ForCausalLMNextN` (deepseek_nextn.py), `DeepseekForCausalLM`, VL2, OCR.
**DeepseekV4 / deepseek_v4: zero hits anywhere in the tree (grep).** MLA backends present (flashmla, cutlass_mla, flashinfer_mla, trtllm_mla) and V3.2-DSA substrate exists (`nsa/nsa_indexer.py` `Indexer`, nsa_backend, hybrid_attn_backend, indexer-aware memory_pool) — but this is V3.2-generation DSA (full-KV + lightning indexer), **not** DSV4's compressed-KV scheme.

## 2. Model requirements (/root/models/DeepSeek-V4-Flash — Q2)

149 GB, 46 safetensors shards. config.json highlights:

- `architectures: ["DeepseekV4ForCausalLM"]`, `model_type: deepseek_v4`; hidden 4096, **43 layers**, vocab 129280, MTP `num_nextn_predict_layers: 1`, YaRN ×16 → 1M ctx.
- **`num_hash_layers: 3`** → layers 0–2 route via **HashTopK** and never call `select_experts` (host: new deepseek_v2.py:679–680 `is_hash = layer_id < n_hash_layers`, :733 `self.topk = HashTopK(...)`; kernel: `sgl_kernel.hash_topk` [0.4.9-only] or JIT `sglang.kernels.ops.moe.dsv4`). All 43 layers still own routed weights (p2l keys 0..42) — confirmed by Phase-1e forensics.
- MoE: **256 routed + 1 shared, top-6**; `topk_method: noaux_tc`, **`scoring_func: sqrtsoftplus`** (18 occurrences in new topk.py; **0 in old topk.py** — old only has softmax/sigmoid); `n_group=8`/`topk_group=8` are config-class defaults (srt/configs/deepseek_v4.py:68/:94 — group limit present but vacuous at 8-of-8; the group-scoring code path still runs); `routed_scaling_factor: 1.5`.
- Quantization: **non-experts FP8** block [128,128] e4m3/`scale_fmt: ue8m0` dynamic (old tree CAN do this: `w8a8_block_fp8` in 0.5.6 fp8.py); **experts packed MXFP4** (`expert_dtype: "fp4"`; per-expert `ffn.experts.N.w{1,2,3}.weight/.scale` in the index).
  - **SM90 constraint (decisive)**: under deepep the auto MoE runner = deep_gemm masked, and packed-FP4 masked GEMM asserts SM100/120 (deepgemm layout.hpp:73) → H20 dead. The working path is **`SGLANG_DSV4_FP4_DEQUANT=1`**: load-time per-expert e2m1→e4m3 cast (new fp8.py:210 `cast_e2m1fn_to_e4m3fn`, call sites :1758/:2091, runner-compat assert :421) → `grouped_gemm_nt_f8f8bf16_masked`. The flashinfer SM90 CUTLASS **W4A16** mxfp4 runner is selected only when `a2a==none` (arg_groups/model_overrides/deepseek_v4.py:77–89).
- Attention: **entirely new layout** — weights `attn.wq_a/wq_b/wo_a/wo_b/wkv/attn_sink` (+ q/k norm), `head_dim: 512`, `o_lora_rank: 1024`, `o_groups: 8`; **C4 indexer** (`index_topk: 512`, `index_head_dim: 128`, `index_n_heads: 64`); `sliding_window: 128`; **per-layer KV compression** `compress_ratios: [0,0,4,128,4,128,…]` + `compress_rope_theta` → requires a bespoke compressed-KV pool; **MHC** hyper-connections (`hc_mult: 4`, `hc_sinkhorn_iters: 20`); **Engram/WKV** memory (`attn.wkv.*` weights; layers/engram.py `EngramHasher`). Nothing in this paragraph maps onto any 0.5.6 module.
- Tokenizer: standard tokenizer.json (DeepSeek BOS/EOS) — loadable by 4.57.1 fast tokenizer, but config parsing (`model_type: deepseek_v4`) needs sglang's own `srt/configs/deepseek_v4.py` (missing in old tree) and practically transformers 5.x (see R5).

**Feature checklist DSV4 model code needs** (all verified present only in 0.5.22): MXFP4 expert loading + `SGLANG_DSV4_FP4_DEQUANT` cast path; HashTopK routing layers; group-limited/noaux_tc+sqrtsoftplus topk; compressed-KV memory pool + DSV4 attention backend; Engram; MHC fused kernels; DP-attention + deepep(2.1.0) a2a; breakable CUDA graph / TBO / batch_overlap; `sglang.kernels` JIT op package; MTP NextN.

## 3. Backport closure: new tree → old tree (Q3)

`deepseek_v4.py` = **5715 LOC with 105 sglang imports**; family total: v4 + mhc(768) + nextn(276) + dspark(1203) + v41_vit(151) = **8113 LOC**, plus `models/deepseek_common/` **6888 LOC**. Tree-wide DSV4 footprint: **159 files under srt/ + 79 files under sglang/kernels/** reference dsv4/deepseek_v4.

Module-existence spot check (0.5.6 vs required):

| Required by DSV4 (0.5.22 path) | LOC/size | In 0.5.6? |
|---|---|---|
| models/deepseek_v4*.py, deepseek_common/ | ~15k | ✗ |
| models/deepseek_v2.py (rewritten; hosts is_hash/HashTopK; `DeepseekV2MoE` is DSV4's MoE block) | 3517 (old 3997, diverged) | ✗ (incompatible rewrite) |
| configs/deepseek_v4.py | 138 | ✗ |
| layers/attention/dsv4/ (compressor ×3, C4Indexer, dsv41_sparse, metadata, sparse_prefill_utils, v41_indexer) | 596 KB (~4.8k+) | ✗ |
| layers/attention/deepseek_v4_backend.py (+trtllm, hip_radix) | 4466 (+553) | ✗ |
| mem_cache/deepseek_v4_memory_pool.py + deepseek_v4_compress_state.py | 2140 + 281 | ✗ (no compressed-KV concept in old memory_pool) |
| layers/engram.py | 1025 | ✗ |
| layers/moe/hash_topk.py | 316 | ✗ (+ needs `sgl_kernel.hash_topk` = 0.4.9-only, or JIT kernels pkg) |
| layers/quantization/mxfp4_flashinfer_trtllm_moe.py, mxfp8_input.py | — | ✗ (old mxfp4.py = gpt-oss triton_kernels path, 870 LOC, unrelated) |
| layers/quantization/fp8.py dequant path (`cast_e2m1fn_to_e4m3fn`, DEQUANT env, masked-fp8 runner asserts) | new fp8.py ≫ old 1491 | ✗ |
| **sglang/kernels/** (JIT csrc+asm: dsv4 wo_a rope/gemm, mhc layernorm ops, mxfp8 epilogue, moe/dsv4.py, expert_pack_mxfp4, deepep_waterfill) | **28 MB**; 34 csrc files for v4 alone | ✗ (package does not exist) |
| model_executor/runner.py, cuda_graph_config.py, runner_backend_utils/breakable_cuda_graph | — | ✗ (old = monolithic model_runner.py, 2911 LOC) |
| layers/cp/ (cp_decode_attn_tp, interleave), layer_boundary | — | ✗ |
| compilation `register_split_op` | — | ✗ (dir exists, symbol absent) |
| hardware_backend/npu/dsv4 | — | ✗ (npu tree exists, no dsv4) |
| batch_overlap/two_batch_overlap, runtime_context, mega_moe, batch_invariant_ops, utils/custom_op | — | ✗ |
| layers/moe/topk.py | 1038 → **2967**; `select_experts` :855 → :2545 (signature+struct churn; StandardTopKOutput exists both) | ~ (heavily diverged) |
| layers/dp_attention.py | 572 → **1266** | ~ (diverged) |
| eplb/ (expert_location_dispatch, topk_ids_logical_to_physical) | present both | ✓ (API drift moderate) |
| token_dispatcher/deepep.py | present both | ✓ (but written against deep_ep **1.2.1** vs DSV4-validated **2.1.0**) |
| managers/pb_oeplb/ | native | ✓ |

Binary/kernel drift (cannot be file-copied): flashinfer 0.5.3 `fused_moe/` = 5 files (core, fused_routing_dsv3, utils) vs 0.7.1 = 20+ (**hash_topk.py router, sparse_mqa.py, auto_candidates, backends/, da_moe, cute_dsl…**) — 0.5.3 does have an early SM90 cutlass fused-MoE + fp4 quant module, but **not** the API surface the DSV4 wrappers call; sgl-kernel 0.3.19 → sglang-kernel 0.4.9; deep_gemm 2.3.0 (`m_grouped_fp8_gemm_nt_masked`) → sgl-deep-gemm 0.2.2 (`m_grouped_fp8_fp4_gemm_nt_masked` + `recipe`/`disable_ue8m0_cast` params); deep_ep 1.2.1 → 2.1.0.

## 4. pb_oeplb native integration in 0.5.6 (Q4)

Confirmed native and lean:
- Package: `srt/managers/pb_oeplb/` (config.py, controller.py, window_v2p1.py, __init__.py + locked backups pre_v21/legacy_eav1).
- Record hook: **topk.py:986–990** — `get_global_expert_distribution_recorder().on_select_experts(...)` then `_ctrl = get_pb_oeplb_controller(); if _ctrl is not None: _ctrl.record_next_layer(topk_ids)` at the tail of `select_experts`.
- Lifecycle: model_runner.py:440–444 (construct under `server_args.enable_pb_oeplb`, non-draft workers only), :2663–2685 (`on_forward_pass_start/end`, lockstep-unconditional end hook), imports :81–83.
- CLI: server_args.py:441–455 (15 `pb_oeplb_*` fields); `ep_dispatch_algorithm` auto→`static` at :1674–1676.
- Controller slot attribution is round-robin `_layer_offset + _layer_counter % _num_moe_layers` (controller.py:404) — **identical frozen code in both trees**.

**Caveat (tree-independent)**: DSV4's 3 HashTopK layers never call `select_experts` in EITHER tree → 40 records/forward vs 43 slots → the same `_layer_counter = layer_id − _layer_offset` alignment patch (validated on 0.5.22: s_change 0.3→0.0006–0.002, z 39–52→0.07–0.29, swaps 25→2, tps 34.5→39.2, gap −19.1%→−8.0%; sha_pre 914e1ed5→sha_post 354b0482) would be required in a 0.5.6 clone too. A DSV4 backport does **not** inherit a free ride on layer attribution.

**The self-cancelling benefit**: native hooks live in old `topk.py`/`model_runner.py`/`server_args.py`. Porting DSV4 forces replacing topk.py (for sqrtsoftplus/new select_experts/HashTopK plumbing), deepseek_v2.py (MoE host), model_runner (compressed-KV pool, dsv4 backend, breakable graphs, TBO) with 0.5.22-lineage code — after which the hooks must be re-derived on the ported hot path, i.e. exactly the state already achieved, patched, and benchmarked in /opt/dsv4_env.

## 5. Environment / kernel risks for a 0.5.6-clone venv on H20/SM90 + CUDA 13 host (Q5)

- **Compute stack baseline is proven**: torch 2.9.1+cu128 + deep_ep 1.2.1 + nvshmem-cu12 3.3.20 ran the entire old campaign on this same 8×H20/driver-580 host; cu12 runtime on a CUDA 13.0 driver is supported (backward compat). Clone mechanics: `cp -a /opt/conda /opt/<clone>` (11 GB) + re-point editable pth's at /workspace/build/{DeepEP,DeepGEMM} (sources intact) — LOW risk, ~half a day.
- **Toolchain gate (CRITICAL)**: the new DSV4 kernel stack JIT-compiles with a **≥12.9 nvcc requirement** (deep_gemm assert, ENV_NOTES try1: all 8 ranks died on nvcc 12.8). Host system nvcc = **12.8** (verified `nvcc --version` V12.8.61). dsv4_env solved this with venv-local nvidia-cuda-nvcc/nvvm **13.0.88** + cu13 runtime headers. A cu128-torch clone running nvcc-13 JIT hits the CCCL "compiler vs toolkit headers incompatible" class (ENV_NOTES try2/3) with **no clean exit**: upgrading the clone's runtime to cu13 means upgrading torch → torch 2.14 → ABI-matches 0.5.22's wheels → the clone converges to dsv4_env and the point of Plan B is lost.
- **MXFP4/FP8 on SM90 in 0.5.6-era code**: old fp8.py has blockwise FP8 (fine for non-experts). Old mxfp4.py targets gpt-oss triton_kernels/AMD, **not** DSV4's packed experts. flashinfer 0.5.3 has `gen_fp4_quantization_sm90_module` + a SM90 cutlass fused-MoE module, but the DSV4 wrappers (mxfp4_flashinfer_trtllm_moe, auto-runner selection) call 0.7.x APIs; DEQUANT→masked-FP8 route needs sgl-deep-gemm-0.2.2-style API on a deep_gemm-2.3.0 base (name/signature shim needed; masked fp8 grouped GEMM itself EXISTS in 2.3.0 — the one favorable finding). No SM90 W4A16 CUTLASS path without flashinfer upgrade (JIT-based, might work on torch 2.9 — unproven, MEDIUM-HIGH).
- **sgl-kernel**: 0.3.19 prebuilt (torch 2.9 ABI) lacks DSV4 ops (`hash_topk` etc.); 0.4.9 wheels are torch-2.14/cu130 ABI → not importable in a clone; source rebuild against torch 2.9 headers = uncharted (HIGH).
- **transformers**: clone needs 4.57.1 → 5.19.0 for `deepseek_v4` config/tokenizer plumbing (0.5.6's hf_transformers_utils assumes 4.x; collateral breakage acceptable in a dedicated clone, MEDIUM).
- **deepep/nvshmem**: clone keeps deep_ep 1.2.1 + nvshmem 3.3.20-cu12 (tolerant transport). But DSV4's dispatcher usage was validated against deep_ep **2.1.0**; 0.5.6's deepep.py dispatcher + DSV4 a2a expectations (waterfill kernels etc.) = compatibility unknown (MEDIUM-HIGH).
- **Validation latency**: GPUs saturated (90–93/97.8 GB) with the live fix-matrix; any clone boot validation queues behind the campaign.

## 6. Concrete minimal path IF FORCED (Plan B execution sketch)

1. `cp -a /opt/conda /opt/dsv4_old_clone` (~11 GB, 0.5 d); fix editable pth's; pip index = tencent (per ENV_NOTES). (LOW)
2. transformers → 5.19.0 in clone. (MEDIUM)
3. Copy leaf/model layer: configs/deepseek_v4.py + weight_utils probe; models/deepseek_v4*.py + deepseek_common/ + new deepseek_v2.py; arg_groups/model_overrides/deepseek_v4.py. (~16k LOC) (HIGH — recursive imports)
4. Copy runtime subsystems: layers/attention/dsv4/ + deepseek_v4_backend*; mem_cache/deepseek_v4_{memory_pool,compress_state}.py + scheduler/schedule_batch/memory_pool integration patches; layers/engram.py; layers/moe/hash_topk.py; fp8.py DEQUANT path + mxfp4/mxfp8 wrappers; dp_attention deltas. (CRITICAL — no 0.5.6 substrate for compressed KV / SWA bounded replay)
5. Kernel wall: sglang/kernels package + JIT csrc (34 v4 files) + sgl-kernel 0.4.9-only ops + flashinfer 0.7.1 surface, all against torch 2.9.1/cu128 with venv-local nvcc 13.0 — or rewrite each op call to a 0.3.19/0.5.3 equivalent where one exists (most don't). (CRITICAL — dominant abort risk)
6. model_runner surgery: port runner.py split / cuda_graph_config / breakable_cuda_graph / compilation register_split_op / TBO-batch_overlap into the 2911-LOC 0.5.6 model_runner. (CRITICAL)
7. pb_oeplb: inherit native package + hooks; apply the **same** `_layer_counter` alignment patch inside the ported topk hot path; re-gate `ExpertLocationDispatchInfo.init_new` construction in ported deepseek_v2.py to `enable_pb_oeplb or pb-active`. (LOW — the only easy item)
8. Re-run full validation: smoke → fp-preservation → swap checksums/mirror symmetry → I-vs-P matrix (methodology already exists). (1 week machine time, queued behind campaign)

**Effort estimate (Plan B)**: 45–70 files, 30–45k LOC ported/rewritten, **3–6 engineer-weeks** + ~1 week validation; success gated on step 5/6 (estimated 40–60% without a torch swap, which voids the premise).
**Effort estimate (Plan A+ alternative, §9)**: 2–3 files, ~50–300 LOC, **1–3 days** + one matrix re-run.

## 7. Top 5 risks (Plan B)

- **R1 (CRITICAL) Kernel/toolchain ABI wall**: sglang-kernel 0.4.9 + flashinfer 0.7.1 API surface + sglang.kernels JIT csrc require torch 2.14/cu130-era ABIs and nvcc ≥12.9; clone base is torch 2.9.1/cu128 with host nvcc 12.8. Every escape route (upgrade torch, upgrade flashinfer, rebuild kernels) either is unproven on the old base or collapses the clone back into 0.5.22.
- **R2 (CRITICAL) No memory/attention substrate**: compressed-KV pool (compress_ratios 4/128), C4 indexer, SWA bounded replay, breakable CUDA graphs, TBO have zero 0.5.6 foundation; this is scheduler+mem_cache+model_runner surgery, and it cannot be validated until GPUs free up.
- **R3 (HIGH, strategic) Self-cancelling benefit**: the port replaces the very hot-path files that host the native pb_oeplb integration; end state ≈ current validated 0.5.22 port on older infrastructure, with the frozen-campaign calibration (Qwen3, near-zero overhead) NOT automatically reproduced because DSV4's remap/record shapes (256 experts, top-6, 43 layers, DEQUANT tensor layouts) differ from Qwen3's regardless of tree.
- **R4 (HIGH) Recursive dependency closure**: deepseek_v4.py's 105 sglang imports chain into runtime_context, mega_moe, batch_overlap, compilation split-ops, custom_op registry, models/utils WeightsMapper, hardware_backend.npu shims… each absent or diverged in 0.5.6; the 159-file srt footprint is the honest lower bound of touched surface.
- **R5 (MEDIUM-HIGH) Numerics/validation debt**: transformers 4.57→5.19 jump, re-derived weight loading (probe_routed_expert_weight_dtype), deep_ep 1.2.1 vs 2.1.0 dispatch semantics → the entire fp-preservation + swap-integrity + matrix validation corpus must be re-run from scratch on an unproven stack, while the 0.5.22 arm already has all of it green (48/48 checksums, 56/56 mirror pairs, token-identical cross-arm probes, post-fix QUIET convergence with 2 swaps).

## 8. What would NOT need redoing (either path)

- **Frozen pb_oeplb v2.1 package**: byte-identical reuse already demonstrated by the 0.5.22 port (12/13 files identical; sole documented delta = fast_metadata `ep_size` field, old semantics preserved). Controller/window/rebalancer/swapper need no changes for Plan B beyond the host hooks.
- **`_layer_counter` alignment fix**: design is tree-independent (round-robin controller identical in both trees); the validated patch block ports as-is.
- **Streams / protocol / campaign harness**: bench_client_v2, frozen E2/E3/E4 stream shas, run_e2_bench.sh / run_ip_matrix.sh / diag-cell scripts, watchers, LEDGER_dsv4.tsv discipline, fresh-boot-per-cell rule, frozen V21 knob envs (PB_OEPLB_*/OEPLB_V21_*) — all reusable verbatim.
- **Environment fix corpus**: NVSHMEM_HCA_LIST=mlx5_bogus P2P-only trick, mem-fraction ceilings (0.80 identity / 0.75 PB), DG_JIT cache pre-warm (compile_dg.sh), tencent pip mirror + nvcc pin recipe, cu13 .so symlink fixes — knowledge transfers to any new venv.
- **Verification methodology**: VERIFY-WEIGHT-MOVE checksums, DIAG-2 mirror symmetry, fp-preservation probes (fp2), old-campaign calibration baselines (Qwen3 main_table.csv medians).
- **Diagnosis assets**: the complete root-cause chain (attribution scramble → s_change/z pinning → planner churn → −450ms/swap-window) and the diag decomposition (static +5.5s / instrumentation +4.2s / swaps) — this is precisely what makes Plan A+ targetable.

## 9. Recommendation — Plan A+ instead of Plan B

The post-fix 0.5.22 residual (−8.0% E2) is exactly `static-dispatch flag (+5.5s) + instrumentation (+4.2s)`; the old campaign paid ≈0 for both **with the same frozen controller**, so the delta lives in 0.5.22's hot-path plumbing, not in pb_oeplb. Targeted micro-backports FROM 0.5.6 INTO dsv4_env (mirror image of Plan B, ~1/20th the surface):

1. **Record path**: restore old-hook input semantics at topk.py:2888–2912 (avoid post-remap routed-column slicing / extra tensor ops per layer per forward; old hook passes `topk_ids` directly with zero measurable cost). Keep the `_layer_counter` alignment.
2. **Static-dispatch remap**: diff 0.5.6's `topk_ids_logical_to_physical` / `ExpertLocationDispatchInfo.init_new` application against 0.5.22's `_post_process_topk_ids` (:2307–:2434, incl. `_eplb_remap_enabled` gating) and port the cheaper implementation/caching into the DSV4 path (both trees share the same eplb primitives, so this is a contained 1–2 file change).
3. Then proceed with the already-planned **Phase-2 knob recalibration** (frozen V21 knobs were fitted on Qwen3 traffic; DSV4 h_rate/z statistics differ) and re-run the fix matrix (r3/r4 already in flight: p_fix_e3_r1, p_fix_e4_r1 rc=0; p_fix_e2_r2 started 17:09Z).

**Decision gate**: if after (1)+(2) the P-vs-I residual is within campaign noise (old-campaign band was +14.6/+2.2/−3.6% tps), parity is declared on 0.5.22 and Plan B is moot. If a residual >5% survives both micro-backports AND recalibration, the remaining suspect is DSV4-intrinsic (43-layer × 256-expert remap shape) and would be paid in a 0.5.6 clone too — again making Plan B pointless. **There is no realistic branch in which Plan B is the cheapest route to parity.**

---

### Evidence index (file:line)

- Old tree: /opt/conda/lib/python3.11/site-packages/sglang/version.py (0.5.6.post2); torch/version.py (2.9.1+cu128); srt/models/deepseek_v2.py:3997 (EntryClass V2/V3/V32); srt/layers/moe/topk.py:986–990 (native record hook), :46–47/:193 (dispatch primitives); srt/model_executor/model_runner.py:81–83/:440–444/:2663–2685; srt/server_args.py:441–455/:1674–1676; srt/managers/pb_oeplb/ (package + backups); pip: flashinfer 0.5.3, sgl-kernel 0.3.19, nvshmem-cu12 3.3.20, editable deep_ep 1.2.1 (/workspace/build/DeepEP), deep_gemm 2.3.0 (/workspace/build/DeepGEMM, masked fp8/fp4 GEMM in deep_gemm/__init__.py:38–72); transformers/models/ (no deepseek_v4); host nvcc V12.8.61.
- Model: /root/models/DeepSeek-V4-Flash/config.json (all fields cited in §2); model.safetensors.index.json (layers.0.attn.{wq_a,wq_b,wo_a,wo_b,wkv,attn_sink}, layers.0.ffn.experts.N.w{1,2,3}.{weight,scale}); tokenizer_config.json (DeepSeek BOS/EOS).
- New tree: /opt/dsv4_env/.../sglang/srt/models/deepseek_v4.py (5715 LOC; EntryClass :5579; import block :1–~200); models/deepseek_v2.py:679–680/:733 (is_hash/HashTopK); layers/moe/topk.py:2888–2912 (ported hook + `_layer_counter = layer_id − _pb_off`), :2307/:2545; layers/moe/hash_topk.py:194/:235 (sgl_kernel.hash_topk | kernels.ops.moe.dsv4); layers/quantization/fp8.py:210/:421/:1758/:2091 (DEQUANT); configs/deepseek_v4.py:60/:68/:94/:104–105 (index_topk, n_group, topk_group, compress); arg_groups/model_overrides/deepseek_v4.py:77–89 (mxfp4 runner selection); mem_cache/deepseek_v4_memory_pool.py (2140); layers/attention/deepseek_v4_backend.py (4466); layers/attention/dsv4/ (596 KB); layers/engram.py (1025); sglang/kernels/ (28 MB; jit/csrc/deepseek_v4 = 34 files); pip: torch 2.14.1, flashinfer 0.7.1 (fused_moe/hash_topk.py, sparse_mqa.py), sglang-kernel 0.4.9, sgl-deep-ep 0.1.2.post1, sgl-deep-gemm 0.2.2, transformers 5.19.0.
- Campaign: ENV_NOTES.md (Phase 1/1b/1c/1d/1e incl. root cause + fix verification), port/PORT_NOTES.md (13-file package staging, 4 host patches, launch-recipe deltas, MXFP4-swapper compat + addenda), bench_matrix_final_table.md (pre-fix −17..−19%), LEDGER_dsv4.tsv (fix-matrix r3/r4 in flight), scripts/ (boot/bench harness).
