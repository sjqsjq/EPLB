# Residual fixed-overhead analysis: (a) static-dispatch flag, (b) instrumentation
Date: 2026-10-22 | Author: main session (read-only analysis; coordinator tasking 16:50Z)
Context: post layer-align fix, P-vs-I E2 gap = -8.0% (tps 39.2 vs 42.6). Earlier decomposition:
i_e2 96.0s -> i_e2_static 101.5s (+5.5s flag) -> p_e2_noswap 105.7s (+4.2s instrumentation).
All evidence file:line below refers to NEW tree N=/opt/dsv4_env/lib/python3.11/site-packages/sglang/srt
and OLD tree O=/opt/conda/lib/python3.11/site-packages/sglang/srt (frozen 0.5.6, read-only).

## (a) static-dispatch flag cost (+5.5s on E2, ~5.7%)

### What the flag turns on, per MoE layer per forward (43 layers on DSV4)
1. `ExpertLocationDispatchInfo.init_new(layer_id)` — N/eplb/expert_location_dispatch.py:41-68.
   Call site on the DSV4 path: N/models/deepseek_v2.py:1505-1515 (`forward_deepep`, :1466),
   UNCONDITIONAL (only `is_nextn` gated) — runs in every arm; returns None early (cheap, ~3us)
   when `ep_dispatch_algorithm is None` (I arm), full build when "static".
   Build cost: 2x get_exec(), get_global_expert_location_metadata(), assert, 3 basic-slicing
   tensor views (:53-66), rank_invariant string compare (:52), dataclass alloc (:50) ≈ 15-30us host.
2. Remap gather: `_topk_ids_logical_to_physical_static` :107-112 —
   `info.partial_logical_to_rank_dispatch_physical_map[topk_ids]` advanced-indexing on
   (tokens, 7) ids ≈ 5-10us GPU launch + dtype check. Reached via CUDA branch of
   `_post_process_topk_ids` (N/layers/moe/topk.py:2344-2385): shared/routed split (:2366-2367),
   `_biased_grouped_topk_postprocess` (:1770-1775, @torch.compile(dynamic=True)) — per-call
   dynamo guard evaluation on a FRESH dataclass+view objects every forward adds ~10-20us.
3. Hash layers 0-2: same gather at N/layers/moe/hash_topk.py:271/:294 (3 layers, minor).
4. No sync points found: init_new/remap are views+gather, no .item()/.cpu().
5. Forward-count bounds for E2: prompt_sum=2,182,939 tok (bench_i_e2_r2 perreq), chunk budget
   1024/request, max_prefill_tokens=16384 → 394 gate opportunities (sp_sample_opportunities,
   p_e2_noswap jsonl) to ~2132 chunk-forwards. 5.5s / (43 layers x 394..2132 fw) = 60..256us
   per layer-forward — the top of that range exceeds pure build+gather (~35-65us), i.e. the
   compiled-frame guard churn with fresh object identities is a likely co-contributor.
   NOTE: zero dynamo/recompile warnings in logs/diag_i_e2_static_boot.log (grep=0) — structural
   guards on same-shape views pass, so no recompile storm; overhead is per-call guard evaluation.

### Old-tree comparison (answer to "did the old campaign pay ~0?")
- Mechanism IDENTICAL: O/eplb/expert_location_dispatch.py:36-63 init_new builds the same
  dataclass + 3 slicing views per call (no cache in 0.5.6 either); old call sites
  O/models/deepseek_v2.py:994 and :1161 — per layer per forward; old remap unconditional in
  every topk variant O/layers/moe/topk.py:427/:482/:551/:621/:697/:720.
- The old campaign NEVER ISOLATED the flag: main_table.csv arms = P/F/I/E/A1..A4 — there is no
  I+static cell. Old I arm ran algo=None (init_new early-return ~3us → "paid ~0" is an I-arm
  statement); old P arm paid the FULL build on 94 layers (≈2.2x DSV4's per-forward build count)
  bundled inside its +14.6% E2 win. So the +5.5s is NOT a port regression; the 0.5.22 tree is
  if anything cheaper per layer (43 vs 94) — the flag cost is intrinsic to static dispatch and
  was simply never visible in old-campaign arm deltas.

### Fixable portion → patch_a_dispatch_info_cache.diff
Cache the dataclass per (layer_id, algo) keyed on metadata OBJECT IDENTITY. Safety argument:
every in-tree mutator writes metadata tensors IN PLACE — ExpertLocationMetadata.update
(N/eplb/expert_location.py:312-341, `self_field[...] = torch.where(...)`) used by
expert_location_updater.py:73 and by the frozen controller (_try_finish_pending_swap
controller.py:2466-2500, warmstart :875/:952) — cached slicing views share storage and follow
swap commits transparently. Object REPLACEMENT only via set_global_expert_location_metadata
(:484; broadcast_:510-528) → `is` check rebuilds; strong module-level ref pins the object so
id() cannot be recycled. Side benefit: stable info identity → dynamo guards hit the fast path.
Predicted saving: build (15-30us) + guard stabilization (5-15us) of the 60-256us envelope →
2.2-3.9s of the 5.5s on E2 (40-70%). Optional A2 hunk (env-gated, default OFF): identity-map
fast path skipping the gather in mutator-free arms (+0.4-0.9s, i_static diag only).

## (b) instrumentation cost (+4.2s on E2, ~4.4%)

### Measured decomposition (rank0 [PB-OEPLB-PROF], logs/diag_p_e2_noswap_boot.log, window#24)
calls=14560 record=2992.30ms allreduce=1555.84ms planbuild=18.67ms finalize=0.00ms
total=4566.81ms avg_record_us=205.52 sp_gate_ms=2.51 — matches the 4.2s wall delta almost exactly.
Shares: frozen record_next_layer 66% | frozen window allreduce/confirm 34% | planbuild+sp_gate <1%.

### Old-campaign baseline (same frozen package, Qwen3-235B)
$F/main/m_E2_P_b1/server.log window#8: calls=11562 record=1883.20ms allreduce=1399.56ms
avg_record_us=162.88. → The protocol premium is INTRINSIC (old P arm paid ~3.4s by window#8 and
still won +14.6% because QUIET convergence gains outweighed it). Not a port regression.

### The +42.6us/call DSV4-vs-old delta — attribution CORRECTED (supersedes earlier guess)
- NOT the shared-strip copy: under static dispatch the recorder input is ALREADY CONTIGUOUS.
  Trace: `_post_process_topk_ids` CUDA branch (topk.py:2363-2372) `routed_cols =
  topk_ids[:, :-num_fused_shared_experts]` (non-contiguous view) is passed THROUGH
  `_biased_grouped_topk_postprocess` → `map[topk_ids]` advanced indexing (:108) returns a FRESH
  CONTIGUOUS (tokens,7) tensor; `recorder_topk_ids = routed_cols` (:2372) is that gather output.
  Frozen `flat = topk_ids.reshape(-1)` (controller.py:410) is therefore a free view. A
  contiguous-staging patch would be a NO-OP (~0us/call saving) — see patch proposal file.
- NOT batch size: old fed MORE entries per call — old ≈128.98M valid rows / ~23.1k calls ≈ 5578
  entries/call (tokens x 8) vs DSV4 64.23M / 14560 = 4411 entries/call (tokens x 7)
  (sp_valid_rows sums from both v2p1_rank0.jsonl).
- Remaining explanation: per-op host-side drift in the 0.5.22 env (heavier torch dispatcher per
  kernel launch, ~11 launches/call inside frozen record_next_layer controller.py:404-431) and/or
  launch-queue backpressure under the deeper DSV4 pipeline. NOT host-patchable.

### Fixable portion → patch_b_recorder_pregate.diff
1. Hoist the per-call local import + getter (topk.py:2892-2894) into a module-level cached
   helper — runs in BOTH arms on every select_experts call (43 x ~2200 x 2-4us ≈ 0.2-0.4s/run).
2. Pre-gate the hook on the frozen function's own two cheap early-exit conditions
   (`_should_record_this_batch`, controller.py:383; `shape[0] < cfg.min_record_tokens`, :389) so
   non-recording batches skip the Python call entirely: ~88k hook entries vs 14,560 recorded
   calls → ~73k skips x 6-10us ≈ 0.4-0.7s (P arm). Behavior-identical: the frozen fn
   early-returns on exactly these conditions BEFORE any side effect; the capturing check
   (:385-386) stays inside; `_layer_counter` pre-set before a capturing early-return is a no-op
   (align patch re-sets it every recorded call).
Combined B saving ≈ 0.6-1.1s (P arm), ≈0.2-0.4s (I arm). NOT fixable host-side: record
internals 2.99s + allreduce 1.56s (frozen; old campaign paid equivalent) → protocol floor ≈4.6s
≈ 4.5% on E2 wall.

## (3) Is there real rebalance headroom left on DSV4? (Phase 2 framing)
Post-fix fixcheck: h_rate converged 0.0202→0.0117 against frozen theta_h_act=0.0114 — the
controller sits AT its activation boundary; z=0.06-0.29 ≪ z_lo=22.64 (QUIET band); swaps_total=2;
steady-state signature grow_blocked_headroom (same as old campaign). Ceiling on further
rebalance gains ≈ h_rate scale ≈ ~1% tps on the frozen E-streams, versus ~8% fixed protocol
cost — the frozen theta calibration is NOT leaving meaningful money on the table for DSV4:
the boot warmstart (controller.py:850-881) plus 2 committed swaps already capture the
exploitable imbalance of these streams. Honest conclusion: post-fix P ≈ I − 8%, of which
~4.5pts is the frozen-protocol floor (record+allreduce, same floor the old campaign paid) and
~2.5-3.5pts is recoverable via patches A(+B). Phase 2 should therefore (i) A/B patches A/B to
reach P ≈ I − 3..5%, and (ii) frame the P arm's value as drift insurance — the old campaign's
decisive win was P-vs-E under shifting traffic (E4: 2767 vs 1000 tps), not steady-state P-vs-I
(old P-vs-I itself was +14.6/+2.2/-3.6% across E2/E3/E4). Knob re-tuning (theta/kappa refit for
DSV4) is NOT recommended: nothing to gain at h_rate≈theta_act, and re-freezing breaks
comparability with the v2.1 campaign.

## Savings summary (E2 wall, 96-101s scale)
| Patch | File | Predicted | Arms | Risk |
|---|---|---|---|---|
| A  | eplb/expert_location_dispatch.py | 2.2-3.9s | static arms (P, i_static) | LOW-MED |
| A2 | same (env-gated, default OFF)   | +0.4-0.9s | mutator-free only | MED |
| B  | layers/moe/topk.py hook site      | 0.6-1.1s (P), 0.2-0.4s (I) | both | LOW |
| —  | frozen record+allreduce           | 0 (not host-patchable; old campaign paid same) | — | — |
