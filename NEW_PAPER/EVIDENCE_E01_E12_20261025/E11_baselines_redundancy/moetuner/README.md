# MoETuner Baseline (for PB-OEPLB Paper Reproduction)

Reproduction of **MoETuner** (Go & Mahajan, arXiv:2502.06643, Feb 2025) as
an offline static baseline for PB-OEPLB.

The original MoETuner patches Megatron-LM; here we reuse the exact same
SGLang stack as PB-OEPLB (`ep_dispatch_algorithm='static'` with a custom
`init_expert_location` permutation) so the two baselines share every
runtime variable except the *placement policy itself*.

## Pipeline

```
(A) profile        →  (B) ILP1              →  (C) emit                 →  (D) bench
    dump P[l,e]        per-layer balanced       init_expert_location.json    2 runs on
    topk_ids trace     clustering (Gurobi)      + greedy cluster→GPU         same rig
                                                                             as PB-OEPLB
```

## Deviations from the paper

1. **No ILP2** — the paper's ILP2 minimizes cross-layer inter-cluster
   communication weighted by `B[g1,g2]`. On a **single 8×H20 node with
   uniform NVLink**, `B[g1,g2]` collapses to a constant, so ILP2's
   objective reduces to "minimize max cross-GPU token volume per layer",
   which is *exactly* what ILP1 already balances. We use greedy assignment
   for cluster→GPU (largest-load cluster to lowest-loaded GPU) and note
   this is the paper's single-node case degenerate solution. This also
   avoids the Gurobi 2000-var restricted-license ceiling.

2. **Added explicit cluster-size constraint** — the paper's printed ILP1
   omits `Σ_e x[c,e,l] = E/G`. Without it the solver returns the
   degenerate "all experts in one cluster" solution. We add it.

3. **Same corpus, disjoint split** — profile on first 2048 requests of
   `L512_O1_realprover_n8192`, benchmark on last 6144 (or full 8192 as a
   *train-set-leaked upper bound*, reported alongside for context).

## Files

```
src/
  profile_hook.py       - SGLang topk.py drop-in that dumps P/R tensors
  solve_ilp.py          - Gurobi ILP1 solver + greedy cluster->GPU
  emit_placement.py     - writes SGLang init_expert_location JSON
configs/
  qwen3_235b_ep8.yaml   - model topology + solver settings
artifacts/
  profile.npz           - filled by (A)
  placement.json        - filled by (C)
logs/
  ilp_solve.log         - Gurobi stdout
```

## Reference

Go, S. & Mahajan, D. "MoETuner: Optimized Mixture of Expert Serving with
Balanced Expert Placement and Token Routing." arXiv:2502.06643, 2025.
