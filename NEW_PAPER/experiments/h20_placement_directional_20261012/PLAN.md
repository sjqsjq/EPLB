# 8×H20 directional expert-placement experiment

- **Date:** 2026-10-12
- **Status:** pre-experiment protocol; E0/E1 results have not yet been measured
- **Hardware scope:** one 8×NVIDIA H20 node, EP=8
- **Execution scope:** DeepEP normal dispatch/combine + DeepGEMM FP8, eager mode
- **Primary shape:** one routed layer, approximately 4096 tokens in the real replay

This protocol tests placement interventions. It does not extend the previous delay
scan and it does not treat raw route imbalance as compute time.

## 1. Questions

E0 and E1 answer three directional questions:

1. Does reducing raw per-rank assignment imbalance necessarily reduce completion
   time when the logical route is unchanged?
2. Do per-expert 128-row padding and actual grouped-GEMM shapes explain placement
   effects better than raw rank totals?
3. Does a frozen cross-rank expert swap release the observed output critical path?

Online migration cost and break-even are E3 questions and are outside this run.

## 2. Intervention and invariants

For every timed layout, freeze all of the following from one real forward:

- rank-local input hidden states;
- logical top-k expert IDs and top-k weights;
- logical expert weights and all quantization scales;
- source rank of every token;
- valid assignment count, top-k, precision, backend, and launch configuration.

The only intervention is the bijection between logical experts and physical
GPU/local slots. For layout `s`, save both complete maps:

- `p2l[p]`: logical expert stored in physical slot `p`;
- `l2p[l]`: physical slot receiving logical expert `l`.

The replay converts frozen logical top-k IDs through `l2p` before dispatch and
places the matching logical weights/scales in the destination physical slots.
It must not regenerate logical top-k IDs to obtain a desired imbalance ratio.

A legal candidate exchanges two complete experts on different ranks while
preserving 16 physical slots per rank. Weight tensors, scales, and logical/physical
metadata move consistently. Migration, compilation, and warmup are outside every
frozen timing block.

## 3. Path constraints

The new replay is a separate path; the historical readiness experiment remains
unchanged.

- no common 20 us delay floor;
- no differential injected delay;
- no barrier or strict gate between expert compute and combine;
- no PB-OEPLB controller or online placement update;
- no CUDA Graph in the first E0/E1 experiment;
- no claim that `max_i elapsed(start_i, end_i)` is a calibrated global makespan.

## 4. Input and layer freeze

The main result must use a snapshot captured from a real SGLang forward. Synthetic
hidden states or routes may validate tools, but cannot replace the main result.

Before any placement timing:

1. freeze the public prompt/workload manifest and all random seeds;
2. capture rank-local hidden states, logical top-k IDs, and logical top-k weights at
   the actual DeepEP entry point;
3. choose one routed layer using a route-count-only rule fixed in advance (never a
   measured placement gain);
4. resolve its real checkpoint expert tensors and quantization scales;
5. record shapes, dtypes, source ranks, checkpoint revision, source paths, and
   SHA-256 hashes in `inputs/manifest.json`;
6. verify that replaying the identity mapping matches the captured path within the
   frozen numerical tolerance.

If a real snapshot cannot be obtained, stop before labeling any synthetic replay
as the E1 main result.

## 5. E0: observation and instrumentation

### 5.1 Event modes

Implement two modes over the same replay:

- **minimal/main:** per-rank block start/end CUDA events plus one cross-rank
  completion fence after the block;
- **detailed/diagnostic:** additionally record same-GPU events at `dispatch_done`,
  `expert/gather_done`, `combine_input_ready`, and `combine_done`.

All elapsed intervals are formed only from events on the same GPU. Detailed events
are never subtracted across devices.

A CUDA-IPC cross-device wait prototype is prohibited on this environment because
it caused process SIGSEGV. The safe common-reference diagnostic is:

1. warm and execute an untimed NCCL start collective;
2. record the reference start event on GPU 0;
3. execute the frozen block on every rank;
4. enqueue one NCCL reduce after each rank's local block completion;
5. record the reference end event on GPU 0 after that reduce.

This produces a GPU-0 same-clock **completion-fenced block interval** that includes
one small end-fence cost. Save and report the warmed empty-fence distribution; do
not silently call it a pure layer makespan. The independent
`max_i local_block_elapsed_i` quantity is reported only as `T_proxy`.

### 5.2 Observer-effect pilot

For identity and one predesignated representative swap, compare minimal and
detailed modes using at least 10 randomized blocks × 50 forwards. Save both the
absolute timing shift and the within-layout swap effect. Use detailed mode only
for representative diagnostics if instrumentation changes the placement effect
materially.

### 5.3 Timing semantics

Per rank, save:

- local layer/block duration;
- dispatch duration;
- expert/gather duration;
- expert-output-to-combine completion duration;
- combine-input-ready to combine completion duration when exposed by the actual
  path.

If the old `layer - combine_after_gate` quantity is reproduced for comparison,
label it only `gate-completion proxy`; it is not direct expert-ready time.

Actual last-ready identity may be claimed only from the validated completion
reference or an independently validated trace. Uncalibrated GPU timestamps are
never directly subtracted.

## 6. E1: frozen directional swaps

Freeze identity plus exactly 12 complete cross-rank swaps before performance
measurement. Enumerate all legal swaps from the frozen route, compute only
route/shape metadata, then select without timing information:

- **A (3):** raw-r and padded maximum work both improve;
- **B (3):** raw-r improvement comparable to A, but padded improvement is small
  or materially different;
- **C (3):** small raw-r change, but substantial padding/expert-composition change;
- **D (3):** uniformly sample legal swaps from the remaining set using a recorded
  seed.

Save the exact category rule, thresholds, tie-break rule, seed, complete maps,
per-expert counts, 128-row aligned counts, per-rank totals, and source→destination
assignment matrices in `placements/manifest.json` before timing.

If a category contains fewer than three legal candidates, report that fact and
fall back to a predefined continuous stratified sample. Never modify the logical
route to populate a category.

## 7. Correctness gate

Use distinct real checkpoint expert weights. For each layout, compare the output
for the identical frozen input against identity and save:

- max absolute error;
- max relative error (with its denominator convention recorded);
- the elementwise `isclose` failure count under the frozen tolerance;
- NaN and Inf counts;
- mapping/inverse and slot-occupancy checks.

Freeze numerical `atol`/`rtol` after the correctness-only pilot and before any
performance timing. FP8 reduction order can change, so bitwise equality is not a
requirement. Any over-tolerance layout is first a correctness/mapping failure; the
tolerance is not relaxed because its timing is favorable.

## 8. Timing and statistics

### Pilot

- at least 10 blocks × 50 forwards per condition;
- randomized interleaving of all frozen layouts;
- identical warmup after every placement activation;
- weight migration and setup excluded from timed blocks.

### Main recommendation after the pilot

- 4 independent starts × 5 blocks × 50 forwards = 20 blocks × 50 per layout;
- preserve all signed effects, including negative values;
- primary estimate: block-paired mean difference from identity;
- median difference: robustness supplement only;
- increase independent blocks/restarts rather than only inner-loop iterations.

Define the frozen-layout effect as

\[
Q(s)=T_{\mathrm{identity}}-T_{\mathrm{layout}(s)}.
\]

Positive `Q` means the layout is faster; negative `Q` means it is slower. Weight
migration is excluded. Where the completion reference is valid, also report

\[
g(s)=\max_iR_i^{\mathrm{before}}-\max_iR_i^{\mathrm{after}(s)}.
\]

Freeze the meaningful-effect tolerance after the pilot and before the main run:

\[
\epsilon=\max(0.01\,T_{\mathrm{reference}},\ 2\ \mu s).
\]

A result is practically negligible only when its confidence interval lies wholly
inside `[-epsilon, +epsilon]`. An interval merely crossing zero means the data are
compatible with zero, not that equivalence has been established.

## 9. Descriptive models

E1 records inputs needed for the later E2 comparison but does not fit a model to
select E1 candidates:

- M0: raw per-rank assignment counts;
- M1: sum of independently 128-row-aligned per-expert counts;
- M2: grouped-GEMM cost calibrated separately on the actual expert shapes;
- optional M3: M2 plus independently calibrated communication/dependency terms.

No model may be calibrated on the same layout effects it is evaluated against
without an explicit train/test split.

## 10. Raw-data and audit policy

Keep raw block/iteration data and representative traces. Files over 100 MiB are
not committed directly to Git. They must be placed in a fetchable archive or
release asset with SHA-256, byte count, URL, schema, and extraction instructions
recorded by `results/raw_index.json` and `traces/index.json`. “Rerun to regenerate”
alone is not sufficient for independent audit.

## 11. Stop conditions

Stop E1 performance interpretation if any of the following holds:

- the input is not a real forward snapshot;
- identity replay does not match the captured path within the frozen tolerance;
- a map is not bijective or does not preserve 16 slots per rank;
- route IDs/weights change logically between layouts;
- migration, compilation, warmup, delay injection, or a mid-layer gate enters the
  timed interval;
- the observation mode materially changes the directional layout effect and no
  low-observer-effect primary mode remains.

No E0/E1 result or paper claim is present in this file; those are written only
after the frozen protocol passes its correctness and audit gates.
