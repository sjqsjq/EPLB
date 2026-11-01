# H20 placement follow-up protocol v1

**Status:** design frozen before the first follow-up performance measurement

**Scope:** one node, exactly 8 × NVIDIA H20, EP=8, DeepEP normal dispatch/combine,
DeepGEMM FP8, Qwen3-235B-A22B-FP8 geometry. No result in this protocol is to be
generalized to another accelerator, EP size, backend, precision, or model.

The original E0/E1 pilot is immutable. Its snapshots, placement manifest,
correctness tolerance, raw records, analyzer, and summaries are not regenerated or
overwritten. Follow-up artifacts use `followup_v1` / `confirmatory_v1` paths.

## Global invariants

Every replay freezes rank-local hidden states, logical top-k IDs, logical top-k
weights, logical FP8 expert weights and scales, source rank, assignment count, and
top-k. A placement is a bijection between 128 logical experts and 128 physical
slots, with exactly 16 slots per rank. Weights and every associated scale tensor
move together. Only this bijection may change.

The operator order remains:

1. DeepEP `dispatch_a`;
2. DeepEP `dispatch_b`;
3. DeepGEMM FP8 `runner.run`;
4. `Buffer.capture`;
5. DeepEP combine;
6. completion-event wait.

No common delay floor, differential delay, sleep/spin, or collective/strict gate
is allowed between expert execution and combine. Allocation, placement setup,
weight movement, JIT, and warmup are outside the execution timing interval.
Cross-device CUDA timestamps are never subtracted. The primary execution measure
is the existing GPU-0 same-clock completion-fenced block interval; `T_proxy` is a
secondary maximum of rank-local same-device elapsed intervals, not a calibrated
global makespan.

The sign convention is fixed globally:

\[
Q = T_{\mathrm{reference}} - T_{\mathrm{candidate}}.
\]

Positive is faster; negative is slower. Negative observations and estimates are
never clipped. Raw route imbalance is assignment metadata, not compute time.
Per-expert 128-row padding and actual grouped-GEMM group shapes remain explicit.
A measured change in global/latest completion is not reduced by another slack
term. A confidence interval that crosses zero is described as compatible with
zero, not as proof of noise or equivalence.

## A. Four-independent-start confirmation

### Frozen inputs

This phase reuses, byte-for-byte:

- the existing layer-24 real-forward snapshots for all eight ranks;
- `placements/manifest.json` with identity plus all 12 E1 layouts;
- `inputs/correctness_tolerance.json` (`atol=2^-8`, `rtol=0`).

Their paths, byte counts, and SHA-256 values are recorded in
`configs/followup_v1/protocol.json`. The existing pilot does not enter the primary
confirmation interval; it supplied the hypotheses and the already declared
practical margin only.

### Execution

Run exactly four fresh sequential `torchrun` launches. Each launch creates eight
new rank processes. For each start:

- validate the GPU model, protocol/source/input hashes, DeepEP mode, and backend;
- preload all layout-specific real weights/scales;
- run 10 initial untimed forwards for every layout to cover JIT and exact shapes;
- pass all 104 rank/layout correctness comparisons using the frozen tolerance;
- run five randomized blocks;
- activate every one of the 13 layouts once per block;
- immediately after every activation, run exactly 10 identical untimed warmup
  forwards and verify the observed aligned expert counts;
- then time exactly 50 forwards in one completion-fenced block.

The four random seeds and all 20 expanded schedules are frozen in the JSON
protocol before timing. No start, block, or repeat is selected based on speed. A
process crash, GPU Xid/ECC error, hash mismatch, topology mismatch, correctness
failure, or structurally incomplete record invalidates the complete start without
consulting its performance values. The failed record is retained and the run
stops; there is no silent replacement or partial rerun.

### Inference

For layout `l`, start `s`, and block `b`:

\[
Q_{s,b,l}=T_{s,b,identity}-T_{s,b,l}.
\]

The five paired block effects are averaged to one value per start. The four
start-level values are the independent inferential sample. The primary interval
is a two-sided 95% Student-t interval with `n=4`, `df=3`. All 20 paired block
values are retained descriptively but are not treated as 20 independent starts.
All 12 identity contrasts are reported, with two-sided Holm-adjusted p-values as
a secondary multiplicity description. The pre-specified A2/A3 contrast is also
reported at the start level.

The practical-equivalence margin is frozen at `epsilon = 22.948043518066406 us`,
the post-pilot rule already recorded for this future confirmation. Only an
interval wholly inside `[-epsilon,+epsilon]` supports practical equivalence.
Crossing zero alone does not.

This phase can confirm only the frozen e089-centered, layer-24, single-forward
case. It cannot validate other hotspots, layers, forwards, prediction models, or
a stopping region.

## B. Out-of-sample prediction study

This phase requires new real model forwards. Its exact request IDs, captures,
layers, layout maps, and all hashes must be frozen in versioned manifests before
the corresponding performance timing. No E1 snapshot is silently counted as an
out-of-sample unit.

### Request split and capture

Use the same hash-identified RealProver dataset and tokenizer policy as E1, but
select disjoint source-line IDs not used by E1. Freeze seven deterministic
request batches of eight approximately 512-token prompts before route profiling:

- one screening batch;
- three calibration batches;
- three locked evaluation batches.

A request group and all forwards derived from it stay in one split. Screening is
used only for route/shape feasibility and layer selection, never for fitting or
locked evaluation. Profile all routed layers for the screening batch. Freeze four
layers from route metadata only: layer 24 as the bridge plus one deterministic
maximum-imbalance layer from each depth third, with duplicate selection resolved
by the next-ranked layer and ties by lower layer ID. Two layers, including layer
24, are calibration layers; two are completely held-out layers. The assignment
rule is frozen before calibration timing.

Capture the two calibration layers for each calibration batch and all four layers
for each evaluation batch, using unique capture IDs and separate directories.
This yields six calibration capture units and 12 locked evaluation capture units.
A capture unit is one real request-batch forward at one routed layer. It includes
all eight ranks' hidden rows, logical top-k IDs/weights, identity output, complete
FP8 expert weights/scales, route provenance, model revision, and hashes. Large
tensors remain local and hash-indexed.

Every new capture/layout must pass correctness with the already frozen
`atol=2^-8`, `rtol=0` before timing. Failure blocks the run; the tolerance is not
relaxed after timing or favorable results are seen.

### Layout split

Calibration placement maps are generated from calibration route/shape metadata
only: identity plus three raw-load designs, three 128-row-padding designs, and
three grouped-shape-coverage designs per capture. Their purpose is model
identification, not measured winner selection.

After calibration is complete and all model code, features, transformations,
coefficients, and hyperparameters are locked, generate evaluation maps with
reserved seeds: identity, two raw-load candidates, two padding candidates, two
locked grouped-GEMM-model candidates, and two stratified random candidates per
capture. Except for identity, evaluation map hashes must be absent from all
calibration manifests. Exact evaluation maps are frozen before any evaluation
performance timing. Timing never determines candidate membership.

### Model definitions

For expert `e`, let `n_e` be its assignment count and `d(e)` its destination rank.
All models receive the same fixed communication covariates derived from the
source-to-destination assignment matrix, including send/receive row vectors,
nonempty source/destination pairs, and derived bytes.

- **M0 raw:** rank loads `R_d=sum(n_e)` for experts owned by `d`, their sorted
  vector, and maximum. No padding or kernel-shape feature is allowed.
- **M1 padding:** `q_e=0` for an empty expert, otherwise
  `q_e=128*ceil(n_e/128)`; use per-rank padded work, maximum, padding waste, and
  active-expert count. Boundary tests cover 0, 1, 127, 128, 129, 255, and 256.
- **M2 grouped-GEMM:** retain each rank's complete multiset of 128-row expert
  blocks, including the block-count histogram, active group count, total blocks,
  maximum blocks/expert, dispersion, and predeclared interactions. Calibrate the
  compute-cost component from same-device CUDA elapsed intervals around the real
  `runner.run` call in a separate diagnostic pass. No cross-device timestamp
  subtraction is allowed.

Completion-delta models are fitted only on the six calibration captures. Grouped
cross-validation holds out complete request batches; layouts from one capture do
not cross folds. M0/M1/M2 use the same split and communication covariates. The
locked evaluation set is opened once; it does not trigger refitting, feature
changes, threshold changes, or new candidate selection.

Each calibration/evaluation timing uses four independent starts, four randomized
blocks, 10 untimed forwards after every activation, and 20 timed forwards per
activation. Primary evaluation uses the unchanged completion-fenced path; M2
kernel diagnostics are separate.

Report locked-evaluation MAE, RMSE, signed bias, Spearman and Kendall ranking,
sign accuracy, selected-layout regret, and paired model-loss differences. Report
new-forward/calibration-layer and held-out-layer strata separately, plus per-layer
results. Average within capture/start before inference. Uncertainty is clustered
over request batches and process starts, never individual tokens or forwards.
The study establishes padding or grouped-model superiority only if locked
out-of-sample paired losses support it; in-sample fit or E1 correlation is not
sufficient. Communication remains recorded as a possible confounder.

## C. Low-r residual swaps and migration cost

### Residual execution study

Use only locked evaluation captures and the locked M2 model. For each designated
capture, construct a deterministic low-r base placement using greedy legal
cross-rank swaps: minimize raw maximum rank assignments; break ties by maximum
padded rows, then the sorted raw-rank-load vector, then placement hash; stop when
no legal single swap improves that lexicographic score. Record the full path and
freeze the final base before timing.

Enumerate every legal cross-rank single swap from the base. Using route/shape
metadata and the locked M2 model only, freeze 12 residual candidates spanning:

- the four largest predicted signed gains, with deterministic hash tie-breaking;
- four predictions closest to zero (retain their signs);
- the four most negative predictions.

Deduplicate candidates and advance deterministically within a stratum when
needed. Freeze exact maps, expert pairs, source/destination ranks, raw/padded/group
features, predicted signed gains, and selection logs before timing. Measured
latency is never used to choose the candidates.

Measure the base and all frozen candidates with four independent starts, five
randomized blocks, 10 activation warmups, and 50 timed forwards. Report every
signed candidate-versus-base effect, including negative effects. Do not select a
best measured prefix. The same frozen `epsilon=22.948043518066406 us` rule is used
for practical equivalence. A zero-crossing interval is not a deadzone. No
one-sided delay slack is converted into a symmetric stopping region, and no slack
is subtracted from measured `Q`.

### Separate real migration experiment

Migration is a separate transaction, never part of execution-only `Q`. The old
`bench_p2p_cost.py` artifact is inadmissible because it reported w13+w2 bytes while
moving only w13 and omitted scale tensors.

For every measured swap, exchange the complete FP8 w13 and w2 tensors plus both
scale tensors between their actual H20 owners. Allocation, communicators, and
staging buffers are prepared before timing. The timed transaction begins after a
start alignment, includes actual transfer, completion acknowledgement, and
mapping publication, and ends on one coordinator's monotonic clock after every
rank acknowledges readiness. Post-transfer tensor hashes and outputs are checked
outside timing. Pointer aliasing or pre-resident duplicate copies do not count as
migration.

Measure each frozen single swap and the actual multi-swap batch with four fresh
process starts and 50 fixed forward/reverse transaction pairs per start. Retain
all repetitions; do not report fastest-repeat bandwidth. Record exact bytes by
tensor and direction, link/rank pairs, coordinator wall time, and same-device CUDA
diagnostics separately.

If execution benefit is `Q>0` per forward and migration cost is `C`, report the
point break-even `C/Q` only with uncertainty. If `Q<=0`, no finite beneficial
break-even exists. If the interval for `Q` crosses zero, break-even is
indeterminate. Do not subtract migration cost per execution sample, subtract a
slack term, or fold migration into execution-only completion effects.

## Versioning and gates

Every phase has a versioned config, manifest, raw directory, source hash, command,
process-start ID, schedule, correctness record, environment snapshot, GPU
inventory/topology, model/capture/placement/tolerance hashes, and completion
marker. Raw result creation is no-clobber. Failed starts and their logs are
retained. Compact JSON/CSV artifacts are tracked; multi-GB captures remain local
and hash-indexed.

The gates are sequential:

1. freeze and commit protocol/code locally;
2. pass static hashes/bijection/source audit;
3. complete and validate all four confirmation starts;
4. freeze request split/captures/calibration manifests;
5. lock M0/M1/M2 before evaluation maps/timing;
6. run and validate locked evaluation;
7. freeze residual candidates before residual timing;
8. validate real full-tensor migration before break-even analysis;
9. reconstruct every table from explicit manifests, not directory globbing.

No push or external publication is part of this protocol without a separate user
instruction.
