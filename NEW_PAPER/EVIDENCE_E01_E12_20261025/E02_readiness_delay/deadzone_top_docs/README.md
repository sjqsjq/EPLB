# DeepEP readiness-slack and global-ready experiment

This directory contains the reproducible 8×H20 experiment used to distinguish
three different quantities that earlier drafts conflated:

1. **one-sided delay tolerance**—extra delay on a currently noncritical rank can
   be absorbed by existing readiness slack;
2. **directional layout benefit**—reducing work on the current slowest rank can
   help immediately and is not the inverse of (1);
3. **real-route layer penalty**—the change in the latest ready time over all
   ranks, including baseline ordering and last-ready switching.

The primary route-level relation tested here is

\[
\Delta T_{\mathrm{layer}}\approx
\Delta R_{\mathrm{global}}
=
\max_i R_i^{\mathrm{route}}-\max_i R_i^{\mathrm{baseline}}.
\]

Once the all-rank latest-ready change has been formed, readiness slack is
already included and **must not be subtracted a second time**. The older
hot-rank-only expression `(δ_hot-H)+` remains in the artifacts only as a labeled
diagnostic ablation.

## Frozen hardware and layer path

The run uses **8×NVIDIA H20 96 GB, EP=8, DeepEP normal mode, DeepGEMM FP8,
eager execution**, with PB-OEPLB and expert exchanges disabled and a fixed
one-copy placement. The measured software environment is SGLang 0.5.6.post2,
DeepEP 1.2.1, DeepGEMM 2.3.0, PyTorch 2.9.1+cu128, and CUDA runtime 12.8.
The harness does not patch the installed libraries.

Each rank executes one Qwen3-235B-A22B-shaped MoE layer:

1. deterministic routing over the eight-rank EP group;
2. SGLang DeepEP normal FP8 dispatch with `expert_alignment=128`;
3. contiguous DeepGEMM FP8 gate/up and down grouped GEMMs;
4. SGLang `ep_gather`;
5. an optional stream-ordered GPU delay after expert compute/gather and before
   combine dependency capture;
6. either the original asynchronous DeepEP combine or a strict GPU-ordered
   NCCL all-rank ready gate followed by the unchanged combine.

The geometry is hidden size 4096, expert intermediate size 1536, 128 routed
experts, top-k 8, and 16 complete experts per rank. Gate/up weights have shape
`[16,3072,4096]`; down weights have shape `[16,4096,1536]`; scales use 128×128
blocks.

## Experimental designs

### 1. Original delay and strict-gate controls

`gpu_delay.cu` launches the same one-block, one-warp kernel on every rank. Every
rank receives a 20 µs common floor; a selected rank receives `20 µs + d`. The
common floor makes small differentials measurable without changing the launch
symbol.

The original coarse/dense scans delay ranks 0, 3, and 7. The strict path inserts
an NCCL `all_reduce(MAX)` ordered after every producer stream, then calls the
same combine. This control shows that useful early combine progress is not
needed for a one-sided delay plateau: under the strict path, selected-rank gate
wait is released before non-target ranks acquire wait.

This finding is intentionally narrow. It shows that **adding delay to a
noncritical rank** can leave the makespan unchanged. It does not show an equally
wide no-benefit region for **reducing work on the current slowest rank**.

### 2. Decisive all-rank readiness scan

The exact `fixed7_one_double`, `r=1`, N=4096 route is frozen. Every rank 0–7 is
independently delayed at

```text
0, 4, 8, 12, 16, 24, 32, 48, 64, 96 µs
```

in both a strict-gate scan and a separately executed original-path scan. Each
scan has 73 conditions (`1 + 8×9`), 5 blocks, and 100 iterations per condition.
The strict scan estimates each rank's effective baseline readiness slack from
same-rank gate-wait changes:

\[
-\Delta W_i(d)\approx\min(d,H_i),\qquad
\Delta W_{-i}(d)\approx(d-H_i)_+.
\]

No timestamps from different GPUs are subtracted. The strict-derived values are
frozen before evaluating the independently executed original scan.

### 3. Real-route global-ready scan

`route_imbalance.py` implements `fixed7_one_double` for requested
`r={1,1.02,1.05,1.10,1.20,1.40}`. Every token keeps exactly seven destination
ranks; only the second expert assignment moves. Thus communication destination
support and the complete source→destination unique-row matrix remain fixed.

For every exact route, the scan runs both original and strict paths. Two
global-ready estimates are retained:

- **direct strict-path estimate**:
  \[
  \Delta\max_i(\text{layer\_ms}_i-
  \text{combine\_after\_gate\_ms}_i);
  \]
- **per-rank reconstruction**:
  \[
  \max_i(-H_i+\Delta R_i)-\max_i(-H_i),
  \quad
  R_i=\text{dispatch}_i+\text{expert}_i+\text{delay}_i.
  \]

The direct estimate is primary. The reconstruction is a secondary test of the
independently estimated rank-specific `H_i` values.

## Padding-aware route semantics

DeepGEMM aligns **each expert's M dimension independently to 128 rows**. Raw
route ratio `r` is therefore metadata, not a linear compute-time variable. The
analysis follows

```text
route counts
→ per-expert 128-row-aligned rows
→ per-rank measured ready changes
→ all-rank latest-ready change
→ layer-time change
```

and never substitutes `δ = S(r-1)`. At N=256 all nonempty expert counts remain
on the same 128-row plateau even at raw `r≈1.398`. At N=4096 the aligned work
changes in discrete steps.

## Estimator and uncertainty

Every production condition uses 5 independently shuffled blocks × 100
iterations. Every observation retains all eight ranks. For each condition and
block, analysis subtracts the matching baseline block median; the main estimate
is the median of five block effects. Confidence intervals cluster-bootstrap the
five blocks. Same-device CUDA-event intervals are used; raw timestamps from
different GPUs are never subtracted.

**Negative baseline-subtracted estimates are retained unchanged.** When an
interval spans zero, the result is reported as compatible with a zero effect;
the experiment does not identify its sign.

## Reproduction

Run all commands from this directory. Main timing and Nsight profiling are
separate runs.

### Earlier coarse/dense controls

```bash
scripts/run_calibration.sh 256
scripts/run_calibration.sh 4096
scripts/run_delay_scan.sh 256
scripts/run_delay_scan.sh 4096
scripts/run_dense_delay_scan.sh 256
scripts/run_dense_delay_scan.sh 4096
scripts/run_matching_route_delay_scan.sh 256
scripts/run_matching_route_delay_scan.sh 4096
scripts/run_load_scan.sh 256
scripts/run_load_scan.sh 4096
```

Analyze the N=4096 controls (use analogous N=256 paths for N=256):

```bash
python scripts/analyze_delay_scan.py \
  --input results/delay_scan_N4096.json \
  --output-dir results/analysis_N4096
python scripts/analyze_delay_scan.py \
  --input results/delay_dense_N4096.json \
  --output-dir results/analysis_dense_N4096
python scripts/merge_delay_analyses.py \
  --coarse-dir results/analysis_N4096 \
  --dense-dir results/analysis_dense_N4096 \
  --output-dir results/analysis_merged_N4096
python scripts/analyze_delay_scan.py \
  --input results/matching_route_delay_N4096.json \
  --output-dir results/matching_route_delay_analysis_N4096
python scripts/analyze_load_scan.py \
  --input results/load_scan_N4096.json \
  --hinge-analysis results/matching_route_delay_analysis_N4096/delay_scan_analysis.json \
  --output-dir results/load_analysis_N4096
```

`analyze_load_scan.py` now labels its output as a **legacy hot-rank-only
ablation**, not the primary causal route model.

### All-rank readiness and global-ready scans

```bash
scripts/run_rank_readiness_scan.sh 4096 strict_barrier
scripts/run_rank_readiness_scan.sh 4096 original

python scripts/analyze_rank_readiness.py \
  --strict-input results/readiness_strict_N4096.json \
  --original-input results/readiness_original_N4096.json \
  --output-dir results/readiness_analysis_N4096

scripts/run_route_ready_scan.sh 4096

python scripts/analyze_route_ready_scan.py \
  --input results/route_ready_N4096.json \
  --readiness-analysis results/readiness_analysis_N4096/rank_readiness_analysis.json \
  --legacy-analysis results/matching_route_delay_analysis_N4096/delay_scan_analysis.json \
  --output-dir results/route_ready_analysis_N4096
```

### Nsight timelines

```bash
scripts/run_nsys_timelines.sh 4096
python scripts/analyze_representative_timelines.py \
  --input results/delay_scan_N4096.json \
  --output-dir results/timeline_analysis_N4096
python scripts/analyze_nsys_traces.py \
  --trace-dir traces/N4096 \
  --manifest results/timeline_plans_N4096/manifest.json \
  --output-dir results/timeline_analysis_N4096
```

Nsight is launched once per torchrun rank and is illustrative only. The 5×100
CUDA-event data, not profiler-inflated layer times, supply numerical estimates.

### Validation

```bash
# Earlier ten raw datasets
python scripts/validate_final_results.py

# Require readiness and route-ready production data as well
python scripts/validate_final_results.py --include-readiness

# Also require the 48 nonempty Nsight reports
python scripts/validate_final_results.py --include-timelines --include-readiness
```

The completed full validation (including timelines and readiness scans) passes
**19 files, 218,006 records, and 1,744,048 rank observations**, with maximum
correctness error zero.

## Main results

### All-rank readiness

The strict scan did not identify one stable singular last-ready rank. The
point-estimate minimum was rank 3 (`H=26.876 µs`), but block winners were
`[1,1,3,3,3]`; the interval-overlap candidate set was `[0,1,3,6]`. Every rank's
4 µs original-path interval spanned zero.

Strict-vs-original fitted-`H_i` correlation was only **0.086**, with
**27.425 µs** RMSE. The frozen curves had **9.269 µs** all-point RMSE
(**9.770 µs** excluding fixed zero anchors), so they capture part of the broad
large-delay rise but do **not** independently validate rank-specific plateau
widths.

### Real routes

The direct all-rank global-ready change predicts the original layer change with
**6.734 µs RMSE** and **5.741 µs MAE**. The per-rank reconstruction has
**21.628 µs RMSE**, consistent with unstable or poorly identified rank-specific
readiness ordering. The legacy hot-expert-minus-`H` ablation has **11.895 µs
RMSE** on the same route-ready scan.

At raw `r=1.10` and `r=1.20`, original layer penalties are nearly identical
(52.432 and 51.664 µs) despite different raw ratios; at `r=1.40` the penalty
jumps to 206.960 µs. This is consistent with the 128-row padding staircase and
inconsistent with a universal `S(r-1)` conversion.

These RMSE values summarize six point estimates. Five-block confidence
intervals are broad at several route points, so they are descriptive model
comparisons rather than high-precision proof at every ratio.

## Repository data policy

The repository versions the harness, condition plans, analysis JSON/CSV,
figures, reports, and validation summary. Large per-iteration raw JSON files,
Nsight traces, logs, and compiled build products are reproducible local
artifacts and are excluded by `.gitignore`; several raw scans individually
exceed GitHub's 100 MiB object limit. Recreate them with the commands above
before rerunning the validators or analyses.

## Primary artifacts

- All-rank readiness: `results/readiness_analysis_N4096/`
- Global-ready route analysis: `results/route_ready_analysis_N4096/`
- Earlier coarse/dense controls: `results/analysis_*` and
  `results/analysis_merged_*`
- Labeled legacy hot-rank ablations: `results/load_analysis_*`
- Static and Nsight timelines: `results/timeline_analysis_N4096/` and
  `traces/N4096/`
- Validation summary: `results/validation.json`
- Full scientific report: `REPORT.md`
