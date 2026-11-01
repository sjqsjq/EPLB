# ADW v2 workload inventory & condition plan (Section 六)

Status: draft_for_step6_freeze. Inventory verified on disk 2026-10-16.
Rule: this plan is frozen into the Step-6 formal protocol BEFORE any formal
v2 run; nothing here was chosen using v2 evaluation results.

## 1. Inventory of available real request files

All under /data/minghua/sjq/OEPLBdata/datasets/multi_domain/ unless noted.
"O1" = max_tokens=1 prefill-replay protocol (same as the full-service campaign:
calibration.jsonl / evaluation.jsonl are both O1, verified from first lines).

| file | n | protocol | structure (verified) | role |
|---|---|---|---|---|
| prefill_heavy_universal.jsonl | 16000 | O1 | 4x4000 contiguous domain blocks (sharegpt/prover/book/code), 3 switches | source of existing splits |
| (EXP)/inputs/fullservice_v2/workload/calibration.jsonl | 8037 | O1 | 3 domain switches (~2000-line blocks) | **W0 DEV/CALIBRATION** (knobs, kappa, F-local-W0; already used for W*/alpha* grid) |
| (EXP)/inputs/fullservice_v2/workload/evaluation.jsonl | 7963 | O1 | 3 domain switches | **known benchmark regression ONLY** - already viewed by v1 work; never "unseen validation" for v2 |
| spliced_code_chinese_O1.jsonl | 8000 | O1 | explicit `domain` field; code x4000 then chinese x4000; exactly 1 switch at line 4000 | **W2-SWITCH holdout** (predefined real change) |
| spliced_code_chinese_O1.jsonl lines 0-3999 | 4000 | O1 | code-only, single domain | **W1-STABLE holdout** (relatively stable) |
| spliced_code_chinese_extreme.jsonl | 2000 | O1 | domain labels; 1 switch at 1000 | reserve (short W2 variant) |
| spliced_code_chinese_big.jsonl | 8000 | O1 | same structure as _O1 | reserve |
| domain_switch_160k.jsonl | 160000 | O1 | NO domain labels; system prompt constant; coarse content signature changes ~every 1000 lines (block structure UNVERIFIED) | EXCLUDED from phase-1 primary comparison (no labels; would require offline content classification); candidate for a later higher-switch-frequency condition |
| multidomain_v2.jsonl / _out1 | 4400 | O32 | interleaved (802 signature transitions), no block structure | excluded (structure + protocol mismatch) |
| spliced_L512_L1024_O1.jsonl | 12288 | O1 | length-band interleaved (6773 transitions), no clean blocks | excluded from primary; candidate for length-mix study |
| frozen_requests_500 | 500 | O512 | single system signature | inventory only (decode-heavy, different protocol) |
| frozen_requests_real_production | 500 | O32 | single signature | inventory only |
| frozen_requests_prefill_heavy | 500 | O32 | 429 signatures | inventory only |
| frozen_requests_{hellaswag,in2048,longbench_4k} | 500 each | mixed | single-domain sets | inventory only |
| frozen_requests_short_in_long_out | 3000 | mixed | | inventory only |
| raw_hellaswag_val.jsonl | 10042 | RAW (ctx/endings keys, not service format) | | excluded (conversion = synthetic processing) |
| prefill_heavy_universal_O256.jsonl | 16000 | O256 | same 4-block structure | candidate protocol-sensitivity condition, not phase-1 |
| triplefixed.jsonl / fixeddata.jsonl | 5120/3072 | O32 | prover-led mixes | inventory only |

## 2. Conditions for v2 phase-1 (window-only isolation)

- **W0-CALIB** = calibration.jsonl (8037, O1, 3 switches). Development and
  knob calibration ONLY (kappa from stationary segments; z/theta_H/dwell/
  confirm counts; W_init confirmation). v2 knobs are frozen here and never
  re-tuned on holdout.
- **W1-STABLE** = spliced_code_chinese_O1.jsonl lines 0-3999 (code block,
  4000 requests, O1, 0 switches). Predefined selection rule: "first
  contiguous labeled single-domain block of the spliced file" - not chosen
  from results.
- **W2-SWITCH** = spliced_code_chinese_O1.jsonl full (8000 requests, O1,
  exactly 1 predefined real domain switch at request 4000).
- Optional secondary (only if budget remains after the primary freeze, never
  part of the main comparison): evaluation.jsonl as known-benchmark
  regression; domain_switch_160k high-frequency condition after offline
  label verification.

Requirements satisfied: >=1 relatively stable condition (W1) + >=1 with
predefined real change (W2); domains/switch scales are the files' real
structure, not tuned for adaptive wins; the controller receives NO domain
labels or switch times (v2 reads routing statistics only); identical
requests/order/service protocol for every arm within a condition.

## 3. Arms per condition (phase-1)

All arms: alpha=0.0 base, D OFF (adaptive_decay False), same planner
(heuristic_first_improvement), same acceptance rules, same launcher, O1
protocol, concurrency 256.

1. **V2** - OEPLB_WINDOW_CONTROLLER=2, knobs frozen on W0 only.
2. **F-global** - fixed W=16 (the single config selected by the W0
   calibration grid across workloads; tests the value of reduced
   cross-workload tuning).
3. **F-local(Wx)** - fixed W = argmin over the {8,16,32,64} x {alpha=0.0}
   grid run ON that same condition (per-workload calibration-selected
   reasonable config; tests net benefit against a strong fixed control).
   Fixed arms are granted the same calibration access as v2 (fairness rule).
4. Post-hoc reference (labeled, never a strategy): best fixed W on holdout
   conditions computed after the fact.

Main pre-declared comparisons per condition: V2 - F-global and
V2 - F-local(Wx). Module ablations (M1/M2/M3) reuse the same campaign's
Full and Fixed runs; no planner x W x D x domain full-factorial expansion.
Start counts/order/failure handling frozen in the Step-6 protocol before
execution; requests are not system replicates; uncertainty reported when
samples are insufficient.
