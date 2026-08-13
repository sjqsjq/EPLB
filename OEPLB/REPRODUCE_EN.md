# Reproduction Guide

This is the **core reproduction suite** for the paper "Adaptive Online Expert Load
Balancing for MoE Inference Serving", organized by the paper's claims. Each section
states the claim being verified, the command, the expected result, and how to
distinguish an environment problem from a genuine mismatch.

Result files live in `benchmarks/results/` under `_dNN_*` prefixes matching the
experiment numbers below. Driver scripts are in `repro/`; raw routing counts are
in `repro/counts*.json`.

## 0. Environment
- 8×NVIDIA H20 96GB, NVLink-only (no IB)
- SGLang 0.5.6.post2 + PB-OEPLB patch, sgl-kernel 0.3.19, torch 2.9.1+cu128, DeepEP v1.2.1, DeepGEMM
- Models: `Qwen3-235B-A22B-FP8`, `Qwen2-57B-A14B-Instruct`, `Qwen3-30B-A3B-FP8`

## 1. Headline: prover single-domain 235B (§1, §5.3)
Claim: +17.5% throughput. `repro/driver38.sh`.
Expected: baseline(identity) ~201s (CV<0.1%), OEPLB ~169s, gain **+17~+20%**.
Measured here: **+19.43%** (r1 +18.43%, r2 +20.46%). `_d38_*`.
Pitfall: baseline must NOT pass `--init-expert-location` (identity), or the gain
reads as negative (see d36 mistake, fixed in d38).

## 2. T(r) sweep: the bound model (§2.4, App. G)
Claim: hinge `T(r)=T_flat+β·T_flat·max(0,r−r_k)`, R²>0.996; fitted bound agrees
with the empirical ceiling (identity→bal) to within 0.4pp.
`repro/driver12.sh`(57B/EP8), `driver13`(EP4), `driver14`(235B/EP8), `driver28`(30B/EP4).
Expected: hinge RSS 7–25× lower than a line; r_k≈1.099/1.032/1.093/1.031;
β≈0.285/0.342/0.352/0.207; held-out identity error <0.5%.
Sanity check: identity-vs-bal empirical ceiling should ≈ the hinge bound
(57B/EP8: +3.82% vs +3.40%, diff 0.42pp).

## 3. r_k EP power law (§2.4)
Claim: r_k−1 = 0.00408·EP^1.52; 235B/EP8 cross-model blind test error 3.8%.
Fit the four r_k values from §2; expect slope ≈1.5.

## 4. Dead-zone threshold lifts η from 26% to 100% (§3.2, D.3)
Claim: default threshold 1.02 sits inside the dead zone and wastes swaps; moving it
to r_k + a swap budget raises η from 26% to 100%. `repro/driver26.sh`.
Expected: threshold=1.02 → +0.98% (η26%); threshold=1.099+budget → +3.81% (η100%).

## 5. Adaptive window (§3.5, App. H)
Claim: M=W/(1−α) is an approximate sufficient statistic (same-M spread <5% for
M≥32); adaptive decay + gate beats the same-starting-point static arm. `repro/driver31.sh`.
Expected: M=32 group spread <5%; adw+adaptive_decay is the fastest arm;
(W=8,α=0.5) is a pathological corner (>5× slower).

## 6. Numerical equivalence (App. E)
Claim: a swap is an identity transform; the ~0.1 nat/token perturbation comes from
FP8 reduction order, not the swap path. `repro/driver15.sh`, `driver19.sh`.
Expected: baseline bit-identical (50385/50385); OEPLB pre→post perturbation ≈
static-placement (zero-swap) perturbation; GSM8K within ±2pp.

## 7. 30B negative gain, correctly attributed (§2.4)
Claim: 30B β=+0.207 (positive), but default-config gain is negative because fixed
overhead exceeds the bound; a swap budget recovers it. `repro/driver30.sh`.
Expected: 1.02 → −3.8%; 1.031 → −2.66%; 1.031+budget → **+0.53%**.

## 8. predict_gain.py: zero-measurement ceiling estimate (§2.4)
`python3 tools/predict_gain.py --model-config config.json --ep 8`
Expected: r_k (EP power law, ±5%), β (FLOP/1.6, ±30%), ceiling. A quantitative
bound still needs the §2 T(r) sweep (β is the only link not predictable from a
single profile, see driver23b).

## Known limitations
1. β cannot be predicted from a single profile (needs a ~2h sweep; profile gives ±30%).
2. The historical multi-domain 235B +14.0% is NOT reproducible (dataset deleted).
   The reproducible multi-domain set gives **+5.80%** (`driver35.sh`).
3. The M* closed form's numerical coefficient is uncalibrated (direction confirmed, no peak seen).
4. All runs on H20 NVLink; the r_k power law is unvalidated on other hardware.
5. r is a weak sufficient statistic for 30B (held-out error 3.2% vs <0.4% for 57B/235B).

## Reviewer priority
1. §1 (prover single-domain +17.5%) — headline, ~1h
2. §2 + §3 (T(r) sweep + r_k power law) — modeling core, ~3h
3. §4 (dead-zone threshold η 26%→100%) — verifiable algorithm improvement, ~1h
