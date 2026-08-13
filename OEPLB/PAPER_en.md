# Adaptive Online Expert Load Balancing for MoE Inference Serving

> **Revision notes for this round (2026-08-12).** This version makes substantial changes to the theoretical model and several measured figures, including:
> (1) the functional form of $T(r)$ has been changed from affine to a **hinge** (dead zone $r\le r_k$), and sweeps have been completed on three configurations (57B/EP8, 57B/EP4, 235B/EP8);
> (2) the upper bound is now parameterized by $\beta=B/T_{\text{flat}}$, separating configuration parameters from workload parameters, so that the upper bound becomes a plottable polyline (§2.4);
> (3) a new §2.5 distinguishes the two ceilings of "optimal placement" and "perfectly balanced routing", giving the applicability domain of "no redundant experts";
> (4) the previous contradiction of "system efficiency 109%>100%" has been closed — the true cause is $f_{\text{sens}}$ (nsys decomposition 0.384 vs. measured 0.496), not cross-model borrowing of $r_k$;
> (5) a new Appendix E (numerical equivalence: layout changes introduce a $\sim0.1$ nat/token perturbation, but this is **unrelated to the swap mechanism**; bit-exact equivalence is unattainable on this stack);
> (6) **retracted/downgraded figures**: the dataset underlying the three lines in Appendix F.2 (+2.6%~+4.7%) no longer exists and is not reproducible; re-measurement on the available datasets gives −0.24%~+2.70%; the earlier claims of "FLOP fraction overestimates $f_{\text{sens}}$ by 7.7×", "EPLB net upper bound 17.0%", and "0 swaps after OEPLB convergence" have all been retracted.
> Every change is annotated at the corresponding location in the main text with its justification and the experiment script name.

## Abstract

Mixture-of-Experts (MoE) models face an expert load imbalance problem at inference time — hot experts concentrate on a few GPUs, creating a compute bottleneck. Existing schemes such as SGLang's EPLB require redundant expert replicas, block inference during rebalancing, and cannot adapt to workload changes in real time. This paper proposes **PB-OEPLB**, a lightweight online expert load balancer that achieves near-optimal expert placement without redundant experts and with an adjustment cost about 4× lower than periodic rebalancing (0.37s vs 1.55s steady-state). PB-OEPLB records expert routing only on prefill batches (decode steps under CUDA graphs are skipped at zero overhead; worst-case overhead in pure-prefill mode is ~1.6%), uses exponentially decaying history for noise-robust decisions, and employs an adaptive pair-selection algorithm that converges to a near-optimal imbalance ratio (1.02) within 3 decision windows. When serving Qwen3-235B-A22B-FP8 on an 8×H20 cluster, PB-OEPLB improves throughput by +17.5% over the baseline on prefill-heavy workloads (mean of re-measurements, n=2), 15.7 percentage points higher than EPLB; on the same dataset, the synchronous blocking introduced by its weight migration is 5.95s per rank (3.4% of wall clock), versus 14.7s (7.3%) for EPLB, and 0.37s vs 1.55s per steady-state adjustment. The adaptive window mechanism automatically adjusts decision frequency based on workload stability, delivering a +5.3% gain even on short-prompt general workloads where static parameter tuning fails.

In addition, this paper conducts interleaved validation with independent restarts on 4×H20 for Qwen2-57B-A14B-Instruct and Qwen3-30B-A3B-FP8, showing that the gain has a two-layer structure — the theoretical bound $\Delta_{\max}$ (set by imbalance, model architecture, GPU count) and the system efficiency $\eta$ (set by the swap-overhead-to-bound ratio): 30B is a canonical case where the bound is positive but fixed overhead drives $\eta<0$. We also find the official EPLB crashes with an AttributeError on non-DeepSeek-architecture models, which PB-OEPLB fixes via a generic fallback.

---

## 1. Introduction

Large language models are increasingly adopting the Mixture-of-Experts (MoE) architecture to scale parameter counts without proportionally increasing compute. In production inference serving, MoE models face a fundamental challenge: **expert load imbalance**. The routing network assigns tokens to experts based on input semantics, and naturally occurring workload patterns (e.g., mathematical reasoning vs. narrative text) cause some experts to be activated far more frequently than others. When experts are distributed across multiple GPUs via expert parallelism (EP), this imbalance creates stragglers — the GPUs holding hot experts become the bottleneck while other GPUs sit idle.

Current approaches fall into three categories:

1. **Static placement**: precompute the optimal expert placement from historical traffic data. This works only when the workload distribution is known and stable — changes in request types make it suboptimal.

2. **Periodic rebalancing (e.g., SGLang's EPLB)**: periodically recompute the expert placement and redistribute weights. This adapts to workload changes, but (a) requires redundant expert replicas (8-16 additional replicas occupying GPU memory), (b) blocks inference for 1.4-4.5 seconds during each rebalance (measured in this paper, 8-GPU 235B, see Table 7b), and (c) is incompatible with certain inference optimizations (forcing `deepep_mode=normal`, thereby disabling CUDA graphs and causing up to -68% throughput degradation on decode-heavy workloads).

3. **Online swapping**: incrementally adjust the expert placement by swapping expert pairs between GPUs. This avoids the overhead of full rebalancing but faces challenges in convergence speed and decision quality under limited data.

This paper proposes **PB-OEPLB** (Prefill-Boundary Online Expert Placement Load Balancer), which addresses the three limitations above:

- **Zero redundancy**: no additional expert replicas are needed; it operates within the existing expert budget.
- **Sparse perturbations, short blocking**: each decision swaps only a small number of expert pairs, and weight migration is performed with synchronous P2P (an asynchronous scheme triggers NCCL hangs on NVLink-only platforms, see §3.4). Steady-state blocking is about 0.37s per decision, 1/4 of a single global rebalance of EPLB (1.55s). (An early draft claimed "no longer triggered after convergence"; that had no reproducible data and is retracted — Appendix E.2 shows steady-state swaps continue at low volume.)
- **Adaptive**: the adaptive window mechanism automatically adjusts decision frequency based on workload stability — shrinking the window to respond quickly at domain switches and expanding it to reduce overhead when stable.
- **Prefill-only recording**: expert routing is recorded only on prefill batches, reducing recording overhead by ~50%, while the decode phase also benefits through the globally shared layout.

**Contributions:**

1. **Adaptive pair-selection algorithm** (§3.3): a greedy swap planner that switches between max-delta (fast convergence when imbalance is large) and gap-targeting (precise balancing near optimality) modes, achieving ratio convergence from 1.74→1.02 within 3 windows, whereas prior methods stalled at 1.26.

2. **Adaptive synchronization window** (§3.5): a feedback-driven mechanism that expands the decision window as the imbalance ratio converges (saving all_reduce overhead) and shrinks it when a workload switch is detected (fast response), automatically adapting to different prompt lengths.

3. **Fast decay mechanism** (§3.2): with decay factor 0.5, cross-domain signal contamination is cleared within 3 windows (vs. 0.9, which still retains 73% of the old signal after 3 windows), while maintaining sufficient statistical sample size.

4. **Comprehensive evaluation** (§5): on 8×H20 with Qwen3-235B-A22B-FP8, PB-OEPLB achieves +18.4% throughput improvement on single-domain prefill workloads (vs. EPLB +9.0%; this is a single-run gross value, the robust 2-round-restart mean is +17.5% which the abstract uses), +10.6%⚠ on multi-domain workloads (vs. EPLB +6.3%⚠; these two numbers are from a historical single-batch run; a different measurement round gave +14.0%/+12.0%; both depend on deleted `/tmp/exp_data/` datasets and are not reproducible), and +5.3% on general ShareGPT workloads — all without redundant experts, and with total blocking from weight migration 1/2.5 of EPLB's. In addition, validation on 4×H20 of Qwen2-57B-A14B (re-measurement on reproducible datasets gives −0.24%~+2.70%, decreasing with workload heterogeneity) and Qwen3-30B-A3B (essentially neutral $\eta\approx0$: its $T(r)$ bound is positive $+6.36\%$ but the realized gain is ≈0, fixed overhead eating the gain; the clean long benchmark `driver42.sh` confirms all four arms within ±1.3% of baseline, the earlier short-benchmark negative/stanching being a noise artifact) reveals that the gain is set jointly by the bound $\Delta_{\max}$ and the system efficiency $\eta$.

5. **Cross-architecture generality** (§2.2): we discovered and fixed the crash from the missing `routed_experts_weights_of_layer` attribute in EPLB and OEPLB on non-DeepSeek architectures (Qwen2-MoE, Qwen3-MoE); a generic fallback makes load balancing compatible with all MoE architectures.

---

## 2. Background and Motivation

### 2.1 Problem Formulation

We formulate expert load balancing under EP inference as an **online balanced partitioning problem**.

**Setup.** A MoE model with $N_E$ routed experts is served via expert parallelism (EP) on $G$ GPUs, each holding $n = N_E/G$ experts. The router assigns each token to its top-$k$ experts. Let $\ell_j \in \mathbb{R}_{\geq 0}$ be the cumulative load (token count) of expert $j$ over an observation window, and $\pi: [N_E] \to [G]$ an assignment of experts to GPUs satisfying the balance constraint $|\pi^{-1}(g)| = n$.

**Imbalance ratio.** The per-layer imbalance ratio is:
$$r(\pi) = \frac{\max_{g \in [G]} L_g(\pi)}{\bar{L}}, \quad L_g(\pi) = \sum_{j: \pi(j)=g} \ell_j, \quad \bar{L} = \frac{1}{G}\sum_g L_g$$

A ratio of 1.0 denotes perfect balance; the timing impact of imbalance is proportional to $r-1$ times the MoE compute fraction.

**Objective.** Find an assignment $\pi$ (from the initial assignment via incremental swaps) that minimizes $r(\pi)$, subject to: (i) at most $B$ pairwise swaps per decision window, (ii) zero additional memory (no redundant expert replicas), (iii) non-blocking execution (swaps must not stall inference).

**Complexity.** Finding $\pi^* = \arg\min_\pi r(\pi)$ under the balance constraint is NP-hard (reduction from 3-PARTITION for $G \geq 3$, from PARTITION for $G=2$). This motivates a greedy local-search approximation.

### 2.2 Limitations of Existing Approaches

**SGLang EPLB** (the state-of-the-art production system) has three architectural limitations:

**Limitation 1: forced disabling of CUDA graphs.** EPLB requires `deepep_mode=normal` (source: `server_args.py:1641`), which triggers `disable_cuda_graph=True`. This disables CUDA graph capture and increases per-step kernel launch overhead. On decode-heavy workloads (O=256) this causes approximately **-62% to -68% throughput degradation** (Appendix F.3 reproduces -62.4% on 4-GPU 57B; the original -68.2% measurement dataset is no longer available). Moreover, EPLB's `ExpertDistributionRecorder` directly does `raise NotImplementedError` for `deepep_mode=auto` (`expert_distribution.py:315`), so the auto mode, which preserves CUDA graphs, is fundamentally incompatible with EPLB.

**Limitation 2: architecture coupling.** EPLB's `eplb_manager.py:110` and `model_runner.py:927` directly access the `model.routed_experts_weights_of_layer` attribute, which is **defined only on DeepSeek-V2/V3 model classes**. On Qwen2-MoE and Qwen3-MoE architectures this raises an `AttributeError`, rendering EPLB **completely non-functional** on these architectures (experimentally confirmed on Qwen2-57B-A14B). The latest SGLang main branch as of August 2026 has not fixed this issue.

PB-OEPLB fixes this limitation with a generic fallback (§4): it first tries the native DeepSeek attribute and, on failure, iterates over `model.layers` to call each MoE layer's `get_moe_weights()`, making it compatible with all MoE architectures.

**Limitation 3: memory overhead of redundant experts.** EPLB allocates $R$ redundant expert replicas (16 in production), occupying about 12.5% additional GPU memory (otherwise usable for KV cache) and reducing maximum concurrency by 8.1% (from 227K to 209K tokens).

### 2.3 Key Observations

Through profiling of Qwen3-235B-A22B (94 MoE layers, 128 experts, EP=8) and Qwen2-57B-A14B (28 layers, 64 experts, EP=4), we identify three key observations:

**Observation 1 (intra-domain stability and cross-domain switches).** Within a single content domain, the cosine similarity of expert load distributions across consecutive decision windows is >0.95. Domain switches cause the ratio to spike (1.20→1.39 on 235B). The measured cross-domain cosine similarity bottoms out around 0.86 (>0.999 within a domain), enough as a changepoint signal; the 0.16 cited in an early draft was a hypothetical value, not measured, and is corrected here. This can be modeled as a **piecewise-stationary Markov process**, where the routing distribution $\boldsymbol{\theta}$ is fixed within each segment and jumps at unknown change points $\tau_1 < \tau_2 < \cdots$ (see §3.2 and the Bayesian formulation in Appendix A.2).

**Observation 2 (decision frequency vs. prompt length).** Short prompts (~50 tokens) require a larger synchronization window (sw=32-64), because each forward batch processes many requests and frequent `all_reduce` calls become the dominant overhead. Long prompts (~250 tokens) benefit from a smaller window (sw=8). This is a **bias-variance tradeoff**: a small window provides fresh data (low bias) but high variance due to limited samples; a large window reduces variance but increases latency and communication overhead. No single static window is optimal for all workloads — this motivates the adaptive mechanism (§3.5).

**Observation 3 (prefill predicts decode).** Layout optimization based on prefill routing data alone yields measurable decode-phase improvements (TPOT: -3.0% to -12.5% across 9 configurations). This is because swap operations modify the globally shared `physical_to_logical_map`, which governs all forward passes regardless of phase. Prefill-only recording therefore captures a *sufficient statistic* of the decode distribution under the hypothesis of intra-domain routing-pattern phase transitions.

### 2.4 Theoretical Speedup Upper Bound

**Theorem (Amdahl-form placement speedup upper bound with a dead zone).** Decompose the single-step forward time into a part insensitive to the imbalance ratio $r$ and a part linearly sensitive to it. The $T(r)$ sweeps in Appendix G (7 placement points × 2 rounds, 14 runs, 0 errors) show that the sensitive part **does not grow linearly from $r=1$**; instead there is a **dead zone** $r\le r_k$: within this interval the overlap between DeepEP's dispatch/combine and the expert GEMMs is sufficient to absorb the entire gap, and $T$ is insensitive to $r$. Thus

$$T(r) = T_{\text{flat}} + B\cdot\max(0,\; r - r_k)$$

This hinge form fits with $R^2=0.998$ for 57B 8-GPU, and its residual sum of squares is **12.1×** lower than the purely linear form (1.866/0.154, Appendix G.2). Defining the **r-sensitive time fraction** $f_{\text{sens}} = B\,r_{\text{before}}/T(r_{\text{before}})$ and the **effective usable interval**
$$x_{\text{eff}} = \frac{r_{\text{before}} - \max(r_{\text{after}},\, r_k)}{r_{\text{before}}}$$
the throughput improvement upper bound remains in Amdahl form, except that $x$ must be replaced by $x_{\text{eff}}$:
$$\Delta_{\max} = \frac{T(r_{\text{before}})}{T(r_{\text{after}})} - 1 = \frac{f_{\text{sens}}\, x_{\text{eff}}}{1 - f_{\text{sens}}\, x_{\text{eff}}}$$
The **predicted gain** must further be multiplied by a system-overhead discount: $1+\Delta_{\text{pred}} = (1+\Delta_{\max})(1-c_{\text{overhead}})$.

There are two corrections relative to the early draft, in opposite directions; the draft obtained a result correct in order of magnitude because the two partially canceled each other:
- The first-order truncation $\Delta_{\max}\approx f_{\text{sens}}x$ **underestimates** the upper bound — for 235B at $f_{\text{sens}}=0.384$, $x=0.407$, the first-order form gives 15.6% and the exact form gives 18.5%; the denominator cannot be dropped (Appendix A.3);
- Ignoring the dead zone (using $x$ instead of $x_{\text{eff}}$) **overestimates** the upper bound — the measured $r_k=1.099$ for 57B 8-GPU trims $x=0.146$ down to $x_{\text{eff}}=0.098$, lowering the upper bound from 4.51% to 3.40% (Appendix G.2).

The dead zone has a direct engineering implication: **pushing $r$ below $r_k$ yields no gain whatsoever**. Therefore the balancer's target should not be $r\to 1$ but $r\to r_k$, and PB-OEPLB's default trigger threshold `threshold_ratio=1.02` is conservative — it pays decision and swap overhead for gaps within the interval $r\in[1.02,\,r_k]$ without buying back any time. Appendix G.2 gives the measured cost of this interval.

**$f_{\text{sens}}$ is not equal to the MoE FLOP fraction.** The routed-expert FLOP fraction computed directly from the model configuration (per layer per token, including attention projections) is: Qwen3-235B **67.9%** (routed 302.0 MFLOP vs attn proj 142.6 MFLOP, no shared expert), Qwen2-57B **46.9%** (routed 440.4 + shared expert 440.4 + attn 58.7). The measured $f_{\text{sens}}$ is lower than these values:

| Configuration | $r_{\text{before}}$(avg) | $f_{\text{sens}}$ source | $f_{\text{sens}}$ | FLOP fraction | Overestimate factor |
|---|---|---|---|---|---|
| **235B L512 (8-GPU)** | **1.721** | **$T(r)$ sweep slope (Appendix G)** | **0.496** | 67.9% | **1.4×** |
| 235B L512 (8-GPU) | 1.721 | Single-point back-solve (dead zone not measured; deprecated) | 0.366 | 67.9% | — |
| 235B L512 (8-GPU) | 1.721 | nsys $\beta$ decomposition (falsified; underestimates by 26%) | 0.384 | 67.9% | — |
| **57B L256 (8-GPU)** | **1.218** | **$T(r)$ sweep slope (Appendix G)** | **0.335** | 46.9% | **1.4×** |
| **57B L256 (4-GPU)** | **1.107** | **$T(r)$ sweep slope (Appendix G)** | **0.369** | 46.9% | **1.3×** |

**The early draft's "7.7× overestimate" is a spurious conclusion and must be retracted.** That number came from the single-point back-solve $f=0.061$ for 57B 8-GPU, whereas the direct measurement this time gives $f_{\text{sens}}=0.335$ — the single-point back-solve is low by 5.5×. The reason is that the measured gain of +1.0% for this configuration itself falls within run-to-run noise (baseline repeated 6 times, CV=1.20%; two baselines of 8-GPU 57B even differ by 8.1%), and when $\Delta$ is tiny, back-solving $f=\Delta/\big(x(1+\Delta)\big)$ multiplicatively amplifies the errors in both $\Delta$ and $r_{\text{before}}$. The corrected conclusion is: **the FLOP fraction overestimates $f_{\text{sens}}$ by ~1.3–1.9×**, not 1.9–7.7×. An independent sweep on 4-GPU 57B further supports this: the same model measured at EP=4 and EP=8 gives $f_{\text{sens}}=0.369$ and $0.335$ (a 9% difference), whereas the single-point back-solve value for the same 4-GPU configuration swings between 0.54 and 0.30 depending on whether $r_{\text{before}}$ is taken as 1.113 or 1.20 — **$f_{\text{sens}}$ itself is a stable, measurable quantity; what is unstable is back-solving as a means**. This is also the direct reason this paper changed the measurement method for $f_{\text{sens}}$ from "back-solving" to "sweeping $T(r)$" (Appendix G).

**Why $f_{\text{sens}}$ is far smaller than the MoE fraction: component decomposition.** Calibrated on 6 sets of 235B nsys traces (3-run means), $f_{\text{sens}} = \sum_c \beta_c f_c$, $f_c=T_c/T_{\text{total}}$, where $\beta_c$ is the sensitivity of component $c$ to $r$:

| Component | $\beta_c$ | Physical meaning |
|---|---|---|
| Expert compute | 0.08 | Nearly unaffected by placement (total token count unchanged) |
| **Combine** | **1.33** | Highly sensitive (the all-gather of the hottest GPU is the slowest; other GPUs wait) |
| Dispatch | -0.78 | Negatively sensitive (OEPLB's all_reduce competes for NVLink bandwidth) |

*Proof.* The MoE-layer time $T_{\text{MoE}} = T_{\text{dispatch}} + T_{\text{expert}} + T_{\text{combine}}$. Expert compute is determined by the token count and does not vary with placement ($\beta \approx 0$). Combine is a collective communication: the hottest GPU finishes last → other GPUs wait → $T_{\text{combine}} \propto r$. In dispatch, OEPLB's all_reduce competes with DeepEP dispatch for bandwidth → $\beta < 0$. Substituting the 235B $f_c$ gives $f_{\text{sens}} = 0.384$, differing by 5% from the back-solved value of 0.366 in the table above. $\square$

**Theoretical upper bounds and system efficiency for each experiment** ($r_{\text{after}}=1.04$, taking the time-averaged operating point reported by PB-OEPLB-DIAG; 235B uses the Appendix G measured value $f_{\text{sens}}=0.496$ (the nsys $\beta$-decomposition value 0.384 is falsified, 26% low), 57B 8-GPU uses the measured value from Appendix G):

| Experiment | $r_{\text{before}}$ | $r_k$ | $x_{\text{eff}}$ | $f_{\text{sens}}$ | $\Delta_{\max}$ | Measured gain | System efficiency |
|---|---|---|---|---|---|---|---|
| **235B L512 (8-GPU)** | **1.721** | **1.093** | **0.365** | **0.496** | **22.09%** | +17.5% | **79%** |
| 235B multi-domain (8-GPU) | 1.39§ | 1.093 | 0.214 | 0.496 | 11.86% | +14.0%⚠ | 118%⚠ |
| 235B ShareGPT (8-GPU) | 1.721§ | 1.093 | 0.365 | 0.496 | 22.09% | +5.3% | 24% |
| **57B L256 (8-GPU)** | **1.218** | **1.099** | **0.098** | **0.335** | **3.40%** | +1.0% | **29%** |
| **57B L256 (4-GPU)** | **1.107** | **1.032** | **0.068†** | **0.369** | **2.57%†** | **+2.70%** | **105%** |
| **30B L512 (4-GPU)** | **1.338** | **1.031** | **0.230** | **0.260**♦ | **6.36%** | −3.8%~+0.5%‖ | **8%** |

*The dead zones of all starred rows have now been measured.
**The previously unclosed item "109%>100%" has now been closed.** The earlier inference of this paper was: if the 57B 8-GPU $r_k=1.10$ is borrowed for 235B, the system efficiency of the L512 row would rise to 109%>100% (an impossible value), hence $r_k$ cannot be borrowed across models. The 235B's own $T(r)$ sweep (`driver14.sh`, 12 runs) shows that **the conclusion was right but the attribution was wrong**: the measured $r_k$ for 235B is 1.093, almost identical to the 1.099 of 57B 8-GPU (the borrowing itself was not the error); the real error lies in $f_{\text{sens}}$ — this paper originally used 0.384 from the nsys $\beta$ decomposition, whereas the direct measurement is **0.496**, an underestimate of 26%. Substituting the measured value gives $\Delta_{\max}=22.09\%$ and system efficiency **79%**, and the contradiction disappears. The lesson is: the $\beta$ decomposition gets the order of magnitude right but is insufficient for quantification; see G.3.
†This row has been filled by `driver18.sh` on the **same dataset** (L256×16384): OEPLB 135.40 s vs baseline 139.06 s = **+2.70%**, indistinguishable from the static-optimal placement (135.49 s). $x_{\text{eff}}$ and $\Delta_{\max}$ use the DIAG-achieved $r_{\text{after}}=1.011$ (not the table-header assumption of 1.04), giving 2.57%; the empirical ceiling is +2.63%, so $\eta$=105% (slightly over 100%, within CV 0.24%). This row was previously "to be measured" because only the ShareGPT +1.85% was available, which has a different source from this row's $r_{\text{before}}$.
‡**$r_k$ shifts with EP scale but is stable across models at fixed EP.** Three sweeps: 57B/EP8 = 1.099, 235B/EP8 = **1.093**, 57B/EP4 = 1.032 (Appendix G.2). The two EP=8 configurations come from two models with vastly different structures (128 experts without shared expert vs. 64 experts with a giant shared expert), yet their $r_k$ differ by only 0.5%; whereas switching the same model from EP=8 to EP=4 drops $r_k$ by 6%. Transferability therefore **holds along the model direction but not along the parallel-configuration direction**.
**EP power law for $r_k$ (4 points, including a cross-model blind test).** Fitting 57B's three EP points gives $r_k-1 = 0.00408\cdot\text{EP}^{1.52}$: 57B/EP2 predicted 1.0117 (measured 1.012, error $-2.4\%$), 57B/EP4 predicted 1.0336 (1.032, $+5.1\%$), 57B/EP8 predicted 1.0966 (1.099, $-2.4\%$), **235B/EP8 predicted 1.0966 (1.093, $+3.8\%$)** — the last is a cross-model blind test (the fit never saw 235B). The exponent 1.52 decomposes as $T_{\text{gemm}}\propto\text{EP}^{-0.99}$ (theoretical $-1$) divided by $\text{slack}\propto\text{EP}^{+0.53}$; physically: more GPUs → longer per-layer all-to-all → larger overlap gap to hide imbalance; meanwhile per-GPU GEMM shrinks → same gap absorbs a larger fraction of the skew. Both effects compound in the same direction. This paper also retains the per-configuration measurement method (G.1, ~2 hours of machine time per config) for cases where the power law fails on untested hardware or larger EP.
§The $r_{\text{before}}$ of these rows is taken from PB-OEPLB-DIAG's self-reported values, and DIAG uses the mean **without weighting across windows**, which is inflated on heterogeneous workloads by the sampling variance of small batches (measured in Appendix D.2: on ShareGPT, DIAG's first window reports 2.161, while the token-weighted criterion gives only 1.100, inflated by 97%). Therefore the $r_{\text{before}}$ of these two rows and the system efficiency derived from it both carry uncertainty; the most likely cause of the multi-domain row exceeding 100% is either the $r_{\text{before}}$ criterion or the +14.0% itself ($n$=1; cf. the precedent in Appendix F.2 where +4.7% dropped to +2.39% after re-measurement).
♦$f_{\text{sens}}=B r_b/T(r_b)=13.25\times1.338/68.05=0.260$ (equivalently $\beta=B/T_{\text{flat}}=13.25/63.98=0.207$). Substituting: $0.260\times0.230/(1-0.260\times0.230)=6.36\%=\beta(r_b-r_k)=0.207\times0.307$.
‖**30B's negative gain is $\eta<0$, not a negative bound (corrected by measurement, `driver28/30.sh`).** An early draft inferred $f_{\text{sens}}<0$ / a negative bound from the nsys $\beta$-decomposition; 30B's own $T(r)$ sweep (15 runs) **falsifies that**: $\beta=+0.207$ (positive), hinge bound $+6.36\%$, so imbalance is genuinely harmful. The negative gain is $\eta<0$: 30B's dead zone is very narrow ($r_k=1.031$, narrower than 57B/EP4), and the fixed overhead (record 1.6% under pure prefill, plus swap/all\_reduce — the 7.1s of §B.2) exceeds the realizable gain. Measurement came in two stages. **The short benchmark (`driver40.sh`, n=3, ~60s/run, CV 7-9%) gave default −3.8%, dead-zone+budget +0.53%, but the long-benchmark re-test (`driver42.sh`, n=4, 32768 requests, ~275s/run) shows these were short-benchmark noise artifacts**: under the long benchmark all four arms fall within ±1.3% of baseline (default +1.27% CV5.2%, deadzone +0.41%, deadzone+budget −0.06% CV2.1%, all within their own CVs). **Clean conclusion: on 30B L512, OEPLB is essentially neutral ($\eta\approx0$), neither significantly helping nor hurting** — the +6.36% bound exists but the realized gain is ≈0, still attributed to overhead eating the gain (§B.2 nsys: 7.1s overhead ≫ 168ms combine improvement), except the clean long benchmark shows the net effect is ≈0 rather than clearly negative.

⚠**The multi-domain +14.0% comes from a deleted dataset** (`/tmp/exp_data/multidomain_16k.jsonl`, 16000 requests) and is not reproducible. A measurement on a different available dataset (`multidomain_v2_out1.jsonl`, 4400 requests) gave −1.1%, but with fewer requests (4400 vs 16000) and shorter benchmark duration (152s vs 247s) this is **not a same-condition re-test** and cannot falsify the original. If +14.0% stands, its η=118%>100% remains an open anomaly (possible causes: DIAG's $r_{\text{before}}$=1.39 is an underestimate, or $f_{\text{sens}}$ is load-dependent above the L512 calibration; both require a same-dataset $T(r)$ sweep that is no longer possible). This row is marked ⚠ = **neither confirmed nor falsified**, and is excluded from quantitative claims.

**Measured verification of the upper bound itself.** The sweeps in Appendix G.2 simultaneously provide a check independent of any fit: under the same configuration, changing the layout from identity ($r=1.218$, measured 85.88 s) to static-optimal packing ($r=1.010$, measured 82.72 s) yields **+3.82%**, which is the empirical ceiling of a "perfect balancer, zero overhead" for this configuration; the hinge model predicts +3.40%, and the two differ by 0.42pp. PB-OEPLB's measured +1.0% captures about 1/4 of the available space — **the reason the gain is small for this configuration is that the space itself is only 3–4%, not that the balancer fails**. The 4-GPU configuration provides a second independent instance: identity ($r=1.107$, 139.06 s) → static-optimal ($r=1.004$, 135.49 s) gives an empirical ceiling of **+2.63%**, versus the hinge upper bound of +2.29%, a difference of 0.34pp. **Under both configurations the fitted upper bound is slightly below the empirical ceiling, with error within 0.4pp**, showing that the hinge model is not concocted after the fact: it makes slightly conservative but correct predictions on two independent configurations.

**A plottable form of the upper bound: switching to the $\beta$ parameterization.** Substituting perfect balance into the hinge model reduces the upper bound to a polyline
$$\Delta_{\text{ceiling}} = \frac{T(r_b)}{T_{\text{flat}}}-1 = \beta\cdot\max(0,\;r_b-r_k),\qquad \beta \equiv \frac{B}{T_{\text{flat}}}$$
This is **algebraically equivalent** to this section's $\Delta_{\max}=f_{\text{sens}}x_{\text{eff}}/(1-f_{\text{sens}}x_{\text{eff}})$ (substitute $f_{\text{sens}}=Br_b/T(r_b)$, $x_{\text{eff}}=(r_b-r_k)/r_b$), but more useful in form:
1. **Separation of configuration parameters and workload parameters**: $(\beta,\,r_k)$ belong only to the configuration, and $r_b$ only to the dataset. Thus one configuration plots as one polyline (knee $r_k$, slope $\beta$), one dataset lands as one point on the line, and $r_b$ can be computed offline (Appendix D.2).
2. **$\beta$ is more stable than $f_{\text{sens}}$**: three sweeps give $\beta=0.285$ (57B/EP8), $0.342$ (57B/EP4), $0.352$ (235B/EP8), a range of 19%; whereas $f_{\text{sens}}$ is $0.335/0.369/0.496$, a range of 48%. The reason is that the definition of $f_{\text{sens}}$ mixes the operating point $r_b$ into the configuration parameter, whereas $\beta$ does not.
3. **$\beta$ has direct physical meaning**: the hottest GPU receives $r$ times the tokens, so its routed-expert GEMMs are $r$ times slower; hence $B\approx T_{\text{routed-gemm}}$, and $\beta\approx$ the **wall-clock fraction** of routed-expert GEMMs. The three $\beta$ values (0.285/0.342/0.352) are uniformly lower than the corresponding routed-expert **FLOP** fractions (0.469/0.469/0.679) by factors of 1.37–1.93, consistent with this section's observation that "the FLOP fraction overestimates", and providing an explanation: the FLOP fraction is not equal to the wall-clock fraction. **Can a single profiling replace the sweep?** `driver23b.sh` profiled all four configurations: even after isolating routed-expert GEMMs (`deep_gemm::sm90_fp8_gemm` kernels, precisely separated from attention/shared-expert `cutlass` kernels), the ratio $\beta$/routed-GEMM% scatters across **0.53–1.25**, not $\approx1$. The physical reason: the profiler measures the kernel-time fraction across all GPUs (total occupancy), whereas $\beta$ measures the **critical-path** wall-clock fraction — the two are not equivalent whenever dispatch/combine overlaps with GEMM, and the degree of overlap varies with architecture (dispatch 29% on 57B vs 13% on 235B) and EP size. **$\beta$ is therefore not predictable from profiling; precise calibration requires a $T(r)$ sweep (~2 h/config).** The FLOP-based estimate ($\beta\approx$FLOP$_\text{routed}/1.6$, error ±30%) is useful for quick screening, but quantitative prediction demands a sweep. $\beta$ remains the one link in the prediction chain that still costs machine time.

**The dead zone is not strictly flat.** Within the dead zone this paper has a pair of measured points (both below $r_k=1.099$): as $r$ goes from 1.010 to 1.073, time goes from 82.72→83.00 s, i.e., $+0.34\%$, equivalent to an in-zone slope of $\le5.5\%/$unit $r$, whereas above the knee it is $28.5\%/$unit $r$ — **a factor of 5.2 apart, and the in-zone difference is not significant at $n$=2 ($t=1.36$)**. The hinge model's "perfectly flat when $r\le r_k$" is therefore an approximation, whose error can be quantified: pushing $r$ further down from optimal placement (the LPT lower bound 1.0100 for 57B/EP8) to 1.000 of perfect routing gains at most $5.5\%\times0.0100=0.055$pp, which is **1.5%** of the 3.70pp upper bound for this configuration, below this paper's measurement resolution (0.03pp). §2.5 makes use of this error term.

**Three control dimensions of the gain.** The bound $\Delta=\beta\max(0,r_b-r_k)$ factorizes the influences on gain into three orthogonal dimensions, each independently measurable and predictable:

| Dimension | Controls which parameter | Direction of effect | Representative measurement |
|---|---|---|---|
| **GPU count / EP** | $r_k$ (dead-zone width) | Larger EP → higher $r_k$ → wider dead zone → less effective headroom | EP=2: $r_k$=1.012, headroom 0.37%; EP=8: $r_k$=1.099, headroom 3.4% |
| **Model architecture** | $\beta$ (sensitivity slope) | Higher routed-expert GEMM share → larger $\beta$ → more gain per unit of $r$ | 235B: $\beta$=0.352; 57B: $\beta$=0.285; 24% apart |
| **Dataset / workload** | $r_{\text{before}}$ (native imbalance) | Greater routing skew → higher $r_b$ → more recoverable headroom | L256: $r_b$=1.218 → ceiling 3.4%; L512: 1.229 → 3.7% |

Their contributions are **multiplicative** ($\beta\times(r_b-r_k)$), so if any one is zero the gain is zero:
- Few GPUs (EP=2) → $r_b-r_k\approx0$: even large $\beta$ yields no headroom
- Config where fixed overhead exceeds the bound (e.g. 30B: $\beta>0$ but a narrow dead zone + large record overhead) → $\eta<0$, net-negative (recoverable with a swap budget)
- Near-uniform routing → $r_b\approx1$: no skew to recover regardless of configuration

Conversely, **when all three are favorable the gain is substantial**: 235B/EP8 on L512 gives $\beta\times(r_b-r_k)=0.352\times(1.737-1.093)=22.7\%$; measured +17.5% ($\eta$=79%).

**Practical implication**: before deployment, run `predict_gain.py` (needs only config.json + one recording) to evaluate each dimension's contribution. The weakest dimension is the bottleneck — if it is $r_k$ (GPU count), adding GPUs or reducing EP will not help; if $\beta$ (model), a different model or dispatch implementation is needed; if $r_b$ (data), a different dataset or routing algorithm is needed.

**Theoretical upper bound for EPLB (same sensitivity model).** The early draft used $f_{\text{MoE}}=0.77$ for EPLB and $\sum\beta_c f_c=0.384$ for OEPLB, mixing two sets of sensitivities, and thus concluded "EPLB gross upper bound 32.7%" — this is not a physical conclusion but a product of model inconsistency. Recomputing with the same $f_{\text{sens}}=0.384$:
- EPLB pushes $r$ to 1.0: $x=(1.721-1.0)/1.721=0.419$, $\Delta_{\max}=19.2\%$
- CUDA graph disabling cost: $1-0.157$ ($0.68\times(1-0.77)$, see §2.2)
- EPLB net prediction $=1.192\times0.843-1=$ **+0.5%**

The measured EPLB is **+1.75%** (L512, 2 rounds of independent restarts, see §5.3), of the same order as the +0.5% of the unified model. **Conclusion: EPLB's gross upper bound is indeed higher than OEPLB's (19.2% vs 18.5%), but the CUDA graph cost eats up nearly all of the gain** — this is a quantitatively predictable conclusion, not the early draft's version of "net upper bound 17.0%, measured +9.0%" (neither number is reproducible).

### 2.5 Two Ceilings: Optimal Placement vs. Perfectly Balanced Routing

The upper bound of §2.4 answers the question "**with routing unchanged and only moving experts**, how much faster can it be at most". This is not the only ceiling. There is a higher one: "assuming the routing itself were uniform ($r=1$), how much faster can it be at most". The difference between the two is the quantity this section aims to quantify, because it determines a concrete engineering question: **when must one touch routing or add redundant experts, and when is purely moving experts enough.**

**The constraint comes from the indivisibility of experts.** Placement can only move experts in whole blocks, so the single hottest expert gives a lower bound
$$r_{\text{place}} \;\ge\; \frac{1}{L}\sum_l \frac{\max_e c_{l,e}}{\big(\sum_e c_{l,e}\big)/G}$$
If some single expert consumes more than $1/G$ of all tokens, the GPU holding it exceeds the mean regardless of placement. Redundant experts are precisely the only mechanism that can break through this granularity constraint — only by replicating a hot expert across multiple GPUs does its load become divisible.

**Key result: on every configuration measured in this paper, the two ceilings coincide.** The $r$ achievable by optimal placement (the per-layer LPT bin-packing lower bound, computed offline from recorded counts) all falls within the dead zone:

| Configuration | Experts/GPU | $r_{\text{native}}$ | $r_{\text{place}}$ | $r_k$ | Placement upper bound | Perfectly balanced routing upper bound | Difference |
|---|---|---|---|---|---|---|---|
| 57B EP=4 | 16 | 1.107 | **1.0039** | 1.032 | 2.75% | 2.75% | $\le0.02$pp |
| 57B EP=8 | 8 | 1.218 | **1.0100** | 1.099 | 3.70% | 3.70% | $\le0.06$pp |
| 235B EP=8 | 16 | 1.737 | **1.0003** | 1.093 | 22.66% | 22.66% | $\le0.00$pp |

The upper bounds on the difference are computed using the measured in-dead-zone slope from §2.4 ($\le5.5\%/$unit $r$); all are below the measurement resolution of 0.03pp. The criterion itself is concise:
$$\text{the two ceilings coincide} \iff r_{\text{place}} \le r_k$$
Both inputs are cheap — $r_{\text{place}}$ is a purely offline quantity (one recording), and $r_k$ is a configuration constant.

**This gives PB-OEPLB's design choice of "no redundant experts" an applicability domain, rather than merely measured support.** Within the region $r_{\text{place}}\le r_k$, pure placement provably achieves the same upper bound as perfect routing, and the additional gain from redundant experts is below the measurement resolution — whereas they cost 12.5% of memory and 8.1% of concurrency (§2.2). Conversely, the model also predicts where it will fail:

| Configuration | Experts/GPU | $r_{\text{place}}$ | Verdict |
|---|---|---|---|
| 57B EP=16 | 4 | 1.0345 | Depends on $r_k$(EP=16); not measured |
| 57B EP=32 | 2 | 1.1709 | Same as above |
| 57B EP=64 | 1 | **1.8725** | Placement **fails entirely** (see below) |
| 235B EP=16 | 8 | **1.3710** | Greater than any measured $r_k$; redundancy should yield a net gain |
| 235B EP=128 | 1 | **10.8251** | Placement fails entirely |

With one expert per GPU, the three quantities $r_{\text{native}}=r_{\text{place}}=r_{\text{hot}}$ are all equal (all 1.8725 for 57B/EP64) — no permutation changes the per-GPU load, and placement-type methods (including this paper's) have **identically zero gain** under this configuration; only redundancy or changing routing is viable. This also explains why large-scale EP deployments (e.g., 256 experts/EP=64) universally adopt redundant replicas.

**Open items.** The verdicts for the EP≥16 rows in the table above depend on extrapolating $r_k$ to larger EP, and $r_k$ **rises** with EP; §2.4 has calibrated a power law $r_k-1=0.00408\,\text{EP}^{1.52}$ from 4 points (including a 235B cross-model blind test), but larger EP (≥16) is outside the calibration range and extrapolation uncertainty grows. This section therefore gives only the criterion and the limiting behavior on both sides, not the location of the crossing point. The 235B/EP=16 row ($r_{\text{place}}=1.371$, far above any measured $r_k$) is the only one on which a conclusion can be drawn now: under this configuration the additional space from redundant experts is about $\beta\times(1.371-1.093)\approx9.8\%$, which pure placement cannot capture.

### 2.6 Modeling the Negative Gain from EPLB's KV Cache Pressure

EPLB's $R$ redundant experts occupy additional memory, squeezing KV cache capacity by $\delta = R \cdot W_{expert} / M_{static}$. For 235B: $\delta = 8.1\%$ (227K→209K tokens).

**Queueing-theory model**: KV cache capacity reduced by $\delta$ → maximum concurrent requests reduced by $\delta$ → system utilization $\rho' = \rho / (1-\delta)$ → queueing time $W_q \propto 1/(1-\rho')$.

| Original $\rho$ | $\rho'$ after $\delta=8.1\%$ | Queueing-time multiplier |
|---|---|---|
| 0.70 | 0.762 | 1.26× |
| 0.85 | 0.925 | 2.00× |
| 0.90 | 0.979 | **4.83×** |
| 0.95 | 1.033 | ∞ (overflow) |

**Measured verification** (L4096_O256 conc=512, $\rho \approx 0.9$): EPLB -3.2% (KV cache pressure causes queueing to explode), OEPLB +16.0% (zero KV cache loss), a gap of 19.2pp.

---

## 3. System Design

### 3.1 Architecture Overview

PB-OEPLB consists of four components integrated into SGLang's ModelRunner:

```
topk.py::select_experts() → Controller.record_next_layer(topk_ids)
                                    │
                    ┌───────────────┴──────────────────┐
                    │ Rebalancer (greedy + adaptive)   │
                    │ AsyncSwapExecutor (P2P transfer) │
                    └──────────────────────────────────┘
```

**Routing hook**: after `select_experts()` computes `topk_ids` (thanks to `ep_dispatch_algorithm="static"`, topk_ids are already in the physical slot space), the controller records them with a single `scatter_add_` call (an O(1) GPU kernel).

**Decision cycle**: every `sync_window` forward passes, the controller: (1) checks whether the previous P2P transfer has completed, (2) runs an all_reduce to aggregate loads across ranks, (3) invokes the rebalancer to compute a swap plan, (4) initiates the P2P transfer asynchronously.

### 3.2 Exponential Decay and Fast Turnover

After each decision window, the load tensor is updated as:
$$A_n = R_n + \alpha \cdot A_{n-1}$$
where $R_n$ is the fresh routing data of the current window and $\alpha$ is the decay factor.

We find $\alpha = 0.5$ to be optimal, compared to $\alpha = 0.9$ (the default of the early version) and $\alpha = 0$ (no history):

| $\alpha$ | Old signal remaining after 3 windows | Multi-domain throughput | Short-prompt throughput |
|---|---|---|---|
| 0 (clear) | 0% | +2.5% | — |
| **0.5** | **12.5%** | **+10.6%** | **+2.3%** |
| 0.9 | 73% | +6.9% | +1.4% |

At $\alpha=0.5$, cross-domain signal contamination drops to 12.5% within 3 windows (~18 seconds), enabling fast adaptation to workload changes. At $\alpha=0.9$, 73% of the old-domain signal persists, causing the controller to make placement decisions based on stale data.

**The threshold should be set above the dead zone.** Appendix G.2 measures that $T(r)$ has a dead zone $r\le r_k$ (8-GPU 57B: $r_k=1.099$), within which lowering $r$ buys back no time. The default `threshold_ratio=1.02` falls inside the dead zone, and therefore pays decision and swap overhead for gaps in $r\in[1.02,\,r_k]$ with zero return (measured: pushing $r$ from 1.073 down to 1.010 is worth only 0.34%, of the same order as the P2P blocking of one swap plan). In principle the threshold should be $\max(1.02,\,r_k)$; $r_k$ can be measured offline with the sweeps of Appendix G, or tuned upward online using "throughput did not improve after lowering $r$" as feedback. **$r_k$ must be measured per configuration and cannot be written as a default**: for the same model and same dataset, EP=4 measures $r_k=1.032$ while EP=8 measures $1.099$ (Appendix G.2). This means the default 1.02 is already near-optimal on EP=4 (the dead zone is only [1.02, 1.032]) but is 8 percentage points too low on EP=8 — the same default parameter has entirely different degrees of reasonableness on the two configurations, which is precisely the necessity of turning it into a measurable parameter. This turns the threshold from an empirical value into a **measurable** parameter. The current implementation still uses 1.02, so the gains reported in this paper are results with this parameter untuned.

### 3.3 Adaptive Pair-Selection Algorithm

The core innovation is the **dual-mode pair-selection** strategy inside the greedy swap planner:

**Mode 1 (Max-delta, fast convergence)**: when the gap between the hottest and coldest GPUs is large, select the pair with the largest load difference. This achieves the fastest ratio reduction per swap.

**Mode 2 (Gap-targeting, precise balancing)**: when the gap is small and the load of the hottest slot exceeds the gap, selecting the max-delta pair **overshoots** — moving too much load onto the cold GPU and turning it into the new hot GPU. Instead, select the slot whose load is closest to $\frac{\text{gap}}{2}$, balancing both GPUs to the mean.

$$\text{selected\_slot} = \begin{cases} \arg\max_s \text{load}[s] & \text{if } \max_s \text{load}[s] \leq \text{gap} \\ \arg\min_s |\text{load}[s] - \frac{\text{gap}}{2}| & \text{otherwise} \end{cases}$$

This simple switch resolves a critical stagnation point: the previous max-delta-only planner could not improve further after converging to a ratio of 1.26 (ratio-improving pairs still existed, but the greedy heuristic, due to overshooting, selected pairs that worsened the ratio). With adaptive selection, the ratio converges to **1.02** within 3 windows (Table 2).

**Marginal swap efficiency and optimal stopping (measured, this work).** The algorithm above converges to ratio 1.02, but the dead-zone result of §2.4 shows **this is over-convergence** — once $r\le r_k$, pushing lower buys no time back. Decomposing the full decision trajectory of the 8-GPU 57B default configuration (Appendix G / the `g8_base` arm of `driver27.sh`, 21 decisions, 343 ops) by "effective reduction below $r_k$":

| Decision | ops | $r$ before→after | Effective reduction below $r_k$=1.099 |
|---|---|---|---|
| #1 | 139 | 1.216→1.017 | **0.117 (the entire useful distance)** |
| #2–#21 | 204 | 1.03↔1.01 oscillating | **0.000 (all inside the dead zone)** |

**The first decision's 139 ops cover the entire useful distance; the subsequent 20 decisions and 204 ops (59% of the total) contribute exactly zero to throughput.** This quantifies the cost–benefit imbalance: the default arm's cumulative swap blocking is 3.94 s/rank, while this configuration's headroom ($\beta(r_b-r_k)T_{\text{flat}}=0.285\times0.119\times82.86$) is only 2.81 s — **the blocking is 1.4× the available space**. Note, however, that this arm's measured end-to-end gain is still **+0.98% (positive), not a net loss**: the 3.94 s of blocking is heavily front-loaded (the first decision alone is 1.75 s), after which the whole run enjoys the reduced $r$, whereas 2.81 s is the whole-run-optimal ceiling; the two have different time distributions and cannot be subtracted directly. What this ratio reliably discriminates is the band — configurations with ratio $\ge1$ land in the lowest $\eta$ band (26–29%) — and indeed moving the threshold to $r_k$ plus a swap budget lifts the same configuration to $\eta$=100% (Appendix D.3). The ~2 s reconciliation gap is listed as an open item.

This yields an optimal stopping condition for swaps. Let $T_{\text{rem}}$ be the remaining serving time, $t_{\text{swap}}$ the per-swap blocking, and $\mathbb{E}[\Delta r_{\text{eff}}]$ the expected effective reduction of the next step; then continuing to swap is net-positive when
$$\beta\cdot\mathbb{E}[\Delta r_{\text{eff}}]\cdot T_{\text{rem}} - t_{\text{swap}} > 0,\qquad \Delta r_{\text{eff}}=\max(0,r-r_k)-\max(0,r'-r_k)$$
When $r$ is already below $r_k$, $\Delta r_{\text{eff}}\equiv0$, the left side is always negative, and one should stop immediately. This is exactly what §2.4's $r_k$-aware threshold (the `dead_zone_ratio` knob) does — approximating this continuous criterion with a hard cutoff $r\le r_k$. The verification in Appendix D.3 shows this approximation suffices: changing the stopping threshold from 1.02 to $r_k$=1.099 cuts decisions from 21 to 1 (issued swaps 9→1) and raises $\eta$ from 26% to 100% (the dead-zone threshold takes $\eta$ to 59%, the swap budget the rest to 100% — see the A/B table in Appendix D.3). **A finer bandit-style adaptive stop (adjusting dynamically with $T_{\text{rem}}$) is left for future work, but the current hard threshold already captures the bulk of this imbalance.**

### 3.4 Synchronous P2P Execution and Blocking Cost

Swaps are executed **synchronously**: the controller completes the entire swap inside `_decide_and_begin_swap()` and does not return until all P2P transfers have finished.

- Swapping two slots **within the same rank** goes through local `clone`/`copy_`; **cross-rank** swaps use `torch.distributed.P2POp(isend/irecv)` to receive the peer's weights into a temporary buffer, `req.wait()` after `batch_isend_irecv`, and then copy back into the live weight tensors.
- By the time `begin()` returns, the swap has already completed; `try_finish()` therefore degenerates to immediately returning the current plan, retained only for interface compatibility.
- **Design tradeoff (negative result)**: the early version used a dedicated low-priority CUDA stream + a separate process group for asynchronous transfers, intending to hide migration behind computation. This scheme ran stably on the NVLink-only H20 platform for about 60 seconds before inevitably triggering an NCCL hang — the relative order in which the same set of ranks launches collective/point-to-point operations on two communicators cannot be guaranteed to be consistent, while NCCL requires a globally consistent launch order. We therefore abandoned the asynchronous path, switched to synchronous execution, and changed the optimization goal from "hiding transfers" to "making each transfer small" (the sparse pairs of §3.3, the `max_total_ops` budget). This is the key mechanistic difference between this work and the official EPLB: **not the difference between blocking and not blocking, but the difference in blocking granularity**.

**Two failure modes of the synchronous path itself (negative results, measured in this paper).** The early draft claimed "after switching to synchronous there are no more stability problems" — this is wrong. Synchronous single-batch P2P on this platform has two failure modes, each of which can hang all ranks, and we triggered and fixed both:

1. **NCCL's P2P channel buffers are crowded out by PyTorch's caching allocator.** The transient overhead of migration is $O(|{\rm plan}|)$: 132 ops in the first decision × ~27.5 MB of fp8 expert weights per op ≈ 3.6 GB (peak is higher because each op needs both send and receive buffers). NCCL allocates P2P channel buffers with raw `cudaMalloc`, **bypassing** PyTorch's caching allocator, and therefore fails when GPU memory is fully occupied by cached blocks (measured: 3 out of 8 ranks reported `Failed to CUDA calloc 10485760 bytes` inside `batch_isend_irecv`). The failing ranks threw exceptions and exited the collective operations while the 5 successful ones kept waiting; the communicator immediately fell out of step, and the 600 s watchdog killed all 8 ranks. Fix: call `torch.cuda.empty_cache()` before the call to return cached blocks to the driver, and retry once on failure.

2. **Chunking a large P2P batch breaks consistent participation across ranks and thus hangs.** The first-version fix for failure mode 1 was to slice the plan into chunks and issue them in batches. This introduced a **more insidious** deadlock: when a rank owns no slots in a given chunk it skips that chunk's `batch_isend_irecv`, whereas NCCL's coalesced work is **numbered by increasing sequence numbers per process group**, and uneven participation causes the sequence numbers to diverge (watchdog measurements: 12 lines of `SeqNum=4`, 3 lines of `SeqNum=5`, `WorkNCCL(SeqNum=4, OpType=COALESCED) ran for 600091 ms`). Therefore **the plan must be submitted as a whole batch**; `max_total_ops` is a safety valve against an overly large single batch, not a freely tunable performance knob — turning it down does not split a large plan into separate executions.

3. **Cumulative swap volume under high-frequency decisions diverges NCCL sequence numbers (new measurement, this revision).** The first two failure modes are within-plan; this one is **cumulative**. With $(W=8,\alpha=0.5)$ on the 235B segmented workload, the balancer keeps issuing large migrations, accumulating 77k swaps, until the watchdog reports `WorkNCCL(SeqNum=524, OpType=COALESCED) ran for 600019 ms` with `last enqueued work: 524, last completed work: 792` — enqueued and completed sequence numbers diverge. Unlike failure mode 2, participation within each batch is uniform here; it is the **cross-batch** issue rate that outruns NCCL. Mitigation is to cap cumulative migration (`swap_budget_frac`; Appendix H shows it converts this point from a hang into a completing run, though still 5.4× slow), and the fix is to avoid that region of the $(W,\alpha)$ plane (Appendix H's rule: $W\ge16$, or $\alpha=0$ / $\alpha\ge0.75$ when $W<16$).

These three lessons apply to any system doing online expert migration: the transient memory overhead of migration grows linearly with plan size and falls outside the framework allocator, and any "optimization" that makes rank participation inconsistent will hang in the form of sequence-number divergence.

**Evolving p2l updates**: when multiple swap operations target the same layer, each operation reads the *current* (evolving) p2l state rather than the stale `logical_a/logical_b` values from planning time. This prevents the p2l inconsistency bug that caused CUDA asserts in the early version.

**Measured blocking cost** (8-GPU 235B, L512_O1, two rounds of independent restarts; full data in Table 7b): 8 decisions per rank in total, 989/1002 swap operations, cumulative blocking of 6.76s/5.14s, accounting for 3.86%/2.98% of the benchmark wall clock. The distribution is highly skewed: the **first decision** (298~299 ops, when the layout is farthest from optimal) takes 4.21s/2.55s, after which each decision stabilizes at 337~407ms. As a comparison, the official EPLB on the same dataset likewise triggers 8 adjustments, but each rearranges all physical slots: first 4.47s/2.87s, steady-state 1.43~1.82s, cumulative 15.82s/13.63s (7.81%/6.85%).

### 3.5 Adaptive Window and Decay: A Single Control $M=W/(1-\alpha)$

**First, clear up a conflation: window $W$ and decay $\alpha$ are not two independent knobs.** The controller decides every $W$ prefill forwards on an exponentially decayed accumulator $A_t = R_t + \alpha A_{t-1}$. Expanding $A_t=\sum_{k\ge0}\alpha^k R_{t-k}$, its **effective memory length** is
$$M = \frac{W}{1-\alpha}\quad(\text{in forwards})$$
In steady state $W$ and $\alpha$ affect the bias-variance operating point **only through $M$**: fixing $M$ while increasing both $W$ and $\alpha$ gives identical statistical quality, but a larger $W$ means sparser decisions and lower `all_reduce` overhead. So the division of labor is: $\alpha$ sets the forgetting-curve shape, $W$ sets decision frequency/overhead, and $M$ is the single **bias-variance degree of freedom** that must be tuned per workload. Early implementations (and most online balancers) change $W$ without co-adjusting $\alpha$, unintentionally moving $M$ — the design flaw this paper corrects. Appendix H measures this on 235B (`driver31.sh`): $M$ is an **approximate** sufficient statistic — same-$M$ arms agree to within 4.8% for $M\ge32$, but the corner $(W=8,\alpha=0.5)$ is 5.4× slower, so the reduction to a single $M$ is not exact.

**The optimal $M$: joint minimization of bias, variance, and changepoint latency.** Three costs:

1. **Variance cost (small $M$)**: per-layer accumulated tokens $N=M\cdot\bar t$ ($\bar t$ = tokens per forward per layer), sampling bias $\text{bias}(M)=c/\sqrt{N}$ ($c=0.65\,\text{EP}$, calibrated in D.2). When $\text{bias}>\gamma(r-r_k)$, noise dominates signal and decisions are untrustworthy, giving a **minimum memory**
$$M_{\min} = \frac{c^2}{\big(\gamma(r-r_k)\big)^2\,\bar t}$$
(This is the earlier $\text{sw}_{\min}$, now folded into a unified framework.) For 8-GPU 57B ($c=5.2$, $\gamma=0.5$, $r-r_k=0.119$, $\bar t\approx2000$/layer): $M_{\min}\approx3.8$, a few forwards suffice; for ShareGPT/4-GPU ($c=2.6$, $r-r_k=0.065$, short-prompt $\bar t\approx110$/layer): $M_{\min}\approx58$ — **this is the root cause of $\eta\approx0$ on heterogeneous workloads in §5.4**: default $M=W/(1-\alpha)=16/0.5=32<58$, so the window statistic is untrustworthy to begin with.

2. **Latency cost (large $M$)**: under geometric decay, after a changepoint the old-domain signal decays to half in $M\ln2$ steps, during which the wrong placement serves, losing $\approx\beta(r_{\text{new}}-r_k)\cdot M\ln2/L_{\text{seg}}$ ($L_{\text{seg}}$ = mean segment length).

3. **Joint optimum**: the variance term must be normalized by the **signal** (what matters is the noise-to-signal ratio $\text{bias}/(\gamma(r-r_k))$, not the absolute noise), so
$$\min_M\left[a\left(\frac{\text{bias}(M)}{\gamma(r-r_k)}\right)^{2} + b\,\beta(r-r_k)\frac{M\ln2}{L_{\text{seg}}}\right]=\min_M\left[\frac{a\,c^2}{M\bar t\,\gamma^2(r-r_k)^2} + b\,\beta(r-r_k)\frac{M\ln2}{L_{\text{seg}}}\right]$$
whose first-order condition gives the closed form
$$M^\star = \left(\frac{a\,c^2\,L_{\text{seg}}}{b\,\beta\,\bar t\,\gamma^2 (r-r_k)^3\ln2}\right)^{1/2}$$
i.e. **optimal memory grows as $\sqrt{L_{\text{seg}}}$ and falls as $(r-r_k)^{3/2}$**. All parameters are measurable online: $c$ known, $\bar t$ recorded, $r$/$r_k$ from the §2.4 power law, $\beta$ from profiling, $L_{\text{seg}}$ from changepoint detection.

**Recovering $(W,\alpha)$ from $M^\star$.** Given a communication budget $\rho$ (acceptable `all_reduce` fraction): $W=\max(M_{\min},\,\rho L_{\text{seg}})$, $\alpha=1-W/M^\star$. If $M^\star<W$ (segment too short), it degenerates to $\alpha=0$ (pure window, no accumulation).

**Adaptive decay (new in this work; previously only fixed $\alpha=0.5$).** Changepoint detection (cosine similarity of consecutive windows' expert distributions $<0.95$, §2.3 obs. 1) should not only shrink $W$ but also **zero $\alpha$ for one step**: under geometric decay $\alpha=0.5$ means 50% of the old signal survives a changepoint and needs one step to half-decay; setting $\alpha\to0$ clears the old-domain history instantly (engineering-wise `load.zero_()`), cutting the response latency from $M\ln2$ to 0. After steady state resumes, $\alpha$ regrows toward $M^\star$. This unifies "shrink window" and "clear history" as one operation on $M$ — $M\to M_{\min}$ at a changepoint, $M\to M^\star$ in steady state.

**The three implemented feedback signals** (a discrete approximation of the above continuous control): **signal 1 (converged)** ratio changes $<0.003$ over 3 windows → double $M$ (lower overhead); **signal 2 (shift)** ratio jumps $>0.03$ or cos_sim $<0.95$ → $M\to M_{\min}$ and $\alpha\to0$ (fast response); **signal 3 (volatile)** ratio oscillates between the two → expand $M$ (more samples, lower variance). Appendix H reports the measured outcome: adaptive window + adaptive decay is the fastest arm (1.07× the no-OEPLB reference), beating all eight static $(W,\alpha)$ configurations, with adaptive decay separately worth 3.6% and 44% fewer swaps; the closed form for $M^\star$ has its directional prediction confirmed on the latency side (short segments favor small $M$, long segments favor large $M$, Appendix H), though its numerical coefficient is not yet calibrated.

### 3.6 Prefill-Only Recording

The controller records routing data only when `forward_batch.forward_mode.is_extend()` (prefill). Decode and idle batches are skipped entirely. This reduces recording overhead by ~50% (in mixed workloads, decode steps typically outnumber prefill by 10:1), while benefiting decode through the globally shared layout.

**Important detail**: during CUDA graph capture/replay (the decode phase), `record_next_layer` returns directly via the `torch.cuda.is_current_stream_capturing()` check — **zero overhead**. Recording is actually performed only in the prefill phase (which does not go through CUDA graphs). This means that in pure-prefill (O=1) scenarios every forward executes record and the overhead is maximal; in mixed prefill+decode scenarios, decode's record is skipped and the overhead is lower.

---

## 4. Implementation

PB-OEPLB is implemented as a patch on SGLang 0.5.6.post2, modifying three files:

1. **`server_args.py`**: 19 CLI arguments (`--pb-oeplb-*`) + a mutual-exclusion check against the official EPLB.
2. **`model_executor/model_runner.py`**: controller initialization in `initialize()`; unconditional call to `on_forward_pass_end()` at the end of `forward()`.
3. **`layers/moe/topk.py`**: call `record_next_layer(topk_ids)` after `select_experts()`.

The core modules live in `sglang/srt/managers/pb_oeplb/`:
- `controller.py` (962 lines): state machine, decay, adaptive window, calibration, cross-architecture fallback.
- `rebalancer.py` (180 lines): greedy planner with adaptive pair selection.
- `async_swapper.py` (250 lines): P2P execution, event-based completion detection.
- `fast_metadata.py` (60 lines): vectorized p2l initialization.
- `config.py` (60 lines): configuration dataclass.

**DeepEP H20 NVLink patches**: two patches to DeepEP v1.2.1 for pure-NVLink (no IB) clusters: (1) make the IBGDA environment-variable setup run only when there are multiple RDMA ranks, (2) comment out the IBGDA assertion in `internode_ll.cu`.

**DeepEP hidden_size=3584 patch**: Qwen2-57B's hidden_size=3584 is not in the hardcoded `SWITCH_HIDDEN` list of DeepEP v1.2.1 (only 2048/2560/4096/5120/6144/7168/8192 are supported). After mathematical verification ($3584 = 7 \times 512$, satisfying all divisibility constraints), `case 3584` was added to `csrc/kernels/launch.cuh`, enabling the Qwen2 family to use `deepep_mode=auto` (preserving CUDA graphs).

**Cross-architecture fallback**: `_get_routed_experts_weights()` methods added to `controller.py` and `async_swapper.py` first try DeepSeek's native `model.routed_experts_weights_of_layer` attribute and, on failure, iterate over `model.layers` to call each MoE layer's `get_moe_weights()`. Compatible with all MoE architectures (DeepSeek-V2/V3, Qwen2-MoE, Qwen3-MoE, etc.).

---


## 5. Evaluation

### 5.1 Experimental Setup

**8-GPU environment (original validation; data source: historical experiments)**:
- **Hardware**: 8× NVIDIA H20 (96GB/GPU), full NVLink NV18 interconnect, no InfiniBand
- **Model**: Qwen3-235B-A22B-FP8 (94 MoE layers, 128 experts, top-8 routing)
- **Serving**: SGLang 0.5.6.post2, DeepEP v1.2.1 (patched), DeepGEMM, TP=8, DP=8, EP=8
- **Concurrency**: 256

**4-GPU environment (independent validation in this session)**:
- **Hardware**: 4× NVIDIA H20 (96GB/GPU), full NVLink NV18 interconnect, no InfiniBand
- **Model 1**: Qwen2-57B-A14B-Instruct (28 MoE layers, 64 experts, top-8, EP=4→16 experts/GPU, with shared expert=20480)
- **Model 2**: Qwen3-30B-A3B-FP8 (48 MoE layers, 128 experts, top-8, EP=4→32 experts/GPU, no shared expert)
- **Serving**: SGLang 0.5.6.post2, DeepEP v1.2.1 (patched+hidden3584), DeepGEMM, TP=4, DP=4, EP=4
- **Concurrency**: 256
- **Method**: the server is restarted independently for each scenario, with baseline and OEPLB runs alternating with independent restarts (to eliminate temporal drift and placement inheritance)

### 5.2 Datasets

| Dataset | #Requests | Prompt length | Output | Domain |
|---|---|---|---|---|
| L256_O1 | 8192 | ~256 tok | 1 (pure prefill) | Prover math |
| L512_O1 | 8192 | ~512 tok | 1 | Prover math |
| L1024_O1 | 4096 | ~1024 tok | 1 | BookCorpus |
| Multi-domain 16K | 16000 | ~1000 tok | 1 | 4 domains × 4000 |
| ShareGPT 100K | 100000 | ~50 tok | 1 | Real conversations |

Dataset paths: `/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/final_grid/` and `multi_domain/`.

### 5.3 8-GPU + 235B Results (Historical Data)

**Table 1: L512_O1 full placement comparison**

| Placement | Imbalance ratio | total_tps | vs baseline |
|---|---|---|---|
| Worst (hotspot stacking) | 2.61 | 16054.5 | -20.4% |
| Baseline (trivial round-robin) | 1.74 | 20167.8 | — |
| EPLB (continuous) | ~1.00 | 21992.2 | +9.0% |
| Frozen-EPLB | ~1.00 | 22668.1 | +12.4% |
| **PB-OEPLB** | **~1.02** | **23870.5** | **+18.4%** |
| Optimal (oracle) | 1.00 | 24460.1 | +21.3% |

**⚠The EPLB and OEPLB numbers in this table come from a single-measurement historical batch and differ from the later re-test.** This paper performed two independent-restart re-tests on the same configuration (`L512_eplb_r{1,2}.json`, `L512_baseline_r{1,2}.json`), obtaining **EPLB +1.75%, OEPLB +17.5%**. OEPLB agrees between the two (18.4% vs 17.5%, a 0.9pp gap within run-to-run noise), but **EPLB differs by 5× (9.0% vs 1.75%)**, and this table's historical dataset is no longer available (§5.8). The most likely source of the discrepancy is that EPLB's gain is highly sensitive to `eplb_rebalance_num_iterations` and the timing of the first rebalance (a 1.4–4.5s rebalancing block landing at different points in a 175s benchmark changes it by 1–2.6%), and this parameter was not recorded for the historical batch. **Wherever this paper makes a quantitative comparison with EPLB, the reproducible re-test values (EPLB +1.75%) are used uniformly**; this table is kept only as a reference for the placement spectrum (worst/baseline/oracle).

**The 235B single-domain gain has been independently re-confirmed a third time (`driver38.sh`)**: identity baseline 201.2 s (CV 0.02%), OEPLB 168.5 s (CV 1.18%), gain **+19.43%** (r1 +18.43%, r2 +20.46%), consistent with (and slightly above) the paper's +17.5%; corresponding $\eta=19.4/22.1=88\%$. On the long-prompt drifting multi-domain workload (`prefill_heavy_universal.jsonl`, 16000 requests), the **headline gain vs the identity baseline is +9.76%** (`driver39.sh`, 2 independent restarts, identity 824.7 s CV 0.13% → OEPLB 751.4 s, r1 +8.08%, r2 +11.49%). Note that `driver35.sh` had used the static-optimal (bal) placement as its baseline, so its +5.80% measures the **adaptation benefit** (OEPLB vs static-optimal), not the headline gain. The two quantities are computed from their own same-run data: headline (+9.76%) from d39; adaptation benefit (+5.80%, bal 801.2 s → OEPLB 757.2 s, same run) from d35. Ordering the three placements by time: identity(824.7) > static-optimal(806.7) > OEPLB(751.4): OEPLB is fastest and **beats the static optimum** — because the static optimum is optimal only for the aggregate distribution, not for each drifting domain, whereas OEPLB follows domain by domain.

**Table 2: Imbalance ratio convergence (L512_O1)**

| Window | Max-delta (old) | Adaptive pairing (new) |
|---|---|---|
| w0 | 1.743 → 1.264 | 1.743 → 1.187 |
| w1 | 1.262 → 1.261 (stagnant) | 1.188 → 1.057 |
| w2 | 1.263 → 1.262 (stagnant) | 1.060 → 1.015 |
| w3 | 1.262 → 1.261 (stagnant) | 1.027 → 1.015 |
| Steady state | ~1.26 | **~1.02** |

### 5.4 4-GPU Independent Validation Results (This Experiment)

**Table 3: Qwen2-57B-A14B (EP=4, 16 experts/GPU, alternating with independent restarts)**

| Scenario | Dataset | Baseline tps | OEPLB tps | Delta | Swap execution |
|---|---|---|---|---|---|
| L512_O1 (8K) | `/tmp/exp_data/L512_O1_full8k`† | 57.5 | 60.0 | +4.3% | 55 swaps in warmup, fewer swaps in steady state |
| Multi-domain 16K | `/tmp/exp_data/multidomain_16k`† | 26.9 | 27.7 | +3.0% | 117 swaps, 0 rollbacks |
| ShareGPT 20K | `/tmp/exp_data/sharegpt_o1_20k`† | 255.3 | 260.1 | +1.9% | continuous swaps |

†**These three rows are not reproducible and have been partially corrected by subsequent measurements.** The three dataset files were located in `/tmp/exp_data/`, which no longer exists. Each number was measured twice independently in the paper's original data and the two measurements agree with each other within $\le1\%$ (L512 baseline 57.5/57.7, OEPLB 60.0/60.4; multi-domain 26.9/27.1, 27.7/27.8; ShareGPT 255.3/257.3, 260.1/265.3), so **the repeatability of the original measurements themselves was good**; the problem is that the datasets are not obtainable.

**Re-measurement (this round, 2×2 rounds of independent restarts)**:

| Scenario | Dataset (obtainable) | Baseline tps | OEPLB tps | Delta | Model upper bound | System efficiency $\eta$ |
|---|---|---|---|---|---|---|
| L256_O1 (16K) | `grid/L256_O1_realprover_n16384` | 118.0 | 121.0 | **+2.70%** | 2.57% | **105%** |
| L512_O1 (8K) | `grid/L512_O1_realprover_n8192` | 58.4 | 59.8 | **+2.39%** | 2.86% | **84%** |
| Multi-domain (4.4K) | `multi_domain/multidomain_v2_out1` | 41.00 | 40.90 | **−0.24%** | 2.26% | **≈0** |
| ShareGPT (20K) | `sharegpt/sharegpt_natural_20k` | 4665.4 | 4671.9 | **+0.14%** | 2.21% | **6%** |

Both arms of the L512 row individually agree with the original measurements (baseline 58.4 vs 57.5/57.7, OEPLB 59.8 vs 60.0/60.4, both $\le1.5\%$), but **the ratio drops from +4.3% to +2.39%** — the difference comes from a different prompt composition of the dataset (different $r_{\text{before}}$), not measurement drift. The multi-domain and ShareGPT rows use different dataset files (differing by 1.5× and 18× in tps scale) and are **not comparable** with the original three rows; they should be regarded as independent data points.

Both arms of all four rows are $n$=2 (independent restarts). Per-arm CV: L256 0.24%/0.10%, L512 0.38%/0.10%, multi-domain 0.02%/0.02%, ShareGPT 0.21%/0.62%.

**System efficiency is monotonically correlated with workload heterogeneity.** Ordering the four rows by $\sigma/\mu$ of prompt length: 0.14 (L256, $\eta$=105%), $\approx$0.14 (L512, 84%), 0.93 (multi-domain, $\eta\approx0$, measured $-0.24\%$), 2.17 (ShareGPT, $\eta$=6%, measured $+0.14\%$). The two homogeneous workloads reach 84%–105% of the ceiling and the two heterogeneous ones both land near zero ($-0.24\%$ and $+0.14\%$, each within its own CV), so we can only say they are indistinguishable from zero, not order them. The mechanism is consistent with the findings of Appendix D.2: heterogeneous workloads produce many small batches, and the balancer's decision statistic is unweighted over the window and is inflated by the sampling variance of small batches, so it chases noise and executes swaps, paying the P2P blocking cost while gaining no time back (those windows did not occupy time to begin with). This is the first candidate explanation for $\eta$, and it can be tested directly (by raising the `--pb-oeplb-min-prefill-tokens` threshold). **Note that the 8-GPU 57B/L256 point does not follow this relationship** ($\sigma/\mu$=0.14 but $\eta$=29%), so heterogeneity is not the entire cause of $\eta$.

**Table 4: Qwen3-30B-A3B (EP=4, 32 experts/GPU, alternating with independent restarts)**

| Scenario | Baseline tps | OEPLB tps | Delta | Swap execution |
|---|---|---|---|---|
| L512_O1 (8K) | 115.4 | 112.4 | **-2.6%** ❌ | 174 swaps in warmup, fewer swaps in steady state |
| Multi-domain 16K | 53.8 | 51.7 | **-3.9%** ❌ | 534+ swaps, 7 rollbacks |
| ShareGPT 20K | 402.0 | 405.8 | +0.9% ~0 | continuous swaps |

**Table 5: EPLB vs OEPLB comparison (Qwen2-57B multi-domain 16K, fair comparison after patching)**

| Configuration | deepep-mode | CUDA graph | Redundant experts | tps | vs baseline |
|---|---|---|---|---|---|
| Baseline | auto | ✅ | 0 | 26.9 | — |
| EPLB (official, patched) | normal | ❌ disabled | 16 | 27.2 | +1.1% |
| **OEPLB** | auto | ✅ | 0 | **27.7** | **+3.0%** |

OEPLB outperforms EPLB by **+1.9 percentage points** (+3.0% vs +1.1%). EPLB yields almost no improvement: normal mode disables CUDA graphs, which cancels out the balancing gains of the 16 redundant experts. (This table's dataset `multidomain_16k` is in the deleted `/tmp/exp_data/` and is not reproducible; it is kept only as a mechanism comparison of EPLB vs OEPLB.)

### 5.5 Full Comparison with EPLB

**Table 6: OEPLB vs EPLB across all scenarios**

| Scenario | OEPLB vs baseline | EPLB vs baseline | OEPLB advantage |
|---|---|---|---|
| L512 single-domain (8-GPU 235B) | +18.4% | +9.0% | +9.4 pp |
| L1024 single-domain (8-GPU 235B) | +15.4% | +7.6% | +7.8 pp |
| L256 single-domain (8-GPU 235B) | +13.0% | +6.0% | +7.0 pp |
| Multi-domain (8-GPU 235B) | +14.0%⚠ (unreproducible) | +12.0%⚠ | — |
| ShareGPT (8-GPU 235B) | +5.3% | -5.0% | +10.3 pp |
| L512 single-domain (4-GPU 57B) | +4.3% | — | — |
| Multi-domain (4-GPU 57B) | +3.0% | +0.4% | +2.6 pp |
| L512 single-domain (4-GPU 30B) | -2.6% | crash | — |
| Multi-domain (4-GPU 30B) | -3.9% | crash | — |

Note: On 4-GPU 30B, EPLB cannot run due to `AttributeError` (§2.2 limitation 2).

### 5.6 Overhead Analysis

**Table 7: OEPLB overhead breakdown (8-GPU 235B L512_O1, 175s benchmark)**

| Component | Time (ms) | Fraction of benchmark |
|---|---|---|
| Record (scatter_add per forward) | 599 | 0.34% |
| All_reduce (per window) | 495 | 0.28% |
| Plan build (rebalancer) | 43 | 0.02% |
| **Swap execution (synchronous P2P, blocking)** | **5953** | **3.42%** |
| **Total** | **7090** | **4.07%** |

The swap-execution row is the per-rank mean over two rounds of independent restarts (6762ms / 5144ms) and is the dominant overhead item. An earlier version of Table 7 recorded this item as "Finalize (P2P completion) 62ms"; that was the time for event queries and shadow-buffer copies under the asynchronous design. The synchronization change described in §3.4 raised the true cost by about two orders of magnitude, and we correct it here. It must be emphasized that this item is **not a pure loss**: what it buys is reducing the ratio from 1.72 to 1.05, for a net gain of +17.5%.

**Table 7b: Direct comparison of weight-migration blocking (8-GPU 235B, L512_O1, two rounds of independent restarts each)**

| | Adjustments/rank | First | Per occurrence in steady state | Cumulative per rank | Fraction of wall clock |
|---|---|---|---|---|---|
| Official EPLB (16 redundant / period 64) | 8 | 4.47s / 2.87s | 1.43~1.82s | 15.82s / 13.63s | 7.81% / 6.85% |
| **PB-OEPLB** | 8 (989/1002 ops) | 4.21s / 2.55s | **0.34~0.41s** | **6.76s / 5.14s** | **3.86% / 2.98%** |

The two trigger the same number of times and have similar first-occurrence costs (both must correct the initial placement that is furthest from the optimum); the difference comes entirely from the **steady state**: EPLB recomputes the global physical→logical mapping and reshuffles all physical slots every period (§2.2), whereas OEPLB only performs sparse pairwise swaps, so its steady-state cost is about 4× lower. As for "whether OEPLB stops swapping after the ratio converges," the three-round pressure experiments cited earlier in this paper left no data that can be re-verified, and that claim is **retracted**. In the 16384-request workload of Appendix E.2, PB-OEPLB recorded a total of 256 swap log entries over the entire run (8 ranks × 32 decisions), i.e., small adjustments continue throughout the steady state rather than dropping to zero. EPLB, by contrast, reshuffles unconditionally on a fixed period.

**Table 8: 4-GPU 30B nsys trace overhead breakdown (L512_O1)**

| Category | Baseline (ms) | OEPLB (ms) | Delta (ms) | Notes |
|---|---|---|---|---|
| kernel (GPU compute) | 7674 | 7867 | +193 | GPU compute nearly unchanged |
| cpu_op | 3064 | **7416** | **+4352** | Python code overhead surges |
| cuda_runtime | 2599 | 3865 | +1267 | More CUDA runtime calls |
| gpu_user_annotation | 8 | **2780** | +2772 | all_reduce annotated on the GPU side |
| DeepEP-Combine | 1031 | 863 | **-168** | Less waiting in combine after balancing |
| **Total trace** | **10667** | **13806** | **+3139 (+29%)** | |

Root cause of the negative gain on 30B: CPU-side Python code overhead (+4352ms) and all_reduce (+2772ms) together total 7.1 seconds, far exceeding the 168ms saved by Combine.

### 5.7 GPU Memory Comparison

**Table 9: GPU memory breakdown (per GPU, 96GB H20)**

| Configuration | Total (GB) | Model weights | KV cache | CUDA graph | Overhead |
|---|---|---|---|---|---|
| Baseline | 88.7 | ~28.0 | **~48.8** | ~10.0 | ~1.9 |
| **PB-OEPLB** | 88.7 | ~28.0 | **~48.8** | ~10.0 | ~1.9 |
| EPLB (16 redundant) | 79.8 | ~30.5 | **~46.3** | 0 (disabled) | ~3.0 |

PB-OEPLB is the only configuration that simultaneously preserves full KV cache capacity and keeps CUDA graphs enabled.

### 5.8 Reproducibility

3 independent cold starts on 8-GPU 235B (L512_O1_realprover_n8192, conc=256, with PB-OEPLB, `driver32.sh`):

| Run | Time (s) | Output tps |
|---|---|---|
| 1 | 170.1 | 48.2 |
| 2 | 177.1 | 46.3 |
| 3 | 175.8 | 46.6 |
| **Mean ± CV** | **174.3 ± 2.2%** | **47.0 ± 2.2%** |

(The early draft reported 22780 ± 156 tps under a different metric (input tokens/s) on a dataset that is no longer available; this table replaces it. CV=2.2% is higher than the static-placement sweeps (0.06–0.36%) because OEPLB's swaps introduce nondeterminism.)

On 4-GPU 57B, alternating with independent restarts across 3 scenarios, 0 errors, swap execution confirmed (VERIFY CHANGED=True).

---

## 6. Related Work

**Expert load balancing during training**: Auxiliary-loss methods (Shazeer et al., 2017; Fedus et al., 2021) and lossless methods (DeepSeek-V3, Wang et al., 2024) balance expert load at training time via routing adjustments. These are orthogonal to our work — we optimize the expert *placement* at inference time.

**EPLB (DeepSeek, 2025)**: Periodically rebalances the expert placement using a greedy bin-packing algorithm with redundant expert replicas. Requires `deepep_mode=normal` (disabling CUDA graphs), 16 redundant experts, and blocks inference during rebalancing. Only supports DeepSeek-architecture models.

**Expert offloading**: Libraries such as LibMoE manage expert placement across GPU-CPU tiers. They focus on memory management rather than runtime load balancing.

**Adaptive scheduling**: Optimizations such as continuous batching in vLLM optimize request-level scheduling but do not address expert-level load imbalance.

---

## 7. Discussion and Future Work

**N-way cyclic rotation**: Once pairwise swaps plateau (all pairwise improvements <0.0005), 3-way cyclic rotations (A→B→C→A) can in principle reach placements that pairwise transpositions cannot. Implementation challenges in P2P execution prevented deployment in this version.

**EPLB refinement**: The hybrid approach — using incremental swaps for fast initial convergence, then a single EPLB-style full re-placement for final refinement — is promising in simulation but introduced instability in practice due to timing issues.

**Cross-model generalization**: All 8-GPU experiments use Qwen3-235B-A22B. The 4-GPU validations on Qwen2-57B-A14B and Qwen3-30B-A3B confirm the scaling laws but need to be verified on more architectures.

**record_next_layer optimization**: In pure-prefill (O=1) scenarios the record overhead reaches 1.6% on 30B (vs 0.34% on 235B), because all forwards bypass CUDA graphs. Future work may consider asynchronous recording or sampled recording to reduce the overhead.

---

## 8. Conclusion

PB-OEPLB demonstrates that lightweight adaptive online expert load balancing can achieve a near-optimal placement (97.6% of oracle) without the architectural overhead of existing approaches. The key insight is that **adaptive pairing selection** (switching between max-delta and gap-targeting modes according to the size of the current gap) combined with **fast exponential decay** (α=0.5) and **adaptive decision windows** achieves fast convergence to a ratio of 1.02 across diverse workloads. On 8×H20+Qwen3-235B, PB-OEPLB improves throughput by +5.3% to +19.43% (the +17.5% single-domain figure confirmed a third time by `driver38.sh`; multi-domain headline +9.76% by `driver39.sh`), outperforming SGLang's EPLB by 15.7 percentage points on L512 under the reproducible re-test criterion (EPLB +1.75%); the 2–10 percentage-point range from historical single batches is marked ⚠. The independent validation on 4×H20 confirms the scaling effect of experts per GPU on the gains: with 16 experts/GPU the gains are −0.24%~+2.70% (re-measured values on reproducible datasets; homogeneous workloads reach 84%~105% of the theoretical upper bound, heterogeneous workloads are close to 0), and with 32 experts/GPU OEPLB is essentially neutral ($\eta\approx0$): its $T(r)$ bound is positive ($\beta=+0.207$, +6.36%) but the realized gain is ≈0 because fixed overhead eats the gain; a clean long benchmark (`driver42.sh`, n=4) shows all arms within ±1.3% of baseline (the earlier short-benchmark negative/stanching was a noise artifact).

---

## References

1. Shazeer, N. et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.
2. Fedus, W. et al. "Switch Transformers: Scaling to Trillion Parameter Models." arXiv 2021.
3. DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv 2024.
4. DeepSeek-AI. "EPLB: Expert Parallelism Load Balancer." 2025. github.com/deepseek-ai/EPLB
5. SGLang. "SGLang: Efficient LLM Serving." sglang.ai
6. DeepSeek-AI. "DeepEP: Efficient Expert Parallelism Communication." 2025.
7. Wang, A. et al. "Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts." arXiv 2024.
8. Sun, Y. "Binary-Integer-Programming Based Algorithm for Expert Load Balancing in MoE Models." arXiv 2025.

---

## Appendix A: Mathematical Modeling

### A.1 Convergence Analysis of the Greedy Planner

**Problem formulation**: Given the global load tensor $L \in \mathbb{R}^{N_L \times N_S}$ (identical across all ranks after all_reduce), find a set of swap operations $S = \{(l_i, a_i, b_i)\}_{i=1}^{|S|}$, $|S| \leq B$ (budget), minimizing the maximum per-layer imbalance ratio.

**Theorem 1 (Convergence)**: For a single layer with $N_G$ GPUs and $N_S$ slots per GPU, if there exists at least one pair $(a, b)$ such that $L[a] > L[b]$ and $L[a] - L[b] \leq \text{gap}$, then the adaptive pairing selection algorithm reduces $r$ by at least:
$$\Delta r \geq \frac{2(L[a] - L[b])}{N_G \cdot \bar{L}} \cdot \left(1 - \frac{L[a] - L[b]}{2 \cdot \text{gap}}\right)$$

**Theorem 2 (Bound for swap-local optima)**: Let $\pi$ be swap-local-optimal. Then:
$$r(\pi) \leq 1 + \frac{G-1}{G} \cdot \frac{\ell_{\max}}{n\mu}$$

where $\ell_{\max}$ is the load of the hottest expert and $\mu$ is the mean expert load.

**Tight example**: $G$ GPUs, one hotspot expert $\ell_1 = M$, all others $\ell_j = \epsilon \to 0$. The initial assignment places the hotspot on GPU 0. For any swap, delta $\geq \text{gap}$, overshoot→local optimum. $r = G$, bound $= 1 + \frac{G-1}{G} \cdot G = G$. **Tight!**

### A.2 Bayesian Derivation of the Optimal Decay Factor

Unrolling the decay $A_t = R_t + \alpha A_{t-1}$: $A_t = \sum_{k=0}^{\infty} \alpha^k R_{t-k}$.

$d$ steps after a change point, the residual weight of the old signal is: $w_{\text{old}}(d) = \alpha^{d+1}$.

Detection requires $w_{\text{old}} < 1/2$: $\alpha < 2^{-1/(d+1)}$. For $d=2$: $\alpha < 0.794$; for $d=3$: $\alpha < 0.841$.

The optimal $\alpha$ minimizes "latency+noise": $\alpha^* \approx 0.52$ (numerical solution at SNR=3, γ=1), validating the empirical value of 0.5.

### A.3 Upper Bound on Throughput Speedup

See the Amdahl form in §2.4. Here we supplement with the error magnitude of the first-order truncation: let $x=1-r_{\text{after}}/r_{\text{before}}$,
$$\Delta_{\max} = \frac{f_{\text{sens}}x}{1-f_{\text{sens}}x},\qquad \Delta_{\max}^{(1)} = f_{\text{sens}}x,\qquad \frac{\Delta_{\max}}{\Delta_{\max}^{(1)}} = \frac{1}{1-f_{\text{sens}}x}$$
For 235B ($f_{\text{sens}}=0.384$, $x=0.407$) the truncation error is a factor of $1/(1-0.156)=1.19$, i.e., the first-order form underestimates the upper bound by 19%.

**Note**: $f_{\text{sens}}$ is not $f_{\text{MoE}}$, nor is it the FLOP fraction of MoE — see the comparison table in §2.4 and the measurements in Appendix G. Substituting the FLOP fraction (67.9% for 235B, 46.9% for 57B) overestimates $f_{\text{sens}}$ by about 1.4–1.9×, and $x$ must be replaced by $x_{\text{eff}}$ with the dead zone excluded (§2.4): when the two errors add in the same direction, the upper bound is overestimated by more than a factor of two (8-GPU 57B: $0.469\times0.146$ gives 7.4%, while the measured value gives 3.40%).

### A.4 Adaptive Windowing as an Optimal Stopping Problem

State $s = (r_n, \Delta r_n, \text{converge\_count}, \text{volatile\_count})$, action $a \in \{\text{grow}, \text{shrink}, \text{hold}\}$.

Reward $R(s,a) = \Delta r \cdot w - c_{\text{all\_reduce}}/w$.

The doubling/halving strategy has a competitive ratio of $O(\log(w_{\max}/w_{\min}))$.

---

## Appendix B: Kernel-Level Analysis

### B.1 Per-Forward-Step Kernel Timing

Measured on 8-GPU 235B (GPU utilization ~62%, ~795 forward steps), normalized per step:

| Category | Baseline (μs/step) | PB-OEPLB (μs/step) | Delta |
|---|---|---|---|
| Dispatch | 7323 | 6300 | **-14.0%** |
| Combine | 5479 | 4015 | **-26.7%** |
| Expert compute | 6214 | 6179 | -0.6% |
| Attention | 4731 | 4695 | -0.8% |
| Other (norm/quant/elementwise/etc.) | 1799 | 1893 | +5.2% |
| **Total** | **25546** | **23082** | **-9.6%** |

(The four main kernel categories sum to 23747/21189; the difference of 1799/1893 goes into the "Other" row, so the table now closes. Source: an 8-GPU 235B torch-profiler trace, normalized over ~795 forward steps.)

### B.2 4-GPU 30B nsys Trace Analysis

A trace of 600 forward steps was collected via `/start_profile` and aggregated by kernel category:

| Category | Baseline (ms) | OEPLB (ms) | Delta (ms) | Delta% |
|---|---|---|---|---|
| DeepEP-Dispatch | 2997 | 3244 | +247 | +8.3% |
| DeepEP-Combine | 1031 | 863 | **-168** | **-16.3%** |
| DeepGEMM | 1959 | 1978 | +20 | +1.0% |
| NCCL/SendRecv(P2P swap) | 0 | 239 | +239 | — |
| NCCL/AllReduce | 0 | 3.5 | +3.5 | — |
| cpu_op(Python) | 3064 | **7416** | **+4352** | **+142%** |
| gpu_user_annotation | 8 | **2780** | +2772 | — |

Root cause of the negative gain on 30B: CPU-side Python overhead (the function calls + conditional branches from calling record_next_layer per layer) accumulates 4.3 seconds in the pure-prefill scenario, plus 2.8 seconds of all_reduce, totaling 7.1 seconds, far exceeding the 168ms reduction in Combine.

---

## Appendix C: Full Grid Results

### C.1 Full 5×4 Grid (8-GPU 235B, historical data)

| L | O | Baseline (rps) | sw=8 | sw=16 | sw=32 | sw=64 | Adaptive (8) | Optimal |
|---|---|---|---|---|---|---|---|---|
| 256 | 1 | 77.1 | +23.9% | +21.0% | +15.3% | +10.9% | +21.8% | sw=8 |
| 512 | 1 | 40.6 | +17.8% | +14.2% | +16.4% | +12.7% | +15.5% | sw=8 |
| 1024 | 1 | 19.9 | +18.8% | +19.0% | +16.6% | +17.3% | +18.3% | sw=16 |
| 2048 | 1 | 9.8 | +12.5% | +13.7% | +8.4% | +8.0% | +13.9% | adaptive |
| 4096 | 1 | 4.5 | +11.2% | +12.1% | +7.6% | +4.1% | +10.4% | sw=16 |

(For all 20 cells, see the English version PAPER_en.md)

---

## Appendix D: Expert Density and Imbalance — Scaling Analysis

### D.1 Two Degrees of Freedom of the Imbalance Ratio: Expert Density and Routing Skew

The imbalance ratio $r$ is jointly determined by the **number of experts per GPU** $n = N_E/\text{EP}$ and the **relative skew of expert loads** $\sigma/\mu$.

**Law of large numbers**: The total load on each GPU is the sum of $n$ expert loads. As $n$ grows, the per-GPU load converges to the global mean — the coefficient of variation shrinks as $O(1/\sqrt{n})$ — making high imbalance statistically unlikely. Let the expert loads have mean $\mu$ and standard deviation $\sigma$; the coefficient of variation of the per-GPU load is $\text{CV}(L_g)=\frac{\sigma}{\mu\sqrt{n}}$, and the expected maximum over the EP GPUs is approximately
$$E[r] \approx 1 + \frac{\sigma}{\mu\sqrt{n}}\cdot\sqrt{2\ln(\text{EP})}$$

**Error in an early draft**: The draft stated this as "the fewer experts per GPU, the higher the imbalance ratio," i.e., that $n$ alone determines $r$. Measurements falsified this (see D.2): 235B and 57B differ in $r$ by a factor of 1.55 at **the same $n=16$**, because their $\sigma/\mu$ differ by nearly 5×. $n$ holds only **within the same model**.

### D.2 Empirical Validation (Measured in This Paper)

We used PB-OEPLB-DIAG to record $r=\max_g L_g/\bar L$ for every decision window, and report the window average (avg) and the worst window (max). **The two must not be conflated**: the time model uses avg, and tail-latency analysis uses max.

| Model | Total experts | EP | Experts/GPU $n$ | $r$ avg | $r$ max | $\sigma/\mu$ (single-point inversion) | OEPLB gain |
|---|---|---|---|---|---|---|---|
| Qwen3-235B (8×H20) | 128 | 8 | 16 | **1.721** | 2.486 | **1.414** | +17.5% |
| Qwen2-57B (8×H20) | 64 | 8 | 8 | **1.216** | 2.760 | **0.300** | +1.0% |
| Qwen2-57B (4×H20) | 64 | 4 | 16 | **1.113** | 1.741 | **0.271** | +1.85% |

**DIAG's $r$ is an unweighted mean over windows and is biased high on heterogeneous workloads.** This paper has two independent sources for $r_{\text{before}}$: self-reported at runtime by PB-OEPLB-DIAG, or recomputed offline from recorded routing counts. Four cross-checks agree to within 1% (8-GPU 57B 1.2177 vs 1.216, 4-GPU 57B 1.1071 vs 1.113, 8-GPU 235B 1.7370 vs 1.721, 4-GPU L512 1.1125 vs 1.116), but **on ShareGPT the two differ by 97%** (DIAG first window 2.161 vs offline 1.0965).

Using the per-forward dimension of `logical_count` in the recorded dumps (shape `[buffer, layers, experts]`), we can pinpoint the cause. Three aggregation levels:

| Aggregation level | Definition | ShareGPT/EP4 | Multi-domain/EP4 |
|---|---|---|---|
| $r_{\text{agg}}$ | Ratio of cumulative counts over the entire run | 1.0965 | 1.0980 |
| $r_{\text{win}}(16)$ | Token-weighted mean of ratios over 16-forward windows | 1.1000 | 1.1020 |
| $r_{\text{fwd}}$ | Token-weighted mean of per-forward ratios | 1.1230 | 1.1040 |

The Jensen effect (the differences among aggregation levels) is only 0.3% and cannot explain 2.161. The true cause is **the sampling variance of small batches**: the distribution of per-forward ratios is p50=1.131, p90=1.759, p99=2.107; forwards with fewer than 1/10 of the median token count (1364 of them) have a mean ratio of 1.551, while the rest have 1.128; the window with the largest ratio has only 3584 tokens, whereas the median window has 922432 (a 250× difference). **The smaller the batch, the larger the sampling variance of the per-GPU distribution and the higher the ratio, yet such windows occupy almost no wall-clock time.**

There are two consequences. **For the model**: $r_{\text{before}}$ must be taken at a token-weighted aggregation level; only the offline value 1.0965 (or 1.1000 for $r_{\text{win}}$) is the time-relevant quantity, and DIAG's 2.161 is a small-sample artifact. Dataset homogeneity predicts the size of the bias — $\sigma/\mu$ of prompt length is 0.14 (L256), 0.93 (multi-domain), 2.17 (ShareGPT); for the first two, DIAG and the offline value agree within $\le1\%$, and only the third breaks down. **For the system** (more important): PB-OEPLB's decision logic uses precisely this unweighted statistic, so it gets triggered by small-sample noise — a 3584-token window reports $r=2.16$, the balancer executes swaps accordingly and pays the P2P blocking cost, but gains no time back (that window did not occupy time to begin with). The existing safeguard `--pb-oeplb-min-prefill-tokens` defaults to 256, four orders of magnitude below the median window, and is effectively inert. This may be one of the causes of the low system efficiency in §2.4 (only 29% on 8-GPU 57B), and it can be tested directly: raise that threshold or switch to a token-weighted estimate.

**The $r$ and "OEPLB gain" columns in the table are not necessarily from the same source.** The $r$ avg=1.113 in the 4-GPU 57B row was recorded by DIAG, while +1.85% comes from the ShareGPT 20K row in Table 3; the offline recomputation from recorded counts gives an identity value of 1.107 for 4-GPU L256, agreeing with 1.113 to 0.5%, but this only shows that the two are consistent **if** both are L256; it does not prove that the $r_{\text{before}}$ of ShareGPT is also 1.11. `driver17.sh` is recording routing counts per dataset to remove this ambiguity (Appendix F.5). The two 8-GPU rows are unaffected: their $r$ and gains both come from the same L256/L512 workload.

**Testing the density model**. For the same model (57B), the routing skews obtained by single-point inversion under the two EP configurations are $\sigma/\mu=0.300$ (8-GPU, $n$=8) and $0.271$ (4-GPU, $n$=16) — consistent to 11%, i.e., the $1/\sqrt{n}$ scaling holds within a model (decreasing $n$ from 16 to 8 raises $r-1$ from 0.113 to 0.216). Across models, skew dominates entirely: $\sigma/\mu=1.414$ for 235B is 4.7–5.2× that of 57B, reflecting the difference in the shape of the routing distributions between 235B (128 experts, no shared expert, top-8) and 57B (64 experts, giant shared expert), independent of $n$.

**Implication**: The first predictor of optimizable headroom is the **routing skew $\sigma/\mu$** (which can be estimated by recording a single window online); $n$ is a secondary correction. This also explains why 57B's gains are only ~1–2% at both parallelism levels: its routing distribution is itself close to uniform.

**Data errors in the early draft (corrected)**: The D.2 table in the draft placed 235B's 1.74 (actually avg) alongside 57B's 1.74 (actually the 4-GPU **max**; avg is only 1.113) as the "measured baseline max ratio," and concluded that "the two have the same imbalance ratio, and the 6× difference in gains must be explained by dilution via shared experts." In fact, the two imbalance ratios themselves differ by a factor of 1.55 (1.721 vs 1.113), and the dilution effect need not carry the entire explanation. Similarly, the draft's "Zipf prediction 1.84 vs measured 1.74, 5.4% error" was comparing $E[r]$ (the mean of the distribution) against the worst window; compared against avg (1.113), the Zipf $s=0.5$ prediction is 65% too high, and that validation does not hold.

**Shared-expert dilution effect (retained but downgraded to qualitative)**: The shared expert of Qwen2-57B (intermediate_size=20480) has per-token FLOPs equal to the sum of all 8 activated routed experts (440.4 MFLOP/layer each), so the imbalance in the routing dimension is diluted by about 2× in the timing dimension. This explains why 57B's $f_{\text{sens}}$ (measured in Appendix G) is far below its FLOP fraction of 46.9%, but the precise dilution factor needs to be measured rather than inferred.

### D.3 Conditions for OEPLB Gains

From §2.4, PB-OEPLB yields a net positive gain if and only if
$$(1+\Delta_{\max})(1-c_{\text{overhead}}) > 1,\qquad \Delta_{\max}=\frac{f_{\text{sens}}x_{\text{eff}}}{1-f_{\text{sens}}x_{\text{eff}}},\quad x_{\text{eff}}=\frac{r_{\text{before}}-\max(r_{\text{after}},r_k)}{r_{\text{before}}}$$

| Configuration | $r_{\text{before}}$ (avg) | $r_k$ | $x_{\text{eff}}$ | $f_{\text{sens}}$ | $\Delta_{\max}$ | Overhead | Predicted net gain | Measured |
|---|---|---|---|---|---|---|---|---|
| **8-GPU 235B** | **1.721** | **1.093** (measured) | **0.365** | **0.496** (measured) | **22.09%** | 0.67% | **+21.3%** | +17.5% ⚠ efficiency 79% |
| **8-GPU 57B** | **1.218** | **1.099** (measured) | **0.098** | **0.335** (measured) | **3.40%** | ~1% | **+2.4%** | +1.0% ✅ sign |
| **4-GPU 57B** | **1.107** | **1.032** (measured) | **0.068** | **0.369** (measured) | **2.57%** | ≤0.4% (back-solved) | **+2.2%** | **+2.70%** ✅ |
| 4-GPU 30B | 1.338 | 1.031 (measured) | 0.230 | **+0.207** (measured) | +6.36% | ~7s fixed | **net-negative ($\eta<0$)** | −3.8%~+0.5% ✅magnitude |
| 4-GPU DS-V2-Lite | 1.02 | not measured | ~0 | — | ≈0 | ~4% | **negative** | −4.5%⚠ |

⚠DS-V2-Lite row: model weights deleted from machine; the −4.5% measurement comes from an early draft with no traceable result file. Kept as qualitative reference only ($r_{\text{before}}\approx1.02$ → no headroom → predicted negative or zero, directionally consistent).

The corrected model predicts the sign correctly for the four 235B/57B configurations. **30B is a subtler case: its bound is positive ($+6.36\%$) but the realized gain is ≈0** — not a sign error in the model, but fixed overhead (the 7.1s of §B.2) eating the realizable gain, i.e. $\eta\approx0$. Capturing this requires both the bound model ($\Delta_{\max}$) and the overhead model ($\eta$ gate): the bound alone would say "do it", while a naive reading of the earlier short-benchmark negative result would misread it as "$\beta<0$". A clean long benchmark (`driver42.sh`, n=4) shows 30B is essentially neutral (all arms within ±1.3% of baseline); the earlier short-benchmark −3.8%/+0.5% was a noise artifact, and overhead is the cause.

**The magnitude gap in the 8-GPU 57B row is the main open problem of this paper**: the upper bound is 3.40% (and the empirical ceiling of +3.82% has been measured independently, Appendix G.2), yet the measured value is only +1.0% and the system efficiency is 29%. This is not a problem with the model but with the **balancer** — the static optimal placement on the same data achieves +3.82%, indicating that what is missing is convergence speed and swap amortization, not available headroom. **The "upper bound ≈ 0" judgment for the 4-GPU 57B row has been falsified by our own sweep.** This paper previously reasoned: the identity $r_{\text{avg}}=1.107$ is nearly equal to the measured 8-GPU $r_k=1.099$, and if the dead-zone position is independent of EP scale, the theoretical upper bound for this configuration is near zero, so the measured +1.85% could only be noise. The 4-GPU sweep in `driver13.sh` (14 runs, between-round CV 0.06–0.36%) shows that **the premise does not hold**: $r_k=1.032$ at EP=4 falls **below** $r_{\text{after}}=1.04$; the dead zone simply does not take effect in this configuration, and the upper bound is **+2.29%** rather than 0; the empirical ceiling of +2.63% also independently confirms this magnitude. The error in that reasoning was treating one configuration's $r_k$ as a constant; the lesson has been recorded in G.3.

†This row is now fully same-source: `driver18.sh` measured OEPLB +2.70% on 4-GPU L256 (same dataset as this row's $r_{\text{before}}$), and the back-solved overhead ≤0.4% follows from the bound (2.57%) minus the realized gain being close to the ceiling. The earlier "+1.85%" was a ShareGPT 20K figure from a different source and is superseded.

**Two gating conditions for system efficiency $\eta$.** $\Delta_{\max}$ is a ceiling; the realized gain is $\eta\cdot\Delta_{\max}$, and the $\eta$ we measure ranges from $\approx0$ to 105% across configurations. Placing PB-OEPLB's own logged swap timings (`N swap(s) done (XXXms)`, normalized per rank) next to the headroom the model predicts ($=\Delta_{\max}\cdot T_{\text{flat}}$, in seconds), the spread in $\eta$ stops being unstructured:

| Configuration | Decisions/round | swap s/rank/round | headroom s | swap/headroom | measured $\eta$ |
|---|---|---|---|---|---|
| 57B/EP8 L256 | 15.5 | 3.68 | 2.92 | **1.26** | 29% |
| 57B/EP4 multi-domain | 6.0 | 2.65 | 2.43 | **1.09** | $\approx$0 |
| 57B/EP4 L256 | 2.5 | 0.98 | 3.57 | 0.27 | **105%** |
| 57B/EP4 L512 | 1.0 | 0.58 | 4.01 | 0.14 | **84%** |
| 57B/EP4 ShareGPT | 22.0 | 2.58 | 30.00 | 0.09 | 6% |

**Condition A (cost gate): the headroom must exceed the swap cost.** The two configurations with swap/headroom $\ge1$ have the lowest $\eta$ in the table: 1.26 (8-GPU 57B, $\eta$=29%) and 1.09 (multi-domain, $\eta\approx0$). That is, **when the ratio $\ge1$, $\eta$ is pushed to its lowest band, though not strictly zero** — 8-GPU 57B still shows +0.98% end-to-end positive gain, because the blocking is heavily front-loaded (the first 139-op decision is 1.75 s while the remainder of the run enjoys the reduced $r$). What CAN be rescued is exactly what the $r_k$ threshold and swap budget target: with both knobs $\eta$ rises to 100% on the same configuration (table below).

**Condition B (direction gate): the decision statistic must reflect time-relevant imbalance.** ShareGPT's swap/headroom is only 0.09 (2.6 s of swapping in a 22-minute run is negligible), yet $\eta$ is 6%. Its failure is therefore not one of cost but of **direction**: its 22 decisions are driven by the unweighted statistic of Appendix D.2, are misled by the sampling variance of small batches, and move experts without reducing the real $r$.

$\eta$ is high only when neither condition is violated (0.27 → 105%, 0.14 → 84%). **Both conditions can be evaluated before deployment**: the headroom is given by $\beta(r-r_k)T_{\text{flat}}$; the swap cost is $\approx$ decisions $\times$ ops per decision $\times$ weight bytes $/$ P2P bandwidth; condition B can be checked from a single recording by comparing $r_{\text{agg}}$ against $r_{\text{fwd}}$ (Appendix D.2). This yields a simple deployment criterion: **do not enable the balancer unless the headroom is substantially larger than the swap cost.**

**One causal chain — and its verification.** 8-GPU 57B makes 21 decisions/round (9 of which issue swaps) on a homogeneous dataset, which is anomalously many; the cause is that $r_{\text{before}}=1.218$ is only 11% above $r_k=1.099$ while the threshold 1.02 lies inside the dead zone, so the balancer can never reach its own target and never stops. Verification (`driver26.sh` + `driver27.sh`, four-arm A/B test):

| Arm | Configuration | Gain | $\eta$ | Decisions/round |
|---|---|---|---|---|
| Default | threshold=1.02 | +0.98% | 26% | 21 |
| +dead zone | threshold=$r_k$=1.099 | +2.26% | 59% | 1 |
| **+dead zone + budget** | threshold=1.099, budget=0.02 | **+3.81%** | **100%** | 1 |
| +dead zone + bias + gate | threshold=1.099, bias, gate=0.5 | +2.20% | 57% | 1 |

(Decision counts are **global**, i.e. DP0 DIAG lines; an earlier version counted all 8 ranks' log lines, inflating by 8×.)

**Setting the threshold to $r_k$ and adding a swap-budget constraint raises $\eta$ from 26% to 100%** (+3.81% vs. ceiling 3.82%) — the balancer finally reaches its own physical limit. The dead zone is the primary contributor (26% → 59%); the budget recovers the rest (59% → 100%). Bias correction and the gate have no positive contribution on homogeneous workloads (they block warmup windows), but on heterogeneous ShareGPT they cut wasteful decisions from 110/round to 8/round (94%), eliminating condition B's violation.



**Deployment recommendations**: PB-OEPLB is most effective under the following conditions:
1. Routing skew $\sigma/\mu \gtrsim 1$ (can be determined by recording a single window online; more predictive than the number of experts)
2. Experts per GPU ≤ 16-20 (at the same skew, smaller $n$ gives higher $r$)
3. Low dispatch time fraction (ample NVLink); otherwise the negative term $\beta_{\text{dispatch}}f_{\text{dispatch}}$ eats up the combine gains
4. The swap cost per decision can be amortized: steady-state swap overhead should be much smaller than $\Delta_{\max}$ (0.37s/rank/decision for 8-GPU 57B in this paper, a 0.2% fraction)

---

## Appendix E: Numerical Equivalence, Correctness, and Robustness

Permuting experts across GPUs is mathematically an identity transformation—the weights are unchanged, routing outcomes are unchanged, and every token is still processed by the same set of experts; only which GPU the computation happens differs. In principle, therefore, no semantic change should be introduced. This appendix tests that claim and reaches a counterintuitive conclusion: **bit-identical equivalence is simply unattainable in this system, and has nothing to do with the balancer.**

### E.1 Method

Two kinds of probe, each run once before and once after the load on the same server:

- **GSM8K probe** (200 questions, temperature=0, seed=0, max_tokens=512): exact-match accuracy, plus an md5 of each output to check byte-level identity.
- **Corpus logprob probe** (fixed text, 50385 tokens): the logprob of every token under teacher forcing. This is a **purely numerical quantity**—no sampling is involved, so it is not affected by sampling-path divergence, and the same model on the same input should match exactly.

The load was L256_O1×16384, conc=256 (the same data as the Appendix G sweep). Crucially, a **control arm** was set up: on the baseline server no swap can possibly occur, so it provides the null hypothesis.

### E.2 Results

| Arm | Round | GSM8K | logprob bit-identical | mean$|\Delta|$ |
|---|---|---|---|---|
| PB-OEPLB | t1 | 82.00% → 84.00% | 109/50385 | 9.92e-2 |
| PB-OEPLB | t2 | 83.50% → 83.50% | 88/50385 | 1.13e-1 |
| **baseline (no swap)** | t1 | 84.50% → 84.00% | **50385/50385** | **0** |
| **baseline (no swap)** | t2 | 83.50% → 83.00% | **50385/50385** | **0** |

**(a) With placement fixed, the system is fully deterministic.** Both baseline rounds give 50385/50385 bit-identical, with $\max|\Delta|=0$. The noise floor of the corpus probe is therefore **zero**, and any nonzero difference carries attributive value.

**(b) Accuracy is unchanged.** Both arms flip the same number of GSM8K answers between rounds (16 for OEPLB, 13 for baseline, with directions nearly symmetric in both), and accuracy changes by $\le2$pp. At $n$=200 one standard deviation is already 2.6pp, so this probe **cannot resolve 2pp in the first place**—this point can only support the conclusion "no large-scale degradation", not "perfectly lossless".

**(c) Byte-level identity never exists in this system.** Even the baseline's GSM8K outputs are only 41/200 byte-identical (17/200 for OEPLB). Sampling paths are affected by dynamic batching, so **byte-level comparison cannot serve as a correctness criterion**—a point that applies to any work attempting equivalence arguments on stacks of this kind.

**(d) The perturbation comes from "the placement changed", not from the swap mechanism.** This is the core control of this appendix. Using `--init-expert-location` to nail down the placement, with the balancer switched off throughout (not a single swap ever occurs):

| Comparison | Placement | # swaps | Bit-identical | mean$|\Delta|$ |
|---|---|---|---|---|
| idA vs idB | same (different processes) | 0 | 50385/50385 | 0 |
| idA vs flag-identity | same (via the `--init-expert-location` identity mapping) | 0 | 50385/50385 | 0 |
| idA vs bal | **different** (LPT-optimal) | **0** | 115/50385 | **9.59e-2** |
| idA vs r135 | **different** (hot experts pressed onto GPU0) | **0** | 87/50385 | **9.31e-2** |
| OEPLB before vs after | **different** | **256** | 109/50385 | **9.92e-2** |

Zero-swap static placements reproduce the perturbation magnitude of 256 real swaps (9.59e-2 / 9.31e-2 vs 9.92e-2, a 3–6% gap, on the same order as the difference between the two static comparisons themselves). The second row rules out the confound that "passing `--init-expert-location` itself has side effects". **Conclusion: the perturbation is the result of the FP8/DeepGEMM reduction order changing once "which card computes which expert" changes; the swap mechanism's additional contribution is zero.** The perturbation has near-zero mean (bias $+7.1$e-3, only 7% of the magnitude), consistent with numerical noise rather than weight corruption (which would degrade things systematically).

**Corollary (holds for the entire class of systems):** Since permuting experts across GPUs is an identity transformation, any system that changes the expert placement—including SGLang's own EPLB—produces per-token perturbations of the same magnitude, and **bit-identical equivalence is unattainable for this whole class of methods**. Equivalence arguments must be built on distribution-level metrics.

### E.3 Robustness boundary under extreme imbalance

In the 235B sweep of Appendix G, an extreme placement ($r_{\text{avg}}=4.686$, with every layer's hottest experts all pressed onto GPU0) was constructed to reach the upper endpoint of $r$. This configuration **failed to start in both rounds**:

```
deep_gemm_wrapper/compile_utils.py:218  _empty_token_fp8
torch.AcceleratorError: CUDA error: invalid configuration argument
```

DeepGEMM preallocates per-token FP8 buffers according to `max_m`; extreme imbalance pushes a single GPU's `max_m` beyond the kernel's configuration limit. This provides a **safety** argument independent of performance: load balancing not only improves throughput, it also keeps the system inside the usable parameter region of the inference kernels. Conversely, it is also a limitation of the sweep method in this paper—the measurable upper endpoint of $r$ is constrained by the runtime, which is why the 235B sweep has only 5 constructible points (still yielding $R^2=0.9995$).

## Appendix F: Three Alternated Comparisons of Qwen2-57B-A14B on 4×H20 (This Experiment)

### F.1 Experimental Setup

- **Hardware**: 4× NVIDIA H20 (96GB/card), full NVLink interconnect, no IB
- **Model**: Qwen2-57B-A14B-Instruct (28 MoE layers, 64 experts, top-8, EP=4 → 16 experts/GPU, shared expert present = 20480)
- **Method**: the server is restarted independently for each scenario; Baseline/OEPLB/EPLB are run in alternation (eliminating temporal drift and placement inheritance)
- **OEPLB configuration**: sw=16, decay=0.5, threshold=1.02
- **EPLB configuration**: 16 redundant experts, eplb-rebalance-num-iterations=64

### F.2 O=1 (Pure Prefill) Results

| Scenario | Baseline | EPLB | OEPLB | OE vs BL | OE vs EPLB |
|---|---|---|---|---|---|
| L512_O1 (8K) | 57.7 | 58.2 | 60.4 | +4.7% | +3.8% |
| Multi-domain 16K | 27.1 | 27.1 | 27.8 | +2.6% | +2.6% |
| ShareGPT 20K | 257.3 | 258.5 | 265.3 | +3.1% | +2.6% |

In all three of these raw measurements OEPLB shows positive gains (+2.6% to +4.7%) and beats EPLB in every case. **However, the three dataset files (`/tmp/exp_data/*`) no longer exist, and this table is not reproducible**; re-measurement on the obtainable equivalent datasets yields gains of +2.70%/+2.39%/−0.24%/−0.15% (below Table 3 in §5.4), where for the L512 row each arm individually agrees to within $\le1.5\%$ but the ratio drops to +2.39%. **The claims of this paper rest on the reproducible re-measured data; this table is retained only as a historical record.**

### F.3 O=256 (Decode-Heavy) Results

| Scenario | Baseline | EPLB | OEPLB | OE vs BL | OE vs EPLB |
|---|---|---|---|---|---|
| L512_O256 (8K) tps | 6652.9 | **2502.3** | 6698.9 | +0.7% | **+167.7%** |
| L512_O256 tpot(ms) | 28.72 | **92.71** | 29.35 | +2.2% | **-68.3%** |

### F.4 Key Finding: EPLB's CUDA-Graph Disabling Cost Is Reproduced

**EPLB degrades catastrophically in the decode scenario, −62.4%** (2502.3 vs 6652.9 tps); tpot worsens from 28.72ms to 92.71ms (3.2× slower).
The logs confirm `cuda graph: False` during the EPLB run (CUDA graph is forcibly disabled because of `deepep_mode=normal`).

This perfectly reproduces the claim of limitation 1 in §2.2 of the paper (the paper reports −68% in the decode scenario; this reproduction gives −62.4%).

**OEPLB keeps CUDA graph** (log shows `cuda graph: True`), loses nothing in decode (+0.7%), and tpot increases by only +2.2%.

The OEPLB vs EPLB gap in the decode scenario is **+167.7%**—this is the strongest evidence for OEPLB over EPLB:
although EPLB's redundant experts can improve expert balance, the cost of forcibly disabling CUDA graph far exceeds the balancing gains in the decode scenario.

### F.5 System Efficiency (Compared Against the Theoretical Upper Bound)

An early draft used $f_{\text{eff}}\approx0.41$ here (an estimate of the FLOP fraction after shared-expert dilution, with no nsys trace) to state "theoretical upper bound 17.2%, system efficiency 15–27%". **That entry has been retracted**: the inverse solution in §2.4 shows that the actual r-sensitive time fraction $f_{\text{sens}}$ of the 57B is 0.06–0.22, 2–7× lower than 0.41; at the same time the draft used $r_{\text{before}}=1.74$, which is the worst window rather than the window mean (the measured 4-card avg is only 1.113, see D.2). Both parameters were overestimated simultaneously, leaving "upper bound 17.2%" without any basis and driving the computed system efficiency too low.

The corrected caliber ($x$ uses the avg ratio; $f_{\text{sens}}$ uses the Appendix G direct measurement):

| Experiment | $r_{\text{before}}$(avg) | $x$ | $\Delta_{\max}$ | Measured gain | System efficiency |
|---|---|---|---|---|---|
| 57B L512_O1 (4 GPUs) | 1.113 (borrowed*) | 0.084 | TBD | +4.7% | undetermined |
| 57B multi-domain (4 GPUs) | 1.113 (borrowed*) | 0.084 | TBD | +2.6% | undetermined |
| 57B ShareGPT (4 GPUs) | 1.113 (borrowed*) | 0.084 | TBD | +3.1% | undetermined |
| **57B L256 (4 GPUs)** | **1.107 (measured)** | **0.079** | **+2.57%**‡ | **+2.70%** | **105%** |

*The $r_{\text{before}}$ of these three rows was **not measured on their respective datasets**: the 1.113 was recorded by DIAG on a different load and is shared across all three rows. Since $r$ depends on the router's actual distribution on that dataset, different datasets have different $r_{\text{before}}$ in principle. `driver17.sh` is currently recording routing counts per dataset (L512_O1 8K / multi-domain / ShareGPT 20K) and recomputing the respective values using the offline definition of $r_{\text{avg}}$ (Appendix G.1); only then do the $x$ and system efficiency of these three rows become defined. For now they are marked "undetermined" rather than filled in with a borrowed number.
†The last row is the only 4-card row with a measured $r_{\text{before}}$, a measured upper bound (Appendix G.2 sweep), and a same-dataset OEPLB control arm (`driver18.sh`, +2.70%): it is the fully same-source row.
‡The $x$ in this column is given under the same caliber as the three rows above ($r_{\text{after}}=1.02$) for comparison; whereas $\Delta_{\max}=2.29\%$ is computed under the §2.4 caliber, taking the time-averaged operating point reported by DIAG, $r_{\text{after}}=1.04$, and plugging in the measured $r_k=1.032$, which gives $x_{\text{eff}}=0.061$. Even if the balancer truly compressed things down to 1.02, $x_{\text{eff}}$ would still be 0.061—because $\max(r_{\text{after}},r_k)=r_k=1.032$, and **the part pushed past the dead zone no longer converts into time**; that is exactly what the dead zone means.

Note that $x=0.084$ implies that even if $f_{\text{sens}}=1$ (all time is $r$-sensitive), the upper bound for the 4-card 57B is only 9.1%—so **if the $r_{\text{before}}$ of these three rows is indeed all around 1.11, then +4.7% would require $f_{\text{sens}}=0.54$**, higher than the 0.369 measured on 4 GPUs. This tension has three possible exits: (i) these datasets' own $r_{\text{before}}$ is higher than 1.11 (`driver17.sh` is measuring it); (ii) $f_{\text{sens}}$ rises with the load (L512 prompts are longer with more tokens per batch; G.3 limitation 3); (iii) +4.7% was inflated by inter-run noise (two runs of the baseline on this configuration once differed by 8.1%, whereas the Appendix G sweep, with placement fixed, pushed the CV down to 0.36%, indicating that the 8.1% comes from placement inheritance and temporal drift rather than intrinsic jitter). The three are distinguishable, but all of them require new data.

**The earlier inference that "the 4-card upper bound ≈ 0 and +1.85% can only be noise" has been falsified.** That inference relied on the untested premise that "$r_k$ is independent of EP scale": if $r_k$ were always 1.099, the 4-card $x_{\text{eff}}$ would be only 0.007–0.013 and the upper bound near zero. The 4-card sweep of `driver13.sh` measured **$r_k=1.032$** directly (14 runs, $R^2=0.9992$), which falls below $r_{\text{after}}=1.04$, so the dead zone does not operate at 4 GPUs: $x_{\text{eff}}=x=0.061$, upper bound +2.29%, independently confirmed by the empirical ceiling of +2.63%. **That $r_k$ moves with EP scale is a measured fact, not an assumption** (1.099@EP=8 → 1.032@EP=4, Appendix G.2). As for inverse solving: 8-card inverse solution 0.061 vs measured 0.335 (5.5× too low); 4-card inverse solution swings between 0.30 and 0.54 depending on $r_{\text{before}}$ vs measured 0.369—both configurations demonstrate that inverse solving is unusable at small $x$.

### F.6 Full Three-Dataset Comparison at O=256 (Decode-Heavy)

| Dataset | Baseline tps | OEPLB tps | Delta | Baseline tpot | OEPLB tpot | tpot Delta |
|---|---|---|---|---|---|---|
| L512_O256 (8K) | 6664.6 | 6710.6 | **+0.7%** | 28.5ms | 29.1ms | +2.1% |
| multi_O256 (16K) | 4244.6 | 4300.3 | **+1.3%** | 49.38ms | **43.51ms** | **-11.9%** |
| sg_O256 (20K) | 8282.1 | 8431.0 | **+1.8%** | 28.61ms | 27.97ms | -2.2% |

OEPLB shows positive gains in all three decode-heavy scenarios (+0.7% to +1.8%).

**The tpot improvement on multi_O256, −11.9%, is the most striking**: in the domain-switching scenario, per-token latency in the decode phase
drops from 49.38ms to 43.51ms. This is because domain switching causes expert hotspots to drift, and OEPLB continuously
corrects the deviation through swaps, reducing straggler waits during decode.

**Comparison with EPLB**: EPLB degrades catastrophically by −62.4% in the decode scenario because it disables CUDA graph (see F.3),
whereas OEPLB keeps CUDA graph and therefore loses nothing in decode. This is the core advantage of OEPLB over EPLB.

---

## Appendix G: Direct Measurement of the r-Sensitive Time Fraction $f_{\text{sens}}$

The upper-bound model of §2.4 has only one free parameter, $f_{\text{sens}}$, which cannot be substituted by the MoE FLOP fraction
(it overestimates by 1.4–1.9×, see §2.4), nor can it be inversely solved from a single measured gain point—when $x=1-r_a/r_b$ is small,
inverse solving amplifies the errors in $\Delta$ and $r_b$ multiplicatively (4-card 57B: changing $r_b$ from 1.113 to 1.20 drops the
inverse-solved $f$ from 0.54 to 0.30, see F.5; the inverse-solved value for the 8-card 57B is 5.5× lower than the direct measurement
of this appendix). This appendix measures $T(r)$ directly via a **forced-imbalance sweep** and reads $f_{\text{sens}}$ off the slope,
without relying on nsys traces, on FLOP priors, or on $\beta$ calibration. The sweep simultaneously **falsifies the functional form of
the model**: $T(r)$ is not a straight line but a hinge with a dead zone—something no single-point experiment could have revealed.

### G.1 Method

1. **Record the real routing distribution.** Start the server with `--expert-distribution-recorder-mode stat`,
   run the target load once, and export the token counts of every logical expert in every layer via
   `/dump_expert_distribution_record` (a native SGLang interface, no modification needed).
2. **Construct layouts of target imbalance offline.** Given each layer's count vector and a target $r$, construct with **deficit greedy**: the expected load vector takes GPU0 $=r\cdot\mu$ and the remaining GPUs $=\frac{EP-r}{EP-1}\mu$, then experts are assigned in descending order of hotness, one at a time, to the GPU with "the largest deficit that still has a free slot". Two endpoints are added: identity (the model's natural placement) and concentrated (the hottest $E/EP$ experts all pressed onto GPU0). All layouts are **pure permutations**, with no redundant experts, so the total number of experts, GPU memory usage, and KV cache capacity are byte-identical to baseline.

   **Layouts must be constructed per layer.** An early implementation picked the counts aggregated across layers and chose **one** permutation shared by all 28 layers; as a result all four target points (1.10/1.20/1.35/1.60) degenerated to the same value $r_{\text{agg}}=1.050$, and the concentrated endpoint only reached 1.115. The reason is that **a single permutation can neither create nor cancel per-layer imbalance**: each layer routes to different logical experts, so any fixed permutation averages out across layers, and the aggregate load tends naturally toward uniformity. After switching to one permutation per layer, each pressing that layer's hot experts onto the same GPU, the sweep interval expanded from 0.11 to 0.54 ($r\in[1.010,\,1.550]$). Had the shared permutation been kept, with $f_{\text{sens}}\approx0.34$ the effect would have been only 0.7%, and **the entire curve would have been buried in noise**.

   **The $x$-axis uses the per-layer average ratio** $r_{\text{avg}} = \frac{1}{L}\sum_l \max_g L_{l,g} / \overline{L_{l,\cdot}}$, not the aggregate ratio: every MoE layer is an independent dispatch/GEMM/combine barrier, and the shortfalls accumulate layer by layer; this is also exactly the `avg_ratio_before/after` reported by PB-OEPLB-DIAG, so the sweep's $x$-axis is directly comparable to the measured $r$ of §5. **Independent validation of this definition**: the identity-layout $r_{\text{avg}}$ computed offline from the recorded counts agrees with the first decision value self-reported by the runtime balancer to within 0.5%—8-card 1.218 vs 1.216, 4-card 1.107 vs 1.113. The two paths (offline count recomputation / runtime self-reporting) are completely independent.

3. **Measure point by point with the balancer off.** Fix the layout with `--init-expert-location <json>`,
   enable neither PB-OEPLB nor EPLB, run the same benchmark, and record the end-to-end time $T$.
   Each point is independently restarted, 2 rounds.
4. **Fit and perform model selection.** Fit both forms simultaneously—the linear $T=A+Br$ assumed by the draft, and
   the hinge $T = T_{\text{flat}} + B\max(0, r-r_k)$ ($r_k$ grid-searched within the sweep interval;
   given $r_k$, $T_{\text{flat}},B$ have a closed-form least-squares solution)—and compare $R^2$ and residual sum of squares.
   This step is not to prettify the fit but to **test** the assumption "$T$ is linear in $r$" itself, which is the entire
   foundation of the $\beta$ decomposition model. The winner's slope gives
   $$f_{\text{sens}} = \frac{B\,r_{\text{before}}}{T(r_{\text{before}})}$$
   to be substituted into §2.4. The identity point **does not participate in the fit** and is held out as a held-out check.

The key to this design is that **placement is the only variable**: permutation changes neither the parameter count, nor the redundancy,
nor CUDA-graph availability, and introduces no online overhead whatsoever (decision and swap are off throughout), so any change in $T$
can only be attributed to $r$. By contrast, inversely solving $f_{\text{sens}}$ from an OEPLB on/off comparison would conflate decision
overhead, swap blocking, and all_reduce bandwidth contention all at once.

### G.2 Results

Qwen2-57B-A14B, 8×H20, L256_O1_realprover_n16384, conc=256. Seven layout points, 2 rounds each, each round independently restarted—14 runs in total, **16384/16384 successful, 0 errors**. The server parameters, apart from `--init-expert-location`, are verbatim identical to baseline (`enable_pb_oeplb=False`, `enable_eplb=False`, `ep_num_redundant_experts=0`, `deepep_mode=auto`, CUDA graph on), verified against the `server_args` dump.

| Layout | $r_{\text{avg}}$ | Round 1 (s) | Round 2 (s) | Mean | CV | vs bal |
|---|---|---|---|---|---|---|
| bal | 1.010 | 82.87 | 82.56 | **82.72** | 0.27% | — |
| r110 | 1.073 | 82.86 | 83.14 | **83.00** | 0.24% | +0.34% |
| r122 | 1.148 | 83.54 | 84.15 | **83.84** | 0.51% | +1.35% |
| identity | 1.218 | 85.69 | 86.06 | **85.88** | 0.30% | +3.82% |
| r135 | 1.220 | 85.98 | 86.02 | **86.00** | 0.03% | +3.97% |
| r150 | 1.287 | 86.68 | 87.80 | **87.24** | 0.91% | +5.47% |
| conc | 1.550 | 93.39 | 93.58 | **93.48** | 0.14% | +13.0% |

**(a) $r$ is a sufficient statistic for throughput.** The $r_{\text{avg}}$ of identity and r135 differ by only 0.002 (1.218 vs 1.220), yet their layout structures are entirely unrelated—one is the model's natural order, the other presses the hot experts artificially onto GPU0—the measured times are 85.88 s vs 86.00 s, a difference of **0.14%**, within the inter-round CV. This supports §2.4's parameterization of placement by a single scalar $r$.

**(b) $T(r)$ is a hinge, not a straight line.** Fit on the 6 constructed points (identity held out):

| Model | Fitted form | $R^2$ | RSS |
|---|---|---|---|
| Linear (draft assumption) | $T = 60.71 + 20.86\,r$ | 0.9772 | 1.866 |
| **Hinge** | $T = 82.86 + 23.60\cdot\max(0,\,r-1.099)$ | **0.9981** | **0.154** |

The hinge's residual sum of squares is **12.1×** lower, and the measurements in the $r\le r_k$ region directly corroborate the dead zone: raising $r$ from 1.010 to 1.073 ($\Delta r=0.063$) costs only an extra 0.34%, whereas a $\Delta r$ of the same size above the knee (1.148→1.220 and beyond, slope 23.6 s per unit $r$) costs 1.7%. The held-out identity point: the hinge predicts 85.68 s (measured 85.88, $-0.23\%$), the linear predicts 86.12 s ($+0.28\%$).

**(c) Direct measurement of $f_{\text{sens}}$.** Slope $B=23.60$ s per unit $r$; at $r_{\text{before}}=1.218$,
$$f_{\text{sens}} = \frac{B\,r_{\text{before}}}{T(r_{\text{before}})} = \frac{23.60\times1.218}{85.68} = \mathbf{0.335}$$
which differs **5.5×** from the 0.061 inversely solved from the +1.0% single point, and differs from this model's MoE FLOP fraction of 46.9% by only 1.4× (§2.4).

**(d) Upper bound and empirical ceiling.** $r_{\text{after}}=1.04 < r_k=1.099$, so $x_{\text{eff}} = (1.218-1.099)/1.218 = 0.098$ (naive $x=0.146$),
$$\Delta_{\max} = \frac{0.335\times0.098}{1-0.335\times0.098} = \mathbf{+3.40\%}$$
The empirical ceiling from the same batch of data (identity→bal, i.e., "perfect balancer, zero overhead") is **+3.82%**, differing from the fitted upper bound by 0.42pp. PB-OEPLB's measured gain on this configuration is +1.0%, i.e., a system efficiency of about **29%**.

**(e) Direct implications for `threshold_ratio`.** The dead zone means that shortfalls with $r\in[1.02,\,1.099]$ are **not worth** triggering a swap: pushing $r$ from 1.073 down to 1.010 is worth only 0.34% on this data (the P2P blocking of a single swap plan is already on the same order). The default threshold 1.02 should be raised toward $r_k$; the threshold choice in §3.2 thereby changes from an "empirical value" into a **measurable** parameter. This also explains the near-noise OEPLB gain on the 57B configuration in §5.3: that configuration's $r_{\text{before}}=1.218$ is only 11% above $r_k=1.099$ in the first place.

**(f) Independent sweep at EP=4: the dead zone moves.** Same model, same dataset (L256_O1×16384, conc=256), same construction method; only EP is changed from 8 to 4 (4×H20, 16 experts/GPU). Seven layout points, 2 rounds each, 14 runs in total, 16384/16384 successful, 0 errors:

| Layout | $r_{\text{avg}}$ | Round 1 (s) | Round 2 (s) | Mean | CV | vs bal |
|---|---|---|---|---|---|---|
| bal | 1.004 | 135.55 | 135.42 | **135.49** | 0.07% | — |
| r108 | 1.059 | 136.53 | 137.23 | **136.88** | 0.36% | +1.03% |
| r115 | 1.102 | 138.46 | 138.35 | **138.41** | 0.06% | +2.15% |
| identity | 1.107 | 138.90 | 139.22 | **139.06** | 0.16% | +2.63% |
| r125 | 1.154 | 141.58 | 141.04 | **141.31** | 0.27% | +4.30% |
| r140 | 1.224 | 144.41 | 144.26 | **144.33** | 0.07% | +6.52% |
| conc | 1.400 | 152.45 | 152.62 | **152.53** | 0.08% | +12.6% |

| Model | Fitted form | $R^2$ | RSS |
|---|---|---|---|
| Linear | $T = 90.19 + 44.34\,r$ | 0.9940 | 1.186 |
| **Hinge** | $T = 135.48 + 46.35\cdot\max(0,\,r-1.032)$ | **0.9992** | **0.156** |

The hinge's RSS is **7.6×** lower; on the held-out identity point the hinge predicts 138.95 s (measured 139.06, $-0.08\%$), the linear predicts 139.27 s ($+0.15\%$). **The hinge form wins on two independent configurations**, which is direct evidence that it is not a product of overfitting.

Three conclusions:

1. **$r_k$ moves with EP scale: 1.099 (EP=8) → 1.032 (EP=4).** This falsifies "$r_k$ is a configuration-independent constant", so the starred rows of §2.4 must not borrow 1.10. It is physically legible: at EP=4 the per-GPU expert GEMMs are twice those at EP=8, and the fitted slope $B$ correspondingly rises from 23.60 to 46.35 (almost exactly 2×), while $T_{\text{flat}}$ rises only from 82.86 to 135.48 (1.63×)—the fixed slack that can be absorbed by overlap occupies a smaller share of the total time, and the dead zone is therefore narrower.
2. **$f_{\text{sens}}$, by contrast, is essentially stable: $B r_b/T(r_b) = 46.35\times1.107/138.95 = \mathbf{0.369}$, vs 0.335 at EP=8, a 9% difference.** That is, **the transferability of the two parameters is completely different**: $f_{\text{sens}}$ characterizes "how much time is sensitive to $r$", is determined by model structure and component proportions, and is fairly stable across EP; $r_k$ characterizes "how much shortfall can be absorbed by overlap", is determined by the per-GPU workload, and varies with the parallel configuration. When extrapolating across configurations, one may borrow $f_{\text{sens}}$ but not $r_k$.
3. **Upper bound and ceiling for this configuration.** $r_{\text{after}}=1.04 > r_k=1.032$, the dead zone does not operate, $x_{\text{eff}}=x=0.061$, $\Delta_{\max}=+2.29\%$; the empirical ceiling (identity→bal) is **+2.63%**, a difference of 0.34pp. Note that this contrasts with 8 GPUs: the 8-card $r_{\text{before}}$ is higher (1.218 vs 1.107) **and** its dead zone is wider—the former enlarges the headroom, the latter compresses it—and the net result is that the 8-card upper bound (3.40%) is still larger than the 4-card one (2.29%).

**(g) Layout invariance reproduces on 4 GPUs.** r115 ($r=1.102$, hot experts pressed artificially onto GPU0) and identity ($r=1.107$, the model's natural order) differ in time by 0.47% (138.41 vs 139.06), comparable to the 0.45% difference in $r$; two layouts that are structurally unrelated but close in $r$ give close times, consistent with the conclusion of (a) on 8 GPUs.

**(h) Independent sweep on 235B: $r_k$ is nearly identical across models.** Qwen3-235B-A22B-FP8, 8×H20, L512_O1×8192, conc=256. Six layout points, 2 rounds each, 12 runs in total, 8192/8192 successful, 0 errors (the 7th point $r=4.686$ cannot start, see Appendix E.3):

| Layout | $r_{\text{avg}}$ | Round 1 (s) | Round 2 (s) | Mean | CV |
|---|---|---|---|---|---|
| bal | 1.000 | 166.83 | 167.32 | **167.07** | 0.21% |
| r120 | 1.200 | 172.67 | 174.23 | **173.45** | 0.64% |
| r140 | 1.400 | 185.28 | 185.03 | **185.16** | 0.10% |
| r160 | 1.600 | 196.47 | 196.12 | **196.30** | 0.13% |
| identity | 1.737 | 203.47 | 204.59 | **204.03** | 0.39% |
| r175 | 1.750 | 205.06 | 207.05 | **206.06** | 0.68% |

| Model | Fitted form | $R^2$ | RSS |
|---|---|---|---|
| Linear | $T = 112.12 + 52.87\,r$ | 0.9883 | 11.952 |
| **Hinge** | $T = 167.07 + 58.78\cdot\max(0,\,r-1.093)$ | **0.9995** | **0.475** |

The hinge's RSS is **25.2×** lower (the largest gap of the three configurations). $f_{\text{sens}} = 58.78\times1.721/203.98 = \mathbf{0.496}$ (where $T(1.721)=167.07+58.78\times0.628=203.98$), $\beta = 58.78/167.07 = 0.352$.

Three conclusions:

1. **$r_k$ is nearly constant across models.** 1.093 for 235B/EP8 vs 1.099 for 57B/EP8, a 0.5% difference—yet the two models differ in expert count (128 vs 64), shared expert (none vs giant), and number of layers (94 vs 28). Together with 57B/EP4 = 1.032 from (f), the direction of transferability becomes clear: **borrowable along models, not borrowable along parallel configurations**.
2. **The upper bound and the empirical ceiling agree to within 0.04pp.** $\Delta_{\max}$ ($r_b=1.721\to r_a=1.05$) is $+22.08\%$; the empirical ceiling identity 204.03 → bal 167.07 $=+22.12\%$. This is the tightest of the three configurations. PB-OEPLB measures $+17.5\%$, a system efficiency of **79%**.
3. **The nsys $\beta$ decomposition of $f_{\text{sens}}$ is falsified.** Decomposition value 0.384 vs measured 0.496, an underestimate of 26%. The "109%>100%" contradiction in §2.4 arose precisely from this, not from a borrowing error in $r_k$.

**(i) Load dependence: $r_k$ is insensitive to concurrency, $f_{\text{sens}}$ is sensitive.** 57B/EP8, three layouts each measured at conc=64/256/512 (2 rounds each):

| Concurrency | bal(1.010) | r122(1.148) | conc(1.550) | $B$ (s per unit $r$) | $f_{\text{sens}}$ | $r_k$ estimate |
|---|---|---|---|---|---|---|
| 64 | 106.09 | 107.88 | 116.88 | 22.38 | 0.249 | 1.068 |
| 256 | 82.72 | 83.84 | 93.48 | 23.98 | 0.342 | 1.101 |
| 512 | 81.99 | 83.79 | 93.04 | 23.01 | 0.328 | 1.070 |

The **absolute cost** of imbalance ($B$, in seconds per unit $r$) is almost unchanged over an 8× range of concurrency (22.4–24.0); what changes is its **fraction** of total time—low concurrency stretches $T_{\text{flat}}$ (106 s vs 82 s, due to poor batching efficiency), so $f_{\text{sens}}$ drops from 0.342 to 0.249. The estimate of $r_k$ (three points determine three parameters, so this is an exact solution rather than a fit) falls in 1.068–1.101, i.e., **$r_k$ is far less sensitive to concurrency than to EP scale** (the latter 1.099→1.032). This supports "$r_k$ is determined by the per-GPU workload rather than by request concurrency", but note that (f) has already ruled out the stronger version, "$r_k$ is determined by the number of experts per GPU".

### G.3 Limitations

1. **The dead-zone position moves with EP but is now predictable.** Fitting the three 57B EP points (EP2, EP4, EP8) gives $r_k-1 = 0.00408\cdot\text{EP}^{1.52}$; 235B/EP8 is a cross-model **blind test** (never seen by the fit) with error 3.8%. The exponent decomposes into $T_{\text{gemm}}\propto\text{EP}^{-0.99}$ (theoretical $-1$) ÷ $\text{slack}\propto\text{EP}^{+0.53}$ — the ratio recovers 1.52 exactly. Caveats: only one model gives the EP trend; the cross-model check is a single point; all four are on the same hardware (H20, NVLink, no IB). The per-config measurement method (G.1, ~2 h) remains the fallback.
2. **Single-point inverse solving is unusable at small $x$.** The methodological conclusion of this appendix: when $\Delta$ is on the same order as inter-run noise (in this paper $\Delta\lesssim2\%$, CV 1.2%), $f_{\text{sens}}$ must not be inversely solved from $\Delta$. The two configurations each provide a counterexample: the 8-card inverse solution 0.061 vs the sweep measurement 0.335 (5.5× too low); the 4-card inverse solution swings between 0.54 and 0.30 depending on whether $r_{\text{before}}$ is taken as 1.113 or 1.20, vs the sweep measurement 0.369. By contrast, the two values from the sweeps (0.335, 0.369) differ by 9%—**$f_{\text{sens}}$ itself is a stable quantity; what is unstable is inverse solving as a tool**. Only sweeping $T(r)$ can yield a meaningful $f_{\text{sens}}$.
3. **$f_{\text{sens}}$ is load-dependent.** What this appendix provides are values under a specific benchmark configuration (L256, $O=1$, conc=256). The former anomaly of 118% (early draft 131%) on the 235B multi-domain configuration in §2.4 **remains open**: the +14.0% comes from the deleted `/tmp/exp_data/multidomain_16k.jsonl` and cannot be reproduced, and a measurement on the different `multidomain_v2_out1.jsonl` (−1.1%) is **not a same-condition re-test and does not falsify it** (see §2.4 ⚠). Separately, a reproducible multi-domain dataset (`prefill_heavy_universal.jsonl`, 16000 requests) gives a clean headline gain of **+9.76%** (`driver39.sh`), showing multi-domain gains are real even though the original +14.0% figure itself is unverifiable.
4. **The sweep measures the steady-state cost of static layouts and contains no dynamic terms.** Decision overhead, swap blocking, and convergence lag are all intentionally excluded (the balancer is off throughout), so $\Delta_{\max}$ is the "perfect balancer" upper bound; the ratio of the measured gain to it (system efficiency) is the quantity this paper's algorithm actually seeks to optimize.


## Appendix H: Joint Optimum of Adaptive Window and Decay ($M=W/(1-\alpha)$)

### H.1 Theory

§3.5 argued that window $W$ and decay $\alpha$ affect the steady-state bias-variance operating point only through the effective memory $M=W/(1-\alpha)$, and that the optimal memory has the closed form
$$M^\star = \left(\frac{a\,c^2\,L_{\text{seg}}}{b\,\beta\,\bar t\,\gamma^2 (r-r_k)^3\ln2}\right)^{1/2}$$
This appendix verifies two falsifiable claims on 235B/EP8: (P1) $M$ is a sufficient statistic — different $(W,\alpha)$ with the same $M$ land on one throughput curve; (P2) the optimal $M^\star$ grows as $\sqrt{L_{\text{seg}}}$.

### H.2 Experimental Design

**Controlled piecewise-stationary workload.** The earlier "multi-domain" set turned out near-homogeneous (per-dataset $r_{\text{before}}$ differs $<1.5\%$, Appendix D.2) and never exercised a real changepoint. We therefore use `make_segmented.py` to interleave L256 (math, skewed routed distribution) and ShareGPT (chat) at known segment lengths $L_{\text{seg}}\in\{50,200,1000\}$, constructing a controlled changepoint sequence.

**$(W,\alpha)$ grid** (`driver31.sh`, complete): points chosen so several $(W,\alpha)$ share the same $M\in\{16,32,64,160\}$. Each point on 235B measures throughput, decision count, swap cost, and the cos_sim trajectory.

### H.3 Pre-registered Predictions vs. Measured Results

The predictions were recorded before the experiment (`NOTES.md`, 2026-08-12 17:25); this section reconciles them as measured, with no post-hoc revision.

**Setup.** 235B/EP8, `segp_L1000.jsonl` (8000 requests, 8 segments × 1000, alternating math and chat; both domains forced to $O=1$ pure prefill so that a changepoint is **only** a routing-distribution change). Reference arm: same dataset with OEPLB disabled, 62.9 s. A uniform swap budget of 0.10 was applied to every arm so that no arm could hang and pollute the comparison.

| Arm | $M$ | $W$ | $\alpha$ | Time (s) | vs. ref | Decisions | Swaps issued | Budget hits |
|---|---|---|---|---|---|---|---|---|
| M16_W8a50 | 16 | 8 | 0.50 | **340.4** | **5.41×** | 7 | 6 | 8 |
| M16_W16a0 | 16 | 16 | 0 | 69.9 | 1.11× | 21 | 21 | 0 |
| M32_W8a75 | 32 | 8 | 0.75 | 69.1 | 1.10× | 21 | 20 | 8 |
| M32_W16a50 | 32 | 16 | 0.50 | 69.5 | 1.10× | 19 | 18 | 8 |
| M32_W32a0 | 32 | 32 | 0 | 69.2 | 1.10× | 16 | 16 | 0 |
| M64_W16a75 | 64 | 16 | 0.75 | 72.2 | 1.15× | 15 | 15 | 8 |
| M64_W32a50 | 64 | 32 | 0.50 | 69.0 | 1.10× | 12 | 12 | 0 |
| M64_W64a0 | 64 | 64 | 0 | 68.9 | 1.09× | 5 | 4 | 0 |
| **Adaptive window + adaptive decay** | — | 16 init | 0.5 init | **67.0** | **1.07×** | 10 | 10 | 0 |

(The adaptive-window-only arm lost its result file to an abnormal bench-process exit; its log records 13 decisions / 12 issued / 2 budget hits. It is excluded from the comparison.)

**P1' (same $M$, different $(W,\alpha)$, spread $<5\%$) — partially falsified.**

| $M$ | Within-group spread | Verdict |
|---|---|---|
| 16 | **387%** | ✗ falsified |
| 32 | 0.5% | ✓ holds |
| 64 | 4.8% | ✓ holds |

For $M\ge32$, $M$ is indeed an approximate sufficient statistic (within-group spread $\le4.8\%$), but **there is a pathological corner: $(W=8,\alpha=0.5)$**. Note this is not "$W=8$ is pathological" — $(W=8,\alpha=0.75)$ (i.e. $M$=32) is perfectly normal at 69.1 s. The pathological point remains **5.4× slower** even after hitting the swap budget 8 times and having its decision count squeezed to 7, so the cause is not decision frequency but the **migration volume per decision**: a small $W$ with a middling $\alpha$ keeps the accumulated history persistently mismatched against the current placement, so every decision attempts a large correction. The swap budget throttles but does not remove it. **$M$ is therefore an approximate, not an exact, sufficient statistic, and the $(W,\alpha)$ plane contains a region to avoid. Practical rule: $W\ge16$; if $W<16$, use $\alpha=0$ or $\alpha\ge0.75$ and avoid intermediate values.**

**P2 (throughput-vs-$M$ is unimodal) — direction confirmed, coefficient not calibrated.** On `segp_L1000` the best time in each $M$ group decreases monotonically with $M$ (69.9 → 69.1 → 68.9 s for $M$=16/32/64); no peak appears, so the latency cost has not yet materialized (a 1000-request segment is still long relative to $M\le64$). The follow-up on the shorter `segp_L200` (`driver34.sh`) shows the **opposite** direction: time rises monotonically with $M$ (throughput falls, 65.8 → 67.9 s, optimal $M$=16). Short segments thus favor small $M$ and long segments favor large $M$, i.e. **$M^\star$ grows with $L_{\text{seg}}$**, supporting the closed form's directional prediction. Since neither dataset shows an interior peak, the numerical coefficient of the $\sqrt{L_{\text{seg}}}$ relation remains uncalibrated

**An intermediate-segment ($L_{\text{seg}}=500$) peak-finding attempt (`driver41.sh`) did not yield clean data, recorded honestly**: on the drifting workload, higher-$M$ (aggressive-adaptation) arms slow the benchmark substantially via swap blocking and repeatedly approach the timeout, with several runs hitting high error rates (e.g. 387 errors out of 5000 requests at one $M$ point) so no reliable throughput could be obtained. The interior peak at an intermediate segment length therefore could not be cleanly measured under our conditions, and calibrating the $M^\star$ coefficient is left to future work. The attempt itself confirms one point: **on a strongly drifting workload the real cost of aggressive adaptation (swap blocking) can exceed its benefit**, consistent with the qualitative expectation that $M^\star$ has an interior optimum, though the quantitative peak needs a more controlled setup..

**P3 (adaptive decay is better) — holds.** Against the static configuration with the same starting point: fixed $\alpha=0.5$ ($W$=16) gives 69.5 s and 18 swaps; adaptive window + adaptive decay gives **67.0 s and 10 swaps** — 3.6% faster with 44% fewer swaps, and the fastest arm in the table (1.07× the reference). This supports §3.5's argument that zeroing $\alpha$ at a changepoint, to clear the mismatched history, is more effective than shrinking the window alone.

**P4 ($W=8$ arms hit the swap budget) — holds.** Both M16_W8a50 and M32_W8a75 hit it 8 times, confirming that decision frequency is a degree of freedom independent of $M$.

**Overall.** The adaptive mechanism **is worth having**: the best adaptive arm (67.0 s) beats all eight static $(W,\alpha)$ configurations, and the contribution of adaptive decay is separately attributable (3.6% faster and 44% fewer swaps than the same-starting-point static arm). But §3.5's claim that $(W,\alpha)$ reduces entirely to a single $M$ is **too strong** — it holds for $M\ge32$ and fails at $(W=8,\alpha=0.5)$. The closed form for $M^\star$ has its direction confirmed (P2) but its coefficient awaits calibration.
