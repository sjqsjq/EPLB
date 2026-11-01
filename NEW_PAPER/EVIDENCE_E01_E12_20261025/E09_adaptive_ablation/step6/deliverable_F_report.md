# Deliverable F — Step-6 Post-Freeze Formal Comparison Results

Date: 2026-10-16 · Protocol: `step6_formal_protocol_E.md` (pre-registered before any formal run) ·
Knob freeze: `../adw_v2/FREEZE_record_v2_knobs.md` · Machine-readable data: `deliverable_F_analysis.json`.
All numbers below are emitted programmatically from raw `bench.json` / `v2_rank0.jsonl` / the analysis JSON — none hand-transcribed.

## 0. Run status & integrity

- **36/36 pre-registered measurement starts completed** (Phase A grid 12 + Phase B 18 + Phase C 6). No additions, no selection, no repeats beyond protocol.
- **1 INFRA-class re-run** (W0_f32 → W0_f32__s1r; EADDRINUSE dist-port race at boot, zero data produced → allowed by protocol §7; deviation memo `step6_deviation_infra_portrace.md`). **0 MECHANISM-class failures** (nothing retired after serving started).
- Every start: `ok == N_rows`, `errors == 0`; code-lock sha256 verified at each phase start; frozen knobs untouched (Phase C used only the 3 pre-registered ABL_* overrides, verified in ACTIVE log lines).
- **Zero desync, zero v2 errors** in all 24 controller-armed starts (V2×6, ablations×6, plus integrity checks on every v2 log).
- **W2_V2 s1/s2: change windows bit-identical** (same window indices, same W transitions) — the mechanism is deterministic under identical workload order.

## T1. Raw full-service times (primary endpoint; includes ALL online overhead)

### Phase A — fixed-W grid (1 start/cell, fixed order W=8,16,32,64 per condition)

| cond | W=8 | W=16 | W=32 | W=64 | spread | F-local argmin |
|---|---|---|---|---|---|---|
| W0 | 366.82 | 366.84 | 368.22 | 368.65 | 0.50% | **W=8** (366.82s) |
| W1 | 351.05 | 353.71 | 353.34 | 356.11 | 1.44% | **W=8** (351.05s) |
| W2 | 713.32 | 708.91 | 705.55 | 721.91 | 2.32% | **W=32** (705.55s) |

Grid spreads 0.5–2.3%: at α=0 on these workloads the window-size time headroom itself is small. This bounds what ANY window controller can win and is the key context for T2/T4.

### Phase B — formal arms (2 starts/cell, round-robin 2 cycles)

| cond | arm | start 1 (s) | start 2 (s) | ok/err |
|---|---|---|---|---|
| W0 | V2 | 370.39 | 362.34 | 8037/0, 8037/0 |
| W0 | FG | 365.63 | 361.29 | 8037/0, 8037/0 |
| W0 | FL | 363.40 | 362.10 | 8037/0, 8037/0 |
| W1 | V2 | 353.64 | 350.89 | 4000/0, 4000/0 |
| W1 | FG | 354.61 | 350.15 | 4000/0, 4000/0 |
| W1 | FL | 357.98 | 354.36 | 4000/0, 4000/0 |
| W2 | V2 | 713.18 | 703.54 | 8000/0, 8000/0 |
| W2 | FG | 712.57 | 715.23 | 8000/0, 8000/0 |
| W2 | FL | 704.57 | 720.00 | 8000/0, 8000/0 |

FG = F-global (controller=1, W=16 on every condition). FL = F-local (controller=1, per-condition grid argmin: W0→8, W1→8, W2→32).

## T2. Per-start-pair deltas (V2 − baseline), pre-registered n=2, no pooling

| cond | Δ vs FG s1 | Δ vs FG s2 | Δ vs FL s1 | Δ vs FL s2 | V2 start spread |
|---|---|---|---|---|---|
| W0 | +4.76s (+1.30%) | +1.05s (+0.29%) | +6.99s (+1.92%) | +0.24s (+0.07%) | 8.05s |
| W1 | -0.97s (-0.27%) | +0.74s (+0.21%) | -4.34s (-1.21%) | -3.47s (-0.98%) | 2.75s |
| W2 | +0.61s (+0.09%) | -11.69s (-1.63%) | +8.61s (+1.22%) | -16.46s (-2.29%) | 9.64s |

**Per-condition reading (compatible-with-zero language; no CI claimable at n=2):**

- **W0 (calibration.jsonl — declared KNOWN benchmark regression, not unseen validation):** V2 slower than FG on both starts (+4.76s/+1.30%, +1.05s/+0.29%); vs FL +6.99s/+0.24s. Sign consistent vs FG, magnitude ≈0 at s2. Reported as-is.
- **W1 (stable code holdout):** ΔFG = −0.97s / +0.74s — sign flip, both <0.3% → **compatible with zero**. ΔFL = −4.34s / −3.47s (V2 faster both starts) — caveat: FL-W1's grid cell measured 351.05s while its two formal starts ran 357.98/354.36s; the W=8 argmin was likely a noise-lucky grid draw (grid spread 1.5%). Post-hoc best fixed cell is a labeled reference only, so ΔFL here must not be read as a V2 win.
- **W2 (predefined one-switch code→chinese):** ΔFG = +0.61s / −11.69s; ΔFL = +8.61s / −16.46s — **sign flips within start spread** (V2 spread 9.64s; FL-W2 spread 15.43s) → **compatible with zero** against both baselines.

**T2 bottom line: no net full-service-time win for V2 is demonstrated at n=2 on any condition; no net loss beyond start spread is demonstrated either — except W0-vs-FG, where both deltas are positive (V2 slower).** With grid headroom ≤2.3%, a time win was not statistically reachable at this n; the protocol therefore pre-declared behavior evidence as the co-primary endpoint (T3/T4).

## T3. Behavior evidence chains (V2 frozen arms)

| cell | windows | W changes | dwell steps by W | c_window median (ms) | desync |
|---|---|---|---|---|---|
| W0_V2__s1 | 65 | 9 | 8:280, 16:232, 32:544 | 0.56 | none |
| W0_V2__s2 | 63 | 11 | 8:272, 16:232, 32:528 | 0.54 | none |
| W1_V2__s1 | 37 | 3 | 8:40, 16:72, 32:880 | 0.62 | none |
| W1_V2__s2 | 35 | 1 | 16:128, 32:848 | 0.62 | none |
| W2_V2__s1 | 75 | 7 | 8:104, 16:144, 32:1728 | 0.66 | none |
| W2_V2__s2 | 75 | 7 | 8:104, 16:144, 32:1728 | 0.57 | none |

Highlights:
- **W1 (stable):** V2 s2 made **1 W-change all run** and spent **87% of steps at W=32** (848/976 dwell steps) — M2's 'stable + low headroom → longer interval' path exercised exactly as designed, zero churn.
- **W0:** 9–11 changes; both real boundaries detected (latency table below); **boundary #3 silent in ALL arms** (z ≪ z_lo — correctly no action, reproduces the calibration behavior run); **3 shrink_rejected_low_headroom** events (z=34.7/55.4/37.6 elevated but h_w below θ_act) — M2's 'high raw signal ≠ time benefit' gate demonstrably active; **14 grow_blocked_headroom** — headroom gate holds W ≤ 32 where M2off drifts to 64.
- **W2:** identical trajectories s1/s2; switch caught at w37/step 1024 (latency 36 steps, z=271.8/244.0, W 32→16); regrow only after the signal settles.
- **Cost:** c_window median 0.54–0.66 ms (max 1.25 excl. the once-per-run lazy-init first window 16.78 ms); allreduce counts symmetric across arms. Overhead is INSIDE the T1 service times, never subtracted.

### Step-aligned detection latency at predefined real changes

(steps = engine steps from run start; boundary step = boundary_row/N_rows × that run's total measured steps. Window-index comparison across arms is INVALID — arms run different W schedules, hence different steps/window. Steps are the only common clock. This table supersedes all live-monitoring latency claims; see §6.)

| change | arm | detect window | detect step | latency (steps past boundary) | z at detect | W transition |
|---|---|---|---|---|---|---|
| W2 switch (row 4000) | V2 (frozen, s1) | w37 | 1024 | 36 | 271.8 (urgent_shrink) | 32→16 |
| W2 switch (row 4000) | V2 (frozen, s2) | w37 | 1024 | 36 | 244.0 (urgent_shrink) | 32→16 |
| W2 switch (row 4000) | M2off | w25 | 1056 | 72 | 876.2 (urgent_shrink) | 64→32 |
| W2 switch (row 4000) | M3off | w35 | 1032 | 44 | 491.3 (urgent_shrink) | 32→16 |
| W2 switch (row 4000) | M1p (beta=0) | w34 | 1024 | 32 | 271.7 (urgent_shrink) | 32→16 |
| W0 B1 (row 2075) | V2 (frozen, s1) | w30 | 320 | 48 | 234.0 (urgent_shrink) | 16→8 |
| W0 B1 (row 2075) | V2 (frozen, s2) | w29 | 336 | 70 | 564.6 (urgent_shrink) | 32→16 |
| W0 B1 (row 2075) | M2off | w29 | 336 | 64 | 559.7 (urgent_shrink) | 32→16 |
| W0 B1 (row 2075) | M3off | w29 | 304 | 34 | 53.0 (confirmed_shrink) | 16→8 |
| W0 B1 (row 2075) | M1p (beta=0) | w21 | 344 | 74 | 322.0 (urgent_shrink) | 32→16 |
| W0 B2 (row 4052) | V2 (frozen, s1) | w43 | 568 | 36 | 204.1 (urgent_shrink) | 32→16 |
| W0 B2 (row 4052) | V2 (frozen, s2) | w42 | 576 | 56 | 419.5 (urgent_shrink) | 32→16 |
| W0 B2 (row 4052) | M2off | w41 | 592 | 60 | 584.4 (urgent_shrink) | 64→32 |
| W0 B2 (row 4052) | M3off | w45 | 576 | 48 | 444.7 (urgent_shrink) | 32→16 |
| W0 B2 (row 4052) | M1p (beta=0) | w31 | 584 | 56 | 490.6 (urgent_shrink) | 32→16 |
| W0 B3 (row 6068) | V2 (frozen, s1) | — | — | **no detection in run (silent)** | z below all thresholds | — |
| W0 B3 (row 6068) | V2 (frozen, s2) | — | — | **no detection in run (silent)** | z below all thresholds | — |
| W0 B3 (row 6068) | M2off | — | — | **no detection in run (silent)** | z below all thresholds | — |
| W0 B3 (row 6068) | M3off | — | — | **no detection in run (silent)** | z below all thresholds | — |
| W0 B3 (row 6068) | M1p (beta=0) | — | — | **no detection in run (silent)** | z below all thresholds | — |

Reading:
- **W2 switch:** Full latency 36 steps on both starts (identical). **M2off latency 72 = 2× Full**, detected at **W=64** — without the headroom gate the controller had over-extended to the coarsest window before the change: exactly the failure mode M2 exists to prevent. M3off 44, M1p 32 — comparable to Full.
- **W0 B1/B2:** all five arms detect both boundaries within 34–74 steps; no arm misses. **M1p is NOT late** (B1 74 vs Full 48/70; B2 56 vs Full 36/56).
- **W0 B3:** silent in every arm — a mild shift with no credible change signal; silence is the correct response (calibration behavior run: z_max 1.95).

## T4. Ablations vs Full (same condition, n=1 each — all times land inside Full's start spread, so ablation conclusions are BEHAVIORAL, as pre-declared)

| ablation | cond | time (s) | Full same-cond (s) | W changes | measured behavioral difference |
|---|---|---|---|---|---|
| M2off (θ_H_low=1e18, θ_act=0) | W0 | 368.31 | 370.39 / 362.34 | 12 | grows to **W=64** (Full: never; 14 grow_blocked_headroom); B2 detected from a 64-window with z=584 |
| M2off | W2 | 715.19 | 713.18 / 703.54 | 8 | sits at **W=64** when the switch hits → **detection latency 2× Full** (72 vs 36 steps), z=876.2; regrows to 64 within 9 windows post-switch |
| M3off (dwell 0/0, m=1/1, cap=99) | W0 | 374.54 | 370.39 / 362.34 | 9 | acts on single-window signal (B1 confirmed_shrink at z=53.0, no 2nd sample); fastest B1 latency (34 steps) — confirmation costs ~1 window and buys noise immunity not exercised on this workload |
| M3off | W2 | 712.12 | 713.18 / 703.54 | 7 | faster post-switch regrow (w42/43 vs Full w45/48) = more churn exposure; no time consequence at this headroom |
| M1p (β=0 only) | W0 | 374.24 | 370.39 / 362.34 | 7 | **no run-start transient shrink** (ref adapts in 1 window); its w11 urgent (step 192, z=127.9) matches a REAL event Full also saw at the same steps (Full w18–20 z=37.9–62.9; Full took no W action only because it was already at the W=8 floor) — **not a false alarm** |
| M1p | W2 | 716.39 | 713.18 / 703.54 | 5 | switch latency same as Full (32 vs 36 steps); post-switch z stays elevated (11.5/6.6/6.6/4.6 > z_lo=3.737) → regrow delayed to w51/58; 46 grow_blocked_headroom |

**M1p honest limitation (pre-declared, restated):** only the EWMA reference β was ablatable via config; fresh-sample normalization and the UNCERTAIN low-sample path could NOT be ablated. M1p ≠ 'M1 off' — it is 'M1 with a 1-window-memory reference'. Conclusions about M1 as a whole are partial.

**T4 bottom line:** at α=0 with grid headroom ≤2.3%, no ablation separates from Full on service time (all inside Full's start spread at n=1). They separate cleanly on **behavior**: M2 prevents over-extension to W=64 and the 2× detection latency that comes with it; M3 trades ~1 window of detection latency for confirmation/churn bounds; M1's β shapes run-start transient handling and post-change regrow timing. This is the resolution the protocol pre-declared as sufficient.

## T5. Applicability scope statement

**Demonstrated** (phase 1: α=0 base planner held constant, window-control-only replacement, 8×H20 EP=8 DeepEP-normal DeepGEMM-FP8 Qwen3-235B-A22B-FP8, real captured workloads):
1. The v2 controller runs as specified across 3 conditions × 24 controller starts: single W authority, bounded response (grid {8,16,32,64}, dwell, confirmation, urgent cap), all-rank consistency (zero desync), per-change evidence chain (z, h_w, reason), deterministic under identical input.
2. Predefined real changes detected with 32–74-step latency; a non-event boundary correctly silent in all arms; monitoring cost ≈0.5–0.7 ms/window, inside service time.
3. M2 headroom gate measurably prevents W=64 over-extension and the 2× detection latency that accompanies it (M2off, W2).

**NOT demonstrated:** a net full-service-time win over fixed baselines at n=2 on any condition (W0: slower both starts vs FG; W1: ≈0 vs FG, FL comparison noise-confounded; W2: sign-flipping within spread). The root cause is measured, not speculative: **the fixed-W grid itself spans only 0.5–2.3%** at α=0 on these workloads — window-size headroom is too small for any controller to convert behavior quality into wall-clock at this n. V2's demonstrable value here is behavioral robustness (detection latency bounds, churn bounds, no over-extension); converting it into time requires regimes where staleness actually costs (larger grids, α>0 rebalancing interaction), which phase 1 explicitly held constant per directive §五.

## 6. Corrections to live-monitoring claims (honest record)

Three mid-run characterizations are corrected by the final step-aligned analysis; recorded here because live-monitoring output is not a deliverable:
1. ~~'M1p missed / was ~1.7× late on W0 boundary #1'~~ — **wrong.** The w31/584-step event is M1p's **B2** detection (latency 56). M1p detects B1 at w21/step 344 (latency 74, comparable to Full's 48/70). The error came from comparing window indices across arms with different steps-per-window.
2. ~~'M1p false-fires urgent in the stationary zone (W0 w11, w21)'~~ — **wrong.** w21 IS B1. w11 (step 192, z=127.9) corresponds to a real routing event Full also observed at the same steps (w18–20, z=37.9–62.9; Full took no W action only because it was already at the W=8 floor).
3. 'M2off detection latency ~2× at W2' — **confirmed** by step-aligned measurement (72 vs 36 steps).

## 7. Budget & hygiene

- 36/36 pre-registered starts used; 0 extra; no significance-driven additions; nothing repeated or dropped for results reasons (only the INFRA zero-data boot re-run, memo on file).
- Datasets frozen by sha256 (`step6_dataset_hashes.txt`), verified at every phase start. No boundary labels given to the controller; known only to analysis.
- Fixed baselines calibrated on the same grid as V2's W_init/bounds (directive §四). W0 restated as known-regression benchmark throughout, never as validation.
- All negative deltas preserved verbatim. **Local commits only; no push** (standing instruction).

*Generated 2026-10-16 from deliverable_F_analysis.json + raw bench.json/v2_rank0.jsonl; boundary steps computed as boundary_row/N_rows × per-run total measured steps.*