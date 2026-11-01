# E12 — DSV4 negative results: consolidated KEY NUMBERS
Recovered 2026-10-25 (see README_provenance.md for source & method).
Verdict of the whole campaign: ALL CELLS NEGATIVE — no configuration of the
ported adaptive controller (P arm) ever beat the static baseline (I arm) on
DeepSeek-V4-Flash; best post-fix overhead was still -3.8% (e2) / -7.8% (e3) /
-14.1% (e4). Function preservation gate: PASS (overhead is real, not a bug).

## 1. 0.5.22-track (new sglang tree) full-campaign medians — P_fixab vs I vs P_fix vs P_broken
Source: RECOVERED__fixab_comparison.md (git blob 2a8e90f4c46300a25caf25a5ac741f0d2d74c635,
sha256 65b01d7644918a5f9244e4073b7d2bc4c11b76e75a0a909efe712b065076cd4e; committed
2026-10-22T18:05Z). VERBATIM:

| stream | arm | rounds | tps | wall_s | errors | vs I | per-round tps |
|---|---|---|---|---|---|---|---|
| e2 | I | 3 | 42.5 | 96.0 | 0 | - | [42.5, 42.8, 42.2] |
| e2 | P_broken | 2 | 34.5 | 118.3 | 0 | -18.8% | [34.9, 34.1] |
| e2 | P_fix | 2 | 39.8 | 102.6 | 0 | -6.4% | [39.2, 40.4] |
| e2 | P_fixab | 1 | 40.9 | 99.9 | 0 | -3.8% | [40.9] |
| e3 | I | 2 | 44.9 | 182.4 | 0 | - | [44.5, 45.3] |
| e3 | P_broken | 2 | 37.2 | 220.6 | 0 | -17.1% | [38.1, 36.3] |
| e3 | P_fix | 2 | 39.7 | 206.8 | 0 | -11.7% | [38.6, 40.7] |
| e3 | P_fixab | 1 | 41.4 | 198.0 | 0 | -7.8% | [41.4] |
| e4 | I | 2 | 3215.1 | 81.5 | 0 | - | [3227.1, 3203.0] |
| e4 | P_broken | 2 | 2624.5 | 100.0 | 0 | -18.4% | [2696.6, 2552.4] |
| e4 | P_fix | 2 | 2755.3 | 95.2 | 0 | -14.3% | [2699.6, 2811.1] |
| e4 | P_fixab | 1 | 2763.1 | 94.9 | 0 | -14.1% | [2763.1] |

Cross-check: every value matches the expected values quoted in the E12 tasking
brief (P_fixab 40.9/42.5 -3.8%, 41.4/44.9 -7.8%, 2763.1/3215.1 -14.1%;
P_fix -6.4/-11.7/-14.3%; P_broken -18.8/-17.1/-18.4%). ✓

Patch effect (P_fixab vs P_fix): e2 39.8->40.9 (+2.8%); e3 39.7->41.4 (+4.4%);
e4 2755.3->2763.1 (+0.3%). Headroom analysis (ledger row 170): recoverable
imbalance ~1% tps vs ~8% fixed frozen-protocol floor => honest conclusion
P ~= I - 8% even with perfect patches; -14.1% on e4 remained.

## 2. Function-preservation gate (fixab E2 cell): PASS=True
Source: RECOVERED__fixab_comparison.md (gate section) + RECOVERED__bench_p_fixab_e2__fixab_gate.json
(git blob a8da5d754af36ea3f04d73d414d006bcdaedeb4e). VERBATIM:
tokens_identical_vs_I=True (pre AND post traffic, 3 probes each, token-id exact);
dpre_pre=0.2027 dpre_post=0.2043 (logprob drift, regrouping-noise scale);
Pfix_pre vs fixab_pre logprobs BITWISE identical (dpre=0.0) -> patches introduce
zero numerical change in steady state; post-traffic tokens == pre-traffic tokens
across 5 committed swaps.
=> The measured slowdown is genuine scheduling/dispatch overhead, NOT numerical
corruption; the negative result stands.

## 3. Broken-arm matrix (r1+r2 medians, pre-fix P vs I) incl. latency deltas
Source: RECOVERED__bench_matrix_final_table.md (git blob
7f9acc29f1b8c9cd6d16ec67cdc6d123d69d3fb5). VERBATIM deltas:
| stream | tps | total_time | ttft_p50 | ttft_p95 | ttft_p99 | tpot_mean |
|---|---|---|---|---|---|---|
| e2 | -19.1% | +23.6% | +23.5% | +39.6% | +47.1% | - |
| e3 | -17.1% | +20.9% | +17.5% | +37.1% | +47.7% | - |
| e4 | -18.4% | +22.6% | +27.7% | +65.9% | +102.8% | +15.8% |
Medians: e2 I=42.6/P=34.5 tps; e3 I=44.9/P=37.2; e4 I=3215.1/P=2624.5.
E2 wall decomposition (diag cells, ledger + OVERHEAD_ANALYSIS): static-dispatch
flag ~26%, instrumentation+windows ~20%, swap effects ~54% of the P-arm penalty.

## 4. Controller statistics path (why swaps never helped)
Source: RECOVERED__STATS_PATH_TRACE.md (git blob c31e69b15ce0e53806f8e044bcf59ddcccf7ec30).
- kappa = 3875.1768798828125 (v2p1_signals.py:87-96); N_MIN = 12320768 gates
  ONLY the state machine, NEVER swap issuance.
- E4: 61/75 init windows = 53 empty + 8 ref-zeroing cascade; ZERO swaps issued
  on the 53 empty windows (planner stats were real).
- E4 s_change median 0.0586 (70x E2's 0.0008) = genuine between-window drift.
- w1 init-window h_rate ~0.094 (E2 warmup) = 10x Poisson floor (0.0093).
- fixab cells controller behavior: e2 59 windows/5 swaps (z_med 0.095),
  e3 74/17 (z_med 2.642), e4 75/11 (z_med 0.663).

## 5. Old-tree (sglang 0.5.6.post2 reference clone) wall-ratio results — NOT RECOVERED
EFFECT_REPORT_old_sglang.md and bench_old_{i,p}_{e2,e3,e4}/ JSON summaries were
created AFTER the last git commit and existed only on disk + in the client-host
session transcript, which is unreachable from this sandbox. Expected values per
the tasking brief (UNVERIFIED, do not cite as evidence until transcript-verified):
  e2 4.3 -> 4.1 tps (-6.0%); e3 4.4 -> 4.1 tps (-6.7%); e4 229.6 -> 203.5 tps (-11.4%).
See UNRECOVERED__EFFECT_REPORT_old_sglang.md for recovery instructions.

## 6. Plan B (adapt DSV4 to old sglang 0.5.6): verdict INFEASIBLE
Source: RECOVERED__planb_assessment.md (git blob f43e514b6c7dc963cc8d3fc0b18495b912e5918d),
2026-10-22, read-only filesystem investigation. NOTE: the M0-M5 gate EXECUTION
rows in the ledger are post-commit and unrecovered (see LEDGER gap marker).

## 7. Environment bring-up (context for reproducibility of the negative result)
Source: RECOVERED__LEDGER_dsv4.tsv + RECOVERED__ENV_NOTES.md (Phases 1/1b/1c/1d/1e).
FIX-NVSHMEM3+4 (HCA_LIST bogus -> P2P/NVLink-only; nvshmem 3.4.5-cu13 ignores
all transport knobs), FIX-FP4DEQUANT (SGLANG_DSV4_FP4_DEQUANT=1, e2m1->e4m3,
+17GB/GPU), FIX-MEMFRAC 0.80->0.75. First identity EP8+DP8+DeepEP boot on H20:
2026-10-22T13:35:51Z PASS (boot 3m50s warm JIT).
