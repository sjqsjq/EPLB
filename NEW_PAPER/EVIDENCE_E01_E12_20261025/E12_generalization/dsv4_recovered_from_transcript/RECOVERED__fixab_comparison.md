# DSV4 adaptation: full campaign table with overhead patches A+B (2026-10-22)

Arms: I = baseline (default dispatch); P_broken = ported controller, pre-fix; P_fix = + layer-align host patch;
P_fixab = + patch A (dispatch-info cache, expert_location_dispatch.py) + patch B (recorder pre-gate, topk.py).
Frozen pb_oeplb package byte-identical throughout; patches are host-side in sglang tree.

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

## Patch effect (P_fixab vs P_fix, same frozen protocol)

- e2: 39.8 -> 40.9 tps (+2.8%), gap vs I -6.4% -> -3.8%
- e3: 39.7 -> 41.4 tps (+4.4%), gap vs I -11.7% -> -7.8%
- e4: 2755.3 -> 2763.1 tps (+0.3%), gap vs I -14.3% -> -14.1%

## Controller behavior (fixab cells)

| cell | windows | swaps | z_med | s_change_med |
|---|---|---|---|---|
| bench_p_fixab_e2 | 59 | 5 | 0.095 | 0.00110 |
| bench_p_fixab_e3_r1 | 74 | 17 | 2.642 | 0.02205 |
| bench_p_fixab_e4_r1 | 75 | 11 | 0.663 | 0.05100 |

## Function preservation gate (fixab E2 cell): PASS=True
tokens_identical_vs_I=True (pre AND post traffic, 3 probes each, token-id exact);
dpre_pre=0.2027 dpre_post=0.2043 (logprob drift, regrouping-noise scale);
Pfix_pre vs fixab_pre logprobs BITWISE identical (dpre=0.0) -> patches introduce zero numerical change in steady state;
post-traffic tokens == pre-traffic tokens across 5 committed swaps.
