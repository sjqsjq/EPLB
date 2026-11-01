# Stats-path trace: v2.1 detection + planner issuance (DSV4, post-layer-alignment-fix)

Date: 2026-10-22. Sources: frozen `pb_oeplb` v2.1 code (controller.py, window_v2.py,
window_v2p1.py, v2p1_signals.py, config.py under /opt/dsv4_env/.../sglang/srt/managers/pb_oeplb/),
boot env (scripts/boot_v4_pb_p.sh), measured jsonl (bench_p_fix_e4_r1/v2p1_rank0.jsonl,
bench_p_fix_e2_r2, fp_fix_art). No frozen file was modified for this analysis.

## 1. Exact formulas (what tensors, what normalization, what N_MIN gates)

Per window boundary (controller.py `_window_control_v2p1`, phase 1-2, :1836-1876):

- `det` = [43, 32] PHYSICAL-slot assignment counts accumulated per rank by
  `record_next_layer` (scatter_add_, :375-445), all-reduced SUM every boundary (:1840).
- `n_fresh` = det.sum() (assignments; includes carried windows when carry_fresh merged).
- `logical` = logical_load(det, cached p2l): scatter det -> LOGICAL expert space via the
  CURRENT physical_to_logical map => [43, 256] routed experts. Shared experts are
  stripped AT THE SOURCE: recorder_topk_ids = routed_cols after physical remap
  (topk.py `_post_process_topk_ids` :2366-2372); hash layers likewise via
  hash_topk.py:271/:294 (topk_ids_logical_to_physical). No logical-id-256 contamination
  anywhere in det/logical/p/s_change. CONFIRMED.
- `p, valid` = row_distribution(logical): per-layer L1 normalization; valid = rowsum>0
  (v2p1_signals.py:42-49).
- `ref` = EWMA of p, beta=0.5, updated ONLY on committed non-uncertain, non-init events
  (controller.py:1992-2000); `init` commit REPLACES ref with p.clone().
- `s_change_logical` = mean over layers with valid & ||p_l||>0 & ||ref_l||>0 of
  (1 - cos(p_l, ref_l)); returns (None, None) if no such layer (v2p1_signals.py:52-64).
- `h_w_assignments` = sum_l max(0, max_r(rank_load_l,r) - max(total_l/EP, max_expert_l)),
  PHYSICAL space, rank = slot//num_local (v2p1_signals.py:67-79; formula identical to
  frozen v2).
- `h_rate` = h_w / n_fresh (0.0 if n_fresh=0) — dimensionless work fraction (F2 scale).
- `z` = s_change / (kappa / n_layer), n_layer = max(n_fresh/43, 1),
  kappa = 3875.1768798828125 (v2p1_signals.py:87-96). z=None <=> s_change=None.
- `N_MIN = 12320768` gates ONLY the state machine (window_v2.py:141:
  n_fresh < k["n_min"] -> uncertain_carry / uncertain_hold / forced_shrink_uncertain;
  z is computed but NOT consulted on that path). Under SP sampling the effective gate is
  n_min_eff = N_MIN // S (controller.py:1869-1875). N_MIN NEVER gates swap issuance.

## 2. E4: why 61/75 windows are `init` with z=None, yet ~10 swaps fired

Measured (bench_p_fix_e4_r1, 75 windows, S=1 throughout, steps_actual=16/window):

- **53 windows are EMPTY** (n_fresh=0, sp_recorded_steps=0). Cause: prefill-focused
  recording by design — the pre-forward SP gate records only extend/prefill forwards
  (`is_prefill or record_all_phase`, controller.py on_forward_pass_start;
  OEPLB_RECORD_ALL_PHASE=0 in this campaign; comment: "OEPLB's differentiator vs EPLB's
  uniform prefill+decode recording"). E4 is decode-heavy: only 183 of 1200 forwards
  (15.25%) are prefill batches >= min_record_tokens=32.
- Empty window -> no valid rows -> change_score returns (None,None) -> z=None ->
  state machine returns EV_INIT ("no_reference_yet", window_v2.py:136-138).
- **ref-zeroing cascade (8 more init windows)**: committing an EMPTY init window sets
  `_v2p1_ref = p.clone()` where p is the ZERO matrix (controller.py:1992-1993). The next
  DATA window then has nr=||ref_l||=0 on every layer -> ok.any()=False -> s_change=None
  -> z=None -> EV_INIT again -> ref re-seeded from that window. Victims: w1 (first),
  w10, w20, w29, w39, w48, w57, w67 — ALL data-bearing (n_fresh 6.8M-22.5M,
  h_rate 0.012-0.087). Total: 61 init = 53 empty + 8 cascade.
- Consequence: in E4 the detection EWMA never accumulates memory; z exists on only 14
  windows (med 1.23, max 12.09 < z_lo=22.64 -> never alerts -> W pinned at 16).
  Working as designed: drift is oscillatory, not a regime shift.

**Swap issuance is a SEPARATE path that never reads z/h_rate/N_MIN.** Order per boundary
(controller.py:1211-1370; comment :1211-1215 "window control ... independent of whether
this window swaps"):
`global_tokens >= min_prefill_tokens(256)` -> swap budget -> [bias_gate / dead_zone /
early-exit: ALL DISABLED this campaign (OEPLB_BIAS_GATE=0.0, OEPLB_BIAS_CORRECT=False,
OEPLB_DEAD_ZONE_RATIO=0.0)] -> `try_build_swap_plan(global_load, threshold_ratio=1.02,
max_total_ops=300, max_swaps_per_layer=64)` -> `len(plan) >= min_swap_ops(8)` -> defer
gate (len<16 && improvement<0.003) -> `async_executor.begin(plan)` (:1367) ->
`_sp_swaps_total += 1`. The planner consumes `global_load` = all-reduced `self.load`
(physical counts of the window's recorded batches).

**E4's 10 swaps**: issued at boundaries w1, w12, w20, w21, w39, w57, w58, w59, w67, w68.
EVERY one of those windows is data-bearing (n_fresh 1.95M-23.3M, h_rate 0.009-0.087).
Five carry the `init` LABEL (w1, w20, w39, w57, w67) purely from the cascade — their
planner stats were real. ZERO swaps were issued on the 53 empty windows
(global_tokens=0 < 256 -> early return).

**jsonl attribution lag**: `sp_swaps_started` is consumed at the NEXT boundary
(controller.py:1898-1907: "a plan started at the previous joint boundary — after that
boundary's window control — is visible to this proposal"). A plan issued at boundary N
is logged on row N+1. The two swap rows sitting on EMPTY init rows (w22, w69) are
issuances at w21 (h_rate 0.0203) and w68 (h_rate 0.0278). NOT stale plans, NOT
thin-stat commits — begin() is synchronous at issuance; only the LOG lags one window.

## 3. Governing knobs + env-level suppression (no frozen-code touch)

- Swap path knobs (all env, boot_v4_pb_p.sh already exports them):
  PB_OEPLB_THRESHOLD_RATIO(1.02), PB_OEPLB_MIN_PREFILL_TOKENS(256),
  PB_OEPLB_MIN_SWAP_OPS(8), PB_OEPLB_MAX_TOTAL_OPS(300),
  PB_OEPLB_MAX_SWAPS_PER_LAYER(64), PB_OEPLB_MAX_TOTAL_SWAP_LAYERS(94).
- Thin-stat protections EXIST but are DISABLED in this campaign (config.py:298-301,
  env-settable): OEPLB_BIAS_CORRECT=False, OEPLB_BIAS_GATE=0.0,
  OEPLB_DEAD_ZONE_RATIO=0.0. Enabled, they block swaps when the token-dependent
  sampling bias is large relative to the ratio margin (controller.py:1231-1258) —
  exactly the thin-stat case.
- Total suppression: PB_OEPLB_MAX_TOTAL_SWAP_LAYERS=0 or MAX_SWAPS_PER_LAYER=0
  (diag-noswap style; kills ALL swaps).
- OEPLB_V21_* (N_MIN/z/thetas) CANNOT suppress swaps — by design no env couples the
  detection state machine to swap issuance.
- **Verdict: suppression NOT warranted for E4.** All 10 swaps fired on real imbalance
  (h_rate 0.009-0.087 vs theta_r_act 0.0114; avg_ratio>1.02), post-fix E4 converged
  (tps 2699.6 vs broken 2624.5; I-arm 3215.1). The "init-window swaps" appearance =
  cascade labeling + one-window logging lag.

## 4. s_change_logical semantics: E4 0.0586 (70x E2's 0.0008) — genuine or artifact?

**GENUINE decode-regime dynamics, cascade-amplified; NOT a residual attribution
artifact.**
- Attribution is verified fixed: record-hook fires exactly 40x/forward with the TRUE
  layer_id (host patch sets _layer_counter = layer_id - _layer_offset at the call site);
  prof ratio 40.0 exact in E4 r1.
- Sampling-noise floor: multinomial E[1-cos] ~ (1/2)(k-1)/n_layer; E4 data windows have
  n_layer = n_fresh/40 ~ 49k-582k assignments/layer -> noise 0.0004-0.0026. Observed
  s_change range 0.0016-0.1609 (med 0.0586) = 20-80x the floor -> real between-window
  distribution shift.
- Why 70x E2: E2 is prefill-heavy with a stable content mix (every forward recorded,
  EWMA ref smooth, consecutive windows near-identical -> 0.0006-0.002). E4 records only
  sporadic prefill bursts; concurrent-request composition genuinely differs between
  bursts. Additionally, post-cascade windows compare against a SINGLE-window ref (no
  EWMA smoothing), inflating amplitude vs a smooth reference.
- z-normalization does its job: max z 12.09 < z_lo 22.64 -> machine stays QUIET ->
  W pinned. The drift is real but oscillatory, not a sustained regime shift.

## 5. w1 init-window h_rate ~ 0.094 (E2 warmup, fp_fix_test / fixcheck)

- w1 n_fresh ~ 129k -> ~12.6 assignments/expert -> ~400/rank. Structural floor: even
  uniform traffic gives max-of-8 rank loads ~ mean + 1.5*sigma ~ +30/layer ->
  h_w ~ 1200 -> h_rate ~ 0.0093 (pure Poisson). Observed 0.094 = 10x floor -> warmup
  traffic CONCENTRATION (synthetic repeated prompts hit hot experts; identity layout
  maps expert blocks contiguously per rank, so hot ids concentrate rank loads).
- h_rate is a RATE: its noise term scales ~1/sqrt(counts) and does not vanish with
  more windows — this is exactly why N_MIN exists (gating z/events on >=12.3M
  assignments) and why the w1-issued first swap (logged on w2) is warmup-driven yet
  functionally harmless (fp_fix_test: all_tokens_same=True across all pairs incl.
  cross-arm and post-traffic).
