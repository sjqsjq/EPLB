# ADW v2 code change & deployment note (Step 3 complete)

Date: 2026-10-16. Status: deployed to site-packages; GPU verification (Step 4) next.
No push performed (standing rule: no push without separate instruction).

## 1. FACT-10 correction (code authority, discovered during this deployment)

Prior claim: "installed controller.py == OEPLB/src/controller.py (git-committed)".
**Corrected**: the installed controller.py contained 23 additional lines NOT in git
HEAD / src: (a) a HOOKDIAG instrumentation block (counters in record_next_layer /
on_forward_pass_end + one INFO log line every 200 fpe calls, added Oct 15 17:23
during hook-coverage debugging); (b) a C3 comment rewording (3→4 lines, identical
functional clamp). Verified by full diff: **diagnostic-only, zero control-flow or
numerical change** — all early-return conditions and arithmetic untouched. All
other 7 installed pb_oeplb files are byte-identical to git HEAD. The five-arm
campaign results therefore remain valid as records of the backed-up code; the
exact campaign code is preserved byte-for-byte in
`.../pb_oeplb/legacy_eav1_locked_20261016/` with sha256 manifest.

## 2. What changed (git-tracked src = deployment base)

- `OEPLB/src/window_v2.py` (NEW): pure-python single-authority window state
  machine `WindowControllerV2` (grid {8,16,32,64}; M1 uncertainty handling via
  n_fresh/N_min/carry<=2/U_max; M2 headroom-gated grow/shrink via theta_h_low /
  theta_h_act; M3 confirm counts, dwell, urgent path with episode cap, bounded
  response, deterministic desync safe mode freezing W at w_init).
- `OEPLB/src/config.py`: +18 fields (window_controller_version=1 default,
  v2_* knobs, v2_beta, v2_log_path) + OEPLB_* env wiring in from_server_args.
- `OEPLB/src/controller.py` (+212/-4 lines, 7 anchored edits):
  P1 __init__ v2 setup (version gate; _init_window_control_v2 when version==2);
  P2 `_last_window_steps` capture at window trigger (log field only);
  P3 legacy S2 gated `self._wctrl_version == 1 and ...`;
  P4 legacy S1 gated `self._wctrl_version == 1 and ...`;
  P5 `_window_control_v2(global_load, global_tokens)` call placed after legacy
     S1 site, BEFORE token/threshold gates (window control independent of swap
     decision); only under version==2;
  P6 end-of-window zeroing: version==2 && carry -> skip zeroing (bounded
     evidence merge); else legacy lines unchanged; `_prefill_batch_counter=0`
     always;
  P7 new methods `_init_window_control_v2` (startup invariants: decay==0.0,
     adaptive_window/adaptive_decay/calibrate all False, kappa>0 else
     RuntimeError — refuses to start with placeholder) and `_window_control_v2`
     (vectorized H_w = sum_l max(0, max_rank_l − max(total_l/EP, max_expert_l));
     detection-only EWMA ref (beta), frozen while uncertain; z = s_change /
     (kappa/n_layer_fresh); sha256 decision hash, one 2-elem all_reduce per 16
     windows, desync -> on_desync() freeze at w_init on all ranks; rank0 JSONL
     per spec §5 incl. steps_actual, N_fresh, S_change, z, H_w (assignments +
     labeled empirical µs conversion, log-only), r_legacy, cos_legacy,
     c_window_ms, full controller record; never raises — errors HOLD W).

**v1 byte-identity**: with `OEPLB_WINDOW_CONTROLLER` unset (default 1), every
legacy path executes exactly as before; all v2 code sits behind
`_wctrl_version == 2` gates. Note the deployed v1 no longer carries the
untracked HOOKDIAG counters (behavior-neutral removal; campaign code preserved
in the backup dir). V3 log-signature check will confirm on GPU.

## 3. Hashes

Backup (installed pre-deploy, = campaign code): see
`legacy_eav1_locked_20261016/MANIFEST_sha256.txt`
(controller.py 9443e6a5..., config.py afb91a5f...).

Deployed (installed post-deploy == OEPLB/src):
- controller.py 789a694e2e791c9dc7e67494e18d555303fde63d88c69578d5cf82401af3be37
- config.py     180fa41eae6e9394186afceb2ceadf9468ebbacda69591fb9b9897a2e6c2680c
- window_v2.py  810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74 

## 4. Verification status

- Unit tests (pure python, installed copy): ALL_TESTS_PASS incl. new T7b
  (desync after W moved -> snaps to deterministic w_init, not current W).
- Import smoke: default version=1; env wiring version=2/kappa/z_hi OK;
  controller module imports with v2 methods; py_compile OK (src + installed).
- Pending (Step 4, needs GPUs): V2 8-rank decision-hash consistency short run;
  V3 v1 log-signature identity vs locked campaign behavior; V4 output sanity;
  V5 switch invariants (arm switch without server restart is NOT supported —
  version is read at controller construction; each arm gets its own boot, same
  as the five-arm campaign).
