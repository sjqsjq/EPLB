# ADW v2 Step-4 verification record (COMPLETE)

Date: 2026-10-16. All smokes on 8xH20 node, real Qwen3-235B-A22B-FP8 service,
real calibration.jsonl requests (dev material; no holdout touched).

## V1 logic unit tests
test_window_v2_logic.py: ALL_TESTS_PASS (T0-T10 + T7b desync-after-move ->
deterministic w_init snap-back).

## V2 multi-rank consistency (smoke C run 2, 4000 requests, w_init=16)
- Decision-hash all_reduce fired at windows 16 and 32: PASSED silently on all
  8 ranks (desync=False everywhere); 35 windows logged.
- W movement exercised: urgent_shrink 16->8 (z=21.0 >= z_urgent), 9x
  grow_blocked_headroom (M2 gate), confirmed_grow 8->16 in low-z tail;
  dwell set after each move; steps_actual tracks W ({8,16}).
- c_window_ms median 0.56, max 26.65 (first-window lazy init). Legacy
  94-layer ratio loop (~14ms/window allreduce-sync cost) intentionally kept
  in BOTH arms (F12 cost symmetry).
- bench 4000/4000 ok, 0 errors, 111.2s. Zero legacy [PB-OEPLB-ADW]/WINDOW/
  RESET lines (single authority verified).

## PRESERVED FAILURE: smoke C run 1 (false desync)
All 8 ranks reported RANK DESYNC at w16 -> froze W at w_init. Root cause:
hash-sum check compared against hardcoded 16*h (16-window check interval
confused with rank count; world=8). NOT a real divergence. Safe-mode itself
verified correct under this event: deterministic freeze, no hang, no
collective divergence, service completed 4000/4000. Fix: world-size from
torch.distributed.get_world_size(); desync log now includes own_h/sum/world.
Artifacts: smoke_step4/smokeC_run1_false_desync_RECORD/ (never overwritten).
Fixed controller hash: 2a4dedb22d4bcaba203233c0f9a5ae6c211370f524addb7d166669406690878b

## V3 v1 byte-identity (smoke A, 1200 requests, W=16 alpha=0.0 ADW=1, OEPLB_WINDOW_CONTROLLER unset)
- Default version=1: legacy window mutations active (40 [PB-OEPLB-ADW]/
  [PB-OEPLB-WINDOW] lines), ZERO [PB-OEPLB-ADW2] lines.
- bench 1200/1200 ok, 0 errors, 44.3s. v1 path intact.
- Note: deployed v1 no longer carries the untracked HOOKDIAG counters found
  in the campaign-installed copy (behavior-neutral; see code_change_note §1;
  campaign code preserved byte-exact in legacy_eav1_locked_20261016/).

## V4 output sanity (smoke B, 1200 requests, w_init=8)
- JSONL schema complete: window_id, steps_actual, n_fresh_assignments,
  s_change, z, h_w (+labeled empirical us conversion, log-only), r_legacy,
  cos_legacy, c_window_ms, prof_allreduce_cum_ms, W before/after, event,
  reasons, alert_k, stable_s, dwell, dwell_dir, uncertain_run,
  episode_shrinks, carry_fresh, desync.
- z with placeholder kappa=1.0 huge (7e3-8e4) as expected -> confirms kappa
  MUST be calibration-fitted (startup guard already refuses kappa<=0).
- s_change series within-domain 0.025-0.147; n_fresh ~49.3M/window at
  saturation (= 94 layers x 8 topk x 65536 tokens; constant under saturated
  chunked prefill - legitimate, not a bug).

## V5 switch invariants
- window_controller_version read at controller construction (per-boot); no
  mid-run switching. Each arm boots its own server exactly like the five-arm
  campaign. Config signatures verified in server logs
  (sync_window/decay_factor/adaptive_window/adaptive_decay).
- History-coordinate remapping (_try_finish_pending_swap), planner,
  rebalancer, async executor: ZERO diff lines (git diff verified).
