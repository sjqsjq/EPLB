import os
from dataclasses import dataclass


def _f(server_args, attr: str, env: str, default: float) -> float:
    """CLI flag if this SGLang registers it, else env var, else default."""
    v = getattr(server_args, attr, None)
    if v is not None and v != default:
        return float(v)
    return float(os.environ.get(env, default))


def _b(server_args, attr: str, env: str, default: bool) -> bool:
    v = getattr(server_args, attr, None)
    if v:
        return True
    return os.environ.get(env, '').lower() in ('1', 'true', 'yes', 'on')


@dataclass
class PBOEPLBConfig:
    enabled: bool = False

    # --- Core parameters ---
    threshold_ratio: float = 1.02
    max_swaps_per_layer: int = 64
    min_prefill_tokens: int = 256
    max_total_swap_layers: int = 94
    max_total_ops: int = 300
    min_swap_ops: int = 8
    always_record: bool = False
    sync_window: int = 8
    decay_factor: float = 0.5
    min_record_tokens: int = 32

    # --- Dead zone / sampling-bias aware decisions (opt-in) -------------------
    # Measured on this stack (paper appendix G): wall time is FLAT in the
    # imbalance ratio for r <= r_k, so pushing r below r_k buys nothing while
    # still paying the P2P blocking cost. r_k is 1.099 at EP=8, 1.032 at EP=4
    # and 1.093 for a different model at EP=8, i.e. it moves with the parallel
    # config but not with the model, and must be measured per config -- hence a
    # knob rather than a formula. 0.0 keeps the old behaviour.
    dead_zone_ratio: float = 0.0

    # The observed per-window ratio carries a positive sampling bias of
    # c/sqrt(N) where N is tokens-per-layer in the window: with few tokens the
    # per-GPU counts are noisy and max/mean is inflated even under perfectly
    # balanced routing. Empirically c is a function of ep_size only, stable to
    # <8% across four datasets (5.10 at ep=8, 2.62 at ep=4), and close to the
    # order-statistics estimate sqrt(2 G ln G) (5.77 / 3.33). Left uncorrected
    # this makes the balancer chase noise on heterogeneous workloads -- a
    # ShareGPT window with 128 tokens/layer reported r=1.355 against a true
    # ~1.10. 0 for bias_coeff means "derive it from ep_size" as 0.65*ep_size,
    # which is the window-level calibration (5.17 at ep=8, 2.66 at ep=4, each
    # within ~10% across four datasets).
    bias_correct: bool = False
    bias_coeff: float = 0.0
    # Correcting the THRESHOLD only protects the trigger. The swap plan is
    # built from the same noisy per-expert counts, so a window that clears the
    # corrected threshold on real imbalance can still produce a plan that
    # chases phantom skew. Require the estimate's noise to be small relative
    # to the margin we are acting on:  bias <= bias_gate * (r_corrected - thr).
    # This must be a RATIO, not an absolute bound: the bias scales with
    # ep_size (median 0.032 at ep=8 vs 0.018 at ep=4 on these workloads), so
    # an absolute cap of 0.02 blocks every window on 8 GPUs -- including the
    # ones with a real 3% of headroom. 0.5 means 'margin at least 2x noise'.
    # 0 disables.
    bias_gate: float = 0.0

    # Cap cumulative swap wall time at this fraction of elapsed serving time.
    # The headroom a balancer can win is beta*(r-r_k), a few percent on most
    # configs, so spending more than that on moving weights is a guaranteed net
    # loss. Two measured configs sat at swap/headroom of 1.26 and 1.09 and both
    # realised ~0 of their ceiling. 0.0 disables the guard.
    swap_budget_frac: float = 0.0

    # --- Adaptive window ---
    adaptive_window: bool = False
    # Adaptive decay: on a detected changepoint, zero the decay for one step to
    # clear stale old-domain history (§3.5). Independent of adaptive_window
    # since the 2026-10-15 E-A switch decoupling (audit finding F2/F3, change
    # C2): RESET domain-shift zeroing and the decay->0 step are gated by this
    # flag alone, so D works with W off and D-off truly disables all live
    # history zeroing.
    adaptive_decay: bool = False
    window_floor: int = 32
    window_shift_cos_threshold: float = 0.85
    window_stable_cos_threshold: float = 0.95
    window_shift_confirm_windows: int = 2
    window_stable_confirm_windows: int = 2

    # --- P11 control planner (E-P arm, 2026-10-15): replaces the pair-selection
    # rule inside the swap planner with best-improvement enumeration within
    # hot/cold rank pairs (rebalancer.try_build_swap_plan_best_improvement,
    # rule name "best_improvement_rank_pair"). Skeleton, acceptance margin,
    # budgets and executor are unchanged. Env-var fallback like adaptive_decay
    # (no server_args patch required): OEPLB_PLANNER_BEST_IMPROVEMENT=1.
    planner_best_improvement: bool = False

    # --- ADW v2 single-authority window controller (2026-10-16) -------------
    # version 1 = legacy behaviour, byte-identical to the five-arm-locked code.
    # version 2 = window_v2.WindowControllerV2 is the ONLY writer of the
    # effective sync window; ALL legacy mutation sites (ratio-jump shrink,
    # converged/volatile grow, cos_sim shift/stable grow) are short-circuited.
    # Env-driven like adaptive_decay (no server_args patch): OEPLB_WINDOW_CONTROLLER=2.
    # v2 phase-1 invariants enforced in controller.__init__: decay_factor==0.0
    # and adaptive_window/adaptive_decay/calibrate_adaptive_sensitivity OFF.
    # All v2_* knobs are calibration-frozen; placeholders refuse to start.
    window_controller_version: int = 1
    v2_kappa: float = 0.0
    v2_z_lo: float = 0.0
    v2_z_hi: float = 0.0
    v2_z_urgent: float = 0.0
    v2_m_shrink: int = 2
    v2_m_grow: int = 3
    v2_n_min: int = 0
    v2_u_max: int = 4
    v2_theta_h_low: float = 0.0
    v2_theta_h_act: float = 0.0
    v2_dwell_shrink: int = 2
    v2_dwell_grow: int = 2
    v2_w_uncertain_target: int = 16
    v2_urgent_episode_cap: int = 3
    v2_beta: float = 0.5
    v2_log_path: str = ""

    # --- Adaptive window sensitivity calibration (opt-in; gated on this flag
    # alone since the 2026-10-15 E-A decoupling, change C7 -- previously also
    # required adaptive_window=True). Measures the prefill:decode forward-pass ratio
    # during an initial calibration window and uses it to pick how
    # aggressively adaptive_window should react (see controller.py's
    # _apply_sensitivity_tier) -- NOT which sync_window value to use (empirically
    # validated: window choice itself doesn't correlate with this ratio, only
    # the overall achievable gain magnitude does). ---
    calibrate_adaptive_sensitivity: bool = False
    calibration_forwards: int = 256

    # --- Tuning parameters (env-var only, rarely changed) ---
    warmup_forwards: int = 10
    sample_interval: int = 1

    @classmethod
    def from_server_args(cls, server_args):
        return cls(
            enabled=server_args.enable_pb_oeplb,
            # Core (CLI → env-var fallback)
            threshold_ratio=getattr(server_args, 'pb_oeplb_threshold_ratio', 1.02),
            max_swaps_per_layer=getattr(server_args, 'pb_oeplb_max_swaps_per_layer', 64),
            min_prefill_tokens=getattr(server_args, 'pb_oeplb_min_prefill_tokens', 256),
            max_total_swap_layers=getattr(server_args, 'pb_oeplb_max_total_swap_layers', 94),
            max_total_ops=getattr(server_args, 'pb_oeplb_max_total_ops', 300),
            min_swap_ops=getattr(server_args, 'pb_oeplb_min_swap_ops', 8),
            always_record=getattr(server_args, 'pb_oeplb_always_record', False),
            sync_window=getattr(server_args, 'pb_oeplb_sync_window', 8),
            decay_factor=getattr(server_args, 'pb_oeplb_decay_factor', 0.5),
            min_record_tokens=getattr(server_args, 'pb_oeplb_min_record_tokens', 32),
            # Dead zone / sampling-bias aware decisions. CLI flag when the
            # running SGLang registers it, env var otherwise -- the vendored
            # SGLang tree in this repo is a different release from the one that
            # may be installed, so relying on an arg-parser patch would make
            # these knobs unreachable on one of them.
            dead_zone_ratio=_f(server_args, 'pb_oeplb_dead_zone_ratio', 'OEPLB_DEAD_ZONE_RATIO', 0.0),
            bias_correct=_b(server_args, 'pb_oeplb_bias_correct', 'OEPLB_BIAS_CORRECT', False),
            bias_coeff=_f(server_args, 'pb_oeplb_bias_coeff', 'OEPLB_BIAS_COEFF', 0.0),
            bias_gate=_f(server_args, 'pb_oeplb_bias_gate', 'OEPLB_BIAS_GATE', 0.0),
            swap_budget_frac=_f(server_args, 'pb_oeplb_swap_budget_frac', 'OEPLB_SWAP_BUDGET_FRAC', 0.0),
            # Adaptive window (CLI)
            adaptive_window=getattr(server_args, 'pb_oeplb_adaptive_window', False),
            adaptive_decay=_b(server_args, 'pb_oeplb_adaptive_decay', 'OEPLB_ADAPTIVE_DECAY', False),
            planner_best_improvement=_b(server_args, 'pb_oeplb_planner_best_improvement', 'OEPLB_PLANNER_BEST_IMPROVEMENT', False),
            # ADW v2 single-authority window controller (env-only knobs;
            # see window_v2.py and results/adw_v2/adw_v2_mechanism_spec.md)
            window_controller_version=int(os.environ.get('OEPLB_WINDOW_CONTROLLER', '1')),
            v2_kappa=float(os.environ.get('OEPLB_V2_KAPPA', '0')),
            v2_z_lo=float(os.environ.get('OEPLB_V2_Z_LO', '0')),
            v2_z_hi=float(os.environ.get('OEPLB_V2_Z_HI', '0')),
            v2_z_urgent=float(os.environ.get('OEPLB_V2_Z_URGENT', '0')),
            v2_m_shrink=int(os.environ.get('OEPLB_V2_M_SHRINK', '2')),
            v2_m_grow=int(os.environ.get('OEPLB_V2_M_GROW', '3')),
            v2_n_min=int(os.environ.get('OEPLB_V2_N_MIN', '0')),
            v2_u_max=int(os.environ.get('OEPLB_V2_U_MAX', '4')),
            v2_theta_h_low=float(os.environ.get('OEPLB_V2_THETA_H_LOW', '0')),
            v2_theta_h_act=float(os.environ.get('OEPLB_V2_THETA_H_ACT', '0')),
            v2_dwell_shrink=int(os.environ.get('OEPLB_V2_DWELL_SHRINK', '2')),
            v2_dwell_grow=int(os.environ.get('OEPLB_V2_DWELL_GROW', '2')),
            v2_w_uncertain_target=int(os.environ.get('OEPLB_V2_W_UNCERTAIN_TARGET', '16')),
            v2_urgent_episode_cap=int(os.environ.get('OEPLB_V2_URGENT_EPISODE_CAP', '3')),
            v2_beta=float(os.environ.get('OEPLB_V2_BETA', '0.5')),
            v2_log_path=os.environ.get('OEPLB_V2_LOG', ''),
            window_floor=getattr(server_args, 'pb_oeplb_window_floor', 32),
            window_shift_cos_threshold=getattr(server_args, 'pb_oeplb_window_shift_cos', 0.85),
            window_stable_cos_threshold=getattr(server_args, 'pb_oeplb_window_stable_cos', 0.95),
            window_shift_confirm_windows=getattr(server_args, 'pb_oeplb_window_shift_confirm', 1),
            window_stable_confirm_windows=getattr(server_args, 'pb_oeplb_window_stable_confirm', 2),
            # Adaptive window sensitivity calibration (CLI)
            calibrate_adaptive_sensitivity=getattr(server_args, 'pb_oeplb_calibrate_adaptive_sensitivity', False),
            calibration_forwards=getattr(server_args, 'pb_oeplb_calibration_forwards', 256),
            # Tuning (env-var only)
            warmup_forwards=int(os.environ.get('OEPLB_WARMUP_FORWARDS', 10)),
            sample_interval=int(os.environ.get('OEPLB_SAMPLE_INTERVAL', 1)),
        )
