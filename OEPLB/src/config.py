from dataclasses import dataclass

@dataclass
class PBOEPLBConfig:
    enabled: bool = False
    threshold_ratio: float = 1.15
    max_swaps_per_layer: int = 3
    min_prefill_tokens: int = 1000
    cooldown_steps: int = 5
    max_total_swap_layers: int = 48
    always_record: bool = False
    log_every_boundary: bool = True
    sync_window: int = 8  # forward passes between rebalance-attempt checkpoints (v0.9)

    # --- Adaptive window (experimental, opt-in) ---
    # Static sync_window=64 was found to protect TTFT (fewer all_reduce/P2P-swap
    # rounds competing with DeepEP dispatch/combine for NVLink bandwidth) but reacts
    # ~2 windows slower to a domain switch than sync_window=32 (peak avg_ratio_before
    # 1.36-1.39 vs 1.27-1.28 on the fixeddata Prover->BookCorpus benchmark).
    # sync_window=32 alone trades that peak-ratio/throughput win for a consistent
    # +21% TTFT regression (more frequent swap traffic contending for NVLink even
    # during steady state, not just during the shift). Adaptive window tries to get
    # both: stay at sync_window (64) during steady state, temporarily shrink to
    # window_floor (32) ONLY while a shift is actively confirmed, grow back once
    # stable. Requires window_confirm_windows (2) CONSECUTIVE low/high-cos_sim
    # windows before switching either direction -- a single-window blip that
    # reverts on its own the next window does not trigger a change, so it can't
    # cause a swap decision to fire based on noise alone. Gated by env vars (not
    # CLI flags), same convention as decay experiments above.
    adaptive_window: bool = False
    window_floor: int = 32
    window_shift_cos_threshold: float = 0.85
    window_stable_cos_threshold: float = 0.95
    # Asymmetric confirmation: shrinking only changes check cadence (safe to act
    # fast), growing back trades a *little* extra overhead for safety (worth
    # being conservative). See controller.py's _decide_and_begin_swap for why.
    window_shift_confirm_windows: int = 1
    window_stable_confirm_windows: int = 2

    @classmethod
    def from_server_args(cls, server_args):
        import os
        return cls(
            enabled=server_args.enable_pb_oeplb,
            threshold_ratio=getattr(server_args, 'pb_oeplb_threshold_ratio', 1.15),
            max_swaps_per_layer=getattr(server_args, 'pb_oeplb_max_swaps_per_layer', 3),
            min_prefill_tokens=getattr(server_args, 'pb_oeplb_min_prefill_tokens', 1000),
            cooldown_steps=getattr(server_args, 'pb_oeplb_cooldown_steps', 5),
            max_total_swap_layers=getattr(server_args, 'pb_oeplb_max_total_swap_layers', 48),
            always_record=getattr(server_args, 'pb_oeplb_always_record', False),
            log_every_boundary=True,
            sync_window=getattr(server_args, 'pb_oeplb_sync_window', 8),
            adaptive_window=os.environ.get('OEPLB_ADAPTIVE_WINDOW', '0') == '1',
            window_floor=int(os.environ.get('OEPLB_WINDOW_FLOOR', 32)),
            window_shift_cos_threshold=float(os.environ.get('OEPLB_WINDOW_SHIFT_COS', 0.85)),
            window_stable_cos_threshold=float(os.environ.get('OEPLB_WINDOW_STABLE_COS', 0.95)),
            window_shift_confirm_windows=int(os.environ.get('OEPLB_WINDOW_SHIFT_CONFIRM', 1)),
            window_stable_confirm_windows=int(os.environ.get('OEPLB_WINDOW_STABLE_CONFIRM', 2)),
        )
