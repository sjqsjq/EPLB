import os
from dataclasses import dataclass


@dataclass
class PBOEPLBConfig:
    enabled: bool = False

    # --- Core parameters ---
    threshold_ratio: float = 1.02
    max_swaps_per_layer: int = 64
    min_prefill_tokens: int = 256
    max_total_swap_layers: int = 94
    max_total_ops: int = 250
    always_record: bool = False
    sync_window: int = 64
    decay_factor: float = 0.9
    min_record_tokens: int = 32

    # --- Adaptive window ---
    adaptive_window: bool = False
    window_floor: int = 32
    window_shift_cos_threshold: float = 0.85
    window_stable_cos_threshold: float = 0.95
    window_shift_confirm_windows: int = 1
    window_stable_confirm_windows: int = 2

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
            max_total_ops=getattr(server_args, 'pb_oeplb_max_total_ops', 250),
            always_record=getattr(server_args, 'pb_oeplb_always_record', False),
            sync_window=getattr(server_args, 'pb_oeplb_sync_window', 64),
            decay_factor=getattr(server_args, 'pb_oeplb_decay_factor', 0.9),
            min_record_tokens=getattr(server_args, 'pb_oeplb_min_record_tokens', 32),
            # Adaptive window (CLI)
            adaptive_window=getattr(server_args, 'pb_oeplb_adaptive_window', False),
            window_floor=getattr(server_args, 'pb_oeplb_window_floor', 32),
            window_shift_cos_threshold=getattr(server_args, 'pb_oeplb_window_shift_cos', 0.85),
            window_stable_cos_threshold=getattr(server_args, 'pb_oeplb_window_stable_cos', 0.95),
            window_shift_confirm_windows=getattr(server_args, 'pb_oeplb_window_shift_confirm', 1),
            window_stable_confirm_windows=getattr(server_args, 'pb_oeplb_window_stable_confirm', 2),
            # Tuning (env-var only)
            warmup_forwards=int(os.environ.get('OEPLB_WARMUP_FORWARDS', 10)),
            sample_interval=int(os.environ.get('OEPLB_SAMPLE_INTERVAL', 1)),
        )
