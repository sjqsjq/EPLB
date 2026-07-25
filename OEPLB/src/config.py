import os
from dataclasses import dataclass


@dataclass
class PBOEPLBConfig:
    enabled: bool = False

    # --- Core parameters (exposed as CLI flags) ---
    threshold_ratio: float = 1.02
    max_swaps_per_layer: int = 64
    min_prefill_tokens: int = 256
    max_total_swap_layers: int = 94
    max_total_ops: int = 250
    always_record: bool = False
    sync_window: int = 64

    # --- Adaptive window (experimental, env-var controlled) ---
    adaptive_window: bool = False
    window_floor: int = 32
    window_shift_cos_threshold: float = 0.85
    window_stable_cos_threshold: float = 0.95
    window_shift_confirm_windows: int = 1
    window_stable_confirm_windows: int = 2

    # --- Tuning parameters (env-var controlled, rarely changed) ---
    decay_factor: float = 0.9
    min_record_tokens: int = 32
    warmup_forwards: int = 10
    sample_interval: int = 1

    @classmethod
    def from_server_args(cls, server_args):
        return cls(
            enabled=server_args.enable_pb_oeplb,
            threshold_ratio=getattr(server_args, 'pb_oeplb_threshold_ratio', 1.02),
            max_swaps_per_layer=getattr(server_args, 'pb_oeplb_max_swaps_per_layer', 64),
            min_prefill_tokens=getattr(server_args, 'pb_oeplb_min_prefill_tokens', 256),
            max_total_swap_layers=getattr(server_args, 'pb_oeplb_max_total_swap_layers', 94),
            max_total_ops=int(getattr(server_args, 'pb_oeplb_max_total_ops',
                                      os.environ.get('OEPLB_MAX_TOTAL_OPS', 250))),
            always_record=getattr(server_args, 'pb_oeplb_always_record', False),
            sync_window=getattr(server_args, 'pb_oeplb_sync_window', 64),
            # Adaptive window (env-var only)
            adaptive_window=os.environ.get('OEPLB_ADAPTIVE_WINDOW', '0') == '1',
            window_floor=int(os.environ.get('OEPLB_WINDOW_FLOOR', 32)),
            window_shift_cos_threshold=float(os.environ.get('OEPLB_WINDOW_SHIFT_COS', 0.85)),
            window_stable_cos_threshold=float(os.environ.get('OEPLB_WINDOW_STABLE_COS', 0.95)),
            window_shift_confirm_windows=int(os.environ.get('OEPLB_WINDOW_SHIFT_CONFIRM', 1)),
            window_stable_confirm_windows=int(os.environ.get('OEPLB_WINDOW_STABLE_CONFIRM', 2)),
            # Tuning (env-var only)
            decay_factor=float(os.environ.get('OEPLB_DECAY_FACTOR', 0.9)),
            min_record_tokens=int(os.environ.get('OEPLB_MIN_RECORD_TOKENS', 32)),
            warmup_forwards=int(os.environ.get('OEPLB_WARMUP_FORWARDS', 10)),
            sample_interval=int(os.environ.get('OEPLB_SAMPLE_INTERVAL', 1)),
        )
