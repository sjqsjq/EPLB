from dataclasses import dataclass

@dataclass
class PBOEPLBv2Config:
    enabled: bool = False
    # V2.1: single threshold (selection AND target) -- compromise between
    # V1's 1.15 (too permissive, ~30% of layers trigger) and V2's 1.30
    # (too restrictive AND capped at 8 swaps). Layers above this ratio get
    # FULLY corrected (multi-round), no global swap budget.
    threshold_ratio: float = 1.22
    max_swaps_per_layer: int = 5   # per-layer round cap (safety valve)
    max_total_swap_layers: int = 48  # safety cap only, effectively unbounded
    sync_window: int = 128
    # V2.1 record: ~2x more frequent than V2's ri=64, but jittered (not a
    # fixed periodic phase) to avoid always sampling the same point in a
    # repeating batch pattern, and requires a higher per-batch token count
    # for quality.
    record_interval: int = 32
    min_record_tokens: int = 64
    min_prefill_tokens: int = 1000
    cooldown_steps: int = 5
    always_record: bool = False

    @classmethod
    def from_server_args(cls, server_args):
        return cls(
            enabled=server_args.enable_pb_oeplb,
            threshold_ratio=getattr(server_args, 'pb_oeplb_threshold_ratio', 1.22),
            max_swaps_per_layer=getattr(server_args, 'pb_oeplb_max_swaps_per_layer', 5),
            max_total_swap_layers=getattr(server_args, 'pb_oeplb_max_total_swap_layers', 48),
            sync_window=getattr(server_args, 'pb_oeplb_sync_window', 128),
            record_interval=getattr(server_args, 'pb_oeplb_record_interval', 32),
            min_record_tokens=getattr(server_args, 'pb_oeplb_min_record_tokens', 64),
            min_prefill_tokens=getattr(server_args, 'pb_oeplb_min_prefill_tokens', 1000),
            cooldown_steps=getattr(server_args, 'pb_oeplb_cooldown_steps', 5),
            always_record=getattr(server_args, 'pb_oeplb_always_record', False),
        )

# Backward compatibility alias
PBOEPLBConfig = PBOEPLBv2Config
