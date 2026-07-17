from dataclasses import dataclass

@dataclass
class PBOEPLBv2Config:
    enabled: bool = False
    # Two-tier threshold: only swap layers with ratio > high_threshold,
    # swap until ratio drops below target_ratio
    high_threshold: float = 1.30
    target_ratio: float = 1.15
    # Global budget: max total swaps across ALL layers per decision window
    max_total_swaps: int = 8
    # How often to check (in forward passes)
    sync_window: int = 128
    # Record only every Nth forward pass (reduce scatter_add_ overhead)
    record_interval: int = 8
    # Minimum accumulated tokens before considering a swap
    min_prefill_tokens: int = 1000
    # Cooldown for prefill batch sampling
    cooldown_steps: int = 5
    # Whether to record on all batches (not just prefill)
    always_record: bool = False

    @classmethod
    def from_server_args(cls, server_args):
        return cls(
            enabled=server_args.enable_pb_oeplb,
            high_threshold=getattr(server_args, 'pb_oeplb_threshold_ratio', 1.30),
            target_ratio=getattr(server_args, 'pb_oeplb_target_ratio', 1.15),
            max_total_swaps=getattr(server_args, 'pb_oeplb_max_total_swaps', 8),
            sync_window=getattr(server_args, 'pb_oeplb_sync_window', 128),
            record_interval=getattr(server_args, 'pb_oeplb_record_interval', 64),
            min_prefill_tokens=getattr(server_args, 'pb_oeplb_min_prefill_tokens', 1000),
            cooldown_steps=getattr(server_args, 'pb_oeplb_cooldown_steps', 5),
            always_record=getattr(server_args, 'pb_oeplb_always_record', False),
        )

# Backward compatibility alias
PBOEPLBConfig = PBOEPLBv2Config
