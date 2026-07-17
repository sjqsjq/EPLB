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

    @classmethod
    def from_server_args(cls, server_args):
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
        )
