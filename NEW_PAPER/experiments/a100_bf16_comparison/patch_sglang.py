#!/usr/bin/env python3
"""Idempotent OEPLB integration patcher for a fresh SGLang 0.5.6.post2 install."""
import os, sglang, sys

S = os.path.join(os.path.dirname(sglang.__file__), "srt")

def patch(path, edits):
    p = os.path.join(S, path)
    src = open(p).read()
    orig = src
    for tag, anchor, insert, mode in edits:
        if tag in src:
            print(f"  [skip] {path}: '{tag}' already present"); continue
        if anchor not in src:
            print(f"  [FAIL] {path}: anchor not found:\n    {anchor[:80]!r}"); sys.exit(1)
        if mode == "after":
            src = src.replace(anchor, anchor + insert, 1)
        elif mode == "replace":
            src = src.replace(anchor, insert, 1)
        print(f"  [ok]   {path}: inserted '{tag}'")
    if src != orig:
        open(p, "w").write(src)
        print(f"  wrote {path}")

# ---- server_args.py ----
DATACLASS = """
    # PB-OEPLB (online expert placement load balancer)
    enable_pb_oeplb: bool = False
    pb_oeplb_threshold_ratio: float = 1.02
    pb_oeplb_max_swaps_per_layer: int = 64
    pb_oeplb_min_prefill_tokens: int = 256
    pb_oeplb_max_total_swap_layers: int = 94
    pb_oeplb_max_total_ops: int = 300
    pb_oeplb_min_swap_ops: int = 8
    pb_oeplb_always_record: bool = False
    pb_oeplb_sync_window: int = 8
    pb_oeplb_decay_factor: float = 0.5
    pb_oeplb_min_record_tokens: int = 32
    pb_oeplb_adaptive_window: bool = False
    pb_oeplb_window_floor: int = 32
    pb_oeplb_window_shift_cos: float = 0.85
    pb_oeplb_window_stable_cos: float = 0.95
    pb_oeplb_window_shift_confirm: int = 1
    pb_oeplb_window_stable_confirm: int = 2
    pb_oeplb_calibrate_adaptive_sensitivity: bool = False
    pb_oeplb_calibration_forwards: int = 256"""

STATIC_OLD = '''        if (self.enable_eplb or (self.init_expert_location != "trivial")) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = "static"

        if self.enable_eplb:
            assert self.ep_size > 1'''
STATIC_NEW = '''        if (self.enable_eplb or self.enable_pb_oeplb or (self.init_expert_location != "trivial")) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = "static"

        if self.enable_eplb:
            assert self.ep_size > 1

        if self.enable_pb_oeplb:
            assert not self.enable_eplb, (
                "--enable-pb-oeplb and --enable-eplb are mutually exclusive."
            )
            assert self.ep_size > 1, "PB-OEPLB requires expert parallelism (ep_size > 1)."'''

ARG_ANCHOR = '''        parser.add_argument(
            "--enable-eplb",
            action="store_true",
            help="Enable EPLB algorithm",
        )'''
ARGS = '''
        # PB-OEPLB (online expert placement load balancer)
        parser.add_argument("--enable-pb-oeplb", action="store_true",
            help="Enable PB-OEPLB (online expert placement load balancer).")
        parser.add_argument("--pb-oeplb-threshold-ratio", type=float,
            default=ServerArgs.pb_oeplb_threshold_ratio, help="Imbalance ratio threshold to trigger a swap.")
        parser.add_argument("--pb-oeplb-max-swaps-per-layer", type=int,
            default=ServerArgs.pb_oeplb_max_swaps_per_layer, help="Max swaps per layer per decision window.")
        parser.add_argument("--pb-oeplb-min-prefill-tokens", type=int,
            default=ServerArgs.pb_oeplb_min_prefill_tokens, help="Min accumulated prefill tokens before deciding a swap.")
        parser.add_argument("--pb-oeplb-max-total-swap-layers", type=int,
            default=ServerArgs.pb_oeplb_max_total_swap_layers, help="Max layers touched by the global swap budget.")
        parser.add_argument("--pb-oeplb-max-total-ops", type=int,
            default=ServerArgs.pb_oeplb_max_total_ops, help="Max total swap ops per decision window.")
        parser.add_argument("--pb-oeplb-min-swap-ops", type=int,
            default=ServerArgs.pb_oeplb_min_swap_ops, help="Skip swap plan smaller than this.")
        parser.add_argument("--pb-oeplb-always-record", action="store_true",
            help="Record routing on decode batches too (default: prefill-only).")
        parser.add_argument("--pb-oeplb-sync-window", type=int,
            default=ServerArgs.pb_oeplb_sync_window, help="Forward passes between decision windows.")
        parser.add_argument("--pb-oeplb-decay-factor", type=float,
            default=ServerArgs.pb_oeplb_decay_factor, help="Exponential decay factor for load history.")
        parser.add_argument("--pb-oeplb-min-record-tokens", type=int,
            default=ServerArgs.pb_oeplb_min_record_tokens, help="Skip recording prefill batch smaller than this.")
        parser.add_argument("--pb-oeplb-adaptive-window", action="store_true",
            help="Enable adaptive sync window (shrink on shift, grow when stable).")
        parser.add_argument("--pb-oeplb-window-floor", type=int,
            default=ServerArgs.pb_oeplb_window_floor, help="Min sync window when adaptive shrinks.")
        parser.add_argument("--pb-oeplb-window-shift-cos", type=float,
            default=ServerArgs.pb_oeplb_window_shift_cos, help="cos_sim below this = workload shift.")
        parser.add_argument("--pb-oeplb-window-stable-cos", type=float,
            default=ServerArgs.pb_oeplb_window_stable_cos, help="cos_sim above this = stable.")
        parser.add_argument("--pb-oeplb-window-shift-confirm", type=int,
            default=ServerArgs.pb_oeplb_window_shift_confirm, help="Consecutive low-cos windows to confirm shift.")
        parser.add_argument("--pb-oeplb-window-stable-confirm", type=int,
            default=ServerArgs.pb_oeplb_window_stable_confirm, help="Consecutive high-cos windows to confirm stable.")
        parser.add_argument("--pb-oeplb-calibrate-adaptive-sensitivity", action="store_true",
            help="Calibrate adaptive window sensitivity from prefill:decode ratio.")
        parser.add_argument("--pb-oeplb-calibration-forwards", type=int,
            default=ServerArgs.pb_oeplb_calibration_forwards, help="Forwards for sensitivity calibration.")'''

patch("server_args.py", [
    ("enable_pb_oeplb: bool", "    enable_eplb: bool = False", DATACLASS, "after"),
    ("self.enable_pb_oeplb or", STATIC_OLD, STATIC_NEW, "replace"),
    ('parser.add_argument("--enable-pb-oeplb"', ARG_ANCHOR, ARGS, "after"),
])

# ---- model_runner.py ----
MR_INIT = """

        # PB-OEPLB (online expert placement load balancer)
        self.pb_oeplb_controller = None
        if self.server_args.enable_pb_oeplb and (not self.is_draft_worker):
            from sglang.srt.managers.pb_oeplb.config import PBOEPLBConfig
            from sglang.srt.managers.pb_oeplb.controller import PBOEPLBController

            self.pb_oeplb_controller = PBOEPLBController(
                PBOEPLBConfig.from_server_args(self.server_args), self
            )
            from sglang.srt.managers.pb_oeplb import set_pb_oeplb_controller

            set_pb_oeplb_controller(self.pb_oeplb_controller)"""

MR_FWD = """
        if self.pb_oeplb_controller is not None:
            self.pb_oeplb_controller.on_forward_pass_end(forward_batch)
"""

patch("model_executor/model_runner.py", [
    ("self.pb_oeplb_controller = None",
     "        self.expert_location_updater = ExpertLocationUpdater()", MR_INIT, "after"),
    ("self.pb_oeplb_controller.on_forward_pass_end",
     "        if self.eplb_manager is not None:\n            self.eplb_manager.on_forward_pass_end()\n",
     MR_FWD, "after"),
])

# ---- topk.py ----
TOPK = """

    from sglang.srt.managers.pb_oeplb import get_pb_oeplb_controller

    _pb_oeplb_ctrl = get_pb_oeplb_controller()
    if _pb_oeplb_ctrl is not None:
        _pb_oeplb_ctrl.record_next_layer(topk_ids)
"""
patch("layers/moe/topk.py", [
    ("get_pb_oeplb_controller",
     "    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)",
     TOPK, "after"),
])

# ---- models/qwen3_moe.py: forward_normal must pass expert-location dispatch ----
# ROOT-CAUSE FIX: the non-DeepEP path (moe-a2a-backend none, used on A100) called
# self.topk(hidden_states, router_logits) WITHOUT expert_location_dispatch_info, so
# topk_ids_logical_to_physical() was never applied. OEPLB's placement swaps (which
# only change logical->physical mapping) were therefore ignored by routing -- load
# never rebalanced (imbalance rebounded every window) AND tokens hit relocated
# weights (silent output corruption). Mirror forward_deepep's call.
QWEN3_TOPK = """        topk_output = self.topk(
            hidden_states,
            router_logits,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=self.layer_id,
            ),
        )"""
patch("models/qwen3_moe.py", [
    ("expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(\n                layer_id=self.layer_id,\n            ),\n        )\n        final_hidden_states = self.experts(hidden_states, topk_output)",
     "        topk_output = self.topk(hidden_states, router_logits)",
     QWEN3_TOPK, "replace"),
])

print("PATCH_DONE")
