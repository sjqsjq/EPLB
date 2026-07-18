"""OEPLB V2.1 Controller: jittered low-overhead record + thorough
threshold-based swap (no artificial global swap budget).

Changes from V2:
- record_next_layer: MIN_RECORD_TOKENS raised (config-driven, default 64)
  so only substantial batches get sampled.
- on_forward_pass_end: record sampling uses a JITTERED interval (randomized
  target within each ~record_interval-sized block) instead of a fixed
  modulo phase, so sampled forward passes don't always land at the same
  relative position in a repeating batch/traffic pattern -- this reduces
  the risk of consistently missing (or consistently over-sampling) a
  particular kind of batch.
- _decide_and_begin_swap: calls try_build_swap_plan_v2 with a single
  threshold_ratio (no high/target split) and no real global swap budget
  (max_total_swap_layers defaults to 48 = effectively unbounded).
"""
import logging
import random
import time
import torch
import torch.distributed
import traceback

from sglang.srt.managers.pb_oeplb.config import PBOEPLBv2Config
from sglang.srt.managers.pb_oeplb.rebalancer import try_build_swap_plan_v2
from sglang.srt.managers.pb_oeplb.async_swapper import AsyncSwapExecutor
from sglang.srt.managers.pb_oeplb.fast_metadata import fast_init_by_mapping

logger = logging.getLogger(__name__)


class PBOEPLBController:
    """V2.1 controller: jittered record sampling + thorough elastic swap."""

    def __init__(self, cfg, model_runner):
        self.cfg = cfg
        self.model_runner = model_runner

        self._meta = self._fetch_metadata()
        self.num_layers = self._meta.num_layers
        self.num_logical_experts = self._meta.num_logical_experts
        self.ep_size = self._meta.ep_size
        self.num_local = self._meta.num_local_physical_experts
        self.dp_size = getattr(model_runner.server_args, 'dp_size', 1)
        self.num_physical_experts = self._meta.num_physical_experts

        self.load = torch.zeros(self.num_layers, self.num_physical_experts,
                                dtype=torch.int64, device="cuda")
        self.total_tokens = 0
        self._layer_counter = 0
        self._cached_p2l = self._meta.physical_to_logical_map

        self.total_swaps = 0
        self.skipped_busy = 0
        self.window_count = 0

        self._prefill_batch_counter = 0
        self._sample_interval = max(1, cfg.cooldown_steps // 5)
        self._should_record_this_batch = False

        self.async_executor = AsyncSwapExecutor(
            model_runner, self.model_runner.moe_ep_rank, self.num_local
        )
        self._pending_plan_start_t = None
        self._prewarmed = (self.dp_size > 1)

        self._forward_id = 0
        self._ready = False
        self._warmup_forwards = 10
        self._steps_since_last_check = 0

        # V2.1: jittered record scheduling. `_record_counter` counts forwards
        # since the last (re)roll; `_next_record_target` is a randomized
        # point within [0.5x, 1.5x] of record_interval, re-rolled every time
        # we either hit the target or overshoot it -- this decorrelates the
        # sampled forward-pass position from any periodic traffic pattern.
        self._record_counter = 0
        self._next_record_target = self._roll_next_record_target()

        # Profiling
        self._prof_record_calls = 0
        self._prof_record_ns = 0
        self._prof_allreduce_ns = 0
        self._prof_planbuild_ns = 0
        self._prof_finalize_ns = 0

        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(f"[OEPLB-V2.1] Init: layers={self.num_layers}, experts={self.num_logical_experts}, "
                    f"ep={self.ep_size}, dp={self.dp_size}, local={self.num_local}, "
                    f"threshold_ratio={cfg.threshold_ratio}, max_swaps_per_layer={cfg.max_swaps_per_layer}, "
                    f"max_total_swap_layers={cfg.max_total_swap_layers}, "
                    f"sync_window={cfg.sync_window}, record_interval={cfg.record_interval} (jittered), "
                    f"min_record_tokens={cfg.min_record_tokens}")

    def _roll_next_record_target(self):
        base = max(1, self.cfg.record_interval)
        jitter = max(1, base // 2)
        return random.randint(max(1, base - jitter), base + jitter)

    def _prewarm_expert_location_updater(self):
        try:
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            updater = getattr(self.model_runner, "expert_location_updater", None)
            if updater is not None:
                updater._first_execution = False
            fast_init_by_mapping(self._cached_p2l.clone(), self.num_logical_experts)
            torch.cuda.synchronize()
            if self.dp_size <= 1:
                self._warmup_all_rank_pairs()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[OEPLB-V2.1] Pre-warmed in {elapsed:.1f}ms")
        except Exception as e:
            logger.warning(f"[OEPLB-V2.1] Pre-warm failed (non-fatal): {e}")

    def _warmup_all_rank_pairs(self):
        if self.ep_size < 2:
            return
        meta = self._fetch_metadata()
        p2l = meta.physical_to_logical_map.clone()
        pairs = [(a, b) for a in range(self.ep_size) for b in range(a + 1, self.ep_size)]
        if len(pairs) > self.num_layers:
            pairs = pairs[:self.num_layers]
        warmup_p2l = p2l.clone()
        update_layers = []
        for layer_id, (ra, rb) in enumerate(pairs):
            phys_a, phys_b = ra * self.num_local, rb * self.num_local
            la = int(warmup_p2l[layer_id, phys_a].item())
            lb = int(warmup_p2l[layer_id, phys_b].item())
            warmup_p2l[layer_id, phys_a], warmup_p2l[layer_id, phys_b] = lb, la
            update_layers.append(layer_id)
        self.model_runner.update_expert_location(fast_init_by_mapping(warmup_p2l, self.num_logical_experts), update_layers)
        self.model_runner.update_expert_location(fast_init_by_mapping(p2l, self.num_logical_experts), update_layers)
        self._meta = self._fetch_metadata()
        self._cached_p2l = self._meta.physical_to_logical_map

    def _fetch_metadata(self):
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        return get_global_expert_location_metadata()

    def record_next_layer(self, topk_ids: torch.Tensor):
        if not self._should_record_this_batch:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if topk_ids.shape[0] < self.cfg.min_record_tokens:
            return

        _t0 = time.perf_counter_ns()
        layer_id = self._layer_counter % self.num_layers
        self._layer_counter += 1
        flat = topk_ids.reshape(-1)
        mask = flat != -1
        self.load[layer_id].scatter_add_(
            dim=0, index=flat.masked_fill(~mask, 0).long(), src=mask.long()
        )
        if layer_id == 0:
            self.total_tokens += topk_ids.shape[0]
        self._prof_record_calls += 1
        self._prof_record_ns += time.perf_counter_ns() - _t0

    def on_forward_pass_end(self, forward_batch):
        self._forward_id += 1
        if not self._prewarmed and not torch.cuda.is_current_stream_capturing():
            self._prewarmed = True
            self._prewarm_expert_location_updater()

        self._try_finish_pending_swap()

        is_idle = forward_batch.forward_mode.is_idle()
        is_prefill = forward_batch.forward_mode.is_extend()
        self._layer_counter = 0

        # V2.1: jittered record trigger -- fire once we've reached the
        # randomized target, then re-roll a fresh (re-randomized) target for
        # the next block. This spreads sampled forward passes irregularly
        # in time rather than locking to a fixed periodic phase.
        self._record_counter += 1
        record_due = self._record_counter >= self._next_record_target
        if record_due:
            self._record_counter = 0
            self._next_record_target = self._roll_next_record_target()

        if is_idle:
            self._should_record_this_batch = False
        elif self.cfg.always_record:
            self._should_record_this_batch = record_due
        elif is_prefill:
            self._prefill_batch_counter += 1
            self._should_record_this_batch = (
                self._prefill_batch_counter % self._sample_interval == 0
                and record_due
            )
        else:
            self._should_record_this_batch = False

        if not self._ready:
            if self._forward_id >= self._warmup_forwards:
                self._ready = True
                logger.info(f"[OEPLB-V2.1] rank={self._rank} ready after {self._forward_id} forwards")
            return

        self._steps_since_last_check += 1
        if self._steps_since_last_check < self.cfg.sync_window:
            return
        self._steps_since_last_check = 0

        self._decide_and_begin_swap()
        self.load.zero_()
        self.total_tokens = 0
        self._prefill_batch_counter = 0

    def _decide_and_begin_swap(self):
        try:
            _t0 = time.perf_counter_ns()
            torch.distributed.all_reduce(self.load, op=torch.distributed.ReduceOp.SUM)
            self.window_count += 1
            self._prof_allreduce_ns += time.perf_counter_ns() - _t0

            global_tokens = int(self.load.sum().item())
            self._maybe_report_profile()

            if global_tokens < self.cfg.min_prefill_tokens:
                return
            if self.async_executor.busy:
                self.skipped_busy += 1
                return

            _t1 = time.perf_counter_ns()
            self._meta = self._fetch_metadata()
            p2l_map = self._meta.physical_to_logical_map.clone()

            plan = try_build_swap_plan_v2(
                logical_count=self.load,
                physical_to_logical_map=p2l_map,
                num_ranks=self.ep_size,
                num_local=self.num_local,
                threshold_ratio=self.cfg.threshold_ratio,
                max_swaps_per_layer=self.cfg.max_swaps_per_layer,
                max_total_swap_layers=self.cfg.max_total_swap_layers,
            )
            self._prof_planbuild_ns += time.perf_counter_ns() - _t1

            if not plan:
                return
            self._pending_plan_start_t = time.perf_counter()
            self.async_executor.begin(plan)
        except Exception as e:
            logger.error(f"[OEPLB-V2.1] error: {e}\n{traceback.format_exc()}")

    def _maybe_report_profile(self):
        if self.window_count == 0 or self.window_count % 4 != 0:
            return
        record_ms = self._prof_record_ns / 1e6
        allreduce_ms = self._prof_allreduce_ns / 1e6
        planbuild_ms = self._prof_planbuild_ns / 1e6
        finalize_ms = self._prof_finalize_ns / 1e6
        total_ms = record_ms + allreduce_ms + planbuild_ms + finalize_ms
        logger.info(
            f"[OEPLB-V2.1-PROF] w#{self.window_count} calls={self._prof_record_calls} "
            f"record={record_ms:.2f}ms allreduce={allreduce_ms:.2f}ms "
            f"planbuild={planbuild_ms:.2f}ms finalize={finalize_ms:.2f}ms "
            f"total={total_ms:.2f}ms swaps_total={self.total_swaps} skipped_busy={self.skipped_busy}"
        )

    def _try_finish_pending_swap(self):
        try:
            plan = self.async_executor.try_finish()
        except Exception as e:
            logger.error(f"[OEPLB-V2.1] finish error: {e}\n{traceback.format_exc()}")
            self.async_executor.pending = None
            return
        if plan is None:
            return
        try:
            self._meta = self._fetch_metadata()
            new_p2l = self._meta.physical_to_logical_map.clone()
            update_layers = set()
            for op in plan:
                new_p2l[op.layer_id, op.phys_slot_a] = op.logical_b
                new_p2l[op.layer_id, op.phys_slot_b] = op.logical_a
                update_layers.add(op.layer_id)
            _tf = time.perf_counter_ns()
            new_meta = fast_init_by_mapping(new_p2l, self.num_logical_experts)
            self._meta.update(new_meta, update_layer_ids=list(update_layers))
            self._cached_p2l = self._meta.physical_to_logical_map
            self._prof_finalize_ns += time.perf_counter_ns() - _tf
            total_ms = (time.perf_counter() - self._pending_plan_start_t) * 1000
            self.total_swaps += len(plan)
            logger.info(f"[OEPLB-V2.1] {len(plan)} swap(s) done ({total_ms:.1f}ms) | total={self.total_swaps}")
        except Exception as e:
            logger.error(f"[OEPLB-V2.1] finalize error: {e}\n{traceback.format_exc()}")
