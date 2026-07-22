import logging
import time
import torch
import torch.distributed
import traceback

from sglang.srt.managers.pb_oeplb.config import PBOEPLBConfig
from sglang.srt.managers.pb_oeplb.rebalancer import try_build_swap_plan
from sglang.srt.managers.pb_oeplb.async_swapper import AsyncSwapExecutor
from sglang.srt.managers.pb_oeplb.fast_metadata import fast_init_by_mapping

logger = logging.getLogger(__name__)

MIN_RECORD_TOKENS = 32


class PBOEPLBController:
    """
    v0.11: v0.10 + exponential decay statistics + fast sync_window.

    Key changes from v0.10:
    - sync_window reduced from 64 to 8 (respond to imbalance 8x faster)
    - load tensor uses exponential decay (decay_factor=0.5) instead of
      zeroing after each swap — this stabilizes decisions against single-
      batch routing noise and preserves historical context
    - All-reduce tensor is now physical-space (matching record), converted
      to logical only at plan-build time

    v0.10: v0.9 + physical-space recording (matches EPLB's on_select_experts
    exactly), eliminating the per-call physical->logical gather.

    Profiling v0.9 (torch wall-clock instrumentation) found record_next_layer
    cost ~800-1000us/call, 5-6x more than the all_reduce it was meant to make
    cheap (370ms allreduce vs ~1.8-2.0s cumulative record over one benchmark
    run). Root cause: each call did reshape + long() + clamp_ + a FANCY-INDEX
    GATHER (p2l[layer_id][flat]) + bincount + add_ — 5-6 separate CUDA kernel
    launches, each paying CPU-side dispatch overhead on the single-threaded
    scheduler's critical path.

    Fix (mirrors EPLB's _SelectExpertsSinglePassGatherer.on_select_experts,
    confirmed via source read of expert_distribution.py:507-511): record
    directly in PHYSICAL expert space via ONE scatter_add_ call — no p2l
    gather, no clamp_, no separate bincount+add_ pair. The physical->logical
    conversion (which EPLB also defers to dump-time, see
    _convert_global_physical_count_to_logical_count) happens ONCE per
    sync_window, as a single vectorized scatter_add_ across all layers,
    instead of once per recorded prefill batch per layer.

    v0.9: EPLB-style local step-counter trigger, single periodic all_reduce.

    Investigation into SGLang's official EPLB (see eplb_manager.py) found that
    its `on_forward_pass_end()` is `next(self._main_generator)` where the
    generator is `for _ in range(rebalance_num_iterations): yield` — a PURE
    LOCAL Python integer loop. Zero communication on every-but-the-Nth call.
    This works because forward passes are inherently synchronized across all
    DP+EP ranks (confirmed empirically: even IDLE batches call
    on_forward_pass_end exactly once per rank per global forward step), so a
    local counter implicitly stays in lockstep across ranks with no all_reduce
    needed to agree on "when".

    v0.8 mistakenly used two all_reduce(MAX) calls EVERY forward pass to reach
    consensus on "is this a P->D boundary" and "are we ready to swap" — this
    was unnecessary (forward passes are already synchronized) and caused the
    observed -16.6% throughput regression under DP=4.

    v0.9 fix: replace the per-forward consensus with a local step counter
    (self._steps_since_last_check), matching EPLB's pattern. Only every
    `sync_window` forwards do we do ONE all_reduce(SUM) on the load tensor —
    this single collective both aggregates the true global load AND serves as
    the "did we accumulate enough tokens" signal (checked locally on the
    already-fetched global sum, no separate readiness round needed).

    Recording itself is unchanged from v0.8 and matches EPLB's own hot path
    (`_SelectExpertsSinglePassGatherer.on_select_experts`, confirmed via source
    inspection): pure local `bincount`/scatter, zero communication, hooked
    into topk.py's post-select_experts callback. OEPLB's differentiator is
    recording ONLY during prefill (targeting the load that determines the
    upcoming decode burst's expert popularity), whereas EPLB records both
    prefill and decode uniformly.
    """

    def __init__(self, cfg: PBOEPLBConfig, model_runner):
        self.cfg = cfg
        self.model_runner = model_runner

        self._meta = self._fetch_metadata()
        self.num_layers = self._meta.num_layers
        self.num_logical_experts = self._meta.num_logical_experts
        self.ep_size = self._meta.ep_size
        self.num_local = self._meta.num_local_physical_experts

        self.dp_size = getattr(model_runner.server_args, 'dp_size', 1)

        self.num_physical_experts = self._meta.num_physical_experts
        # v0.10: load is recorded in PHYSICAL expert space (see class
        # docstring) — converted to logical space once per sync_window,
        # not once per record_next_layer call.
        self.load = torch.zeros(self.num_layers, self.num_physical_experts,
                                dtype=torch.int64, device="cuda")
        self.total_tokens = 0
        self._layer_counter = 0
        self._cached_p2l = self._meta.physical_to_logical_map

        self.total_swaps = 0
        self.skipped_busy = 0
        self.window_count = 0

        self._prefill_batch_counter = 0
        # Sample every Nth prefill batch to reduce record overhead.
        # With sync_window=256 and 48 layers, recording every batch costs
        # ~82us × 48 × 256 = ~1s per window on scheduler critical path.
        # Sampling every 4th batch cuts this to ~0.25s while keeping
        # routing statistics representative (still ~3000 calls per window).
        self._sample_interval = 1  # record every prefill batch
        self._should_record_this_batch = False

        self.async_executor = AsyncSwapExecutor(
            model_runner, self.model_runner.moe_ep_rank, self.num_local
        )
        self._pending_plan_start_t = None
        self._prewarmed = (self.dp_size > 1)

        # Pure local counters — no cross-rank consensus needed (see class docstring)
        self._forward_id = 0
        self._ready = False
        self._warmup_forwards = 10
        self._steps_since_last_check = 0
        self._decay_factor = 0.9  # best tested value
        # Adaptive window (experimental, opt-in via cfg.adaptive_window): shrink
        # sync_window temporarily during a CONFIRMED workload shift (>=2
        # consecutive low-cos_sim windows, not a single blip), grow it back once
        # stable (>=2 consecutive high-cos_sim windows). See _decide_and_begin_swap
        # for where this is updated and on_forward_pass_end for where it's read.
        self._effective_sync_window = cfg.sync_window
        self._last_cos_sim = None
        self._window_shift_count = 0
        self._window_stable_count = 0

        # Profiling counters (wall-clock, CPU-side critical-path time)
        self._prof_record_calls = 0
        self._prof_record_ns = 0
        self._prof_allreduce_ns = 0
        self._prof_planbuild_ns = 0
        self._prof_finalize_ns = 0
        self._prof_last_report_forward = 0

        self._prev_load = None  # for routing stability tracking
        self._stability_history = []  # cosine similarities
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(f"[PB-OEPLB] Init v0.11 (fast-response + decay): "
                    f"layers={self.num_layers}, experts={self.num_logical_experts}, "
                    f"ep={self.ep_size}, dp={self.dp_size}, local={self.num_local}, "
                    f"thresh={cfg.threshold_ratio}, min_tok={cfg.min_prefill_tokens}, "
                    f"sync_window={cfg.sync_window}, always_record={cfg.always_record}")

    def _prewarm_expert_location_updater(self):
        try:
            t0 = time.perf_counter()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            updater = getattr(self.model_runner, "expert_location_updater", None)
            if updater is not None:
                updater._first_execution = False
            fast_init_by_mapping(self._meta.physical_to_logical_map.clone(), self.num_logical_experts)
            torch.cuda.synchronize()
            if self.dp_size <= 1:
                self._warmup_all_rank_pairs()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[PB-OEPLB] Pre-warmed in {elapsed:.1f}ms")
        except Exception as e:
            logger.warning(f"[PB-OEPLB] Pre-warm failed (non-fatal): {e}")

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
        """Zero-communication hot path — direct scatter_add_ on PHYSICAL expert
        ids, exactly matching EPLB's on_select_experts (expert_distribution.py:
        507-511). No p2l gather here (see class docstring for why this
        replaced the old bincount+gather approach)."""
        if not self._should_record_this_batch:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        if topk_ids.shape[0] < MIN_RECORD_TOKENS:
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

        # Recording decision — local-only, prefill-focused (OEPLB's differentiator
        # vs EPLB's uniform prefill+decode recording). No cross-rank agreement needed:
        # this only affects what THIS rank writes into its own load buffer.
        if is_idle:
            self._should_record_this_batch = False
        elif self.cfg.always_record:
            self._should_record_this_batch = True
        elif is_prefill:
            self._prefill_batch_counter += 1
            self._should_record_this_batch = (self._prefill_batch_counter % self._sample_interval == 0)
        else:
            self._should_record_this_batch = False

        if not self._ready:
            if self._forward_id >= self._warmup_forwards:
                self._ready = True
                logger.info(f"[PB-OEPLB] rank={self._rank} ready after {self._forward_id} forwards")
            return

        # --- Pure local step counter (EPLB-style, zero communication) ---
        # Implicitly synchronized across ranks: every rank calls this exactly
        # once per global forward step (including IDLE batches), so this
        # counter advances in lockstep with every other rank's counter with
        # NO all_reduce needed to agree on "when to check".
        self._steps_since_last_check += 1
        if self._steps_since_last_check < self._effective_sync_window:
            return
        self._steps_since_last_check = 0
        self._decay_factor = 0.9  # best tested value

        self._decide_and_begin_swap()
        # Exponential decay instead of zeroing: preserve routing history
        # so swap decisions reflect the stable/dominant pattern, not
        # single-window noise. decay=0.3 means 70% of current window's
        # signal is retained, 30% of history bleeds through.
        self.load.copy_((self.load.float() * self._decay_factor).long())
        self.total_tokens = 0
        self._prefill_batch_counter = 0

    def _decide_and_begin_swap(self):
        try:
            # DEADLOCK FIX: force any still-pending swap from the PREVIOUS window
            # to genuinely finish (blocking if necessary) before this rank issues
            # the collective all_reduce below. See async_swapper.py's try_finish()
            # docstring for the full root-cause explanation (cross-rank NCCL op
            # ordering divergence between the async P2P stream and the default
            # stream's collectives). This is the ONLY blocking point -- every
            # other forward pass in between windows still uses the non-blocking
            # try_finish() path via on_forward_pass_end()'s _try_finish_pending_swap().
            self._try_finish_pending_swap(force_wait=True)

            _t0 = time.perf_counter_ns()
            # BUGFIX: all_reduce must NOT mutate self.load in place. self.load
            # is a per-rank LOCAL decayed accumulator (see on_forward_pass_end);
            # once all_reduce sums it across ranks, every rank ends up holding
            # an IDENTICAL GLOBAL value. If that global value is decayed and
            # left in self.load, the NEXT window's all_reduce sums 8 copies of
            # an already-globally-summed quantity again -- compounding by
            # ~num_ranks x decay_factor EVERY window (confirmed empirically:
            # observed tok_global growing ~7.2x/window with decay=0.9, exactly
            # matching 8 ranks x 0.9). Fix: reduce a CLONE for the decision;
            # self.load itself stays per-rank-local and only ever decays its
            # own local history.
            global_load = self.load.clone()
            torch.distributed.all_reduce(
                global_load, op=torch.distributed.ReduceOp.SUM
            )
            self.window_count += 1
            self._prof_allreduce_ns += time.perf_counter_ns() - _t0

            global_tokens = int(global_load.sum().item())
            self._maybe_report_profile()
            # Track drift every window (even quiet ones) so adaptive window sizing
            # has an unbroken cos_sim history to react to.
            self._last_cos_sim = self._track_routing_stability(global_load)
            if self.cfg.adaptive_window and self._last_cos_sim is not None:
                if self._last_cos_sim < self.cfg.window_shift_cos_threshold:
                    self._window_shift_count += 1
                    self._window_stable_count = 0
                    # ASYMMETRIC confirmation: shrinking only changes CADENCE (how
                    # soon we check again), not what gets swapped -- rebalancer's
                    # own threshold_ratio still gates every actual swap decision
                    # independent of window size. A false-positive shrink costs at
                    # most one extra all_reduce; it can't mis-place an expert. So
                    # shrinking doesn't need the multi-window confirmation that
                    # decay/swap-aggressiveness changes would (those DO directly
                    # cause swaps, where a false positive wastes real P2P bandwidth
                    # and can't be un-done cheaply). React on window_shift_confirm
                    # (default 1) low-cos_sim window(s) -- this window's cos_sim and
                    # its own avg_ratio_before spike are computed from the SAME
                    # global_load snapshot, so reacting immediately costs zero extra
                    # lag beyond the one window of latency that's unavoidable (you
                    # can't know a window is anomalous before it's finished).
                    if self._window_shift_count >= self.cfg.window_shift_confirm_windows:
                        if self._effective_sync_window != self.cfg.window_floor:
                            logger.info(f"[PB-OEPLB-WINDOW] shift confirmed "
                                        f"({self._window_shift_count} low-cos_sim "
                                        f"window(s)) -- shrinking sync_window "
                                        f"{self._effective_sync_window} -> {self.cfg.window_floor}")
                        self._effective_sync_window = self.cfg.window_floor
                elif self._last_cos_sim > self.cfg.window_stable_cos_threshold:
                    self._window_stable_count += 1
                    self._window_shift_count = 0
                    # Recovery (growing back to the larger, TTFT-friendlier window)
                    # keeps the stricter multi-window confirmation
                    # (window_stable_confirm_windows, default 2) -- erring toward
                    # staying small a little longer costs a bit of extra check
                    # overhead, which is cheap, so there's no reason to rush this
                    # direction the way there is for reacting to a real shift.
                    if self._window_stable_count >= self.cfg.window_stable_confirm_windows:
                        if self._effective_sync_window != self.cfg.sync_window:
                            logger.info(f"[PB-OEPLB-WINDOW] stable confirmed "
                                        f"({self._window_stable_count} consecutive "
                                        f"high-cos_sim windows) -- restoring sync_window "
                                        f"{self._effective_sync_window} -> {self.cfg.sync_window}")
                        self._effective_sync_window = self.cfg.sync_window
                else:
                    # Ambiguous middle band: require a FRESH streak of confirmations
                    # in either direction -- don't let a borderline value count
                    # towards either side's confirmation tally.
                    self._window_shift_count = 0
                    self._window_stable_count = 0
            if global_tokens < self.cfg.min_prefill_tokens:
                return

            if self.async_executor.busy:
                self.skipped_busy += 1
                return

            _t1 = time.perf_counter_ns()
            self._meta = self._fetch_metadata()
            p2l_map = self._meta.physical_to_logical_map.clone()

            # BUGFIX: topk_ids reaching record_next_layer() are PHYSICAL slot
            # ids (fused_topk's topk_ids_logical_to_physical() already ran,
            # since ep_dispatch_algorithm="static" whenever --enable-pb-oeplb
            # is set) -- NOT logical expert ids. self.load (and global_load,
            # its all-reduced snapshot) is indexed by physical slot. Pass
            # global_load (the all-reduced decision snapshot), not the
            # per-rank-local self.load, to the plan builder.
            plan = try_build_swap_plan(
                logical_count=global_load,
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
            logger.info(f"[PB-OEPLB] window#{self.window_count}: "
                       f"issued {len(plan)} swap(s) ({global_tokens} tok global, dp={self.dp_size})")
        except Exception as e:
            logger.error(f"[PB-OEPLB] error: {e}\n{traceback.format_exc()}")

    def _dump_detailed_load(self, global_tokens, global_load):
        """Verification A+B: cross-check token count aggregation methods."""
        if self._rank != 0 or self.window_count not in (5, 10):
            return
        if global_tokens < self.cfg.min_prefill_tokens:
            return
        from sglang.srt.managers.pb_oeplb.rebalancer import compute_gpu_load
        p2l = self._cached_p2l

        for layer_id in [24]:
            # VERIFY-A: use rebalancer compute_gpu_load (proven correct for swap decisions)
            gpu_load_reb = compute_gpu_load(
                global_load[layer_id], p2l[layer_id],
                num_ranks=self.ep_size, num_local=self.num_local
            )
            logger.info(f"[VERIFY-A] w#{self.window_count} L{layer_id} rebalancer={gpu_load_reb.tolist()}")

            # VERIFY-B: BUGFIXED -- self.load is PHYSICAL-slot indexed (topk_ids
            # reaching record_next_layer are already logical->physical converted
            # by fused_topk/topk_ids_logical_to_physical since ep_dispatch_algorithm
            # ="static"). No p2l gather needed; physical slot i belongs to rank
            # i // num_local directly.
            v2s, v2m = [], []
            for r in range(self.ep_size):
                s = r * self.num_local
                e = (r + 1) * self.num_local
                pe = global_load[layer_id][s:e]
                v2s.append(int(pe.sum().item()))
                v2m.append(int(pe.max().item()))
            logger.info(f"[VERIFY-B] w#{self.window_count} L{layer_id} v2_sums={v2s} v2_maxes={v2m}")

            # VERIFY-B manual: rank3 raw detail (physical slots, not logical ids)
            r3s = 3 * self.num_local
            r3e = 4 * self.num_local
            r3_counts = [int(global_load[layer_id][slot].item()) for slot in range(r3s, r3e)]
            logger.info(f"[VERIFY-B-R3] w#{self.window_count} L{layer_id} "
                        f"slots=[{r3s}..{r3e-1}] "
                        f"counts={r3_counts[:5]}..{r3_counts[-3:]} "
                        f"sum={sum(r3_counts)} max={max(r3_counts)}")

            # META: load tensor info
            total = int(global_load[layer_id].sum().item())
            nz = int((global_load[layer_id] > 0).sum().item())
            logger.info(f"[VERIFY-META] w#{self.window_count} L{layer_id} "
                        f"shape={list(self.load[layer_id].shape)} total={total} "
                        f"nonzero={nz}/{self.load.shape[1]}")

    def _track_routing_stability(self, global_load):
        """Track cosine similarity between consecutive windows' load distributions.
        High similarity (>0.95) means the routing pattern is stable; low similarity
        means the workload just shifted (e.g. domain switch). Returns the cos_sim
        for THIS window (or None if there's no previous window yet).

        BUGFIX: this used to be gated by `if self._rank != 0: return`, which meant
        only rank 0 ever advanced self._prev_load / computed cos_sim. global_load is
        already all-reduced (identical bit-for-bit on every rank) by the time this
        is called, so every rank can safely compute the SAME cos_sim from the SAME
        tensors -- the rank-0 guard now only protects the periodic log line (avoid
        8x spam), not the computation itself. Kept as a general-purpose diagnostic
        signal (independent of decay/window logic) so any future consumer can rely
        on every rank holding a consistent, unbroken cos_sim history.
        """
        current = global_load.float()
        cos_sim = None
        if self._prev_load is not None:
            # Per-layer cosine similarity, then average
            dot = (current * self._prev_load).sum(dim=1)
            norm_c = current.norm(dim=1)
            norm_p = self._prev_load.norm(dim=1)
            valid = (norm_c > 0) & (norm_p > 0)
            if valid.any():
                cos_sim = (dot[valid] / (norm_c[valid] * norm_p[valid])).mean().item()
                self._stability_history.append(cos_sim)
                if self._rank == 0 and len(self._stability_history) % 4 == 0:
                    recent = self._stability_history[-8:]
                    logger.info(
                        f"[PB-OEPLB-STABILITY] w#{self.window_count} "
                        f"cos_sim={cos_sim:.4f} "
                        f"recent_avg={sum(recent)/len(recent):.4f} "
                        f"all_avg={sum(self._stability_history)/len(self._stability_history):.4f} "
                        f"(>0.95 = stable pattern, window can be larger)"
                    )
        self._prev_load = current.clone()
        return cos_sim

    def _dump_heatmap_data(self, global_tokens, global_load):
        """Dump per-rank load for each layer to analyze hot-spots."""
        if self._rank != 0:
            return
        if global_tokens < self.cfg.min_prefill_tokens:
            return
        # BUGFIX (corrected from prior version): self.load IS in PHYSICAL slot
        # space -- topk_ids reaching record_next_layer() have already been
        # converted logical->physical by fused_topk()'s call to
        # topk_ids_logical_to_physical() (topk.py), since ep_dispatch_algorithm
        # ="static" whenever --enable-pb-oeplb is set. Physical slot i belongs
        # to rank i // num_local directly -- no p2l gather needed.
        per_rank_load = torch.zeros(self.num_layers, self.ep_size, dtype=torch.int64, device=global_load.device)
        for r in range(self.ep_size):
            s, e = r * self.num_local, (r + 1) * self.num_local
            per_rank_load[:, r] = global_load[:, s:e].sum(dim=1)
        # per_rank_load: [num_layers, ep_size]
        # Pick layer 0 and layer 24 (middle) as representative
        for layer_id in [0, 24, 47]:
            loads = per_rank_load[layer_id].tolist()
            total = sum(loads)
            if total == 0:
                continue
            avg = total / self.ep_size
            ratios = [l / max(avg, 1) for l in loads]
            max_r = max(ratios)
            std_r = (sum((r - 1.0)**2 for r in ratios) / len(ratios)) ** 0.5
            logger.info(
                f"[PB-OEPLB-HEATMAP] window#{self.window_count} layer={layer_id} "
                f"rank_loads={[int(l) for l in loads]} "
                f"ratios=[{', '.join(f'{r:.2f}' for r in ratios)}] "
                f"max_ratio={max_r:.3f} std={std_r:.3f}"
            )

    def _maybe_report_profile(self):
        """Log cumulative wall-clock cost breakdown every 4 windows."""
        if self.window_count == 0 or self.window_count % 4 != 0:
            return
        record_ms = self._prof_record_ns / 1e6
        allreduce_ms = self._prof_allreduce_ns / 1e6
        planbuild_ms = self._prof_planbuild_ns / 1e6
        finalize_ms = self._prof_finalize_ns / 1e6
        total_ms = record_ms + allreduce_ms + planbuild_ms + finalize_ms
        logger.info(
            f"[PB-OEPLB-PROF] window#{self.window_count} calls={self._prof_record_calls} "
            f"record={record_ms:.2f}ms allreduce={allreduce_ms:.2f}ms "
            f"planbuild={planbuild_ms:.2f}ms finalize={finalize_ms:.2f}ms "
            f"total={total_ms:.2f}ms avg_record_us={1000*record_ms/max(1,self._prof_record_calls):.2f}"
        )

    def _try_finish_pending_swap(self, force_wait: bool = False):
        try:
            plan = self.async_executor.try_finish(force_wait=force_wait)
        except Exception as e:
            logger.error(f"[PB-OEPLB] finish error: {e}\n{traceback.format_exc()}")
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

            # BUGFIX: remap self.load (decay history) to match the new placement.
            # Without this, decay history has counts indexed by OLD physical slots,
            # but new forward passes record into slots under the NEW placement.
            # The mismatch makes ratio_before stuck at the pre-swap level because
            # 90% of the signal (decay history) still reflects the old layout.
            # Fix: for each swapped pair, swap their counts in self.load too,
            # so the history "follows" the experts to their new physical slots.
            for op in plan:
                layer = op.layer_id
                a, b = op.phys_slot_a, op.phys_slot_b
                self.load[layer, a], self.load[layer, b] = (
                    self.load[layer, b].clone(), self.load[layer, a].clone()
                )

            self._prof_finalize_ns += time.perf_counter_ns() - _tf
            total_ms = (time.perf_counter() - self._pending_plan_start_t) * 1000
            self.total_swaps += len(plan)
            logger.info(f"[PB-OEPLB] {len(plan)} swap(s) done ({total_ms:.1f}ms) | total={self.total_swaps}")
        except Exception as e:
            logger.error(f"[PB-OEPLB] finalize error: {e}\n{traceback.format_exc()}")
