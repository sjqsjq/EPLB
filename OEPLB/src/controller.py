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



class PBOEPLBController:
    """
    在线 MoE expert 负载均衡状态机：周期性检查各 rank 的专家负载不均衡度，
    超过阈值时通过 rank 间 P2P 交换物理专家权重来纠正。当前架构的关键设计点：

    1. 本地计数器触发，零额外通信。每个 rank 只看自己的本地 forward 计数器
       (`_steps_since_last_check`) 判断"是否到 sync_window"，不需要每次
       forward 都做跨 rank 共识。这样做是安全的：forward pass 本身就是
       DP+EP 全体 rank 隐式同步的（即使是 IDLE batch，每个 rank 每个 global
       forward step 也恰好调用一次 `on_forward_pass_end`），所以本地计数器
       天然和其他 rank 保持一致，不需要额外的 all_reduce 来对齐"什么时候
       检查"。每 `sync_window` 个 forward 才做一次 all_reduce(SUM)，这一次
       collective 同时完成"聚合全局负载"和"判断是否积累够 token"两件事，
       不需要单独一轮"是否ready"的共识。这个设计对齐了 SGLang 官方 EPLB
       自己的本地计数器触发机制(`eplb_manager.py`)。

    2. 物理空间直接记录，最小化 hot path 开销。`record_next_layer` 直接对
       physical slot 做一次 `scatter_add_`，不做逐次的物理↔逻辑 id 转换、
       不做 clamp_、不做单独的 bincount+add_。物理转逻辑的转换只在每个
       sync_window 做一次向量化批处理（覆盖所有层），而不是每个 prefill
       batch 每层都做一次。这样设计的原因：更早期的写法每次
       `record_next_layer` 调用要花 ~800-1000us（reshape+long()+clamp_+
       fancy-index gather+bincount+add_，5-6 次独立 CUDA kernel launch，
       每次都要在单线程 CPU 调度器的关键路径上付 dispatch 开销），比它想要
       摊薄的 all_reduce 本身还贵 5-6 倍（370ms allreduce vs 单次 benchmark
       累计 ~1.8-2.0s 的 record 开销）。现在的写法直接匹配 SGLang 官方
       EPLB 自己的热路径(`_SelectExpertsSinglePassGatherer.on_select_experts`)：
       纯本地 scatter，零通信。

    3. 指数衰减代替硬清零。每个 sync_window 做完决策后，load 历史按
       `decay_factor`（默认 0.9）衰减而不是清零。这样单批次的路由噪声不会
       主导决策，但最近的负载模式依然占主导——在"响应速度"和"抗噪声"之间
       取一个折中，而不是二选一。

    4. 异步 P2P swap + 精确一处的强制同步，避免死锁。swap 操作在独立的低
       优先级 CUDA stream 上发起，非阻塞返回；`try_finish()` 默认只做非阻塞
       的 `event.query()` 检查，真正确认 GPU 上传输完成才做 shadow buffer
       → live weight 的拷贝并翻转路由表。但在每轮决策发起下一次 all_reduce
       之前，必须先用 `force_wait=True` 阻塞等上一轮 P2P 传输在 GPU 上真正
       确认完成——原因是 NCCL 要求同一个通信组里所有 rank 发起 collective
       的"相对顺序"完全一致，不管是哪个本地 CUDA stream 发起的；如果一个
       rank 的 CPU 线程先跑到了下一轮的 all_reduce（默认 stream），而另一个
       rank 上一轮的 P2P op（低优先级 stream）还没真正提交给 NCCL，两个
       rank 在 NCCL 眼里的 op 序列类型就会分叉（一个是"collective"，另一个
       还是"P2P"），通信组永远等不到对方，直接死锁。这个强制同步只发生在
       这一个点上，其余每次 forward pass 之间仍然走非阻塞的快路径。

    5. 只在 prefill 阶段记录路由决策（OEPLB 相对官方 EPLB 的差异化设计）。
       官方 EPLB 在 prefill 和 decode 阶段均匀记录；OEPLB 只记录 prefill，
       因为 prefill 阶段的专家热度分布决定了紧随其后的 decode 阶段会调用
       哪些专家——这是"提前纠偏"而不是"事后纠偏"。
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
        self._sample_interval = cfg.sample_interval
        self._should_record_this_batch = False

        self.async_executor = AsyncSwapExecutor(
            model_runner, self.model_runner.moe_ep_rank, self.num_local
        )
        self._pending_plan_start_t = None
        self._prewarmed = (self.dp_size > 1)

        # Pure local counters — no cross-rank consensus needed (see class docstring)
        self._forward_id = 0
        self._ready = False
        self._warmup_forwards = cfg.warmup_forwards
        self._steps_since_last_check = 0
        self._decay_factor = cfg.decay_factor
        # Adaptive window (experimental, opt-in via cfg.adaptive_window): shrink
        # sync_window temporarily during a CONFIRMED workload shift (>=2
        # consecutive low-cos_sim windows, not a single blip), grow it back once
        # stable (>=2 consecutive high-cos_sim windows). See _decide_and_begin_swap
        # for where this is updated and on_forward_pass_end for where it's read.
        self._effective_sync_window = cfg.sync_window
        self._last_cos_sim = None
        self._window_shift_count = 0
        self._window_stable_count = 0
        self._last_avg_ratio = None

        # Sensitivity calibration (opt-in via cfg.calibrate_adaptive_sensitivity,
        # requires adaptive_window=True). Measures the prefill:decode
        # forward-pass ratio over the first `calibration_forwards` non-idle
        # forwards, then picks a sensitivity tier (see _apply_sensitivity_tier)
        # that overrides window_floor/window_shift_confirm_windows/
        # window_stable_confirm_windows for the rest of this run. Deliberately
        # does NOT touch sync_window itself -- validated empirically (L512,
        # O=1/64/256, bracketed baseline measurements) that the achievable gain
        # magnitude scales cleanly with this ratio (~+18% at O=1 down to ~+4%
        # at O=256) but WHICH static window is best does not correlate with it.
        self._calib_done = not (cfg.adaptive_window and cfg.calibrate_adaptive_sensitivity)
        self._calib_prefill_fwd = 0
        self._calib_decode_fwd = 0

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

        if not self._calib_done:
            self._update_sensitivity_calibration(is_idle, is_prefill)

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
        self._decay_factor = self.cfg.decay_factor

        self._decide_and_begin_swap()
        # Exponential decay instead of zeroing: preserve routing history
        # so swap decisions reflect the stable/dominant pattern, not
        # single-window noise. decay=0.3 means 70% of current window's
        # signal is retained, 30% of history bleeds through.
        self.load.copy_((self.load.float() * self._decay_factor).long())
        self.total_tokens = 0
        self._prefill_batch_counter = 0

    def _update_sensitivity_calibration(self, is_idle: bool, is_prefill: bool):
        """Count prefill vs decode forward passes (non-idle only, mirrors the
        denominator used elsewhere for recording decisions), then once the
        calibration window has elapsed, all_reduce the counts across ranks and
        pick a sensitivity tier from the GLOBAL aggregate. See
        _apply_sensitivity_tier for the empirical basis.

        BUGFIX (found via live test on 2026-07-30): trigger timing must be a
        purely local step-counter check (self._forward_id increments
        unconditionally every call, so it's implicitly lockstep-synchronized
        across ranks -- same pattern as _steps_since_last_check), and the
        decision must be made from all-reduced GLOBAL counts, not each rank's
        own local counts. In DP mode, different ranks can genuinely be in
        different phases (prefill vs decode) at the same global step, so local
        counts are real per-rank views, not just noise around a shared number.
        Confirmed empirically: on a near-0.5-boundary workload, two ranks
        computed decode_fraction=0.496 and 0.500 from their own local counts
        and picked DIFFERENT tiers -- which would have desynchronized
        window_floor/window_shift_confirm_windows across ranks, breaking the
        lockstep invariant every other part of this controller depends on."""
        if not is_idle:
            if is_prefill:
                self._calib_prefill_fwd += 1
            else:
                self._calib_decode_fwd += 1
        if self._forward_id < self._warmup_forwards + self.cfg.calibration_forwards:
            return
        counts = torch.tensor(
            [self._calib_prefill_fwd, self._calib_decode_fwd],
            dtype=torch.float32, device=self.load.device,
        )
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        total = counts[0].item() + counts[1].item()
        decode_fraction = counts[1].item() / total if total > 0 else 0.0
        self._apply_sensitivity_tier(decode_fraction)
        self._warm_start_placement()
        self._calib_done = True

    def _apply_sensitivity_tier(self, decode_fraction: float):
        """Pick how aggressively adaptive_window should react, based on the
        measured prefill:decode forward-pass ratio -- NOT which sync_window
        value to use.

        Empirical basis (bracketed baseline measurements, L=512, final_grid
        O=1/64/256, this-session validation -- see COMPREHENSIVE_EXPERIMENT_LOG
        entry for the full data): achievable OEPLB gain over baseline scales
        cleanly with output length / decode-heaviness (~+14-22% at O=1,
        ~+8-10% at O=64, ~+3-4.5% at O=256), but which specific static
        sync_window wins does NOT correlate with this ratio (differences
        between windows within a single O tier were 1.5-8 points, no
        monotonic trend). So this tier selection only touches how eagerly
        adaptive_window shrinks/holds its window, not sync_window itself.

        Tier boundaries calibrated from direct measurement on this session's
        validation runs (L512, final_grid, same tuned params, calibration_forwards=256):
        O=1 -> decode_fraction~0.03, O=64 -> ~0.78, O=256 -> ~0.93. Note these are
        NOT evenly spread -- decode_fraction saturates toward 1.0 quickly once O
        exceeds single digits (each request contributes one prefill-forward-pass
        share but O decode-forward-pass shares), so the O=64/O=256 boundary sits
        much higher (~0.86, roughly midway between the two measured values) than
        a naive guess would suggest."""
        if decode_fraction < 0.5:
            tier = "prefill-heavy"
            self.cfg.window_floor = 8
            self.cfg.window_shift_confirm_windows = 1
        elif decode_fraction < 0.86:
            tier = "balanced"
            self.cfg.window_floor = 32
            self.cfg.window_shift_confirm_windows = 1
        else:
            tier = "decode-heavy"
            self.cfg.window_floor = self.cfg.sync_window  # effectively disables shrinking
            self.cfg.window_shift_confirm_windows = 3
        logger.info(f"[PB-OEPLB-CALIB] rank={self._rank} GLOBAL decode_fraction={decode_fraction:.3f} "
                    f"(this rank's local counts: prefill_fwd={self._calib_prefill_fwd}, "
                    f"decode_fwd={self._calib_decode_fwd}) "
                    f"-> tier={tier}, window_floor={self.cfg.window_floor}, "
                    f"shift_confirm={self.cfg.window_shift_confirm_windows}")

    def _warm_start_placement(self):
        """After calibration, use the accumulated load stats to compute a
        better-than-trivial initial placement via EPLB's greedy algorithm,
        then apply it using SGLang's official ExpertLocationUpdater (which
        handles weight movement layer-by-layer, avoiding the superlinear
        overhead of batching 200+ P2P ops into one massive batch_isend_irecv).

        This replaces the expensive cold-start swap burst (previously 223 ops,
        1.8s of batch_isend_irecv overhead alone) with a single bulk re-assignment
        that uses the same code path as the official EPLB rebalance."""
        try:
            import time as _time
            _t0 = _time.perf_counter()
            global_load = self.load.clone()
            torch.distributed.all_reduce(global_load, op=torch.distributed.ReduceOp.SUM)

            if global_load.sum() == 0:
                logger.info("[PB-OEPLB-WARMSTART] no load data yet, skipping warm-start")
                return

            from sglang.srt.eplb import eplb_algorithms
            current_p2l = self._meta.physical_to_logical_map

            physical_to_logical_map_new, _, _ = eplb_algorithms.rebalance_experts(
                tokens_per_expert=global_load.unsqueeze(0),
                num_physical_experts=self.num_local * self.ep_size,
                num_local_physical_experts=self.num_local,
                num_groups=None,
                num_nodes=1,
                algorithm=eplb_algorithms.EplbAlgorithm.deepseek,
            )
            physical_to_logical_map_new = physical_to_logical_map_new.squeeze(0).to(current_p2l.device)

            # Check if the new placement is actually different
            changed_layers = (physical_to_logical_map_new != current_p2l).any(dim=1)
            num_changed = changed_layers.sum().item()
            if num_changed == 0:
                logger.info("[PB-OEPLB-WARMSTART] placement unchanged, skipping")
                return

            update_layer_ids = changed_layers.nonzero(as_tuple=True)[0].tolist()
            new_meta = fast_init_by_mapping(physical_to_logical_map_new, self.num_logical_experts)

            # Use SGLang's official ExpertLocationUpdater (layer-by-layer P2P,
            # NOT our AsyncSwapExecutor's single massive batch)
            self.model_runner.update_expert_location(new_meta, update_layer_ids)
            self._meta = self._fetch_metadata()
            self._cached_p2l = self._meta.physical_to_logical_map

            # Reset load to match new placement (old counts are meaningless now)
            self.load.zero_()
            self.total_tokens = 0

            elapsed = (_time.perf_counter() - _t0) * 1000
            logger.info(f"[PB-OEPLB-WARMSTART] bulk re-placement done: "
                       f"{num_changed} layers changed in {elapsed:.1f}ms")
        except Exception as e:
            logger.warning(f"[PB-OEPLB-WARMSTART] failed (non-fatal, falling back to "
                          f"incremental swap): {e}")

    def _decide_and_begin_swap(self):
        if not self._calib_done:
            return

        # Hard-reset detection: if ratio jumped significantly after a period
        # of convergence, the workload domain likely shifted. Clear the decayed
        # history (which is now contaminated with old-domain signal) and let
        # this window's fresh data speak for itself on the NEXT window.
        if not hasattr(self, '_prev_window_ratio'):
            self._prev_window_ratio = None
            self._converged_ratio = None
            self._skip_next_for_reset = False

        if self._skip_next_for_reset:
            self._skip_next_for_reset = False
            # This window has clean-only data after the reset. Proceed normally.

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

            # Compute current avg_ratio for hard-reset detection
            _layer_ratios = []
            for _lid in range(self.num_layers):
                _lc = global_load[_lid]
                _loads = [_lc[r*self.num_local:(r+1)*self.num_local].sum().item() for r in range(self.ep_size)]
                _avg = max(sum(_loads) / self.ep_size, 1.0)
                _layer_ratios.append(max(_loads) / _avg)
            _current_avg_ratio = sum(_layer_ratios) / len(_layer_ratios) if _layer_ratios else 1.0

            if self._prev_window_ratio is not None:
                # Track "converged" as the best ratio we've seen recently
                if self._converged_ratio is None or self._prev_window_ratio < self._converged_ratio + 0.005:
                    self._converged_ratio = min(self._converged_ratio or 99, self._prev_window_ratio)

                # Detect domain shift: ratio jumped >0.03 above our converged level
                if (self._converged_ratio is not None and
                    _current_avg_ratio > self._converged_ratio + 0.03 and
                    self._prev_window_ratio < self._converged_ratio + 0.015):
                    logger.info(f"[PB-OEPLB-RESET] domain shift detected: "
                               f"converged={self._converged_ratio:.3f} "
                               f"prev={self._prev_window_ratio:.3f} "
                               f"current={_current_avg_ratio:.3f} "
                               f"-> zeroing load history for clean re-profiling")
                    self.load.zero_()
                    self.total_tokens = 0
                    self._converged_ratio = None
                    self._prev_window_ratio = _current_avg_ratio
                    self._skip_next_for_reset = True
                    return
            self._prev_window_ratio = _current_avg_ratio
            self._prof_allreduce_ns += time.perf_counter_ns() - _t0

            global_tokens = int(global_load.sum().item())
            # Track current avg imbalance ratio for adaptive window grow decision
            from sglang.srt.managers.pb_oeplb.rebalancer import compute_gpu_load
            _ratios = []
            for _lid in range(0, self.num_layers, max(1, self.num_layers // 8)):
                _gl = compute_gpu_load(global_load[_lid], self._cached_p2l[_lid], self.ep_size, self.num_local)
                _avg = max(float(_gl.float().mean().item()), 1.0)
                _ratios.append(float(_gl.max().item()) / _avg)
            self._last_avg_ratio = sum(_ratios) / len(_ratios) if _ratios else None
            self._maybe_report_profile()
            # Track drift every window (even quiet ones) so adaptive window sizing
            # has an unbroken cos_sim history to react to.
            self._last_cos_sim = self._track_routing_stability(global_load)
            if self.cfg.adaptive_window and self._last_cos_sim is not None:
                window_ceiling = self.cfg.sync_window * 2
                if self._last_cos_sim < self.cfg.window_shift_cos_threshold:
                    self._window_shift_count += 1
                    self._window_stable_count = 0
                    if self._window_shift_count >= self.cfg.window_shift_confirm_windows:
                        new_window = max(self.cfg.window_floor,
                                         self._effective_sync_window // 2)
                        if new_window != self._effective_sync_window:
                            logger.info(f"[PB-OEPLB-WINDOW] shift confirmed "
                                        f"({self._window_shift_count} low-cos_sim "
                                        f"window(s)) -- halving sync_window "
                                        f"{self._effective_sync_window} -> {new_window}")
                        self._effective_sync_window = new_window
                elif self._last_cos_sim > self.cfg.window_stable_cos_threshold:
                    self._window_stable_count += 1
                    self._window_shift_count = 0
                    # Below sync_window: recover with 2-window confirmation.
                    # At or above sync_window: grow with stricter 4-window
                    # confirmation, capped at window_ceiling (2x sync_window).
                    if self._effective_sync_window < self.cfg.sync_window:
                        confirm_needed = self.cfg.window_stable_confirm_windows
                    else:
                        confirm_needed = 4
                    # OPTIMIZATION: don't grow if imbalance ratio is still high.
                    # Stable cos_sim only means the PATTERN isn't changing, not
                    # that the imbalance is resolved. If ratio is still above
                    # threshold, there's ongoing value in frequent checks.
                    if self._last_avg_ratio is not None and self._last_avg_ratio > self.cfg.threshold_ratio * 1.05:
                        self._window_stable_count = 0  # reset, don't grow yet
                    elif self._window_stable_count >= confirm_needed:
                        new_window = min(window_ceiling,
                                         self._effective_sync_window * 2)
                        if new_window != self._effective_sync_window:
                            logger.info(f"[PB-OEPLB-WINDOW] stable confirmed "
                                        f"({self._window_stable_count} consecutive "
                                        f"high-cos_sim windows) -- doubling sync_window "
                                        f"{self._effective_sync_window} -> {new_window}")
                        self._effective_sync_window = new_window
                        self._window_stable_count = 0
                else:
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
                max_total_ops=self.cfg.max_total_ops,
            )
            self._prof_planbuild_ns += time.perf_counter_ns() - _t1
            if not plan:
                return
            if len(plan) < self.cfg.min_swap_ops:
                # Not worth the batch_isend_irecv + cross-rank sync overhead for
                # this few ops (observed tail windows doing 1-6 swaps at ~330ms
                # each, purely from all_reduce+begin() overhead, for negligible
                # further ratio improvement once the bulk correction already
                # landed in window#1/#2). Decision was already made (DIAG logged
                # above), we just skip issuing the P2P transfer itself.
                logger.info(f"[PB-OEPLB] window#{self.window_count}: skipped "
                           f"{len(plan)} swap(s) (< min_swap_ops={self.cfg.min_swap_ops}, "
                           f"not worth the P2P overhead)")
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
