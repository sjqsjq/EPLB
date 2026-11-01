import logging
import math
import time
import torch
import torch.distributed
import traceback

from sglang.srt.managers.pb_oeplb.config import PBOEPLBConfig
from sglang.srt.managers.pb_oeplb.rebalancer import try_build_swap_plan, try_build_swap_plan_best_improvement
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
        self._routed_weights_cache = None
        self.num_layers = self._meta.num_layers
        self.num_logical_experts = self._meta.num_logical_experts
        self.ep_size = self._meta.ep_size
        self.num_local = self._meta.num_local_physical_experts

        self.dp_size = getattr(model_runner.server_args, 'dp_size', 1)

        self.num_physical_experts = self._meta.num_physical_experts

        # BUGFIX: models with leading dense layers (e.g. DeepSeek-V2/V3's
        # first_k_dense_replace) have fewer real MoE layers than
        # self.num_layers (=metadata's num_hidden_layers, which counts the
        # dense layers too). record_next_layer() is only invoked by
        # select_experts(), which never fires for a dense layer -- so a
        # naive `_layer_counter % self.num_layers` cycle mis-shifts every
        # recorded index by the number of leading dense layers, silently
        # writing MoE layer K's data into self.load[K-1] instead of
        # self.load[K]. This desyncs self.load from physical_to_logical_map
        # (which IS indexed by real transformer layer_id), and downstream
        # swap plans end up citing dense layer_ids that have no entry in
        # model_runner.model.routed_experts_weights_of_layer -> KeyError at
        # swap execution time. Fix: derive the leading-dense-layer offset
        # from the actual set of layer_ids that have routed expert weights,
        # and shift record_next_layer's cycle to start there (see below).
        # Computed lazily on first record_next_layer() call (NOT here): at
        # __init__ time, initialize() hasn't called load_model() yet (see
        # model_runner.py's initialize(): controller is constructed before
        # self.model exists), so self.model_runner.model.routed_experts_
        # weights_of_layer would raise/be absent here. record_next_layer()
        # is only ever invoked from a live forward pass, by which point the
        # model is guaranteed fully loaded.
        self._layer_offset = None

        # v0.10: load is recorded in PHYSICAL expert space (see class
        # docstring) — converted to logical space once per sync_window,
        # not once per record_next_layer call.
        self.load = torch.zeros(self.num_layers, self.num_physical_experts,
                                dtype=torch.int64, device="cuda")

        # Pre-allocated buffers for record_next_layer hot path.
        # Avoids creating 4 temporary tensors per call (reshape+mask+masked_fill+long),
        # cutting per-call overhead from ~100us to ~25us (1 kernel launch vs 4).
        # Sized for max_batch * top_k; resized on demand if exceeded.
        self._record_index_buf = torch.zeros(1, dtype=torch.int64, device="cuda")
        self._record_src_buf = torch.ones(1, dtype=torch.int64, device="cuda")
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
        # final_plan_20261019 arm flags (all default OFF => sealed rev1
        # behavior bit-identical; see results/final_plan_20261019/C_configs)
        self._record_all_phase = bool(int(getattr(cfg, "record_all_phase", 0)))
        self._no_commit = bool(int(getattr(cfg, "no_commit", 0)))
        self._nocommit_planned_ops = 0
        self._decode_record_batches = 0
        # rank gate not yet available here (_rank assigned later); boot-time
        # echo on every rank is acceptable (8 lines, once per launch).
        logger.info(
            "[PB-OEPLB-ARMFLAGS] sp_mode=%s sp_s=%s sp_h=%s sp_w=%s "
            "sp_rev=%s force_s=%s record_all_phase=%s no_commit=%s "
            "pair_rule=%s",
            getattr(cfg, "sp_mode", "?"), getattr(cfg, "sp_s", "?"),
            getattr(cfg, "sp_h", 0), getattr(cfg, "sp_w", "?"),
            getattr(cfg, "sp_rev", 0), int(getattr(cfg, "sp_force_s", 0)),
            int(self._record_all_phase), int(self._no_commit),
            getattr(cfg, "pair_rule", "gap"))

        self.async_executor = AsyncSwapExecutor(
            model_runner, self.model_runner.moe_ep_rank, self.num_local
        )
        self._pending_plan_start_t = None
        # Swap-budget accounting (cfg.swap_budget_frac). _cum_swap_s is this
        # rank's own blocking time inside batch_isend_irecv, which is what
        # competes with the headroom -- not the sum across ranks.
        self._serving_t0 = time.perf_counter()
        self._cum_swap_s = 0.0
        self._budget_warned = False
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
        # Auto-compute dead zone ratio from EP power law (§2.4, Appendix G):
        # r_k - 1 = 0.00408 * EP^1.52, calibrated from 4 configs incl. cross-model blind test (±3-5%)
        if cfg.dead_zone_ratio <= 0.0:
            auto_rk = 1.0 + 0.00408 * (self.ep_size ** 1.52)
            cfg.dead_zone_ratio = auto_rk
            logger.info(f"[PB-OEPLB] auto dead_zone_ratio = {auto_rk:.3f} from EP power law (EP={self.ep_size})")
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
        # C7 (E-A switch decoupling, 2026-10-15): the calibration warmup is its
        # own mechanism; gate it on its own flag instead of adaptive_window.
        # With calibrate_adaptive_sensitivity=False (as launched in v3 and in
        # all planned E-A arms) the value is True either way -> A11 identical.
        self._calib_done = not cfg.calibrate_adaptive_sensitivity
        self._calib_prefill_fwd = 0
        self._calib_decode_fwd = 0

        # --- ADW v2 single-authority window controller (see window_v2.py and
        # results/adw_v2/adw_v2_mechanism_spec.md). version 1 (default) keeps
        # every legacy path byte-identical; version 2 short-circuits ALL legacy
        # window mutation sites and routes the single W decision through
        # _window_control_v2 below. ---
        self._wctrl_version = int(getattr(cfg, 'window_controller_version', 1))
        self._v2_ctrl = None
        self._v2_ref = None
        self._v2_carry = False
        self._v2_hash_count = 0
        self._v2_log_fh = None
        self._last_window_steps = None
        # sp_coupling_v1 defaults. Built only under window_controller=21;
        # SP_MODE=off leaves every SP path inert and the legacy path
        # bit-identical (PD-12/PD-14).
        self._sp = None
        self._sp_active = False
        self._sp_fwd = 0              # PD-9 all-rank lockstep forward number
        self._sp_h_cycle = 0          # H-boundary (detection cycle) id
        self._steps_since_h = 0
        self._steps_since_w = 0
        self._sp_opps_h = 0           # PD-13: S-gate opportunities, this H cycle
        self._sp_recorded_h = 0       # PD-13: recorded forwards, this H cycle
        self._sp_opps_total = 0
        self._sp_recorded_total = 0
        self._sp_swaps_total = 0
        self._sp_swap_started_h = 0   # plan starts since previous H proposal
        self._sp_plan_opp_h = False   # a W boundary fired within this H cycle
        self._sp_recorded_flag = False
        self._sp_cover_w = 0          # W-steps covered at boundary (seal log)
        self._sp_seal_log = []
        self._sp_last_commit = None
        self._sp_n_min_full = 0
        self._sp_dumped = False
        self._sp_valid_rows_h = None  # GPU 0-dim accumulator (built in v2p1 init)
        self._prof_sp_gate_ns = 0
        self._v2p1_fatal_flag = False
        if self._wctrl_version == 2:
            self._init_window_control_v2(cfg)
        elif self._wctrl_version == 21:
            self._init_window_control_v2p1(cfg)
        elif self._wctrl_version != 1:
            raise ValueError(
                "OEPLB_WINDOW_CONTROLLER=%r: supported versions are 1 "
                "(legacy), 2 (frozen ADW v2), 21 (ADW v2.1 bounded revision)"
                % self._wctrl_version)

        # Profiling counters (wall-clock, CPU-side critical-path time)
        self._prof_record_calls = 0
        self._prof_record_ns = 0
        self._prof_allreduce_ns = 0
        self._prof_planbuild_ns = 0
        self._prof_finalize_ns = 0
        self._prof_last_report_forward = 0
        # v2.2-calib (Route-B probe, spec v1.1-calib) instrumentation --
        # decision tier d1-d3 / diag tier g1. _true_steps_total: independent
        # true-forward counter (d2), advances in lockstep with
        # _steps_since_last_check after ready; carried in the extended confirm
        # reduce for the MIN==MAX lockstep audit.
        self._true_steps_total = 0
        self._prof_probe_ns = 0             # probe-instrument self-cost ledger
        self._prev_boundary_mono_ns = None  # d3: previous boundary monotonic stamp
        self._v22_last_bcost_ns = 0         # g1: previous boundary full wall (ns)
        self._probe_dumped = False

        self._prev_load = None  # for routing stability tracking
        self._stability_history = []  # cosine similarities
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(f"[PB-OEPLB] Init v0.11 (fast-response + decay): "
                    f"layers={self.num_layers}, experts={self.num_logical_experts}, "
                    f"ep={self.ep_size}, dp={self.dp_size}, local={self.num_local}, "
                    f"thresh={cfg.threshold_ratio}, min_tok={cfg.min_prefill_tokens}, "
                    f"sync_window={cfg.sync_window}, always_record={cfg.always_record}")
        logger.info("[PB-OEPLB-PLANNER] rule="
                    + ("best_improvement_rank_pair (P11 control)"
                       if getattr(cfg, 'planner_best_improvement', False)
                       else "heuristic_first_improvement (default)"))

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
        if getattr(self, '_wctrl_version', 1) == 21:
            # ADW v2.1 (F1): net-zero warmup relayout, but bump the mapping
            # version and clear the (still empty pre-ready) detection
            # accumulator so versions stay locked by construction.
            self._p2l_version += 1
            if getattr(self, '_det_load', None) is not None:
                self._det_load.zero_()
            self._det_map_version = self._p2l_version

    def _fetch_metadata(self):
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        return get_global_expert_location_metadata()


    def _get_routed_experts_weights(self):
        """Get {layer_id: List[Tensor]} for all MoE layers, model-agnostic.

        DeepSeek-V2/V3 models expose `model.routed_experts_weights_of_layer`
        directly (a LazyValue). Other MoE models (Qwen2-MoE, Qwen3-MoE, etc.)
        don't expose this attribute, but their MoE layer class implements
        `get_moe_weights()`. This helper provides a uniform interface so swap
        execution works across all MoE architectures, not just DeepSeek.

        Returns a dict {layer_id: List[Tensor]} where each Tensor is one
        weight matrix of the experts on that layer (shape: (num_local, ...)).
        Cached after first successful call.
        """
        if self._routed_weights_cache is not None:
            return self._routed_weights_cache
        model = self.model_runner.model
        # Try the native DeepSeek attribute first
        if hasattr(model, "routed_experts_weights_of_layer"):
            self._routed_weights_cache = model.routed_experts_weights_of_layer
            return self._routed_weights_cache
        # Fallback: build it by walking model.layers and calling get_moe_weights()
        result = {}
        layers = getattr(getattr(model, "model", model), "layers", None)
        if layers is None:
            raise RuntimeError(
                "PB-OEPLB: cannot find model.layers to build routed expert weights"
            )
        for layer_id, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "get_moe_weights"):
                result[layer_id] = mlp.get_moe_weights()
        if not result:
            raise RuntimeError(
                "PB-OEPLB: no layer with get_moe_weights() found; "
                "this model architecture may not be supported"
            )
        self._routed_weights_cache = result
        logger.info(f"[PB-OEPLB] built routed_experts_weights via fallback: "
                    f"{len(result)} MoE layers")
        return self._routed_weights_cache

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

        if self._layer_offset is None:
            try:
                _routed_layer_ids = sorted(
                    self._get_routed_experts_weights().keys()
                )
                self._layer_offset = _routed_layer_ids[0] if _routed_layer_ids else 0
            except Exception:
                self._layer_offset = 0
            if self._layer_offset:
                logger.info(f"[PB-OEPLB] detected {self._layer_offset} leading "
                           f"dense layer(s); record_next_layer offset accordingly")

        _t0 = time.perf_counter_ns()
        _num_moe_layers = max(1, self.num_layers - self._layer_offset)
        layer_id = self._layer_offset + (self._layer_counter % _num_moe_layers)
        self._layer_counter += 1
        flat = topk_ids.reshape(-1)
        mask = flat != -1
        self.load[layer_id].scatter_add_(
            dim=0, index=flat.masked_fill(~mask, 0).long(), src=mask.long()
        )
        if self._wctrl_version == 21:
            # ADW v2.1 (F1): separate detection accumulator. Mirrors the
            # count-follows-expert remap at every mapping-commit site, so the
            # window-end logical gather under the CURRENT p2l is exact. The
            # planner accumulator (self.load) is untouched by v2.1.
            self._det_load[layer_id].scatter_add_(
                dim=0, index=flat.masked_fill(~mask, 0).long(), src=mask.long()
            )
        if self._sp_active:
            # sp_coupling_v1 PD-13: separated ledgers. Opportunities are
            # counted at the gate (on_forward_pass_start), recorded forwards
            # at forward end, valid rows accumulate ON GPU and are read once
            # per H boundary (no per-call sync). Never folded into each other.
            self._sp_recorded_flag = True
            self._sp_valid_rows_h += mask.sum()
        if layer_id == 0:
            self.total_tokens += topk_ids.shape[0]
        self._prof_record_calls += 1
        self._prof_record_ns += time.perf_counter_ns() - _t0

    def on_forward_pass_start(self, forward_batch):
        """sp_coupling_v1 (plan Sec.2, PD-9/PD-15): PRE-forward sampling gate.

        Called from model_runner.forward immediately after forward_pass_id
        += 1 and BEFORE this forward's MoE record hooks. SP modes only --
        with SP_MODE=off this is a no-op and the legacy post-hoc decision in
        on_forward_pass_end stays bit-identical.

        _sp_fwd advances on EVERY global forward on EVERY rank (IDLE dummy
        batches included), i.e. in the all-rank-consistent global forward
        domain the plan requires; the gate is a deterministic function of
        (SP_SEED, _sp_fwd, committed S), identical on every rank with no
        communication. Inside a selected opportunity the actual recording
        still follows real backend semantics (non-capture, prefill/extend,
        >= min_record_tokens, valid-row mask); idle ranks contribute zero
        rows instead of diverging the schedule. Samples are never inflated
        by S: a skipped forward is skipped for BOTH accumulators (the single
        gate in record_next_layer covers the planner and detection
        scatters)."""
        if not self._sp_active:
            return
        _tg = time.perf_counter_ns()
        self._sp_fwd += 1
        gate = self._sp.gate(self._sp_fwd)
        if gate:
            self._sp_opps_h += 1
            self._sp_opps_total += 1
        is_idle = forward_batch.forward_mode.is_idle()
        is_prefill = forward_batch.forward_mode.is_extend()
        self._should_record_this_batch = bool(
            gate and (not is_idle)
            and (is_prefill or self._record_all_phase))
        if (self._should_record_this_batch and self._record_all_phase
                and not is_prefill):
            self._decode_record_batches += 1
        self._prof_sp_gate_ns += time.perf_counter_ns() - _tg

    def _sp_boundary_dispatch(self):
        """sp_coupling_v1: split H/W boundary schedule (plan Sec.3/Sec.4.2).

        _steps_since_h/_steps_since_w advance in lockstep on every global
        forward after ready. The state table guarantees W % H == 0 and
        committed transitions restart both counters, so W boundaries are a
        strict subset of H boundaries and the per-rank collective schedule
        is a fixed function of the boundary type alone:
          H-only : [detSUM, confirmMIN, confirmMAX]   (detection cycle)
          H+W    : [plannerSUM, detSUM, confirmMIN, confirmMAX]
                   (existing _decide_and_begin_swap order :852 -> :967)
        """
        self._steps_since_h += 1
        self._steps_since_w += 1
        h_eff = self._sp.H
        w_eff = self._sp.W
        hb = self._steps_since_h >= h_eff
        wb = self._steps_since_w >= w_eff
        if wb and not hb:
            # mechanism error (W % H == 0 is enforced at construction and
            # transitions restart both counters): fail loudly, never drift.
            raise RuntimeError(
                "[PB-OEPLB-SP] mechanism error: W boundary without H "
                "boundary (steps_h=%d/%d steps_w=%d/%d state=%s)"
                % (self._steps_since_h, h_eff, self._steps_since_w, w_eff,
                   self._sp.state))
        if not hb:
            return
        self._last_window_steps = self._steps_since_h
        self._sp_cover_w = self._steps_since_w
        self._steps_since_h = 0
        self._decay_factor = self.cfg.decay_factor
        _tb0 = time.monotonic_ns()
        if wb:
            self._steps_since_w = 0
            self._sp_plan_opp_h = True
            self._decide_and_begin_swap()
        else:
            # detection-only cycle: no planner collective, no plan. Called
            # OUTSIDE the error-swallowing try of _decide_and_begin_swap, so
            # r3 communicator-fatal semantics propagate into the scheduler
            # loop exactly as designed.
            self._window_control_v2p1(None, None)
        self._v22_last_bcost_ns = time.monotonic_ns() - _tb0
        # PD-7: a committed transition restarts BOTH counters -- the new
        # state's H/W schedule begins fresh at this boundary.
        if (self._sp_last_commit or {}).get("transition"):
            self._steps_since_h = 0
            self._steps_since_w = 0
        self._sp_post_h_boundary()
        if wb:
            self._sp_post_w_boundary()

    def _sp_post_h_boundary(self):
        """SP H-boundary bookkeeping (all ranks, local-only -- no collective
        here; runs after _window_control_v2p1 consumed/logged the ledgers)."""
        # detection accumulator carry/zero -- v21 semantics, per H cycle
        # (detection fresh stats are H-cycle scoped, plan Sec.4.2).
        if getattr(self, '_v2p1_carry', False):
            self._v2p1_carry = False
        else:
            self._det_load.zero_()
        # PD-13 per-cycle ledger reset (values already in the jsonl line).
        self._sp_opps_h = 0
        self._sp_recorded_h = 0
        if self._sp_valid_rows_h is not None:
            self._sp_valid_rows_h.zero_()
        self._sp_plan_opp_h = False

    def _sp_post_w_boundary(self):
        """SP W boundary: planner accumulator zeroing -- identical form to
        the legacy v21 boundary (decay_factor==0.0 under the v2.1 phase-1
        invariant, so this is an exact per-window zero). The planner and
        its budgets are untouched by the coupling (plan Sec.4.2)."""
        self.load.copy_((self.load.float() * self._decay_factor).long())
        self.total_tokens = 0

    def _sp_write_summary(self):
        """sp_coupling_v1 Sec.9 run-summary writer (rank0). Called
        incrementally at every committed transition (teardown is SIGKILL, so
        the timeline must not depend on atexit) and best-effort at exit."""
        try:
            import json as _json
            import os as _os
            import hashlib as _hl
            summary = {
                "mode": self._sp.mode,
                "seed": self._sp.seed,
                "state_final": self._sp.state,
                "S_final": self._sp.S, "H_final": self._sp.H,
                "W_final": self._sp.W,
                "sp_fwd_total": self._sp_fwd,
                "h_cycles": self._sp_h_cycle,
                "window_count": self.window_count,
                "true_steps_total": self._true_steps_total,
                "sample_opportunities": self._sp_opps_total,
                "recorded_steps": self._sp_recorded_total,
                "swaps_started_total": self._sp_swaps_total,
                "nocommit_planned_ops_total": self._nocommit_planned_ops,
                "decode_record_batches": self._decode_record_batches,
                "residency_committed_h": dict(self._sp.residency),
                "transitions": list(self._sp.transitions),
                "seals": list(self._sp_seal_log),
                "p2l_version": self._p2l_version,
                "p2l_fp8": _hl.blake2b(
                    self._cached_p2l.cpu().numpy().tobytes(),
                    digest_size=8).hexdigest(),
                "prof_sp_gate_ms": self._prof_sp_gate_ns / 1e6,
                "prof_record_calls": self._prof_record_calls,
            }
            path = _os.path.join(
                _os.path.dirname(self._v2p1_log_path) or ".",
                "sp_run_end.json")
            with open(path, "w") as f:
                _json.dump(summary, f, indent=1, default=str)
        except Exception:
            logger.error("[PB-OEPLB-SP] summary write failed: %s",
                         traceback.format_exc())

    def _sp_run_end_dump(self):
        if not self._sp_active or self._rank != 0 or self._sp_dumped:
            return
        self._sp_dumped = True
        self._sp_write_summary()
        logger.info("[PB-OEPLB-SP] run-end summary dumped (state=%s "
                    "transitions=%d h_cycles=%d)", self._sp.state,
                    len(self._sp.transitions), self._sp_h_cycle)

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
        if self._sp_active:
            # sp_coupling_v1: the gate for THIS forward was decided
            # PRE-forward (on_forward_pass_start, plan Sec.2). Here only the
            # PD-13 recorded-steps ledger closes; the legacy rank-local
            # prefill counter is inert (PD-15).
            if self._sp_recorded_flag:
                self._sp_recorded_h += 1
                self._sp_recorded_total += 1
                self._sp_recorded_flag = False
        elif is_idle:
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
        self._true_steps_total += 1   # d2: independent true-forward counter
        if self._sp_active:
            # sp_coupling_v1: split H/W schedule replaces the single-counter
            # boundary check (off mode: unchanged below).
            self._sp_boundary_dispatch()
            return
        if self._steps_since_last_check < self._effective_sync_window:
            return
        self._last_window_steps = self._steps_since_last_check  # actual steps of the closing window (v2 log)
        self._steps_since_last_check = 0
        self._decay_factor = self.cfg.decay_factor

        # Per PAPER.md §3.1: every sync_window forward passes, unconditionally
        # (1) checks pending P2P, (2) all_reduce, (3) rebalance, (4) async P2P.
        # No skip-cycle/stall-detection -- that was undocumented, and empirically
        # found to suppress response to LATER workload shifts (its reactivation
        # check uses a coarsely-sampled LOCAL ratio, not an all-reduced global
        # one, and in a multi-domain run it entered stall after the cold-start
        # convergence and never woke back up for the remaining domain switches).
        _tb0 = time.monotonic_ns()
        self._decide_and_begin_swap()
        # g1: full boundary wall (entry->exit of _decide_and_begin_swap, incl.
        # plan build + async begin). Readable as prev_bcost_us during the NEXT
        # window's log line; monotonic clock (comparable across processes).
        self._v22_last_bcost_ns = time.monotonic_ns() - _tb0

        if not hasattr(self, '_adw_converge_count'):
            self._adw_converge_count = 0
            self._adw_volatile_count = 0
            self._adw_last_ratio = None
        if self._wctrl_version == 1 and hasattr(self, '_prev_window_ratio') and self._prev_window_ratio is not None:
            cur_r = self._prev_window_ratio
            # Cooldown: skip ADW shrink after RESET to prevent thrashing
            if getattr(self, '_reset_cooldown', 0) > 0:
                self._reset_cooldown -= 1
                self._adw_last_ratio = cur_r
            elif self._adw_last_ratio is not None:
                delta = abs(cur_r - self._adw_last_ratio)
                # C1 (E-A switch decoupling, 2026-10-15, audit finding F1):
                # window mutations run ONLY under cfg.adaptive_window. The
                # changepoint tracking (_adw_last_ratio) and the decay->0 step
                # below stay under cfg.adaptive_decay so D works independently
                # of W. With both flags True this is behaviourally identical
                # to the v3-locked code (same mutations, same log strings).
                _adw_on = getattr(self.cfg, 'adaptive_window', False)
                if delta > 0.03:
                    if _adw_on:
                        self._effective_sync_window = max(8, self._effective_sync_window // 2)
                    self._adw_converge_count = 0
                    self._adw_volatile_count = 0
                    # Adaptive decay (opt-in via cfg.adaptive_decay): a changepoint
                    # makes the accumulated history stale, so shrinking the window
                    # alone still leaves alpha*old_signal in the accumulator (50%
                    # at alpha=0.5, needing one more window to half-decay). Zeroing
                    # the decay for this one step clears the old-domain signal
                    # immediately, cutting response latency from M*ln2 to 0. This
                    # is the alpha->0 step of §3.5; steady state restores alpha below.
                    if getattr(self.cfg, 'adaptive_decay', False):
                        self._decay_factor = 0.0
                        if _adw_on:
                            logger.info(f"[PB-OEPLB-ADW] ratio jump {delta:.3f} -> shrink sw to {self._effective_sync_window}, decay->0 (clear stale history)")
                        else:
                            logger.info(f"[PB-OEPLB-ADW] ratio jump {delta:.3f} -> decay->0 (clear stale history; window fixed)")
                    elif _adw_on:
                        logger.info(f"[PB-OEPLB-ADW] ratio jump {delta:.3f} -> shrink sw to {self._effective_sync_window}")
                elif delta < 0.003:
                    if _adw_on:
                        self._adw_converge_count += 1
                        self._adw_volatile_count = 0
                        if self._adw_converge_count >= 3:
                            new_sw = min(128, self._effective_sync_window * 2)
                            if new_sw != self._effective_sync_window:
                                self._effective_sync_window = new_sw
                                self._adw_converge_count = 0
                                logger.info(f"[PB-OEPLB-ADW] converged 3x -> grow sw to {self._effective_sync_window}")
                else:
                    if _adw_on:
                        self._adw_volatile_count += 1
                        self._adw_converge_count = 0
                        if self._adw_volatile_count >= 3:
                            new_sw = min(128, self._effective_sync_window * 2)
                            if new_sw != self._effective_sync_window:
                                self._effective_sync_window = new_sw
                                self._adw_volatile_count = 0
                                logger.info(f"[PB-OEPLB-ADW] volatile 3x -> grow sw to {self._effective_sync_window} for stability")
            self._adw_last_ratio = cur_r

        if self._wctrl_version == 2 and self._v2_carry:
            # ADW v2 UNCERTAIN carry: keep this window's fresh counts so the
            # next window merges evidence (bounded to 2 windows by the logic).
            # Deterministic across ranks (same bit-identical decision record).
            self._v2_carry = False
        else:
            self.load.copy_((self.load.float() * self._decay_factor).long())
            self.total_tokens = 0
        if self._wctrl_version == 21:
            # ADW v2.1: detection-accumulator window boundary (separate from
            # the planner history above). UNCERTAIN carry merges at most one
            # extra window (bounded by the inherited logic); carry no longer
            # touches planner state (detection/planning accumulators separate).
            if getattr(self, '_v2p1_carry', False):
                self._v2p1_carry = False
            else:
                self._det_load.zero_()
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
            new_meta = fast_init_by_mapping(physical_to_logical_map_new, self.num_logical_experts, existing_meta=self._meta)

            # Use SGLang's official ExpertLocationUpdater (layer-by-layer P2P,
            # NOT our AsyncSwapExecutor's single massive batch)
            self.model_runner.update_expert_location(new_meta, update_layer_ids)
            self._meta = self._fetch_metadata()
            self._cached_p2l = self._meta.physical_to_logical_map

            # Reset load to match new placement (old counts are meaningless now)
            self.load.zero_()
            self.total_tokens = 0
            if self._wctrl_version == 21:
                # ADW v2.1 (F1): bulk relayout invalidates accumulated counts
                # (same rule as the planner accumulator above); mapping
                # versions stay locked. A low-n_fresh window is handled by the
                # inherited UNCERTAIN path.
                self._det_load.zero_()
                self._p2l_version += 1
                self._det_map_version = self._p2l_version

            elapsed = (_time.perf_counter() - _t0) * 1000
            logger.info(f"[PB-OEPLB-WARMSTART] bulk re-placement done: "
                       f"{num_changed} layers changed in {elapsed:.1f}ms")
        except Exception as e:
            logger.warning(f"[PB-OEPLB-WARMSTART] failed (non-fatal, falling back to "
                          f"incremental swap): {e}")

    def _check_eplb_refinement(self, global_load):
        """When incremental swap reaches a plateau (ratio won't improve further
        via pairwise swaps), trigger ONE EPLB-style full re-placement to push
        ratio from ~1.07 down to ~1.00. This bridges the gap between swap-only
        optimization and the global optimum that only a full recomputation can reach.

        Only fires ONCE per serving lifetime (not repeatedly)."""
        if getattr(self, '_eplb_refined', False):
            return
        if not hasattr(self, '_plateau_count'):
            self._plateau_count = 0

        # Count consecutive windows where swap produced <8 ops (plateau)
        self._plateau_count += 1
        if self._plateau_count < 999:  # Effectively disabled
            return  # Need 3 consecutive plateau windows to confirm

        # Don't refine if ratio is already good enough
        _layer_ratios = []
        for _lid in range(self.num_layers):
            _lc = global_load[_lid]
            _loads = [_lc[r*self.num_local:(r+1)*self.num_local].sum().item() for r in range(self.ep_size)]
            _avg = max(sum(_loads) / self.ep_size, 1.0)
            _layer_ratios.append(max(_loads) / _avg)
        _avg_ratio = sum(_layer_ratios) / len(_layer_ratios)
        if _avg_ratio < 1.03:
            self._eplb_refined = True  # Already near optimal, skip
            return

        try:
            import time as _time
            _t0 = _time.perf_counter()

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

            changed_layers = (physical_to_logical_map_new != current_p2l).any(dim=1)
            num_changed = changed_layers.sum().item()
            if num_changed == 0:
                self._eplb_refined = True
                return

            update_layer_ids = changed_layers.nonzero(as_tuple=True)[0].tolist()
            new_meta = fast_init_by_mapping(physical_to_logical_map_new, self.num_logical_experts, existing_meta=self._meta)
            self.model_runner.update_expert_location(new_meta, update_layer_ids)
            self._meta = self._fetch_metadata()
            self._cached_p2l = self._meta.physical_to_logical_map
            self.load.zero_()
            self.total_tokens = 0
            if self._wctrl_version == 21:
                # ADW v2.1 (F1): see warmstart hook -- bulk relayout clears
                # the detection accumulator and bumps mapping versions.
                self._det_load.zero_()
                self._p2l_version += 1
                self._det_map_version = self._p2l_version

            elapsed = (_time.perf_counter() - _t0) * 1000
            logger.info(f"[PB-OEPLB-REFINE] EPLB refinement done: "
                       f"{num_changed} layers changed, ratio {_avg_ratio:.3f}->~1.00, "
                       f"elapsed={elapsed:.0f}ms")
            self._eplb_refined = True
        except Exception as e:
            logger.warning(f"[PB-OEPLB-REFINE] failed (non-fatal): {e}")
            self._eplb_refined = True  # Don't retry

    def _effective_threshold(self, global_tokens: int):
        """Threshold the observed window ratio must clear before we swap.

        Two corrections, both opt-in, both raising the bar:

        1. Dead zone. Wall time is flat in r below r_k (appendix G), so a swap
           that moves r from inside the dead zone to further inside it buys
           nothing and still blocks on P2P. Floor the threshold at r_k.

        2. Sampling bias. The ratio we observe is max_g/mean over per-GPU token
           counts accumulated in this window. With N tokens per layer those
           counts carry multinomial noise, and taking a max over ep_size groups
           turns that noise into a POSITIVE bias of ~c/sqrt(N) -- present even
           under perfectly uniform routing. Rather than correcting the ratio
           (which would mean touching the plan builder's per-layer arithmetic)
           we raise the threshold by the same amount, which gates identically:
               r_obs - bias > thr   <=>   r_obs > thr + bias
           c depends only on ep_size (measured 5.10 at ep=8, 2.62 at ep=4;
           order-statistics estimate sqrt(2 G ln G) gives 5.77 / 3.33).

        Returns (threshold, bias) so the caller can log both.
        """
        thr = max(self.cfg.threshold_ratio, self.cfg.dead_zone_ratio)
        bias = 0.0
        if self.cfg.bias_correct:
            c = self.cfg.bias_coeff
            if c <= 0:
                # Calibrated at window level against recorded routing counts by
                # regressing the observed ratio on 1/sqrt(N) across window sizes
                # 1..128 and four datasets: c = 5.17 +-11% at ep=8 and
                # 2.66 +-9% at ep=4, i.e. c ~= 0.65*ep_size. The order-statistics
                # estimate sqrt(2 G ln G) gives 5.77 / 3.33 -- the right shape but
                # 12-25% high, so it over-corrects. Only two ep values were
                # measured, so treat the linear form as a fit, not a law, and
                # override with bias_coeff when a config has been calibrated.
                c = 0.65 * self.ep_size
            n_per_layer = max(global_tokens / max(self.num_layers, 1), 1.0)
            bias = c / math.sqrt(n_per_layer)
            thr += bias
        return thr, bias

    def _swap_budget_exhausted(self):
        """True when cumulative swap time has eaten its allowance.

        beta*(r-r_k) bounds what perfect balancing can win, and it is a few
        percent on every config we measured, so once the balancer has spent
        more than that fraction of elapsed serving time inside
        batch_isend_irecv it cannot come out ahead no matter how good the
        remaining decisions are. Measured: two configs at swap/headroom 1.26 and
        1.09 both realised ~0 of their ceiling.
        """
        if self.cfg.swap_budget_frac <= 0:
            return False
        elapsed = time.perf_counter() - self._serving_t0
        if elapsed <= 0:
            return False
        spent_frac = (self._cum_swap_s / elapsed)
        if spent_frac <= self.cfg.swap_budget_frac:
            return False
        if not self._budget_warned:
            self._budget_warned = True
            logger.info(f"[PB-OEPLB] swap budget exhausted: spent "
                        f"{self._cum_swap_s:.2f}s of {elapsed:.1f}s "
                        f"({100*spent_frac:.2f}% > {100*self.cfg.swap_budget_frac:.2f}%) "
                        f"-- no further swaps will be issued")
        return True

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
            try:
                torch.distributed.all_reduce(
                    global_load, op=torch.distributed.ReduceOp.SUM
                )
            except Exception as _e_sum:
                if self._sp_active:
                    # PD-10: under SP modes a planner-SUM failure is a
                    # communicator-level fault -- swallowing it would risk
                    # misaligning the cross-rank collective schedule at the
                    # next boundary (H-only cycles keep running detSUM +
                    # confirm). FATAL: flag + re-raise through the outer
                    # handler into the scheduler loop. Off mode: bare
                    # re-raise, outer handler swallows exactly as the
                    # frozen code does (bit-identical).
                    self._v2p1_fatal_flag = True
                    logger.error(
                        "[PB-OEPLB-SP] FATAL: planner SUM all_reduce failed "
                        "-- communicator-level fault, terminating run: "
                        "%s\n%s", _e_sum, traceback.format_exc())
                raise
            self.window_count += 1

            # final_plan_20261019: incremental summary dump every 16 windows
            # (teardown is SIGKILL so atexit is unreliable; short/smoke runs
            # may commit no SP transition at all).
            if (self._sp_active and self._rank == 0
                    and self.window_count % 16 == 0):
                self._sp_write_summary()

            # Compute current avg_ratio for hard-reset detection
            _layer_ratios = []
            for _lid in range(self.num_layers):
                _lc = global_load[_lid]
                _loads = [_lc[r*self.num_local:(r+1)*self.num_local].sum().item() for r in range(self.ep_size)]
                _avg = max(sum(_loads) / self.ep_size, 1.0)
                _layer_ratios.append(max(_loads) / _avg)
            _current_avg_ratio = sum(_layer_ratios) / len(_layer_ratios) if _layer_ratios else 1.0

            # C2 (E-A switch decoupling, 2026-10-15, audit finding F2):
            # RESET domain-shift history zeroing is history management ->
            # gated by cfg.adaptive_decay (was cfg.adaptive_window). With both
            # flags True (A11) this fires exactly as in the v3-locked code.
            if self._prev_window_ratio is not None and self.cfg.adaptive_decay:
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
                    self._eplb_refined = False  # allow re-refinement after domain shift
                    self._plateau_count = 0
                    self._reset_cooldown = 3  # cooldown: skip ADW shrink for 3 windows to prevent thrashing
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
            if self._wctrl_version == 1 and self.cfg.adaptive_window and self._last_cos_sim is not None:
                window_ceiling = self.cfg.sync_window * 2
                if self._last_cos_sim < self.cfg.window_shift_cos_threshold:
                    self._window_shift_count += 1
                    self._window_stable_count = 0
                    if self._window_shift_count >= self.cfg.window_shift_confirm_windows:
                        # FIX (found by GB200/EP4 repro): when window_floor >
                        # sync_window, max(floor, W//2) made "shrink" EXPAND the
                        # window. Clamp so shrinking is monotone non-expanding.
                        new_window = min(self._effective_sync_window,
                                         max(self.cfg.window_floor,
                                             self._effective_sync_window // 2))
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
            if self._wctrl_version == 2:
                # ADW v2: single-authority window decision (M1/M2/M3). Runs
                # BEFORE the token/threshold gates so window control is
                # independent of whether this window ends up swapping.
                self._window_control_v2(global_load, global_tokens)
            if self._wctrl_version == 21:
                # ADW v2.1 bounded revision: logical-identity detection (F1),
                # rate-scale headroom (F2), confirm-then-commit (F3). Same
                # call position as v2: BEFORE the token/threshold gates so
                # window control is independent of whether this window swaps.
                self._window_control_v2p1(global_load, global_tokens)
            if global_tokens < self.cfg.min_prefill_tokens:
                return

            _thr, _bias = self._effective_threshold(global_tokens)
            if self._swap_budget_exhausted():
                return
            # Plan quality, not just trigger: the plan is built from the same
            # counts that produced the biased ratio, so when the estimate is
            # coarse relative to the margin we are acting on, the per-expert
            # ordering is noise and the migration corrects skew that is not
            # there. Sit the window out and let the accumulator gather tokens.
            # The comparison is against the MARGIN, not an absolute bound --
            # the bias scales with ep_size, so a fixed cap would block every
            # window on wide EP including those with real headroom.
            if self.cfg.bias_gate > 0 and _bias > 0:
                _margin = (_current_avg_ratio - _bias) - max(
                    self.cfg.threshold_ratio, self.cfg.dead_zone_ratio)
                if _margin <= 0 or _bias > self.cfg.bias_gate * _margin:
                    if self.window_count % 8 == 1:
                        logger.info(f"[PB-OEPLB] window#{self.window_count}: skipped "
                                    f"-- noise/margin too high (bias={_bias:.4f}, "
                                    f"margin={_margin:.4f}, {global_tokens} tok)")
                    return
            # With the bias term the corrected ratio is what we should be
            # judging, so bail out early when even the raw ratio cannot clear
            # the bar -- saves building a plan that would come back empty.
            #
            # Gated on the new knobs being active on purpose. This is a
            # WINDOW-AVERAGE test, while the plan builder applies the threshold
            # PER LAYER, so it is strictly more aggressive: a window whose mean
            # ratio sits under the bar can still contain a badly skewed layer
            # that the old path would have fixed. Enabling it unconditionally
            # would silently change default behaviour and invalidate every
            # measurement taken so far.
            _early_exit_active = self.cfg.bias_correct or self.cfg.dead_zone_ratio > 0
            if _early_exit_active and _current_avg_ratio <= _thr:
                if _bias > 0 and self.window_count % 8 == 1:
                    logger.info(f"[PB-OEPLB] window#{self.window_count}: "
                                f"r={_current_avg_ratio:.4f} <= thr={_thr:.4f} "
                                f"(bias={_bias:.4f} from {global_tokens} tok) -- no swap")
                self._check_eplb_refinement(global_load)
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
            _planner_fn = (try_build_swap_plan_best_improvement
                           if getattr(self.cfg, 'planner_best_improvement', False)
                           else try_build_swap_plan)
            plan = _planner_fn(
                logical_count=global_load,
                physical_to_logical_map=p2l_map,
                num_ranks=self.ep_size,
                num_local=self.num_local,
                threshold_ratio=_thr,
                max_swaps_per_layer=self.cfg.max_swaps_per_layer,
                max_total_swap_layers=self.cfg.max_total_swap_layers,
                max_total_ops=self.cfg.max_total_ops,
            )
            self._prof_planbuild_ns += time.perf_counter_ns() - _t1
            # DIAG-1 (final_followup_20261020 r1): per-rank plan-content
            # hash -> cross-rank plan identity becomes measured, not
            # inferred. Pure diagnostic log; no behavior change.
            if plan is not None:
                import hashlib as _hl_plan
                _plan_ph = _hl_plan.sha256(repr(
                    [(op.layer_id, op.phys_slot_a, op.phys_slot_b)
                     for op in plan]
                ).encode()).hexdigest()[:16]
                logger.info(f"[PB-OEPLB-PLANHASH] window#{self.window_count} "
                            f"rank={self._rank} n_ops={len(plan)} "
                            f"plan_hash={_plan_ph}")
            # Track improvement for defer logic
            if plan:
                _pre_ratios = []
                for _lid in set(op.layer_id for op in plan[:10]):
                    _lc = global_load[_lid]
                    _loads = [_lc[r*self.num_local:(r+1)*self.num_local].sum().item() for r in range(self.ep_size)]
                    _avg = max(sum(_loads) / self.ep_size, 1.0)
                    _pre_ratios.append(max(_loads) / _avg)
                self._last_diag_improvement = (sum(_pre_ratios)/len(_pre_ratios) - 1.0) * 0.1 if _pre_ratios else 0
            else:
                self._last_diag_improvement = 0
            if not plan:
                self._check_eplb_refinement(global_load)
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
                self._check_eplb_refinement(global_load)
                return
            self._pending_plan_start_t = time.perf_counter()
            # Skip swap if improvement is negligible: save P2P overhead,
            # let next window accumulate more data for a better decision.
            from sglang.srt.managers.pb_oeplb.rebalancer import compute_gpu_load as _cgl
            _ratios_before = []
            _ratios_after_sim = []
            for _op in plan[:5]:  # sample first 5 ops for quick ratio check
                _lid = _op.layer_id
                _lc = global_load[_lid]
                _loads = [_lc[r*self.num_local:(r+1)*self.num_local].sum().item() for r in range(self.ep_size)]
                _avg = max(sum(_loads) / self.ep_size, 1.0)
                _ratios_before.append(max(_loads) / _avg)
            if _ratios_before:
                _sampled_ratio_before = sum(_ratios_before) / len(_ratios_before)
            else:
                _sampled_ratio_before = 1.0

            # If plan is small AND ratio improvement from DIAG is tiny, skip
            if len(plan) < 16 and hasattr(self, '_last_diag_improvement'):
                if self._last_diag_improvement < 0.003:
                    logger.info(f"[PB-OEPLB] window#{self.window_count}: deferred "
                               f"{len(plan)} swap(s) (improvement={self._last_diag_improvement:.4f} < 0.003)")
                    return

            self._plateau_count = 0
            if self._no_commit:
                # final_plan_20261019 A4 (P_noCommit): identical record/decide/
                # planbuild path; ONLY the physical commit is suppressed. The
                # executor is never engaged, so try_finish() has no pending
                # plan and no mapping/p2l commit occurs. swaps_started stays 0
                # (physical truth); planned ops are tallied separately.
                self._nocommit_planned_ops += len(plan)
                logger.info(
                    "[PB-OEPLB-NOCOMMIT] window#%d: plan built, %d op(s) "
                    "suppressed (observe-only arm)",
                    self.window_count, len(plan))
            else:
                self.async_executor.begin(plan)
                if self._sp_active:
                    self._sp_swap_started_h += 1
                    self._sp_swaps_total += 1
            logger.info(f"[PB-OEPLB] window#{self.window_count}: "
                       f"issued {len(plan)} swap(s) ({global_tokens} tok global, dp={self.dp_size})")
        except Exception as e:
            if self._sp_active and getattr(self, "_v2p1_fatal_flag", False):
                # PD-10/r3: collective faults in this path TERMINATE the run
                # under SP modes -- continuing on a suspect communicator
                # would misalign the cross-boundary NCCL schedule. Off-mode
                # behavior is the frozen swallow (bit-identical).
                self._v2p1_fatal_flag = False
                logger.error("[PB-OEPLB-SP] FATAL: propagating collective "
                             "failure from swap-decision path (run "
                             "terminates): %s", e)
                raise
            logger.error(f"[PB-OEPLB] error: {e}\n{traceback.format_exc()}")

    def _init_window_control_v2(self, cfg):
        """Validate ADW v2 phase-1 invariants and build the single-authority
        window controller (window_v2.WindowControllerV2).

        Phase-1 isolation (user rule): v2 replaces ONLY the window control.
        Required frozen base config: decay_factor==0.0 (the all-reduced window
        stats ARE the fresh detection stats - no second accumulator, no extra
        hot-path cost; alpha>0 would need one and is deferred to the D-phase
        study with its record cost accounted), and adaptive_window /
        adaptive_decay / calibrate_adaptive_sensitivity all OFF so every
        legacy mutation site is unreachable (single authority, M3/F1).
        """
        from sglang.srt.managers.pb_oeplb.window_v2 import WindowControllerV2
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if float(cfg.decay_factor) != 0.0:
            raise RuntimeError(
                "ADW v2 phase 1 requires decay_factor==0.0 (window-only "
                "isolation; alpha>0 needs a separate fresh accumulator with "
                "its record cost accounted - deferred to the D-phase study)")
        if cfg.adaptive_window or cfg.adaptive_decay or cfg.calibrate_adaptive_sensitivity:
            raise RuntimeError(
                "ADW v2 requires adaptive_window=False, adaptive_decay=False, "
                "calibrate_adaptive_sensitivity=False (single-authority rule M3)")
        if float(getattr(cfg, 'v2_kappa', 0.0)) <= 0.0:
            raise RuntimeError(
                "ADW v2: OEPLB_V2_KAPPA must be a calibration-frozen positive "
                "noise coefficient; refusing to start with a placeholder")
        knobs = dict(
            kappa=cfg.v2_kappa, z_lo=cfg.v2_z_lo, z_hi=cfg.v2_z_hi,
            z_urgent=cfg.v2_z_urgent, m_shrink=cfg.v2_m_shrink,
            m_grow=cfg.v2_m_grow, n_min=cfg.v2_n_min, u_max=cfg.v2_u_max,
            theta_h_low=cfg.v2_theta_h_low, theta_h_act=cfg.v2_theta_h_act,
            dwell_shrink=cfg.v2_dwell_shrink, dwell_grow=cfg.v2_dwell_grow,
            w_uncertain_target=cfg.v2_w_uncertain_target,
            urgent_episode_cap=cfg.v2_urgent_episode_cap)
        self._v2_beta = float(cfg.v2_beta)
        self._v2_kappa = float(cfg.v2_kappa)
        self._v2_ctrl = WindowControllerV2(cfg.sync_window, knobs, beta=self._v2_beta)
        self._v2_log_path = cfg.v2_log_path or ("/tmp/oeplb_v2_window_rank%d.jsonl" % rank)
        logger.info("[PB-OEPLB-ADW2] single-authority window controller v2 ACTIVE: "
                    f"w_init={cfg.sync_window} grid=(8,16,32,64) beta={self._v2_beta} "
                    f"knobs={knobs} log={self._v2_log_path}")

    def _window_control_v2(self, global_load, global_tokens):
        """ADW v2 per-window decision (the ONLY writer of _effective_sync_window
        under version 2). Signals (mechanism spec M1/M2), all derived from the
        all-reduced bit-identical fresh load tensor, vectorized (no per-layer
        .item() storm):

          h_w     = sum_l max(0, max_rank_load_l - max(total_l/EP, max_expert_l))
                    -- recoverable excess above the indivisible-expert work
                    floor, in assignment units (conservative proxy; limits
                    documented in the spec; count->time conversion is LOG-ONLY)
          s_change = mean_l (1 - cos(p_l, ref_l)) over per-layer normalized
                    fresh distributions; ref = detection-only EWMA (beta),
                    frozen while uncertain; NOT the planner history
          z       = s_change / (kappa / n_layer_fresh) -- noise-normalized, so
                    shrinking the window (fewer samples) does not inflate z
                    (removes the v1 shrink->noise->shrink feedback, F2)
          n_fresh = global_tokens (fresh recorded assignments; equals the
                    window total because phase-1 alpha==0; includes carry)

        Cross-rank safety: a decision hash is computed every window from
        bit-identical inputs; every 16th window one small all_reduce verifies
        all ranks hold the same hash. Mismatch -> deterministic safe mode
        (W frozen at w_init on every rank; window_v2.on_desync). Never raises:
        internal errors HOLD W and log; serving continues.
        """
        import hashlib
        import json as _json
        import struct
        t0 = time.perf_counter_ns()
        try:
            gl = global_load.float()
            L = gl.shape[0]
            R = self.ep_size
            S = gl.shape[1] // R
            rank_loads = gl.view(L, R, S).sum(-1)
            total = rank_loads.sum(-1)
            max_expert = gl.max(-1).values
            floor = torch.maximum(total / R, max_expert)
            excess = (rank_loads.max(-1).values - floor).clamp(min=0)
            h_w = float(excess.sum().item())

            rowsum = gl.sum(-1, keepdim=True)
            valid = rowsum.squeeze(-1) > 0
            p = gl / rowsum.clamp(min=1.0)
            z = None
            s_change = None
            if self._v2_ref is not None:
                dot = (p * self._v2_ref).sum(-1)
                nc = p.norm(dim=-1)
                nr = self._v2_ref.norm(dim=-1)
                ok = valid & (nc > 0) & (nr > 0)
                if bool(ok.any().item()):
                    cos = dot[ok] / (nc[ok] * nr[ok])
                    s_change = float((1.0 - cos).mean().item())
                    n_layer = max(global_tokens / float(L), 1.0)
                    z = s_change / (self._v2_kappa / n_layer)

            rec = self._v2_ctrl.step(z, h_w, int(global_tokens))
            ev = rec["event"]

            # detection EWMA (M1): seeded on init; frozen while uncertain
            if ev == "init":
                self._v2_ref = p.clone()
            elif ev not in ("uncertain_carry", "uncertain_hold",
                            "forced_shrink_uncertain", "w_desync_freeze"):
                if self._v2_ref is None:
                    self._v2_ref = p.clone()
                else:
                    self._v2_ref.mul_(self._v2_beta).add_(p, alpha=1.0 - self._v2_beta)

            self._effective_sync_window = rec["W_after"]
            self._v2_carry = bool(rec["carry_fresh"])

            # cross-rank decision-hash check (one 2-elem all_reduce per 16 windows;
            # all ranks reach it in the same window -> NCCL order consistent;
            # pending P2P already force-waited at the top of _decide_and_begin_swap)
            hbytes = hashlib.sha256(struct.pack(
                ">qdddi", self.window_count, float(rec["W_after"]),
                (z if z is not None else -1.0), h_w, int(global_tokens))).digest()
            h = int.from_bytes(hbytes[:8], "big") % (1 << 58)
            self._v2_hash_count += 1
            desync = False
            if self._v2_hash_count % 16 == 0:
                M = 1 << 58
                world = torch.distributed.get_world_size()
                t = torch.tensor([h, M - h], dtype=torch.int64, device=gl.device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                # sum over the world group must equal world * own h (h is
                # bit-identical across ranks by construction). BUGFIX
                # (smoke C, 2026-10-16): was hardcoded 16 (window interval
                # confused with rank count) -> false desync at EP=8.
                if int(t[0].item()) != world * h:
                    desync = True
                    self._v2_ctrl.on_desync()
                    self._effective_sync_window = self._v2_ctrl.W
                    logger.error("[PB-OEPLB-ADW2] RANK DESYNC on decision hash "
                                 f"(w#{self.window_count}) -> safe mode: W frozen "
                                 f"at w_init={self._v2_ctrl.W} on all ranks "
                                 f"(own_h={h} sum={int(t[0].item())} world={world})")

            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            if self._rank == 0:
                if self._v2_log_fh is None:
                    self._v2_log_fh = open(self._v2_log_path, "a")
                line = {
                    "window_id": self.window_count,
                    "steps_actual": self._last_window_steps,
                    "n_fresh_assignments": int(global_tokens),
                    "s_change": s_change,
                    "z": z,
                    "h_w_assignments": h_w,
                    "h_w_us_empirical_conversion": h_w * 0.1354912244734509,
                    "h_w_conversion_note": "E-T slope; empirical predictor with role bias (FACT-07); log-only, never an online decision input",
                    "r_legacy": self._prev_window_ratio,
                    "cos_legacy": self._last_cos_sim,
                    "c_window_ms": elapsed_ms,
                    "prof_allreduce_cum_ms": self._prof_allreduce_ns / 1e6,
                    "W_before": rec["W_before"],
                    "W_after": rec["W_after"],
                    "event": ev,
                    "reasons": rec["reasons"],
                    "alert_k": rec["alert_k"],
                    "stable_s": rec["stable_s"],
                    "dwell": rec["dwell"],
                    "dwell_dir": rec["dwell_dir"],
                    "uncertain_run": rec["uncertain_run"],
                    "episode_shrinks": rec["episode_shrinks"],
                    "carry_fresh": rec["carry_fresh"],
                    "desync": desync,
                }
                self._v2_log_fh.write(_json.dumps(line) + "\n")
                self._v2_log_fh.flush()
                logger.info(f"[PB-OEPLB-ADW2] w#{self.window_count} "
                            f"z={'%.2f' % z if z is not None else 'NA'} "
                            f"H={h_w:.0f} N={global_tokens} "
                            f"W {rec['W_before']}->{rec['W_after']} ev={ev}")
        except Exception as e:
            logger.error(f"[PB-OEPLB-ADW2] v2 error (W HELD, serving continues): {e}\n"
                         f"{traceback.format_exc()}")

    def _init_window_control_v2p1(self, cfg):
        """ADW v2.1 (limited revision of frozen v2 -- see
        NEW_PAPER/.../results/adw_v2p1/minimal_diff.md). Addresses exactly the
        three audit-verified findings (2026-10-16):
          F1 detection unified in logical-expert identity (separate detection
             accumulator + commit-site count remap + mapping-version guard);
          F2 headroom on the work-fraction scale h_w/n_fresh (thresholds are
             rates, re-calibrated pre-registered on W0 only);
          F3 window & control state committed only after a per-window
             cross-rank unanimity check (confirm-then-commit; error paths
             participate with a deterministic sentinel so the collective
             schedule can never misalign).
        Same phase-1 invariants as v2: decay_factor==0.0 and adaptive flags
        OFF; planner path (self.load, ratio gates, swap acceptance, executor)
        is NOT touched -- v2.1 replaces detection/window control only.
        Framing: correctness revision, not a promised performance change.
        """
        from sglang.srt.managers.pb_oeplb.window_v2p1 import WindowControllerV2P1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if float(cfg.decay_factor) != 0.0:
            raise RuntimeError(
                "ADW v2.1 phase 1 requires decay_factor==0.0 (window-only isolation)")
        if cfg.adaptive_window or cfg.adaptive_decay or cfg.calibrate_adaptive_sensitivity:
            raise RuntimeError(
                "ADW v2.1 requires adaptive_window=False, adaptive_decay=False, "
                "calibrate_adaptive_sensitivity=False (single-authority rule M3)")
        if float(getattr(cfg, 'v21_kappa', 0.0)) <= 0.0:
            raise RuntimeError(
                "ADW v2.1: OEPLB_V21_KAPPA must be a calibration-frozen positive "
                "noise coefficient (logical-space signal); refusing placeholder")
        knobs = dict(
            kappa=cfg.v21_kappa, z_lo=cfg.v21_z_lo, z_hi=cfg.v21_z_hi,
            z_urgent=cfg.v21_z_urgent, m_shrink=cfg.v21_m_shrink,
            m_grow=cfg.v21_m_grow, n_min=cfg.v21_n_min, u_max=cfg.v21_u_max,
            theta_h_low=cfg.v21_theta_r_low, theta_h_act=cfg.v21_theta_r_act,
            dwell_shrink=cfg.v21_dwell_shrink, dwell_grow=cfg.v21_dwell_grow,
            w_uncertain_target=cfg.v21_w_uncertain_target,
            urgent_episode_cap=cfg.v21_urgent_episode_cap)
        self._v2p1_beta = float(cfg.v21_beta)
        self._v2p1_kappa = float(cfg.v21_kappa)
        self._v2p1_ctrl = WindowControllerV2P1(cfg.sync_window, knobs, beta=self._v2p1_beta)
        # F1: logical-space detection state. The ref EWMA lives in LOGICAL
        # identity -> stable across swap commits, never remapped (that was the
        # v2 artifact). Separate from the planner accumulator by construction.
        self._v2p1_ref = None
        self._v2p1_carry = False
        self._det_load = torch.zeros(self.num_layers, self.num_physical_experts,
                                     dtype=torch.int64, device=self.load.device)
        self._p2l_version = 0       # bumped at EVERY mapping-commit site
        self._det_map_version = 0   # _det_load contents consistent with this version
        self._v2p1_log_path = cfg.v21_log_path or ("/tmp/oeplb_v2p1_window_rank%d.jsonl" % rank)
        self._v2p1_log_fh = None
        logger.info("[PB-OEPLB-ADW21] limited-revision controller v2.1 ACTIVE: "
                    f"w_init={cfg.sync_window} grid=(8,16,32,64) beta={self._v2p1_beta} "
                    f"knobs={knobs} (theta_* in RATE units) log={self._v2p1_log_path}")
        # -- Route-B bounded probe admission (spec v1.1-calib; frozen protocol
        # PRE_REG_calibration_m1m2m3.md). mode=off -> detector path untouched
        # (M1 determinism gate vs sealed v2.1 r3). mode=gate -> ABA arm
        # (whole-run growth gating + one bounded W16->W32->W16 probe, commit
        # disabled). mode=aaa -> null arm (whole-run growth gating, matched
        # segment forward budgets, no W override anywhere). The ACTIVE banner
        # above is UNTOUCHED (integrity_check greps it); probe config logs on
        # its own line.
        from sglang.srt.managers.pb_oeplb.probe_admission import ProbeAdmission
        if getattr(cfg, "probe_mode", "off") != "off" and int(cfg.sync_window) != 16:
            raise RuntimeError(
                "v1 probe is frozen at the W16 baseline (EA_W=16); got "
                "sync_window=%r" % cfg.sync_window)
        self._probe = ProbeAdmission(
            mode=getattr(cfg, "probe_mode", "off"),
            r=int(getattr(cfg, "probe_r", 12)),
            n_ref=int(getattr(cfg, "probe_n_ref", 4)),
            n_cand=int(getattr(cfg, "probe_n_cand", 3)),
            n_fb=int(getattr(cfg, "probe_n_fb", 4)),
            w_base=int(cfg.sync_window), w_cand=32,
            eps_risk_us=int(getattr(cfg, "probe_eps_risk_us", 2000000)),
            eps_prac_frac=float(getattr(cfg, "probe_eps_prac_frac", 0.01)),
            force_abort_at=int(getattr(cfg, "probe_force_abort_at", 0)),
            commit_enabled=bool(int(getattr(cfg, "probe_commit", 0))),
            calib_trigger=bool(int(getattr(cfg, "probe_calib_trigger", 1))),
            dwell_grow=int(cfg.v21_dwell_grow), k_tb=8)
        if self._probe.mode != "off":
            logger.info("[PB-OEPLB-ADW21-PROBE] route-B bounded probe ACTIVE: "
                        f"mode={self._probe.mode} R={self._probe.r} "
                        f"n_ref={self._probe.n_ref} n_cand={self._probe.n_cand} "
                        f"(cand_windows_this_mode={self._probe.cand_window_target()}) "
                        f"n_fb={self._probe.n_fb} "
                        f"w={self._probe.w_base}->{self._probe.w_cand} "
                        f"eps_risk_us={self._probe.eps_risk_us} "
                        f"force_abort_at={self._probe.force_abort_at} "
                        f"commit={int(self._probe.commit_enabled)} "
                        f"calib_trigger={int(self._probe.calib_trigger)} "
                        f"eps_prac_frac={self._probe.eps_prac_frac}")
        import atexit as _atexit
        _atexit.register(self._probe_run_end_dump)
        # -- sp_coupling_v1 (plan 2026-10-18): sampling/planning coupling.
        # PD-12 conflicts are rejected at config construction; re-asserted
        # here so a directly-built config cannot bypass them.
        from sglang.srt.managers.pb_oeplb.sp_state import (
            SpController, SP_TABLE)
        _sp_mode = str(getattr(cfg, "sp_mode", "off"))
        if _sp_mode != "off":
            if self._probe.mode != "off":
                raise RuntimeError(
                    "sp_coupling_v1: SP_MODE=%r requires probe_mode=off "
                    "(PD-12)" % _sp_mode)
            if cfg.always_record or int(cfg.sample_interval) != 1:
                raise RuntimeError(
                    "sp_coupling_v1: SP_MODE=%r requires always_record=False "
                    "and sample_interval=1 (PD-12/PD-15)" % _sp_mode)
        self._sp = SpController(
            mode=_sp_mode, sp_s=int(getattr(cfg, "sp_s", 1)),
            sp_w=int(getattr(cfg, "sp_w", 16)),
            seed=int(getattr(cfg, "sp_seed", 20261018)),
            quiet_streak=int(getattr(cfg, "sp_quiet_streak", 3)),
            recover_min_cycles=int(getattr(cfg, "sp_recover_min_cycles", 2)),
            z_lo=cfg.v21_z_lo, z_hi=cfg.v21_z_hi,
            theta_h_low=cfg.v21_theta_r_low, n_min=cfg.v21_n_min,
            theta_h_act=cfg.v21_theta_r_act,
            rev=int(getattr(cfg, "sp_rev", 0)),
            sp_h=int(getattr(cfg, "sp_h", 0)),
            force_s=int(getattr(cfg, "sp_force_s", 0)))
        self._sp_active = (_sp_mode != "off")
        self._sp_n_min_full = int(cfg.v21_n_min)
        if self._sp_active:
            if int(cfg.sync_window) != 16:
                raise RuntimeError(
                    "sp_coupling_v1: SP modes require the frozen w_init=16 "
                    "baseline (EA_W=16); got sync_window=%r"
                    % cfg.sync_window)
            self._sp_valid_rows_h = torch.zeros(
                (), dtype=torch.int64, device=self.load.device)
            logger.info(
                "[PB-OEPLB-SP] sampling/planning coupling ACTIVE: mode=%s "
                "S=%d H=%d W=%d seed=%d quiet_streak=%d "
                "recover_min_cycles=%d n_min_eff=%d rev=%d theta_out=%.6f "
                "table=%s "
                "(pre-forward gate PD-9; W_after neutralized PD-8; "
                "conflicts rejected PD-12)",
                _sp_mode, self._sp.S, self._sp.H, self._sp.W, self._sp.seed,
                self._sp.quiet_streak, self._sp.recover_min_cycles,
                self._sp.n_min_eff(), self._sp.rev, self._sp.theta_h_act,
                sorted(SP_TABLE.items()))
            _atexit.register(self._sp_run_end_dump)

    def _v2p1_confirm_reduce(self, pair, op):
        """F3 (r2) confirm collective: elementwise MIN (then MAX) all_reduce
        on the 2-element int64 pair [h, prop_valid]. Fixed order on every
        rank, unconditional, one pair of collectives per window."""
        t = torch.tensor([pair[0], pair[1]], dtype=torch.int64,
                         device=self.load.device)
        rop = (torch.distributed.ReduceOp.MIN if op == "min"
               else torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(t, op=rop)
        pair[0] = int(t[0].item())
        pair[1] = int(t[1].item())

    def _window_control_v2p1(self, global_load, global_tokens):
        """ADW v2.1 per-window decision -- the ONLY writer of
        _effective_sync_window under version 21. Revision r2 (2026-10-17,
        pre-freeze): collective schedule UNCONDITIONAL + symmetric verdict.
        Revision r3 (2026-10-17, pre-freeze): communicator-fatal boundary
        -- collective failures TERMINATE the run (re-raise into the
        scheduler loop) instead of holding and continuing the schedule.

        Phases:
          1 (collective, unconditional): det SUM all_reduce. Every rank
             participates every window REGARDLESS of local state -- r1 ran
             the mapping-version tripwire before this collective, so an
             asymmetric local failure could misalign it (NCCL size-mismatch
             hang); in r2+ no local check precedes the collective. A
             failure HERE is a COMMUNICATOR-LEVEL fault: r3 logs it FATAL
             and RE-RAISES -- the exception propagates through
             on_forward_pass_end (called unwrapped from model_runner.
             forward) into the scheduler loop, crashing the process and
             terminating the run, per NCCL fatal-error guidance (abort
             operations; recovery only by rebuilding the communicator =
             job restart). The r2 behavior (hold + still attempt phase 3
             to preserve schedule alignment) is explicitly NOT a safe
             handling principle and was removed.
          2 (local, RECOVERABLE scope): mapping-version tripwire, signal
             computation, ctrl.step() proposal, decision hash. NO collective
             inside this scope. Any exception -> rollback + deterministic
             error sentinel + prop_valid=0 -> joint HOLD; serving continues.
          3 (collective, unconditional): confirm MIN + MAX all_reduce on
             [h, prop_valid]. Verdict commit = (h_min==h_max) AND (v_min==1)
             -- a function of reduced (bit-identical) data only, so every
             rank derives the SAME verdict by construction.
          4 (symmetric commit): commit -> apply W / carry / ref EWMA (ref
             frozen on uncertain/desync events, inherited); hold -> rollback
             (identical on every rank), streak+1; streak>=2 -> inherited
             safe mode (freeze at w_init).
        Fault boundary (r3), explicitly:
          RECOVERABLE = phase-2 local computation exceptions on a HEALTHY
             communicator: rollback + error sentinel + prop_valid=0 ->
             joint HOLD, serving continues. (The 11-window fault-injection
             test covers exactly this class.)
          NON-RECOVERABLE = any exception from the collectives themselves
             (phase 1 det SUM, phase 3 confirm MIN/MAX): FATAL -- log +
             re-raise, TERMINATE this run; no further ordinary confirm or
             commit is attempted on a suspect communicator. Recovery
             belongs to the framework job-restart flow, not the controller.
        Phase 2 never raises by construction; all online costs (det SUM +
        confirm MIN + confirm MAX + local compute) are inside c_window_ms.
        """
        import time as _time
        import json as _json
        from sglang.srt.managers.pb_oeplb import v2p1_signals as _sig
        from sglang.srt.managers.pb_oeplb.window_v2p1 import (
            decision_hash, error_sentinel, confirm_commit)
        if self._sp_active:
            # sp_coupling_v1: H-cycle id advances at EVERY H boundary on
            # every rank (window_count counts joint planning boundaries
            # only, so it is NOT the detection-cycle id under SP modes).
            self._sp_h_cycle += 1
        t0 = _time.perf_counter_ns()
        # v2.2-calib d1/d3: per-boundary monotonic stamps -> window span (us).
        # CLOCK_MONOTONIC is comparable across processes on the same node and
        # aligns with the bench client's mono_start_ns/mono_end_ns fields.
        # First window logs span_us_local=0 (sentinel; never inside a segment
        # because the trigger floor R >= 1).
        _mono_now = _time.monotonic_ns()
        _span_us = ((_mono_now - self._prev_boundary_mono_ns) // 1000
                    if self._prev_boundary_mono_ns is not None else 0)
        self._prev_boundary_mono_ns = _mono_now
        _t_phase = {"det": 0, "local": 0, "confirm": 0, "commit": 0}
        ctrl = self._v2p1_ctrl
        snap = ctrl.snapshot()
        rec = None
        z = None
        s_change = None
        h_w = None
        h_rate = 0.0
        n_fresh = 0
        p = None
        err_reason = None
        prop_valid = 0
        _wid_sp = (self._sp_h_cycle if self._sp_active
                   else self.window_count)
        h = error_sentinel(_wid_sp)
        _sp_prop = None
        _sp_swaps_since = 0
        _sp_valid_rows = 0
        _sp_state_before = (self._sp.state if self._sp_active else "")

        def _v2p1_fatal_record(phase, exc):
            # best-effort evidence line BEFORE termination; a logging
            # failure must never mask the fatal error itself.
            try:
                if self._rank == 0 and self._v2p1_log_path:
                    if self._v2p1_log_fh is None:
                        self._v2p1_log_fh = open(self._v2p1_log_path, "a")
                    self._v2p1_log_fh.write(_json.dumps({
                        "protocol_rev": "r3", "event": "fatal_collective",
                        "phase": phase, "window_id": self.window_count,
                        "W": ctrl.W, "err": str(exc)}) + "\n")
                    self._v2p1_log_fh.flush()
            except Exception:
                pass

        # -- phase 1: det collective, unconditional on every rank ------------
        # A collective failure HERE is a communicator-level (NON-RECOVERABLE)
        # fault, not a local computation error: per NCCL fatal-error guidance
        # the run must terminate -- abort operations, recovery only by
        # rebuilding the communicator (= job restart). Log FATAL and
        # RE-RAISE into the scheduler loop; continuing to the ordinary
        # confirm MIN/MAX on a suspect communicator is NOT a safety policy.
        try:
            _tp1 = _time.perf_counter_ns()
            det = self._det_load.clone()
            torch.distributed.all_reduce(det, op=torch.distributed.ReduceOp.SUM)
            _t_phase["det"] = (_time.perf_counter_ns() - _tp1) // 1000
        except Exception as e:
            logger.error(
                "[PB-OEPLB-ADW21] FATAL: det SUM all_reduce failed -- "
                "communicator-level fault, terminating run (no further "
                "ordinary confirm/commit will be attempted): %s\n%s",
                e, traceback.format_exc())
            _v2p1_fatal_record(1, e)
            self._v2p1_fatal_flag = True
            raise
        # -- phase 2: local recoverable scope (NO collective inside) ---------
        try:
            if self._det_map_version != self._p2l_version:
                raise RuntimeError(
                    "mapping version mismatch: det=%d p2l=%d (unhandled "
                    "commit path -- holding W this window)" % (
                        self._det_map_version, self._p2l_version))
            n_fresh = int(det.sum().item())
            logical = _sig.logical_load(det, self._cached_p2l)
            h_w = _sig.headroom(det, self.ep_size)
            h_rate = _sig.headroom_rate(h_w, n_fresh)
            p, p_valid = _sig.row_distribution(logical)
            if self._v2p1_ref is not None:
                s_change, _nv = _sig.change_score(p, self._v2p1_ref, p_valid)
                z = _sig.z_from_signals(s_change, n_fresh, logical.shape[0],
                                      self._v2p1_kappa)
            if self._sp_active:
                # PD-1/PD-8: under sampling S the frozen dense coverage gate
                # applies to S*H forwards of opportunities; the effective
                # per-H-cycle gate is n_min // S (pre-declared; subject to
                # the Sec.4.3 dev-data noise check, never tuned on eval
                # data). Derived from COMMITTED state, so every rank sets
                # the identical value and ctrl stays a deterministic pure
                # function of the reduced signals.
                ctrl.k["n_min"] = max(
                    1, self._sp_n_min_full // max(1, self._sp.S))
            rec = ctrl.step(z, h_rate, n_fresh)
            # -- Route-B probe (v1.1-calib): PRE-HASH proposal. The machine is
            # a deterministic function of the reduced window series (mutated
            # only in phase-4 advance below), so every rank derives the same
            # override by construction; W overrides are applied BEFORE
            # decision_hash -> the confirm protocol gains no new mismatch
            # class. A probe exception here stays inside the recoverable
            # phase-2 scope (rollback + error sentinel + joint HOLD).
            _probe2 = getattr(self, "_probe", None)
            if _probe2 is not None and _probe2.mode != "off":
                _tpn = _time.perf_counter_ns()
                try:
                    _pact2 = _probe2.propose(self.window_count, rec)
                finally:
                    self._prof_probe_ns += _time.perf_counter_ns() - _tpn
                if _pact2.get("override"):
                    _ov = _pact2["override"]
                    _gated = dict(rec)
                    _gated["W_after"] = int(_ov["W_after"])
                    _gated["event"] = _ov["event"]
                    _gated["reasons"] = (list(rec.get("reasons", []))
                                         + [_ov.get("reason", "probe_override")])
                    rec = _gated
            if self._sp_active:
                # sp_coupling_v1 phase-2: coupling proposal from the reduced
                # signals + committed machine state. swaps_started is
                # consumed and re-zeroed HERE: it must count plan starts
                # since the PREVIOUS H-boundary proposal (a plan started at
                # the previous joint boundary -- after that boundary's
                # window control -- is visible to this proposal).
                _sp_swaps_since = self._sp_swap_started_h
                self._sp_swap_started_h = 0
                _sp_state_before = self._sp.state
                _sp_prop = self._sp.propose({
                    "z": z, "h_rate": h_rate, "n_fresh": n_fresh,
                    "event": rec["event"],
                    "swap_pending": (self.async_executor.pending
                                     is not None),
                    "swaps_started": _sp_swaps_since,
                    "plan_opp": self._sp_plan_opp_h,
                })
                _sp_hash = self._sp.proposal_hash(_sp_prop)
            h = decision_hash(_wid_sp, rec)
            if self._sp_active:
                # PD-6: the coupling proposal folds into the SAME single
                # hash -- any cross-rank divergence in the machine demotes
                # this boundary to a joint HOLD (state unchanged), exactly
                # like a ctrl-proposal divergence.
                h = (h ^ _sp_hash) % (1 << 58)
            prop_valid = 1
            _t_phase["local"] = ((_time.perf_counter_ns() - t0) // 1000
                                 - _t_phase["det"])
        except Exception as e:
            err_reason = ((err_reason + "; " if err_reason else "")
                          + "local_recoverable: "
                          f"{e}\n{traceback.format_exc()}")
            ctrl.restore(snap)
            rec = None
            prop_valid = 0
            _sp_prop = None
            h = error_sentinel(_wid_sp)

        # -- phase 3: confirm collectives, unconditional on every rank -------
        # v2.2-calib (author point 3): confirm payload EXTENDED from
        # [h, prop_valid] to the fixed-order 6-element int64 vector
        # [h, prop_valid, span_us, bcost_us, steps_w, true_steps_total].
        # Same collective schedule as r3 (one MIN then one MAX, fixed order,
        # unconditional); verdict UNCHANGED and computed from elements 0/1
        # only: commit <=> h_min==h_max AND v_min==1 -- timing fields can
        # never influence the decision hash. Timing series used for probe
        # accounting = elementwise MAX (slowest rank governs; NEVER summed
        # across ranks). steps/true_steps MIN==MAX is the lockstep audit.
        # bcost_us = partial boundary wall (entry -> this reduce); the FULL
        # boundary wall of the previous window is logged as prev_bcost_us.
        commit = False
        h_min = h_max = v_min = None
        span_min = span_max = bcost_max = None
        steps_min = steps_max = ts_min = ts_max = None
        lockstep_ok = None
        _bcost_partial_us = (_time.perf_counter_ns() - t0) // 1000
        try:
            _tp3 = _time.perf_counter_ns()
            _payload = [h, prop_valid, _span_us, _bcost_partial_us,
                        int(self._last_window_steps or 0),
                        int(self._true_steps_total)]
            _tmin = torch.tensor(_payload, dtype=torch.int64,
                                 device=self.load.device)
            _tmax = _tmin.clone()
            torch.distributed.all_reduce(_tmin, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(_tmax, op=torch.distributed.ReduceOp.MAX)
            h_min, v_min = int(_tmin[0].item()), int(_tmin[1].item())
            h_max = int(_tmax[0].item())
            span_min, span_max = int(_tmin[2].item()), int(_tmax[2].item())
            bcost_max = int(_tmax[3].item())
            steps_min, steps_max = int(_tmin[4].item()), int(_tmax[4].item())
            ts_min, ts_max = int(_tmin[5].item()), int(_tmax[5].item())
            commit = (h_min == h_max) and (v_min == 1)
            lockstep_ok = (steps_min == steps_max) and (ts_min == ts_max)
            _t_phase["confirm"] = (_time.perf_counter_ns() - _tp3) // 1000
        except Exception as e:
            logger.error(
                "[PB-OEPLB-ADW21] FATAL: confirm MIN/MAX all_reduce failed "
                "-- communicator-level fault, terminating run (no ordinary "
                "commit attempted on a suspect communicator): %s\n%s",
                e, traceback.format_exc())
            _v2p1_fatal_record(3, e)
            self._v2p1_fatal_flag = True
            raise
        # -- phase 4: symmetric commit ---------------------------------------
        committed = False
        desync = False
        if commit and rec is not None and prop_valid == 1:
            if not self._sp_active:
                self._effective_sync_window = rec["W_after"]
            self._v2p1_carry = bool(rec["carry_fresh"])
            ev = rec["event"]
            if ev == "init":
                self._v2p1_ref = p.clone()
            elif ev not in ("uncertain_carry", "uncertain_hold",
                            "forced_shrink_uncertain", "w_desync_freeze"):
                if self._v2p1_ref is None:
                    self._v2p1_ref = p.clone()
                else:
                    self._v2p1_ref.mul_(self._v2p1_beta).add_(
                        p, alpha=1.0 - self._v2p1_beta)
            ctrl.confirm_fail_streak = 0
            committed = True
            proposed = rec["event"]
            w_after = rec["W_after"]
        else:
            if commit and (rec is None or prop_valid != 1):
                # unreachable under the protocol (commit => v_min==1 on every
                # rank => prop_valid==1 locally); logged, then held.
                err_reason = ((err_reason + "; " if err_reason else "")
                              + "INTERNAL: commit verdict without local proposal")
            ctrl.restore(snap)   # idempotent; error path already restored
            self._v2p1_carry = False
            ctrl.confirm_fail_streak += 1
            proposed = rec["event"] if rec is not None else "error"
            w_after = snap["W"]
            if ctrl.confirm_fail_streak >= 2:
                ctrl.on_desync()
                self._effective_sync_window = ctrl.W
                desync = True
                ev = "confirm_fail_freeze"
            else:
                ev = "confirm_hold"
        # -- v2.2-calib: probe state-machine advance (phase-4 mutator).
        # Consumes ONLY the reduced series, so all ranks mutate identically.
        # ctrl directives (restore / set_W / dwell) are applied ONLY on
        # commit; holds re-propose idempotently at the next window.
        _probe = getattr(self, "_probe", None)
        _pinfo = None
        _probe_err = False
        _probe_state_at_window = ""
        if _probe is not None and _probe.mode != "off":
            _probe_state_at_window = _probe.state
            _tpn = _time.perf_counter_ns()
            try:
                _pinfo = _probe.advance(
                    self.window_count, committed,
                    steps=(steps_max if steps_max is not None
                           else int(self._last_window_steps or 0)),
                    span_us=(span_max if span_max is not None else _span_us),
                    bcost_us=(bcost_max if bcost_max is not None else 0),
                    true_steps_total=(ts_max if ts_max is not None
                                      else self._true_steps_total),
                    p2l_version=self._p2l_version)
            except Exception:
                # deterministic => symmetric on all ranks: disable the probe,
                # keep serving, flag every subsequent line with probe_error.
                _probe_err = True
                logger.error("[PB-OEPLB-ADW21-PROBE] advance failed -- probe "
                             "disabled for this run (symmetric on all ranks): "
                             "%s", traceback.format_exc())
                _probe.mode = "off"
            finally:
                self._prof_probe_ns += _time.perf_counter_ns() - _tpn
            if committed and _pinfo and _pinfo.get("directive"):
                _pd = _pinfo["directive"]
                if _pd.get("restore"):
                    ctrl.restore(snap)
                if "set_W" in _pd:
                    ctrl.W = int(_pd["set_W"])
                if "dwell" in _pd:
                    ctrl.dwell = int(_pd["dwell"])
                if "dwell_dir" in _pd:
                    ctrl.dwell_dir = _pd["dwell_dir"]
            # immediate run-end dump at terminal states (rank0): teardown is
            # SIGKILL, so atexit alone would lose the summary.
            if (self._rank == 0 and not self._probe_dumped
                    and _probe.mode != "off"
                    and _probe.state in ("DONE", "ABORTED")):
                self._probe_dumped = True
                self._probe_run_end_dump()
        # -- sp_coupling_v1 phase-4: coupling transitions apply ONLY on
        # joint commit (PD-6); on HOLD the proposal is discarded and the
        # machine is untouched (streaks neither advance nor reset).
        if self._sp_active:
            # PD-8: ctrl internal W pinned at w_init -- the SP schedule
            # comes from the coupling table and rec["W_after"] is NEVER
            # applied; only z/alert_k/stable_s/shrink-confirm signals are
            # consumed from ctrl, under its original confirmation mechanics.
            ctrl.W = ctrl.w_init
            _sp_s_before = self._sp.S
            if commit and prop_valid == 1 and _sp_prop is not None:
                _sp_info = self._sp.apply(_sp_prop, h_id=self._sp_h_cycle)
                _sp_seal_reason = ""
                _sp_cover = int(self._sp_cover_w)
                if _sp_info["S_changed"]:
                    # PD-7: seal the planner accumulator on an S change --
                    # its contents were sampled at the OLD rate; mixing
                    # rates inside one planning window would bias the
                    # ratio. Detection ref-EWMA rules are UNAFFECTED
                    # (logical space, no sampling-rate identity), and the
                    # seal is logged with the covered step count.
                    self.load.zero_()
                    self.total_tokens = 0
                    _sp_seal_reason = "sp_seal_S_change"
                    self._sp_seal_log.append({
                        "h_id": self._sp_h_cycle,
                        "reason": _sp_seal_reason,
                        "cover_steps": _sp_cover,
                        "S_before": _sp_s_before,
                        "S_after": self._sp.S})
                self._sp_last_commit = {
                    "transition": bool(_sp_info["transition"]),
                    "S_changed": bool(_sp_info["S_changed"]),
                    "state_before": _sp_state_before,
                    "state_after": self._sp.state,
                    "seal_reason": _sp_seal_reason,
                    "cover_steps": _sp_cover,
                    "reasons": list(_sp_prop.get("reasons", [])),
                }
            else:
                self._sp_last_commit = {
                    "transition": False, "S_changed": False,
                    "state_before": _sp_state_before,
                    "state_after": self._sp.state,
                    "seal_reason": "", "cover_steps": 0, "reasons": [],
                }
            # PD-13 valid-rows read: once per H boundary on every rank (a
            # sync point anyway; the accumulator is per-rank local).
            _sp_valid_rows = int(self._sp_valid_rows_h.item())
        _t_phase["commit"] = ((_time.perf_counter_ns() - t0) // 1000
                              - _t_phase["det"] - _t_phase["local"]
                              - _t_phase["confirm"])
        elapsed_ms = (_time.perf_counter_ns() - t0) / 1e6
        if self._rank == 0:
            if self._v2p1_log_fh is None:
                self._v2p1_log_fh = open(self._v2p1_log_path, "a")
            # d4: low-cost layout fingerprint (rank0; blake2b-8B of the p2l
            # bytes). Timed into the probe self-cost ledger. Cross-rank
            # mapping consistency remains guarded by the p2l_version
            # tripwire; the fingerprint is for cross-arm comparability.
            _p2l_fp8 = ""
            if _probe is not None and (_probe.mode != "off" or _probe_err):
                _tfp = _time.perf_counter_ns()
                try:
                    import hashlib as _hl
                    _p2l_fp8 = _hl.blake2b(
                        self._cached_p2l.cpu().numpy().tobytes(),
                        digest_size=8).hexdigest()
                except Exception:
                    _p2l_fp8 = "err"
                self._prof_probe_ns += _time.perf_counter_ns() - _tfp
            _probe_active = (_probe is not None and _probe.mode != "off")
            line = {
                "window_id": self.window_count,
                "steps_actual": self._last_window_steps,
                "protocol_rev": "r3",
                "n_fresh_assignments": n_fresh,
                "s_change_logical": s_change,
                "z": z,
                "h_w_assignments": h_w,
                "h_rate": h_rate,
                "W_before": snap["W"],
                "W_after": w_after,
                "proposed_event": proposed,
                "event": ev,
                "committed": committed,
                "prop_valid": prop_valid,
                "h_min": h_min,
                "h_max": h_max,
                "v_min": v_min,
                "confirm_fail_streak": ctrl.confirm_fail_streak,
                "p2l_version": self._p2l_version,
                "det_map_version": self._det_map_version,
                "alert_k": ctrl.alert_k,
                "stable_s": ctrl.stable_s,
                "dwell": ctrl.dwell,
                "dwell_dir": ctrl.dwell_dir,
                "uncertain_run": ctrl.uncertain_run,
                "episode_shrinks": ctrl.episode_shrinks,
                "carry_fresh": bool(self._v2p1_carry),
                "c_window_ms": elapsed_ms,
                "desync": desync,
                # ---- v2.2-calib extensions (spec v1.1-calib d1-d5/g1) ----
                "spec_rev": "v1.1-calib",
                "probe_mode": (_probe.mode if _probe is not None else "none"),
                "probe_state": _probe_state_at_window,
                "probe_state_after": (_probe.state if _probe_active else ""),
                "probe_action": (_pinfo or {}).get("action", ""),
                "probe_seg": (_pinfo or {}).get("seg", ""),
                "probe_error": _probe_err,
                "true_steps_total": self._true_steps_total,
                "steps_min": steps_min,
                "steps_max": steps_max,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "lockstep_ok": lockstep_ok,
                "span_us_local": _span_us,
                "span_us_min": span_min,
                "span_us_max": span_max,
                "bcost_partial_us_local": _bcost_partial_us,
                "bcost_partial_us_max": bcost_max,
                "prev_bcost_us": self._v22_last_bcost_ns // 1000,
                "t_det_us": _t_phase["det"],
                "t_local_us": _t_phase["local"],
                "t_confirm_us": _t_phase["confirm"],
                "t_commit_us": _t_phase["commit"],
                "prof_probe_cum_ms": self._prof_probe_ns / 1e6,
                "p2l_fp8": _p2l_fp8,
                "probe_excess_cum_us": (_probe.excess_cum_us if _probe_active else None),
                "probe_overshoot_us": (_probe.overshoot_us if _probe_active else None),
                "probe_pending_abort": (_probe.pending_abort if _probe_active else None),
                "probe_flags": (_probe.flags_all() if _probe_active else []),
                "probe_eval": (_probe.eval if (_probe_active
                               and _probe.eval is not None
                               and _probe.eval_window == self.window_count)
                               else None),
            }
            if self._sp_active:
                # ---- sp_coupling_v1 Sec.8 minimal per-H-boundary logging.
                # Rank0's OWN ledgers (per-rank values; NEVER summed across
                # ranks). Timing fields elsewhere in this line stay
                # MAX-reduced; CPU-side costs are not GPU critical-path
                # evidence and are reported separately.
                _lc = self._sp_last_commit or {}
                line.update({
                    "sp_mode": self._sp.mode,
                    "sp_h_id": self._sp_h_cycle,
                    "sp_state_before": _lc.get("state_before",
                                               self._sp.state),
                    "sp_state_after": self._sp.state,
                    "sp_s": self._sp.S,
                    "sp_h": self._sp.H,
                    "sp_w": self._sp.W,
                    "sp_transition": bool(_lc.get("transition")),
                    "sp_reasons": _lc.get("reasons", []),
                    "sp_streak": self._sp.streak,
                    "sp_recover_cycles": self._sp.recover_cycles,
                    "sp_n_min_eff": self._sp.n_min_eff(),
                    "sp_exit_streak": self._sp.exit_streak,
                    "sp_sample_opportunities": self._sp_opps_h,
                    "sp_recorded_steps": self._sp_recorded_h,
                    "sp_valid_rows": _sp_valid_rows,
                    "sp_swaps_started": _sp_swaps_since,
                    "sp_swap_pending": (self.async_executor.pending
                                        is not None),
                    "sp_plan_opportunity": bool(self._sp_plan_opp_h),
                    "sp_prop_valid_cycle": bool(
                        (_sp_prop or {}).get("valid_cycle")),
                    "sp_prop_quiet_cond": bool(
                        (_sp_prop or {}).get("quiet_cond")),
                    "sp_prop_clean_cycle": bool(
                        (_sp_prop or {}).get("clean_cycle")),
                    "sp_seal_reason": _lc.get("seal_reason", ""),
                    "sp_cover_steps": _lc.get("cover_steps", 0),
                    "sp_residency": dict(self._sp.residency),
                })
            self._v2p1_log_fh.write(_json.dumps(line) + "\n")
            self._v2p1_log_fh.flush()
            if self._sp_active:
                # Incremental run-summary persistence (SIGKILL-safe).
                # P0e revision (pre-timing, documented): teardown is
                # pkill -9 (atexit never runs) and fixed arms / quiet
                # runs may never transition, so refresh at EVERY
                # committed H boundary -- rank0-only local write, same
                # frequency class as the jsonl line above.
                self._sp_write_summary()
            _sp_log_part = ""
            if self._sp_active:
                _lc2 = self._sp_last_commit or {}
                _sp_log_part = (
                    f" sp_h#{self._sp_h_cycle} "
                    f"{_lc2.get('state_before', '?')}->{self._sp.state}"
                    f" S/H/W={self._sp.S}/{self._sp.H}/{self._sp.W}"
                    + (" TRANSITION" if _lc2.get("transition") else ""))
            logger.info(f"[PB-OEPLB-ADW21] w#{self.window_count} "
                        f"z={'%.2f' % z if z is not None else 'NA'} "
                        f"hr={h_rate:.5f} N={n_fresh} "
                        f"W {snap['W']}->{w_after} ev={ev} committed={committed}"
                        + _sp_log_part
                        + (f" probe={_probe_state_at_window}/"
                           f"{(_pinfo or {}).get('action') or '-'}/"
                           f"{(_pinfo or {}).get('seg') or '-'}"
                           if _probe_active else ""))
        if err_reason:
            logger.error("[PB-OEPLB-ADW21] v2.1 window issue (held unless "
                         f"jointly committed; serving continues): {err_reason}")

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

    def _probe_run_end_dump(self):
        """v2.2-calib: best-effort probe run-end summary (rank0). Written
        IMMEDIATELY when the machine reaches a terminal state (DONE/ABORTED)
        -- teardown is SIGKILL, so the atexit registration alone would lose
        the summary. atexit re-writes it (idempotent) for non-killed exits."""
        try:
            probe = getattr(self, "_probe", None)
            if probe is None or probe.mode == "off" or self._rank != 0:
                return
            import json as _json
            import os as _os
            summary = probe.summary()
            summary["true_steps_total"] = self._true_steps_total
            summary["window_count"] = self.window_count
            summary["prof_probe_cum_ms"] = self._prof_probe_ns / 1e6
            path = _os.path.join(
                _os.path.dirname(self._v2p1_log_path) or ".",
                "probe_run_end.json")
            with open(path, "w") as f:
                _json.dump(summary, f, indent=1, default=str)
            logger.info("[PB-OEPLB-ADW21-PROBE] run-end summary -> %s "
                        "(state=%s license_used=%s trigger_window=%s)",
                        path, probe.state, probe.license_used,
                        probe.trigger_window)
        except Exception:
            logger.error("[PB-OEPLB-ADW21-PROBE] run-end dump failed: %s",
                         traceback.format_exc())

    def _maybe_report_profile(self):
        """Log cumulative wall-clock cost breakdown every 4 windows."""
        if self.window_count == 0 or self.window_count % 4 != 0:
            return
        record_ms = self._prof_record_ns / 1e6
        allreduce_ms = self._prof_allreduce_ns / 1e6
        planbuild_ms = self._prof_planbuild_ns / 1e6
        finalize_ms = self._prof_finalize_ns / 1e6
        total_ms = record_ms + allreduce_ms + planbuild_ms + finalize_ms
        probe_ms = self._prof_probe_ns / 1e6
        _sp_part = ""
        if self._sp_active:
            # Sec.8: gate cost is this process's CPU clock around the gate
            # computation -- NOT evidence of GPU critical-path exposure;
            # reported separately, never summed across ranks.
            _sp_part = (f" sp_opps={self._sp_opps_total}"
                        f" sp_rec={self._sp_recorded_total}"
                        f" sp_swaps={self._sp_swaps_total}"
                        f" sp_gate_ms={self._prof_sp_gate_ns / 1e6:.2f}"
                        f" sp_state={self._sp.state}"
                        f" sp_h={self._sp_h_cycle}")
        logger.info(
            f"[PB-OEPLB-PROF] window#{self.window_count} calls={self._prof_record_calls} "
            f"record={record_ms:.2f}ms allreduce={allreduce_ms:.2f}ms "
            f"planbuild={planbuild_ms:.2f}ms finalize={finalize_ms:.2f}ms "
            f"total={total_ms:.2f}ms avg_record_us={1000*record_ms/max(1,self._prof_record_calls):.2f} "
            f"probe={probe_ms:.2f}ms avg_probe_us={1000*probe_ms/max(1,self.window_count):.2f} "
            f"last_bcost_us={self._v22_last_bcost_ns // 1000}"
            + _sp_part
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
                # Read current logical mappings from the EVOLVING new_p2l
                # (not from op.logical_a/b which are stale for multi-step ops)
                cur_a = new_p2l[op.layer_id, op.phys_slot_a].item()
                cur_b = new_p2l[op.layer_id, op.phys_slot_b].item()
                new_p2l[op.layer_id, op.phys_slot_a] = cur_b
                new_p2l[op.layer_id, op.phys_slot_b] = cur_a
                update_layers.add(op.layer_id)
            _tf = time.perf_counter_ns()
            new_meta = fast_init_by_mapping(new_p2l, self.num_logical_experts, existing_meta=self._meta)
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
            if self._wctrl_version == 21:
                # ADW v2.1 (F1): mirror the count-follows-expert remap into
                # the detection accumulator so the window-end logical gather
                # under the CURRENT p2l attributes every count (including
                # pre-commit counts) to the right logical expert. The ref
                # EWMA lives in logical space and needs NO remap.
                for op in plan:
                    layer = op.layer_id
                    a, b = op.phys_slot_a, op.phys_slot_b
                    self._det_load[layer, a], self._det_load[layer, b] = (
                        self._det_load[layer, b].clone(),
                        self._det_load[layer, a].clone()
                    )
                self._p2l_version += 1
                self._det_map_version = self._p2l_version

            self._prof_finalize_ns += time.perf_counter_ns() - _tf
            total_ms = (time.perf_counter() - self._pending_plan_start_t) * 1000
            self._cum_swap_s += total_ms / 1000.0
            self.total_swaps += len(plan)
            logger.info(f"[PB-OEPLB] {len(plan)} swap(s) done ({total_ms:.1f}ms) | total={self.total_swaps}")
        except Exception as e:
            logger.error(f"[PB-OEPLB] finalize error: {e}\n{traceback.format_exc()}")
