import logging
import torch
import torch.distributed

logger = logging.getLogger(__name__)


class AsyncSwapExecutor:
    """
    Self-contained async pairwise expert-weight swap executor, purpose-built for
    PB-OEPLB's SwapOp semantics (exactly two physical slots exchanging content
    between exactly two ranks). Does NOT touch SGLang's shared ExpertLocationUpdater
    (used by the official EPLB feature) to avoid risking that code path.

    Design (per the diagnosed root cause: batch_isend_irecv on model_runner's
    forward_stream + a synchronous req.wait() blocks the single-threaded scheduler
    event loop for the full P2P transfer duration):

      1. begin(plan): issue all P2P ops for `plan` on a dedicated low-priority
         CUDA stream, record a completion event. Returns immediately (no CPU wait).
      2. try_finish(): non-blocking `event.query()`. Only when the event has
         actually fired (GPU-confirmed complete, not merely enqueued) does it do
         the (fast, no-NCCL) shadow-buffer -> live-weight copy_() and return the
         plan so the caller can flip ExpertLocationMetadata's routing tables.

      Until try_finish() confirms completion, the routing tables are NOT flipped,
      so concurrent forward passes keep dispatching to the OLD physical slots
      (whose weights have not yet been overwritten) -- no torn-read/misroute
      hazard. The event-based cross-stream check (rather than a synchronize())
      guarantees the shadow->live copy only starts after the transfer has
      genuinely completed on the GPU.

    BUGFIX (found via live diagnostic on 2026-07-15): temp buffers used to be
    keyed by (layer_id, 'a'|'b') in a shared dict. When max_swaps_per_layer>=2
    and the same rank keeps the same hot/cold role across multiple rounds
    within one layer (the common case -- one swap rarely fully drains a
    genuinely hot rank), the second round's buffer silently overwrote the
    first round's dict entry before try_finish() ever read it back. Confirmed
    empirically: 4100 collisions in 80s of traffic with max_swaps_per_layer=5
    (148 completed swap decisions). Net effect: one physical slot's rightful
    incoming weights were discarded and replaced with a *different* expert's
    weights, while ExpertLocationMetadata's physical_to_logical_map was
    updated as if the swap succeeded correctly -- i.e. the routing table and
    the actual weight data silently diverged. Fix: store temps per-op, indexed
    by position in `plan` (op_temps[i] corresponds 1:1 to plan[i]), not in a
    dict keyed by anything that could collide across ops.
    """

    def __init__(self, model_runner, my_rank: int, num_local: int):
        self.model_runner = model_runner
        self.my_rank = my_rank
        self.num_local = num_local
        self.stream = torch.cuda.Stream(priority=-1)
        self.pending = None  # dict: plan, op_temps, event

    @property
    def busy(self) -> bool:
        return self.pending is not None


    def _get_routed_weights(self):
        """Model-agnostic access to routed expert weights (see controller's
        _get_routed_experts_weights for full rationale). Returns
        {layer_id: List[Tensor]}."""
        model = self.model_runner.model
        if hasattr(model, "routed_experts_weights_of_layer"):
            return model.routed_experts_weights_of_layer
        result = {}
        layers = getattr(getattr(model, "model", model), "layers", None)
        if layers is None:
            raise RuntimeError("PB-OEPLB async_swapper: cannot find model.layers")
        for layer_id, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "get_moe_weights"):
                result[layer_id] = mlp.get_moe_weights()
        return result

    def begin(self, plan):
        """Issue P2P ops for `plan` (List[SwapOp]) on the dedicated stream. Non-blocking."""
        if self.busy:
            raise RuntimeError("AsyncSwapExecutor: begin() called while a swap is in flight")

        import time as _time
        _t_begin_start = _time.perf_counter()
        routed_weights = self._get_routed_weights()
        op_temps = [None] * len(plan)  # op_temps[i] = {"a": [Tensor,...]} or {"b": [...]} or {} or {"local": True}
        p2p_ops = []

        # VERIFY-WEIGHT-MOVE: checksum the first op's weight slots BEFORE the
        # swap, on whichever rank(s) are actually involved (rank_a or rank_b).
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        if plan and (self.my_rank == plan[0].rank_a or self.my_rank == plan[0].rank_b):
            op0 = plan[0]
            w0 = routed_weights[op0.layer_id]
            la0 = op0.phys_slot_a - op0.rank_a * self.num_local
            lb0 = op0.phys_slot_b - op0.rank_b * self.num_local
            local_slot = la0 if self.my_rank == op0.rank_a else lb0
            checksum_before = float(w0[0][local_slot].float().sum().item())
            _logger.info(f"[VERIFY-WEIGHT-MOVE] rank={self.my_rank} BEFORE swap "
                        f"layer={op0.layer_id} phys_a={op0.phys_slot_a}(rank{op0.rank_a}) "
                        f"phys_b={op0.phys_slot_b}(rank{op0.rank_b}) "
                        f"my_local_slot={local_slot} weight0_checksum={checksum_before:.6f}")
            self._verify_checksum_before = checksum_before
            self._verify_local_slot = local_slot
            self._verify_layer_id = op0.layer_id
        else:
            self._verify_checksum_before = None

        for i, op in enumerate(plan):
            weights = routed_weights[op.layer_id]  # List[Tensor], each (num_local, ...)
            local_a = op.phys_slot_a - op.rank_a * self.num_local
            local_b = op.phys_slot_b - op.rank_b * self.num_local

            if self.my_rank == op.rank_a == op.rank_b:
                # Same-rank swap: pure local copy, no P2P needed. Handle immediately
                # via a same-shape temp swap (cheap, on default stream is fine since
                # it doesn't involve any other rank).
                for w in weights:
                    tmp = w[local_a].clone()
                    w[local_a].copy_(w[local_b])
                    w[local_b].copy_(tmp)
                op_temps[i] = {}
                continue

            if self.my_rank == op.rank_a:
                temp = [torch.empty_like(w[local_a]) for w in weights]
                op_temps[i] = {"a": temp}
                for wi, w in enumerate(weights):
                    p2p_ops.append(torch.distributed.P2POp(torch.distributed.isend, w[local_a], op.rank_b))
                    p2p_ops.append(torch.distributed.P2POp(torch.distributed.irecv, temp[wi], op.rank_b))
            elif self.my_rank == op.rank_b:
                temp = [torch.empty_like(w[local_b]) for w in weights]
                op_temps[i] = {"b": temp}
                for wi, w in enumerate(weights):
                    p2p_ops.append(torch.distributed.P2POp(torch.distributed.irecv, temp[wi], op.rank_a))
                    p2p_ops.append(torch.distributed.P2POp(torch.distributed.isend, w[local_b], op.rank_a))
            else:
                op_temps[i] = {}  # this rank is not involved in this particular SwapOp.

        _t_temps_done = _time.perf_counter()
        with torch.cuda.stream(self.stream):
            reqs = torch.distributed.batch_isend_irecv(p2p_ops) if p2p_ops else []
            event = torch.cuda.Event()
            event.record(self.stream)
        _t_issue_done = _time.perf_counter()
        logger.info(f"[PB-OEPLB-TIMING] begin(): temp_alloc={(_t_temps_done-_t_begin_start)*1000:.1f}ms "
                    f"batch_isend_irecv={(_t_issue_done-_t_temps_done)*1000:.1f}ms "
                    f"total_begin={(_t_issue_done-_t_begin_start)*1000:.1f}ms n_ops={len(plan)}")

        self.pending = {"plan": plan, "op_temps": op_temps, "event": event, "reqs": reqs}

    def try_finish(self, force_wait: bool = False):
        """Non-blocking check by default. Returns the completed plan (List[SwapOp])
        once the transfer is confirmed done, else None (caller should retry later).

        BUGFIX (deadlock, found via py-spy dump showing all 8 ranks hung inside
        the SAME torch.distributed.all_reduce call in controller.py): P2P ops are
        issued on a dedicated low-priority CUDA stream so begin() doesn't block
        the scheduler. But NCCL requires operations on a shared communicator to
        be issued in the SAME RELATIVE ORDER by every rank, regardless of which
        local CUDA stream enqueued them. If one rank's CPU thread reaches the
        NEXT window's all_reduce (default stream) before another rank's P2P ops
        from the PREVIOUS window have actually been submitted to NCCL (low-priority
        stream scheduling can lag arbitrarily under load), the ranks' respective
        NCCL op sequences diverge in TYPE (one rank's next op is "collective
        all_reduce", another's is still "P2P send/recv") -- the communicator can
        never resolve this and hangs forever, on every rank, in whichever call
        happens to be next. force_wait=True (used right before issuing the next
        window's all_reduce -- see controller.py's _decide_and_begin_swap) blocks
        until the pending transfer is GPU-confirmed complete, restoring strict
        cross-rank ordering at that one synchronization point, without giving up
        the non-blocking fast path for every other forward pass in between."""
        if not self.busy:
            return None
        if force_wait:
            self.pending["event"].synchronize()
        elif not self.pending["event"].query():
            return None  # still in flight; do not block

        plan = self.pending["plan"]
        op_temps = self.pending["op_temps"]
        routed_weights = self._get_routed_weights()

        for i, op in enumerate(plan):
            weights = routed_weights[op.layer_id]
            local_a = op.phys_slot_a - op.rank_a * self.num_local
            local_b = op.phys_slot_b - op.rank_b * self.num_local
            temp = op_temps[i]
            if "a" in temp:
                for wi, w in enumerate(weights):
                    w[local_a].copy_(temp["a"][wi])
            if "b" in temp:
                for wi, w in enumerate(weights):
                    w[local_b].copy_(temp["b"][wi])

        if getattr(self, '_verify_checksum_before', None) is not None:
            import logging as _logging
            _logger = _logging.getLogger(__name__)
            w0 = routed_weights[self._verify_layer_id]
            checksum_after = float(w0[0][self._verify_local_slot].float().sum().item())
            changed = abs(checksum_after - self._verify_checksum_before) > 1e-3
            _logger.info(f"[VERIFY-WEIGHT-MOVE] rank={self.my_rank} AFTER swap "
                        f"layer={self._verify_layer_id} local_slot={self._verify_local_slot} "
                        f"weight0_checksum_before={self._verify_checksum_before:.6f} "
                        f"weight0_checksum_after={checksum_after:.6f} "
                        f"CHANGED={changed}")
            self._verify_checksum_before = None

        self.pending = None
        return plan
