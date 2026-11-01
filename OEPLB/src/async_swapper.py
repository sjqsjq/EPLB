import logging
import time
import torch
import torch.distributed

logger = logging.getLogger(__name__)


class AsyncSwapExecutor:
    """
    Synchronous pairwise expert-weight swap executor for PB-OEPLB.

    Previous async design (dedicated stream + separate PG) caused NCCL hangs
    after ~60s of normal forward passes post-swap. Root cause: multi-PG
    concurrent usage on the same GPU corrupts NCCL internal state (observed
    as allgather timeout on the forward-pass PG, not the swap PG itself).

    New design: swap is performed SYNCHRONOUSLY during _decide_and_begin_swap()
    -- all ranks are already synchronized at that point (just did all_reduce),
    so batch_isend_irecv on the default PG is safe. The blocking duration is
    acceptable (~200ms for typical 9-92 ops, measured empirically on H20).

    This matches the official ExpertLocationUpdater's approach (synchronous
    batch_isend_irecv + req.wait()) which has no stability issues.

    IMPORTANT -- the plan must be transferred in a SINGLE batch_isend_irecv.
    Splitting it into chunks to bound the receive-buffer memory looks safe but
    is not: within a chunk a rank posts ops only if it owns one of the two slots
    being swapped, so ranks that own nothing in that chunk skip the call
    entirely. ProcessGroupNCCL numbers each coalesced work item per process
    group, so uneven participation makes the sequence numbers diverge and the
    next collective deadlocks (measured: ranks 0/1 completed all 9 chunks while
    2/5/6 stalled, watchdog reported SeqNum=4 on some ranks and SeqNum=5 on
    others, 600 s timeout). With one batch every rank participates exactly once.
    Bound the memory with --pb-oeplb-max-total-ops instead.
    """

    def __init__(self, model_runner, my_rank: int, num_local: int):
        self.model_runner = model_runner
        self.my_rank = my_rank
        self.num_local = num_local
        self.pending = None  # dict: plan, result_ready

    @property
    def busy(self) -> bool:
        return self.pending is not None

    def _get_routed_weights(self):
        """Model-agnostic access to routed expert weights."""
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
        """Execute swap SYNCHRONOUSLY. All ranks must call this together.

        This replaces the old async design. The swap runs on the default stream
        and default PG, blocking until complete. Since all ranks are synchronized
        (they just did all_reduce in _decide_and_begin_swap), this is safe.
        """
        if self.busy:
            raise RuntimeError("AsyncSwapExecutor: begin() called while a swap is in flight")

        _t_begin = time.perf_counter()
        routed_weights = self._get_routed_weights()
        p2p_ops = []
        op_temps = [None] * len(plan)

        # VERIFY-WEIGHT-MOVE: checksum before swap
        if plan and (self.my_rank == plan[0].rank_a or self.my_rank == plan[0].rank_b):
            op0 = plan[0]
            w0 = routed_weights[op0.layer_id]
            la0 = op0.phys_slot_a - op0.rank_a * self.num_local
            lb0 = op0.phys_slot_b - op0.rank_b * self.num_local
            local_slot = la0 if self.my_rank == op0.rank_a else lb0
            checksum_before = float(w0[0][local_slot].float().sum().item())
            logger.info(f"[VERIFY-WEIGHT-MOVE] rank={self.my_rank} BEFORE swap "
                        f"layer={op0.layer_id} phys_a={op0.phys_slot_a}(rank{op0.rank_a}) "
                        f"phys_b={op0.phys_slot_b}(rank{op0.rank_b}) "
                        f"my_local_slot={local_slot} weight0_checksum={checksum_before:.6f}")
            self._verify_checksum_before = checksum_before
            self._verify_local_slot = local_slot
            self._verify_layer_id = op0.layer_id
        else:
            self._verify_checksum_before = None

        for i, op in enumerate(plan):
            weights = routed_weights[op.layer_id]
            local_a = op.phys_slot_a - op.rank_a * self.num_local
            local_b = op.phys_slot_b - op.rank_b * self.num_local

            if self.my_rank == op.rank_a == op.rank_b:
                # Same-rank swap: pure local copy
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
                op_temps[i] = {}

        _t_alloc = time.perf_counter()

        # NCCL allocates its P2P channel buffers with a raw cudaMalloc, i.e.
        # outside the PyTorch caching allocator. After a large prefill the
        # allocator can be holding every free block, so that raw alloc fails
        # (observed: "Failed to CUDA calloc 10485760 bytes" on 3 of 8 ranks on
        # the first, largest swap plan -- 132 ops x ~27.5 MB of fp8 expert
        # weights). Returning cached blocks to the driver first gives NCCL room.
        # Cost is a few ms and it only matters on the first swap, when the
        # channel buffers are created.
        if p2p_ops:
            torch.cuda.empty_cache()
            # MIGRATION-RACE FIX-1 (final_followup_20261020 r1): drain all
            # in-flight serving kernels before posting the coalesced P2P
            # group so the rendezvous runs on quiesced GPUs. Root cause of
            # the 7 retired final_plan boots: a large-plan (~300 op)
            # batch_isend_irecv group overlapped with DeepEP spin kernels
            # of the NEXT forward on ranks that returned from begin()
            # early -> circular SM/rendezvous wait -> 600s NCCL watchdog.
            # Sync cost is intentionally retained (per followup directive).
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Synchronous P2P on default PG -- all ranks participate.
        # MIGRATION-RACE FIX-3 (final_followup_20261020 r2): post the plan as
        # PER-PEER SUBGROUPS in ascending peer order instead of one giant
        # coalesced group. Root cause of the B1_P_E3 window#17 hang (same
        # signature as retired m_E3_P_b3): torch batch_isend_irecv() wraps all
        # ops in a single ncclGroup; NCCL splits large p2p groups into
        # multiple SEQUENTIAL kernel plans via an internal budget
        # (scheduleP2pTasksToPlan), and its own source notes the cut point
        # "could vary by rank" (p2p work isn't collective). With the
        # asymmetric per-rank participation of real swap plans, matched
        # send/recv pairs land in different plan indices on the two sides ->
        # circular wait between sequentially-executing plan kernels ->
        # silent all-posted deadlock (upstream: pytorch/pytorch#174288, open,
        # repro'd on NCCL<=2.29; this deployment: torch 2.9.1 / NCCL 2.27.5).
        # Proven offline on 8xGPU (followup correctness/gpu_harness/
        # test_split_deadlock_gpu.py): identical 299-op partitioned skewed
        # plans deadlock the single-group path at iter 0 with the exact B1
        # signature (plan hashes identical, all stuck ranks enqueued the
        # COALESCED work, kernels resident, no NCCL WARN, watchdog timeout,
        # opposite cluster completes and waits) and pass the per-peer path
        # 40/40 iters with canary_bad=0.
        # Why per-peer is safe: both sides of a pair derive mirror-symmetric
        # op lists (same count/size/order) from the identical plan -> equal
        # budget cuts -> matched plan boundaries. Ascending peer order with a
        # host wait per subgroup is deadlock-free (any blocked chain strictly
        # descends in rank, contradiction at the minimum). Same-peer
        # send/recv stay in one group (gang safety preserved). Plan content,
        # ordering, candidate rules, thresholds and action budgets are
        # untouched; the added subgroup serialization cost is intentionally
        # retained and measured in the TIMING line (per followup directive).
        # A per-rank failure here would leave the ranks desynced (some
        # complete their transfers, some do not) and hang the server until
        # the watchdog fires, so retry each subgroup once rather than letting
        # the exception escape to the caller's blanket `except Exception`.
        if p2p_ops:
            _by_peer = {}
            for _pop in p2p_ops:
                _by_peer.setdefault(_pop.peer, []).append(_pop)
            _peers = sorted(_by_peer)
            # DIAG-2 (r2): per-peer send/recv histogram -> pairwise mirror
            # symmetry (rank i n_send->j == rank j n_recv<-i) becomes
            # measured on every live window. Pure logging, no behavior change.
            _hist = [[p,
                      sum(1 for o in _by_peer[p] if o.op is torch.distributed.isend),
                      sum(1 for o in _by_peer[p] if o.op is torch.distributed.irecv)]
                     for p in _peers]
            logger.info(f"[PB-OEPLB-P2PPEERS] rank={self.my_rank} "
                        f"n_p2p_ops={len(p2p_ops)} n_subgroups={len(_peers)} "
                        f"peer_send_recv={_hist}")
            for _peer in _peers:
                _sub = _by_peer[_peer]
                try:
                    reqs = torch.distributed.batch_isend_irecv(_sub)
                    for req in reqs:
                        req.wait()
                except Exception as ex:
                    logger.error(f"[PB-OEPLB] rank={self.my_rank} P2P swap "
                                 f"subgroup(peer={_peer}) failed "
                                 f"({type(ex).__name__}: {ex}); retrying this "
                                 f"subgroup once after empty_cache()")
                    torch.cuda.empty_cache()
                    reqs = torch.distributed.batch_isend_irecv(_sub)
                    for req in reqs:
                        req.wait()
            # MIGRATION-RACE FIX-2 (final_followup_20261020 r1, retained):
            # req.wait() on NCCL work is stream-ordered (host non-blocking);
            # enforce true completion before begin() returns so no rank can
            # run ahead into the next forward while peers' group kernels are
            # still pending.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        _t_p2p = time.perf_counter()

        # Copy received data to live weights
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

        # VERIFY-WEIGHT-MOVE: checksum after swap
        if getattr(self, '_verify_checksum_before', None) is not None:
            w0 = routed_weights[self._verify_layer_id]
            checksum_after = float(w0[0][self._verify_local_slot].float().sum().item())
            changed = abs(checksum_after - self._verify_checksum_before) > 1e-3
            logger.info(f"[VERIFY-WEIGHT-MOVE] rank={self.my_rank} AFTER swap "
                        f"layer={self._verify_layer_id} local_slot={self._verify_local_slot} "
                        f"weight0_checksum_before={self._verify_checksum_before:.6f} "
                        f"weight0_checksum_after={checksum_after:.6f} "
                        f"CHANGED={changed}")
            self._verify_checksum_before = None

        _t_done = time.perf_counter()
        logger.info(f"[PB-OEPLB-TIMING] begin(): temp_alloc={(_t_alloc-_t_begin)*1000:.1f}ms "
                    f"batch_isend_irecv={(_t_p2p-_t_alloc)*1000:.1f}ms "
                    f"copy={(_t_done-_t_p2p)*1000:.1f}ms "
                    f"total_begin={(_t_done-_t_begin)*1000:.1f}ms n_ops={len(plan)}")

        # Mark as complete immediately (synchronous execution)
        self.pending = {"plan": plan, "result_ready": True}

    def try_finish(self, force_wait: bool = False):
        """Return completed plan. Since begin() is now synchronous, this always
        returns the plan immediately if one is pending."""
        if not self.busy:
            return None
        plan = self.pending["plan"]
        self.pending = None
        return plan
