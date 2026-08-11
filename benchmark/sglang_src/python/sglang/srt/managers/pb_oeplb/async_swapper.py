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

        # Synchronous P2P on default PG -- all ranks participate
        if p2p_ops:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

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
