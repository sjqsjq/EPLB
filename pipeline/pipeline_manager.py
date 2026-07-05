"""
Pipeline with forward-pass-boundary sync (Option A):

- Per LAYER (no cross-GPU collective, safe inside the DeepEP hot path):
  * Cross-layer routing prediction (local GPU tensor ops)
  * Local expert-count accumulation for this forward pass
  * Token sharding decision for THIS layer, using the placement agreed at
    the previous forward-pass boundary (already identical on all ranks)

- Per FORWARD PASS boundary (layer wraps back to 0 — a point where DeepEP's
  own per-layer collective has just completed on ALL ranks, i.e. a natural
  barrier, so our own collective cannot race with DeepEP's communicator):
  * AllReduce the accumulated local demand -> identical global demand on
    every rank
  * Run the Expert Migration solver ONCE with that global demand -> every
    rank computes the IDENTICAL new placement + P2P ops
  * Execute the real P2P weight swap for those ops (all ranks participate
    with the same ops, so no NCCL ordering mismatch)
  * The new placement takes effect for the NEXT forward pass's per-layer
    token sharding decisions
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Optional

import numpy as np
import torch

from .cross_layer_predictor import CrossLayerPredictor
from .algorithms import (
    ExpertMigrationSolver,
    TokenShardingSolver,
    make_initial_placement,
    apply_ops_to_phy2log,
)
from .weight_swapper import WeightSwapper

logger = logging.getLogger(__name__)


class ExpertMigrationPerLayer:
    """Expert migration solver — invoked once per forward pass boundary."""

    def __init__(self, num_experts: int, num_gpus: int, n_local: int,
                 n_max: int = 8, m_budget: int = 159):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.n_local = n_local
        self.n_max = n_max
        self._solver = ExpertMigrationSolver(
            num_experts=num_experts, num_gpus=num_gpus,
            n_local=n_local, n_max=n_max, m_budget=m_budget,
        )
        self._total_time_us = 0.0
        self._call_count = 0
        self._total_cost = 0
        self._total_ops = 0

    def compute_mapping(self, demand_cpu: torch.Tensor,
                        current_placement: np.ndarray,
                        initial_owner: np.ndarray):
        t0 = time.perf_counter()
        counts = demand_cpu.numpy().astype(np.float64)
        new_placement, ops, cost = self._solver.solve(
            demand=counts, current_placement=current_placement,
            initial_owner=initial_owner,
        )
        self._total_time_us += (time.perf_counter() - t0) * 1e6
        self._call_count += 1
        self._total_cost += cost
        self._total_ops += len(ops)
        return new_placement, ops, cost

    def get_stats(self):
        n = self._call_count
        return {
            "call_count": n,
            "total_ms": round(self._total_time_us / 1000, 2),
            "avg_us": round(self._total_time_us / n, 1) if n else 0,
            "avg_transfer_cost": round(self._total_cost / n, 2) if n else 0,
            "avg_ops": round(self._total_ops / n, 2) if n else 0,
        }


class TokenShardingPerLayer:
    """Token sharding — invoked every layer, purely local (no collective)."""

    def __init__(self, num_experts: int, num_gpus: int):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self._solver = TokenShardingSolver(num_experts=num_experts, num_gpus=num_gpus)
        self._total_time_us = 0.0
        self._call_count = 0
        self._total_imbalance = 0.0
        self._total_makespan = 0.0

    def compute_sharding(self, demand_cpu: torch.Tensor,
                         placement: np.ndarray, initial_owner: np.ndarray,
                         topk_ids_cpu: torch.Tensor):
        t0 = time.perf_counter()
        counts = demand_cpu.numpy().astype(np.float64)
        topk_ids_np = topk_ids_cpu.numpy()
        token_to_gpu, makespan, imbalance = self._solver.solve(
            demand=counts, placement=placement,
            initial_owner=initial_owner, topk_ids=topk_ids_np,
        )
        self._total_time_us += (time.perf_counter() - t0) * 1e6
        self._call_count += 1
        self._total_imbalance += imbalance
        self._total_makespan += makespan
        return token_to_gpu, makespan, imbalance

    def get_stats(self):
        n = self._call_count
        return {
            "call_count": n,
            "total_ms": round(self._total_time_us / 1000, 2),
            "avg_us": round(self._total_time_us / n, 1) if n else 0,
            "avg_imbalance": round(self._total_imbalance / n, 4) if n else 0,
            "avg_makespan": round(self._total_makespan / n, 1) if n else 0,
        }


def _sharding_cpu_work(local_topk_ids_cpu, demand_cpu, placement,
                       initial_owner, sharding):
    """Background thread: per-layer LOCAL token sharding (no collective)."""
    token_to_gpu, makespan, imbalance = sharding.compute_sharding(
        demand_cpu, placement, initial_owner, local_topk_ids_cpu
    )
    return {"token_to_gpu": token_to_gpu, "makespan": makespan, "imbalance": imbalance}


class PipelineManager:
    """Per-layer local prediction/sharding + forward-pass-boundary global sync."""

    _instance: Optional["PipelineManager"] = None

    def __init__(self, num_layers: int, num_experts: int, top_k: int,
                 num_gpus: int = 4, n_max: int = 8, m_budget: int = 159):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_gpus = num_gpus

        n_local = num_experts // num_gpus

        self.predictor = CrossLayerPredictor.initialize(num_layers, num_experts, top_k)
        self.migration = ExpertMigrationPerLayer(num_experts, num_gpus, n_local, n_max, m_budget)
        self.sharding = TokenShardingPerLayer(num_experts, num_gpus)

        num_local_physical = (num_experts + n_max * num_gpus) // num_gpus
        self.weight_swapper = WeightSwapper(num_experts, num_gpus, num_local_physical)
        self._expert_weights_ref = None

        self._pg = None
        self._pg_init_attempted = False

        base_placement, initial_owner = make_initial_placement(num_experts, num_gpus)
        self._base_placement = base_placement
        self._current_placement = base_placement.copy()  # agreed at last boundary
        self._initial_owner = initial_owner

        # Per-LAYER demand accumulator for THIS forward pass (GPU tensor,
        # no collective per layer — pure local writes). Shape [num_layers, num_experts].
        # NEVER left unusable across a boundary — every rank must call the
        # boundary's all_reduce unconditionally (even with an all-zero
        # contribution), or ranks that skip it (e.g. an empty local batch on
        # some layers) will deadlock ranks that do call it.
        self._layer_demand_matrix = None  # lazily created on the right GPU device
        self._accum_device = None
        self._matrix_touched = False

        # GLOBALLY-SYNCED per-layer demand from the PREVIOUS forward pass
        # (all_reduced across ranks at the last boundary). Used by the
        # CURRENT pass's per-layer token-sharding decisions so every rank
        # makes a CONSISTENT choice about which replica to prefer for a
        # given expert, without needing a live collective every layer.
        self._last_pass_layer_demand = torch.zeros(num_layers, num_experts)

        # Weight swap is real NCCL P2P (unlike the Gloo demand sync) and is
        # the most collision-prone part under heavy concurrent load. Throttle
        # how often we actually touch physical weights; the demand sync +
        # migration DECISION still happens every boundary (cheap, Gloo-safe).
        self._swap_interval = 1  # STRESS TEST: swap every boundary
        self._boundary_count = 0

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._pending_future: Optional[Future] = None
        self._last_result = None  # exposed to dispatch hook

        # Disabled by default — enabled explicitly after CUDA graph
        # capture/warmup completes (see launch_with_pipeline.py monkeypatch).
        # Warmup runs 2x dry-run forward passes per cuda_graph_bs OUTSIDE any
        # torch.cuda.graph() context (so is_current_stream_capturing()==False),
        # on synthetic data with no real rank-to-rank sync guarantees — running
        # our cross-rank collectives / P2P weight swap there causes deadlocks.
        self.enabled = False
        self._forward_count = 0
        self._current_layer = -1
        self._per_layer_total_us = []
        self._boundary_time_us = []

        logger.info(
            f"[PipelineManager] forward-pass-boundary sync pipeline: "
            f"layers={num_layers}, experts={num_experts}, top_k={top_k}, "
            f"gpus={num_gpus}, n_local={n_local}, n_max={n_max}, m_budget={m_budget}"
        )

    def _ensure_process_group(self):
        """Dedicated Gloo (CPU/network) process group for the pipeline's own
        all_reduce. Gloo never touches the GPU's CUDA stream, so it cannot
        create false in-order dependencies with DeepEP's NCCL collectives —
        NCCL-backed groups (even distinct ones) still enqueue kernels on the
        same physical CUDA stream and can block behind/ahead of DeepEP's own
        dispatch/combine ops depending on per-rank timing skew."""
        if self._pg_init_attempted:
            return self._pg
        self._pg_init_attempted = True
        try:
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                self._pg = torch.distributed.new_group(
                    ranks=list(range(world_size)), backend="gloo"
                )
                logger.info(f"[PipelineManager] created dedicated GLOO process group, world_size={world_size}")
        except Exception as e:
            logger.warning(f"[PipelineManager] failed to create process group: {e}")
            self._pg = None
        return self._pg

    def register_model_weights(self, routed_experts_weights_of_layer):
        if self._expert_weights_ref is None:
            self._expert_weights_ref = {}
        if isinstance(routed_experts_weights_of_layer, dict):
            self._expert_weights_ref.update(routed_experts_weights_of_layer)
        logger.info(f"[PipelineManager] registered expert weights for {len(self._expert_weights_ref)} layers")

    @classmethod
    def get_instance(cls): return cls._instance

    @classmethod
    def initialize(cls, num_layers, num_experts, top_k, num_gpus=4, n_max=8, m_budget=159):
        if cls._instance is None:
            cls._instance = cls(num_layers, num_experts, top_k, num_gpus, n_max, m_budget)
        return cls._instance

    def on_moe_layer(self, layer_id, gate_input, gate_weight, actual_topk_ids,
                     correction_bias=None):
        if not self.enabled:
            return

        t0 = time.perf_counter()

        # Forward pass boundary: layer wraps back to 0 → do the global sync now,
        # BEFORE this layer's local work, using data accumulated over the PREVIOUS
        # forward pass's layers.
        if layer_id == 0 and self._current_layer > 0:
            self._on_forward_pass_boundary()
        self._current_layer = layer_id

        # --- Per-layer LOCAL work (no cross-GPU collective) ---

        # Capture device unconditionally, so the boundary can ALWAYS build a
        # zero-fallback tensor for ranks that saw no real prediction this pass.
        self._accum_device = gate_input.device

        # 1) Cross-layer routing prediction
        prediction = self.predictor.on_moe_forward(
            layer_id, gate_input, gate_weight, actual_topk_ids, correction_bias
        )

        if prediction and "predicted_topk_ids" in prediction:
            predicted_topk_ids = prediction["predicted_topk_ids"]
            self._accum_device = predicted_topk_ids.device

            # Write THIS layer's local demand into the per-layer matrix
            # (pure GPU op, no sync) — used for the migration decision (summed
            # across layers) AND, from the NEXT pass onward, as the reference
            # for OTHER ranks' sharding decisions at this same layer_id.
            local_counts = torch.bincount(
                predicted_topk_ids.flatten(), minlength=self.num_experts
            ).float()
            if self._layer_demand_matrix is None:
                self._layer_demand_matrix = torch.zeros(
                    self.num_layers, self.num_experts, device=predicted_topk_ids.device
                )
            if layer_id < self._layer_demand_matrix.shape[0]:
                self._layer_demand_matrix[layer_id] = local_counts  # stays on GPU, no sync
                self._matrix_touched = True

            # 2) Per-layer token sharding: use the GLOBALLY-SYNCED demand for
            #    THIS layer from the PREVIOUS pass (identical on all ranks,
            #    no live collective needed) so every rank makes a CONSISTENT
            #    replica choice, instead of only seeing its own local tokens.
            if self._pending_future is not None:
                try:
                    prev = self._pending_future.result(timeout=0.1)
                    if prev is not None:
                        self._last_result = prev
                except Exception:
                    pass
                self._pending_future = None

            if layer_id < self._last_pass_layer_demand.shape[0]:
                global_demand_cpu = self._last_pass_layer_demand[layer_id]
            else:
                global_demand_cpu = local_counts.cpu()

            topk_ids_cpu = predicted_topk_ids.cpu()
            self._pending_future = self._executor.submit(
                _sharding_cpu_work, topk_ids_cpu, global_demand_cpu,
                self._current_placement, self._initial_owner, self.sharding,
            )

        total_us = (time.perf_counter() - t0) * 1e6
        self._per_layer_total_us.append(total_us)

    def on_empty_layer(self, layer_id):
        """Called when this rank's local batch is empty at this layer.
        Still advances layer tracking and participates in the boundary's
        cross-rank collective (with zero contribution), so ranks with empty
        batches never silently skip a collective that other ranks call."""
        if not self.enabled:
            return
        if layer_id == 0 and self._current_layer > 0:
            self._on_forward_pass_boundary()
        self._current_layer = layer_id

    def _check_map_consistency(self, pg, layer_id, current_map):
        """Verify all ranks have the IDENTICAL old physical_to_logical_map
        for this layer before attempting P2P weight swap. Uses the same safe
        Gloo channel. Returns True iff all ranks match."""
        if pg is None:
            return True
        try:
            local = torch.tensor(current_map, dtype=torch.int64)
            gathered = [torch.zeros_like(local) for _ in range(torch.distributed.get_world_size(group=pg))]
            torch.distributed.all_gather(gathered, local, group=pg)
            ref = gathered[0]
            for r, g in enumerate(gathered[1:], start=1):
                if not torch.equal(ref, g):
                    diff_idx = (ref != g).nonzero(as_tuple=True)[0].tolist()
                    logger.warning(
                        f"[PipelineManager] map mismatch layer={layer_id} rank0_vs_rank{r}: "
                        f"{len(diff_idx)} differing slots, e.g. idx={diff_idx[:5]} "
                        f"rank0={[ref[i].item() for i in diff_idx[:5]]} "
                        f"rank{r}={[g[i].item() for i in diff_idx[:5]]}"
                    )
                    return False
            return True
        except Exception as e:
            logger.warning(f"[PipelineManager] consistency check failed: {e}")
            return False

    def _on_forward_pass_boundary(self):
        """Safe cross-GPU sync point: runs right after the previous forward
        pass's last layer combine, before this forward pass's layer-0 dispatch.
        No DeepEP collective is in flight at this exact point."""
        t0 = time.perf_counter()
        self._forward_count += 1
        self.predictor.reset_state()

        if self._pending_future is not None:
            try:
                self._pending_future.result(timeout=1.0)
            except Exception:
                pass
            self._pending_future = None

        self._boundary_count += 1
        do_swap_this_boundary = (self._boundary_count % self._swap_interval == 0)

        if self._expert_weights_ref is not None:
            try:
                pg = self._ensure_process_group()

                # ONE Gloo transfer per boundary for the WHOLE per-layer matrix
                # (num_layers x num_experts, a few tens of KB) — replaces what
                # would otherwise be a per-layer sync.
                if self._matrix_touched and self._layer_demand_matrix is not None:
                    layer_matrix_cpu = self._layer_demand_matrix.cpu()
                else:
                    layer_matrix_cpu = torch.zeros(self.num_layers, self.num_experts)

                if pg is not None:
                    torch.distributed.all_reduce(
                        layer_matrix_cpu, op=torch.distributed.ReduceOp.SUM, group=pg
                    )

                # This becomes the reference for the NEXT pass's per-layer
                # sharding decisions on every rank (identical, already synced).
                self._last_pass_layer_demand = layer_matrix_cpu

                # Aggregate demand across all layers -> migration decision input
                global_counts_cpu = layer_matrix_cpu.sum(dim=0)

                new_placement, ops, cost = self.migration.compute_mapping(
                    global_counts_cpu, self._base_placement, self._initial_owner
                )

                if ops and do_swap_this_boundary:
                    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
                    metadata = get_global_expert_location_metadata()
                    if metadata is not None:
                        total_phy = metadata.physical_to_logical_map_cpu.shape[1]
                        per_gpu = total_phy // self.num_gpus
                        for layer_id in list(self._expert_weights_ref.keys())[:1]:
                            current_map = metadata.physical_to_logical_map_cpu[layer_id].tolist()

                            # DIAGNOSTIC: verify all ranks agree on the OLD map
                            # before attempting P2P. A prior swap round that left
                            # metadata inconsistent would make ranks compute
                            # asymmetric (non-matching) send/recv ops here,
                            # deadlocking batch_isend_irecv permanently.
                            consistent = self._check_map_consistency(
                                pg, layer_id, current_map
                            )
                            if not consistent:
                                logger.warning(
                                    f"[PipelineManager] SKIPPING swap for layer {layer_id}: "
                                    f"ranks disagree on current physical_to_logical_map"
                                )
                                continue

                            new_map = apply_ops_to_phy2log(ops, current_map, self.num_gpus, per_gpu)
                            changes = sum(1 for a, b in zip(current_map, new_map) if a != b)
                            if 0 < changes <= 30:
                                # Drain this rank's own GPU queue before issuing
                                # the real NCCL P2P weight transfer, to reduce
                                # cross-rank timing skew relative to DeepEP's
                                # own collective at the moment of the swap.
                                torch.cuda.synchronize()
                                self.weight_swapper.swap_for_layer(
                                    layer_id, new_map, self._expert_weights_ref,
                                    gloo_group=pg,
                                )

                self._current_placement = new_placement
            except Exception as e:
                logger.warning(f"[PipelineManager] boundary sync error: {e}")

        self._layer_demand_matrix = None
        self._matrix_touched = False

        elapsed_us = (time.perf_counter() - t0) * 1e6
        self._boundary_time_us.append(elapsed_us)

        if self._forward_count % 20 == 0:
            avg_layer = sum(self._per_layer_total_us) / len(self._per_layer_total_us) if self._per_layer_total_us else 0
            avg_boundary = sum(self._boundary_time_us) / len(self._boundary_time_us) if self._boundary_time_us else 0
            logger.info(
                f"[PipelineManager] fwd#{self._forward_count} | "
                f"avg_per_layer={avg_layer:.0f}us avg_boundary={avg_boundary:.0f}us | "
                f"predict: {self.predictor.get_stats()} | "
                f"migrate: {self.migration.get_stats()} | "
                f"shard: {self.sharding.get_stats()} | "
                f"swap: {self.weight_swapper.get_stats()}"
            )
            self._per_layer_total_us.clear()
            self._boundary_time_us.clear()

    def get_last_result(self):
        return self._last_result

    def get_stats(self):
        avg_layer = sum(self._per_layer_total_us) / len(self._per_layer_total_us) if self._per_layer_total_us else 0
        return {
            "forward_count": self._forward_count,
            "avg_per_layer_us": round(avg_layer, 1),
            "predictor": self.predictor.get_stats(),
            "migration": self.migration.get_stats(),
            "sharding": self.sharding.get_stats(),
            "weight_swapper": self.weight_swapper.get_stats(),
        }
