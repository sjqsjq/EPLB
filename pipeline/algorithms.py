"""
Expert Migration & Token Sharding Algorithms for per-layer load balancing.

Expert Migration: three-phase greedy algorithm that adjusts expert placement
(replicate/evict/migrate) based on predicted token demand.

Token Sharding: two-phase algorithm that assigns tokens to GPUs to minimize
makespan (closed-form + water-filling) then minimize communication (locality-first).
"""
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np
import torch


class OpType(Enum):
    REPLICATE = "replicate"
    EVICT = "evict"
    MIGRATE = "migrate"


@dataclass
class MigrationOp:
    op_type: OpType
    expert_id: int
    src_gpu: int
    dst_gpu: int  # -1 for evict


class ExpertMigrationSolver:
    """
    Three-phase greedy expert migration algorithm.

    Phase A: Identify bottleneck experts and GPUs.
    Phase B: Replicate hot experts onto underloaded GPUs (up to max_replicate_iters).
    Phase C: Evict/migrate cold experts off overloaded GPUs (up to max_migrate_iters).
    """

    def __init__(
        self,
        num_experts: int,
        num_gpus: int,
        n_local: int,
        n_max: int,
        m_budget: int,
        max_replicate_iters: int = 32,
        max_migrate_iters: int = 16,
    ):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.n_local = n_local
        self.n_max = n_max
        self.capacity = n_local + n_max
        self.m_budget = m_budget
        self.max_replicate_iters = max_replicate_iters
        self.max_migrate_iters = max_migrate_iters

    def solve(
        self,
        demand: np.ndarray,
        current_placement: np.ndarray,
        initial_owner: np.ndarray,
    ) -> Tuple[np.ndarray, List[MigrationOp], int]:
        """
        Args:
            demand: [E] float, predicted token count per expert.
            current_placement: [E, R] bool, current expert-GPU assignment.
            initial_owner: [E] int, home GPU for each expert.

        Returns:
            new_placement: [E, R] bool, updated placement.
            ops: list of MigrationOp.
            cost: total number of P2P transfers.
        """
        E, R = self.num_experts, self.num_gpus
        placement = current_placement.copy()
        d = demand.astype(np.float64)
        ops: List[MigrationOp] = []
        cost = 0

        # --- Phase A: compute bookkeeping ---
        k = placement.sum(axis=1).astype(np.float64)  # [E] replica counts
        k = np.maximum(k, 1.0)
        slots_used = placement.sum(axis=0).astype(np.int64)  # [R]

        # gpu_load[r] = sum of d[e]/k[e] for experts on GPU r
        share = d / k  # [E] per-replica share
        gpu_load = (placement * share[:, None]).sum(axis=0).astype(np.float64)  # [R]

        total_demand = d.sum()
        mu = total_demand / R

        # --- Phase B: replicate hot experts ---
        tried = np.zeros(E, dtype=bool)
        for _ in range(self.max_replicate_iters):
            if cost >= self.m_budget:
                break

            effective_load = np.where(tried, 0.0, d / k)  # [E]
            best_e = int(np.argmax(effective_load))

            if effective_load[best_e] <= mu:
                break

            candidates_mask = (~placement[best_e]) & (slots_used < self.capacity)
            if not candidates_mask.any():
                tried[best_e] = True
                continue

            candidate_loads = np.where(candidates_mask, gpu_load, np.inf)
            best_r = int(np.argmin(candidate_loads))

            old_k = k[best_e]
            new_k = old_k + 1
            new_target_load = gpu_load[best_r] + d[best_e] / new_k
            current_max = gpu_load.max()

            if new_target_load >= current_max:
                tried[best_e] = True
                continue

            # Execute replication
            src_gpu = int(np.argmax(placement[best_e]))
            ops.append(MigrationOp(OpType.REPLICATE, best_e, src_gpu, best_r))
            cost += 1

            placement[best_e, best_r] = True
            slots_used[best_r] += 1

            # Incremental load update
            old_share = d[best_e] / old_k
            new_share = d[best_e] / new_k
            delta = old_share - new_share
            holders = np.where(placement[best_e])[0]
            gpu_load[holders] -= delta
            gpu_load[best_r] += new_share

            k[best_e] = new_k
            tried[:] = False  # reset tried flags after a successful replication

        # --- Phase C: evict/migrate cold experts off overloaded GPUs ---
        for _ in range(self.max_migrate_iters):
            if cost >= self.m_budget:
                break

            current_max = gpu_load.max()
            threshold = mu * 1.05
            if current_max <= threshold:
                break

            overloaded_r = int(np.argmax(gpu_load))

            experts_on_gpu = np.where(placement[:, overloaded_r])[0]
            if len(experts_on_gpu) == 0:
                break

            expert_loads_on_gpu = d[experts_on_gpu]
            coldest_idx = int(np.argmin(expert_loads_on_gpu))
            coldest_e = int(experts_on_gpu[coldest_idx])

            if k[coldest_e] > 1:
                # Evict replica (zero cost)
                ops.append(MigrationOp(OpType.EVICT, coldest_e, overloaded_r, -1))
                placement[coldest_e, overloaded_r] = False
                slots_used[overloaded_r] -= 1

                old_k = k[coldest_e]
                new_k = old_k - 1
                old_share = d[coldest_e] / old_k
                new_share = d[coldest_e] / new_k

                gpu_load[overloaded_r] -= old_share
                holders = np.where(placement[coldest_e])[0]
                gpu_load[holders] += (new_share - old_share)

                k[coldest_e] = new_k
            else:
                # Migrate to least-loaded GPU with free slot
                target_mask = (slots_used < self.capacity)
                target_mask[overloaded_r] = False
                if not target_mask.any():
                    break

                target_loads = np.where(target_mask, gpu_load, np.inf)
                target_r = int(np.argmin(target_loads))

                ops.append(MigrationOp(OpType.MIGRATE, coldest_e, overloaded_r, target_r))
                cost += 1

                share_val = d[coldest_e] / k[coldest_e]
                placement[coldest_e, overloaded_r] = False
                placement[coldest_e, target_r] = True
                slots_used[overloaded_r] -= 1
                slots_used[target_r] += 1
                gpu_load[overloaded_r] -= share_val
                gpu_load[target_r] += share_val

        return placement, ops, cost


class TokenShardingSolver:
    """
    Two-phase token sharding algorithm.

    Phase 1+2: Minimize makespan via closed-form lower bound, then assign
    tokens with locality-first priority.
    Phase 3: Vectorized token-to-GPU assignment via matrix multiplication.
    """

    def __init__(self, num_experts: int, num_gpus: int):
        self.num_experts = num_experts
        self.num_gpus = num_gpus

    def solve(
        self,
        demand: np.ndarray,
        placement: np.ndarray,
        initial_owner: np.ndarray,
        topk_ids: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Args:
            demand: [E] float, actual token count per expert.
            placement: [E, R] bool, expert-GPU assignment (with replicas).
            initial_owner: [E] int, home GPU for each expert.
            topk_ids: [B, K] int, actual token routing (optional).

        Returns:
            token_to_gpu: [B] int or None.
            makespan: optimal L*.
            imbalance: makespan / mean_load.
        """
        E, R = self.num_experts, self.num_gpus
        d = demand.astype(np.float64)

        k = placement.sum(axis=1).astype(np.float64)  # [E]
        k = np.maximum(k, 1.0)

        total_demand = d.sum()
        mu = total_demand / R if R > 0 else 0.0

        # --- Phase 1: closed-form makespan ---
        per_replica = d / k  # [E]
        L_star = max(mu, float(per_replica.max())) if E > 0 else mu

        # --- Phase 2: locality-first allocation ---
        n_alloc = np.zeros((E, R), dtype=np.float64)
        gpu_load = np.zeros(R, dtype=np.float64)

        expert_order = np.argsort(-d)

        for e in expert_order:
            e = int(e)
            if d[e] <= 0:
                continue

            replicas = np.where(placement[e])[0]
            if len(replicas) == 0:
                continue

            home = int(initial_owner[e])
            remaining = d[e]

            # Home GPU first, then least-loaded
            is_home = (replicas == home).astype(np.int32)
            replica_loads = gpu_load[replicas]
            sort_key = np.lexsort((replica_loads, -is_home))
            sorted_replicas = replicas[sort_key]

            for r in sorted_replicas:
                capacity = L_star - gpu_load[r]
                assign = min(remaining, max(capacity, 0.0))
                if assign > 0:
                    n_alloc[e, r] = assign
                    gpu_load[r] += assign
                    remaining -= assign
                if remaining <= 1e-9:
                    break

            if remaining > 1e-9:
                fallback_r = int(sorted_replicas[np.argmin(gpu_load[sorted_replicas])])
                n_alloc[e, fallback_r] += remaining
                gpu_load[fallback_r] += remaining

        actual_makespan = float(gpu_load.max())
        imbalance = actual_makespan / mu if mu > 0 else 1.0

        # --- Phase 3: token-to-GPU assignment ---
        token_to_gpu = None
        if topk_ids is not None and len(topk_ids) > 0:
            token_to_gpu = self._assign_tokens(topk_ids, placement)

        return token_to_gpu, actual_makespan, imbalance

    def _assign_tokens(
        self, topk_ids: np.ndarray, placement: np.ndarray
    ) -> np.ndarray:
        """Vectorized token-to-GPU assignment via torch gather + sum."""
        B, K = topk_ids.shape
        E, R = placement.shape

        placement_t = torch.from_numpy(placement.astype(np.int8))
        topk_t = torch.from_numpy(topk_ids).long()

        selected = placement_t[topk_t.reshape(-1)]  # [B*K, R]
        score = selected.reshape(B, K, R).sum(dim=1)  # [B, R]

        return score.argmax(dim=1).numpy()

def build_physical_to_logical_map(
    placement: np.ndarray, num_gpus: int, capacity_per_gpu: int
) -> torch.Tensor:
    """
    Convert [E, R] bool placement to sglang's physical_to_logical_map format.

    Returns:
        [num_physical_experts] int64 tensor where
        phy2log[r * capacity + slot] = logical expert ID.
        Empty slots are filled with trivial mapping (slot % num_logical_experts)
        to avoid crashes in update_expert_weights_single_layer.
    """
    num_logical = placement.shape[0]
    num_physical = num_gpus * capacity_per_gpu
    # Start with trivial mapping as fallback
    phy2log = np.array([i % num_logical for i in range(num_physical)], dtype=np.int64)

    for r in range(num_gpus):
        experts_on_r = np.where(placement[:, r])[0]
        for slot, e in enumerate(experts_on_r):
            if slot >= capacity_per_gpu:
                break
            phy2log[r * capacity_per_gpu + slot] = e

    return torch.from_numpy(phy2log)


def make_initial_placement(num_experts: int, num_gpus: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create initial contiguous expert placement.

    Returns:
        placement: [E, R] bool.
        initial_owner: [E] int.
    """
    E, R = num_experts, num_gpus
    n_local = E // R

    placement = np.zeros((E, R), dtype=bool)
    initial_owner = np.zeros(E, dtype=np.int64)

    for r in range(R):
        start = r * n_local
        end = start + n_local
        placement[start:end, r] = True
        initial_owner[start:end] = r

    return placement, initial_owner


def apply_ops_to_phy2log(
    ops: list, 
    current_phy2log: list,
    num_gpus: int,
    per_gpu: int,
) -> list:
    """
    Incrementally update phy2log based on migration ops.
    Only modifies the slots affected by the ops, not the entire map.
    
    Returns:
        new_phy2log: updated copy with minimal changes
    """
    new_phy2log = list(current_phy2log)
    
    for op in ops:
        expert_id = op.expert_id
        src_gpu = op.src_gpu
        dst_gpu = op.dst_gpu
        
        if op.op_type == OpType.REPLICATE:
            # Find an empty or duplicate slot on dst_gpu
            dst_start = dst_gpu * per_gpu
            dst_end = dst_start + per_gpu
            # Find a slot that has a duplicate (expert appears elsewhere too)
            best_slot = -1
            for slot in range(dst_start, dst_end):
                # Prefer a slot whose expert is cold or already replicated
                log_e = new_phy2log[slot]
                # Count how many times this expert appears in the map
                count = sum(1 for v in new_phy2log if v == log_e)
                if count > 1:
                    best_slot = slot
                    break
            if best_slot == -1:
                # Use the last slot as fallback
                best_slot = dst_end - 1
            new_phy2log[best_slot] = expert_id
            
        elif op.op_type == OpType.EVICT:
            # Find this expert on src_gpu and revert to trivial
            src_start = src_gpu * per_gpu
            src_end = src_start + per_gpu
            for slot in range(src_start, src_end):
                if new_phy2log[slot] == expert_id:
                    new_phy2log[slot] = slot % 128  # revert to trivial
                    break
                    
        elif op.op_type == OpType.MIGRATE:
            # Swap: find expert on src_gpu, find a cold expert on dst_gpu, exchange
            src_start = src_gpu * per_gpu
            src_end = src_start + per_gpu
            dst_start = dst_gpu * per_gpu
            dst_end = dst_start + per_gpu
            
            src_slot = -1
            for slot in range(src_start, src_end):
                if new_phy2log[slot] == expert_id:
                    src_slot = slot
                    break
            if src_slot == -1:
                continue
            
            # Find coldest expert on dst_gpu to swap with
            dst_slot = dst_end - 1  # default: last slot
            for slot in range(dst_start, dst_end):
                # Find a slot with a replicated expert (can be safely overwritten)
                log_e = new_phy2log[slot]
                count = sum(1 for v in new_phy2log if v == log_e)
                if count > 1:
                    dst_slot = slot
                    break
            
            # Swap the two slots
            new_phy2log[src_slot], new_phy2log[dst_slot] = new_phy2log[dst_slot], new_phy2log[src_slot]
    
    return new_phy2log
