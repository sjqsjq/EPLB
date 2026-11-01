#!/usr/bin/env python3
"""PB-OEPLB online decision path ported to single-layer full-service replay.

Faithful port of the DEFAULT PBOEPLBController decision path
(/workspace/EPLB/OEPLB/src/controller.py with /workspace/EPLB/OEPLB/src/config.py
defaults) to a single-MoE-layer replay service on 8xH20. Line references (L###)
are to the vendored OEPLB tree at protocol-freeze time.

PORTED (production defaults only):
  * physical-slot-space count recording, min_record_tokens=32 gate (L288-327);
    SGLang receives already-physical topk ids (static dispatch); this port
    converts captured LOGICAL ids through the current l2p at record time --
    arithmetically identical physical-space counts (fidelity note FN-1).
  * _should_record_this_batch set at end of pass N governs pass N+1 (L340-352);
    all replayed forwards are prefill-shaped, sample_interval=1 -> always True
    once set (fidelity note FN-2: no prefill/decode distinction exists here).
  * warmup_forwards=10: ready flips at end of pass 10; step counting only while
    ready (L354-358); recording itself is NOT ready-gated (matches production).
  * sync_window=8 local step counter; no cross-rank agreement needed for when
    to check (L365-369); window order: finish-pending (no-op here), all_reduce,
    decide, issue (L371-377).
  * all_reduce(SUM) over a CLONE; local load never mutated by the reduce
    (L736-751); window_count increments after the reduce (L751).
  * min_prefill_tokens=256 gate on global_tokens = global_load.sum() (L788,L848).
  * threshold_ratio=1.02; bias_correct/dead_zone OFF by default so the
    early-exit path is inactive and _effective_threshold returns 1.02 (L851,L882).
  * try_build_swap_plan(global_load[None-shaped 1x128 physical space], p2l,
    num_ranks=8, num_local=16, 1.02, max_swaps_per_layer=64,
    max_total_swap_layers=94, max_total_ops=300) from the VENDORED unmodified
    rebalancer (L907-915).
  * _last_diag_improvement = (mean pre-ratio over plan layers - 1.0) * 0.1
    (L917-925); min_swap_ops=8 skip (L940-950); defer when len(plan)<16 and
    improvement<0.003 (L961-965).
  * after swap execution: self.load counts remapped for swapped pairs (L1135-
    1146); decay self.load = (load.float()*0.5).long() at EVERY window boundary
    regardless of decision outcome (L428); total_tokens reset (L429).

NOT PORTED (inactive at production defaults or SGLang-integration specific):
  adaptive window/decay, bias correction, bias gate, dead_zone_ratio,
  swap_budget_frac, warm-start bulk re-placement, EPLB refinement, sensitivity
  calibration, multi-layer scheduling, idle-batch handling, CUDA-graph capture
  checks (replay never captures graphs).

PRE-REGISTERED FIDELITY LIMITS:
  FN-3: production executes swaps ASYNC on a low-priority stream overlapping
  later forwards; this port executes boundary-synchronously between service
  forwards (all ranks wait for their P2P completion before the map flips). The
  measured online overhead is therefore a NO-OVERLAP UPPER BOUND; the runner
  also reports service time net of migration wall time as a lower-bound view.
  FN-4: production issues one batch_isend_irecv per plan; this port stages the
  plan so that no physical slot appears twice within one stage (greedy
  stage coloring), executing one batch per stage sequentially. When the plan
  has no slot conflicts (logged), this is operation-identical to production.
  FN-5: no explicit barrier after migration; the next forward's DeepEP dispatch
  collective provides the natural cross-rank synchronization, matching
  production lockstep flip semantics.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any

import torch
import torch.distributed as dist

from vendor_pb_oeplb_rebalancer import SwapOp, try_build_swap_plan

# Frozen production defaults (OEPLB/src/config.py PBOEPLBConfig)
THRESHOLD_RATIO = 1.02
MAX_SWAPS_PER_LAYER = 64
MIN_PREFILL_TOKENS = 256
MAX_TOTAL_SWAP_LAYERS = 94
MAX_TOTAL_OPS = 300
MIN_SWAP_OPS = 8
SYNC_WINDOW = 8
DECAY_FACTOR = 0.5
MIN_RECORD_TOKENS = 32
WARMUP_FORWARDS = 10
SAMPLE_INTERVAL = 1
DEFER_MIN_IMPROVEMENT = 0.003
DEFER_PLAN_LEN = 16

WEIGHT_FIELDS = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")


@dataclasses.dataclass(frozen=True)
class PortConfig:
    threshold_ratio: float = THRESHOLD_RATIO
    max_swaps_per_layer: int = MAX_SWAPS_PER_LAYER
    min_prefill_tokens: int = MIN_PREFILL_TOKENS
    max_total_swap_layers: int = MAX_TOTAL_SWAP_LAYERS
    max_total_ops: int = MAX_TOTAL_OPS
    min_swap_ops: int = MIN_SWAP_OPS
    sync_window: int = SYNC_WINDOW
    decay_factor: float = DECAY_FACTOR
    min_record_tokens: int = MIN_RECORD_TOKENS
    warmup_forwards: int = WARMUP_FORWARDS


class OnlinePlanner:
    """Single-layer PB-OEPLB online decision path (see module docstring)."""

    def __init__(
        self,
        *,
        rank: int,
        group: dist.ProcessGroup,
        device: torch.device,
        num_logical: int = 128,
        num_ranks: int = 8,
        num_local: int = 16,
        cfg: PortConfig | None = None,
    ) -> None:
        self.cfg = cfg or PortConfig()
        self.rank = rank
        self.group = group
        self.device = device
        self.num_logical = num_logical
        self.num_ranks = num_ranks
        self.num_local = num_local
        # physical-slot indexed local decayed accumulator (controller L112)
        self.load = torch.zeros(num_logical, dtype=torch.int64, device=device)
        self.p2l = list(range(num_logical))
        self.l2p = list(range(num_logical))
        self.l2p_gpu = torch.arange(num_logical, dtype=torch.int64, device=device)
        self.forward_id = 0
        self.ready = False
        self.steps_since_check = 0
        self.window_count = 0
        self.total_tokens = 0
        self.should_record = False  # init False; set at end of each pass
        self.prefill_batch_counter = 0
        self.prev_window_ratio: float | None = None
        self.windows: list[dict[str, Any]] = []
        self.migrations: list[dict[str, Any]] = []

    # ------------------------------------------------------------- recording
    def record_next_layer(self, logical_topk_ids: torch.Tensor) -> None:
        """Hot path during a forward: GPU-only scatter_add, zero communication.

        Mirrors controller.record_next_layer (L288) with FN-1 logical->physical
        conversion through the current map.
        """
        if not self.should_record:
            return
        if logical_topk_ids.shape[0] < self.cfg.min_record_tokens:
            return
        flat = logical_topk_ids.reshape(-1)
        mask = flat != -1
        phys = self.l2p_gpu[flat.masked_fill(~mask, 0).long()]
        self.load.scatter_add_(0, phys, mask.long())
        self.total_tokens += int(logical_topk_ids.shape[0])

    def on_forward_pass_end(self) -> dict[str, Any] | None:
        """End-of-pass bookkeeping; returns a window event dict on boundaries."""
        self.forward_id += 1
        # recording decision for the NEXT pass (L340-352); all replay forwards
        # are prefill-shaped, sample_interval=1
        self.prefill_batch_counter += 1
        self.should_record = (self.prefill_batch_counter % SAMPLE_INTERVAL == 0)
        if not self.ready:
            if self.forward_id >= self.cfg.warmup_forwards:
                self.ready = True
            return None
        self.steps_since_check += 1
        if self.steps_since_check < self.cfg.sync_window:
            return None
        self.steps_since_check = 0
        event = self._decide_and_begin_swap()
        # decay at every window boundary regardless of decision (L428-430)
        self.load.copy_((self.load.float() * self.cfg.decay_factor).long())
        self.total_tokens = 0
        self.prefill_batch_counter = 0
        self.windows.append(event)
        return event

    # ------------------------------------------------------------- decision
    def _rank_ratio(self, counts: list[int]) -> float:
        loads = [
            sum(counts[r * self.num_local : (r + 1) * self.num_local])
            for r in range(self.num_ranks)
        ]
        avg = max(sum(loads) / self.num_ranks, 1.0)
        return max(loads) / avg

    def _decide_and_begin_swap(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "window_index": self.window_count + 1,
            "forward_id": self.forward_id,
        }
        t0 = time.perf_counter_ns()
        global_load = self.load.clone()  # never mutate local load (L736-747)
        dist.all_reduce(global_load, op=dist.ReduceOp.SUM, group=self.group)
        self.window_count += 1
        global_counts = global_load.tolist()
        avg_ratio = self._rank_ratio(global_counts)
        self.prev_window_ratio = avg_ratio
        global_tokens = int(sum(global_counts))
        event.update(
            avg_ratio=avg_ratio,
            global_tokens=global_tokens,
            allreduce_plus_sync_us=(time.perf_counter_ns() - t0) / 1000.0,
        )
        if global_tokens < self.cfg.min_prefill_tokens:
            event.update(decision="below_min_prefill_tokens", plan_len=0)
            return event
        t1 = time.perf_counter_ns()
        plan = try_build_swap_plan(
            logical_count=global_load.unsqueeze(0),
            physical_to_logical_map=torch.tensor(
                [self.p2l], dtype=torch.int64, device=global_load.device
            ),
            num_ranks=self.num_ranks,
            num_local=self.num_local,
            threshold_ratio=self.cfg.threshold_ratio,
            max_swaps_per_layer=self.cfg.max_swaps_per_layer,
            max_total_swap_layers=self.cfg.max_total_swap_layers,
            max_total_ops=self.cfg.max_total_ops,
        )
        event["plan_build_us"] = (time.perf_counter_ns() - t1) / 1000.0
        if plan:
            pre_ratio = avg_ratio  # single layer: plan[:10] layers == {0}
            self._last_diag_improvement = (pre_ratio - 1.0) * 0.1
        else:
            self._last_diag_improvement = 0.0
        event["plan_len"] = len(plan)
        if not plan:
            event["decision"] = "no_plan"
            return event
        if len(plan) < self.cfg.min_swap_ops:
            event["decision"] = "skipped_min_swap_ops"
            return event
        if len(plan) < DEFER_PLAN_LEN and self._last_diag_improvement < DEFER_MIN_IMPROVEMENT:
            event["decision"] = "deferred_small_improvement"
            event["diag_improvement"] = self._last_diag_improvement
            return event
        event["decision"] = "executed"
        return event  # runner executes via execute_plan (needs weight handles)

    # ------------------------------------------------------------- execution
    def stage_plan(self, plan: list[SwapOp]) -> list[list[SwapOp]]:
        """Greedy stage coloring so no physical slot repeats within a stage (FN-4)."""
        stages: list[list[SwapOp]] = []
        for op in plan:
            touched = {op.phys_slot_a, op.phys_slot_b}
            placed = False
            for stage in stages:
                if all(
                    not (touched & {o.phys_slot_a, o.phys_slot_b}) for o in stage
                ):
                    stage.append(op)
                    placed = True
                    break
            if not placed:
                stages.append([op])
        return stages

    def execute_plan(
        self,
        plan: list[SwapOp],
        weights: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Boundary-synchronized real-weight swap execution (FN-3, FN-4, FN-5).

        `weights` is the rank-local mutable store: field -> [num_local, ...]
        tensors for the 16 physical slots owned by this rank.
        """
        stages = self.stage_plan(plan)
        conflict_free = len(stages) == 1
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter_ns()
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        ops_moved = 0
        bytes_moved = 0
        for stage in stages:
            requests = []
            staging: list[tuple[SwapOp, str, torch.Tensor]] = []
            my_ops = [op for op in stage if self.rank in (op.rank_a, op.rank_b)]
            for op in my_ops:
                peer = op.rank_b if self.rank == op.rank_a else op.rank_a
                my_slot_full = op.phys_slot_a if self.rank == op.rank_a else op.phys_slot_b
                peer_slot_full = op.phys_slot_b if self.rank == op.rank_a else op.phys_slot_a
                my_local = my_slot_full % self.num_local
                for field in WEIGHT_FIELDS:
                    src = weights[field][my_local]
                    buf = torch.empty_like(src)
                    requests.append(dist.P2POp(dist.isend, src, peer, group=self.group))
                    requests.append(dist.P2POp(dist.irecv, buf, peer, group=self.group))
                    staging.append((op, field, buf))
                    # isend payload + irecv payload, both directions counted
                    bytes_moved += 2 * src.numel() * src.element_size()
            if requests:
                reqs = dist.batch_isend_irecv(requests)
                for req in reqs:
                    req.wait()
                for op, field, buf in staging:
                    my_slot_full = (
                        op.phys_slot_a if self.rank == op.rank_a else op.phys_slot_b
                    )
                    weights[field][my_slot_full % self.num_local].copy_(buf)
                ops_moved += len(my_ops)
        ev_end.record()
        torch.cuda.synchronize(self.device)
        wall_us = (time.perf_counter_ns() - t0) / 1000.0
        gpu_us = ev_start.elapsed_time(ev_end) * 1000.0
        # deterministic map flip on every rank (identical plan everywhere)
        for op in plan:
            a, b = op.phys_slot_a, op.phys_slot_b
            self.p2l[a], self.p2l[b] = self.p2l[b], self.p2l[a]
            la, lb = self.p2l[a], self.p2l[b]
            self.l2p[la], self.l2p[lb] = a, b
            # remap local decayed load history (L1135-1146)
            tmp = self.load[a].clone()
            self.load[a] = self.load[b]
            self.load[b] = tmp
        self.l2p_gpu = torch.tensor(self.l2p, dtype=torch.int64, device=self.device)
        record = {
            "window_index": self.window_count,
            "forward_id": self.forward_id,
            "plan_len": len(plan),
            "stage_count": len(stages),
            "conflict_free_single_stage": conflict_free,
            "rank_local_ops": ops_moved,
            "rank_local_bytes_sent_and_received": bytes_moved,
            "migration_wall_us": wall_us,
            "migration_gpu_us": gpu_us,
            "plan": [
                {
                    "phys_slot_a": op.phys_slot_a,
                    "phys_slot_b": op.phys_slot_b,
                    "rank_a": op.rank_a,
                    "rank_b": op.rank_b,
                    "imbalance": op.imbalance,
                }
                for op in plan
            ],
            "map_sha256_after": self.map_sha256(),
        }
        self.migrations.append(record)
        return record

    def map_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(
            ",".join(str(v) for v in self.p2l).encode("utf-8")
        ).hexdigest()
