"""One-shot real-forward capture for the H20 placement replay."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
from typing import Any

import torch
import torch.distributed as dist

_LOCK = threading.Lock()
_PROFILE_COUNTS: dict[str, dict[int, dict[str, Any]]] = {}
_WRITTEN: set[tuple[str, int]] = set()


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    value = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))
    return int(value)


def _read_trigger() -> dict[str, Any] | None:
    path = pathlib.Path(os.environ["H20_PLACEMENT_CAPTURE_TRIGGER"])
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _plain_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return repr(value)


def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def _save_atomic(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _capture_topk(module, args, output, captured: dict[str, torch.Tensor]) -> None:
    del module, args
    if not hasattr(output, "topk_weights") or not hasattr(output, "topk_ids"):
        raise TypeError(f"unexpected top-k output type: {type(output)!r}")
    captured["topk_weights"] = output.topk_weights.detach().clone()
    captured["topk_ids"] = output.topk_ids.detach().clone()


def _profile_layer(
    *,
    capture_id: str,
    rank: int,
    layer_id: int,
    expected_layers: int,
    num_experts: int,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    topk_ids: torch.Tensor,
    output_dir: pathlib.Path,
) -> None:
    key = (capture_id, rank)
    if key in _WRITTEN:
        return
    counts = torch.bincount(
        topk_ids.to(torch.int64).flatten(), minlength=num_experts
    )[:num_experts].cpu()
    entry = {
        "layer_id": layer_id,
        "hidden_rows": int(hidden_states.shape[0]),
        "num_token_non_padded": _plain_value(
            getattr(forward_batch, "num_token_non_padded", None)
        ),
        "logical_expert_counts": counts,
        "topk_shape": list(topk_ids.shape),
    }
    with _LOCK:
        profile = _PROFILE_COUNTS.setdefault(capture_id, {})
        profile[layer_id] = entry
        if len(profile) != expected_layers:
            return
        layers = [profile[index] for index in range(expected_layers)]
        payload = {
            "schema_version": 1,
            "capture_id": capture_id,
            "mode": "profile",
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
            "expected_layers": expected_layers,
            "num_experts": num_experts,
            "identity_map_required": True,
            "layers": layers,
        }
        _save_atomic(output_dir / f"profile_rank{rank}.pt", payload)
        _WRITTEN.add(key)


def _snapshot_layer(
    *,
    capture_id: str,
    rank: int,
    layer_id: int,
    hidden_states: torch.Tensor,
    hidden_snapshot: torch.Tensor,
    forward_batch: Any,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_output: torch.Tensor,
    experts: Any,
    output_dir: pathlib.Path,
    map_sha256: str | None,
) -> None:
    key = (capture_id, rank)
    with _LOCK:
        if key in _WRITTEN:
            return
        _WRITTEN.add(key)
    required = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    )
    missing = [name for name in required if not hasattr(experts, name)]
    if missing:
        raise AttributeError(f"missing live expert fields: {missing}")
    quant_method = getattr(experts, "quant_method", None)
    quant_config = getattr(quant_method, "quant_config", None)
    block_shape = getattr(quant_config, "weight_block_size", None)
    if block_shape is None:
        raise AttributeError("active expert quantization has no weight_block_size")
    block_shape = [int(value) for value in block_shape]

    tensors = {
        "hidden_states": hidden_snapshot.cpu(),
        "logical_topk_weights": topk_weights.cpu(),
        "logical_topk_ids": topk_ids.to(torch.int32).cpu(),
        "routed_output": routed_output.detach().cpu(),
        "w13_weight": experts.w13_weight.detach().cpu(),
        "w2_weight": experts.w2_weight.detach().cpu(),
        "w13_weight_scale_inv": experts.w13_weight_scale_inv.detach().cpu(),
        "w2_weight_scale_inv": experts.w2_weight_scale_inv.detach().cpu(),
    }
    descriptions = {
        name: _tensor_description(tensor) for name, tensor in tensors.items()
    }
    local_experts = int(tensors["w13_weight"].shape[0])
    payload = {
        "schema_version": 1,
        "capture_id": capture_id,
        "mode": "snapshot",
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
        "layer_id": layer_id,
        "identity_map_required": True,
        "identity_map_sha256": map_sha256,
        "num_token_non_padded": _plain_value(
            getattr(forward_batch, "num_token_non_padded", None)
        ),
        "hidden_rows": int(hidden_states.shape[0]),
        "logical_expert_ids": list(
            range(rank * local_experts, (rank + 1) * local_experts)
        ),
        "weight_block_size": block_shape,
        "tensor_descriptions_before_cpu_save": descriptions,
        **tensors,
    }
    _save_atomic(output_dir / f"snapshot_rank{rank}.pt", payload)


def install() -> None:
    from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    if getattr(Qwen3MoeSparseMoeBlock, "_h20_capture_installed", False):
        return
    original = Qwen3MoeSparseMoeBlock.forward_deepep

    def wrapped(self, hidden_states, forward_batch):
        trigger = _read_trigger()
        layer_id = int(self.layer_id)
        should_capture = False
        if trigger is not None:
            mode = str(trigger.get("mode"))
            capture_id = str(trigger.get("capture_id"))
            rank = _rank()
            min_tokens = int(trigger.get("min_hidden_rows", 1))
            target_layer = trigger.get("target_layer")
            should_capture = (
                (capture_id, rank) not in _WRITTEN
                and hidden_states.shape[0] >= min_tokens
                and (mode == "profile" or layer_id == int(target_layer))
            )
        if not should_capture:
            return original(self, hidden_states, forward_batch)

        captured: dict[str, torch.Tensor] = {}
        handle = self.topk.register_forward_hook(
            lambda module, args, output: _capture_topk(
                module, args, output, captured
            )
        )
        hidden_snapshot = (
            hidden_states.detach().clone() if mode == "snapshot" else None
        )
        try:
            routed_output = original(self, hidden_states, forward_batch)
        finally:
            handle.remove()
        if "topk_ids" not in captured or "topk_weights" not in captured:
            raise RuntimeError("top-k hook did not observe a selection")

        output_dir = pathlib.Path(trigger["output_dir"])
        if mode == "profile":
            _profile_layer(
                capture_id=capture_id,
                rank=rank,
                layer_id=layer_id,
                expected_layers=int(trigger.get("expected_layers", 94)),
                num_experts=int(trigger.get("num_experts", 128)),
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                topk_ids=captured["topk_ids"],
                output_dir=output_dir,
            )
        elif mode == "snapshot":
            _snapshot_layer(
                capture_id=capture_id,
                rank=rank,
                layer_id=layer_id,
                hidden_states=hidden_states,
                hidden_snapshot=hidden_snapshot,
                forward_batch=forward_batch,
                topk_weights=captured["topk_weights"],
                topk_ids=captured["topk_ids"],
                routed_output=routed_output,
                experts=self.experts,
                output_dir=output_dir,
                map_sha256=trigger.get("identity_map_sha256"),
            )
        else:
            raise ValueError(f"unknown capture mode {mode!r}")
        return routed_output

    Qwen3MoeSparseMoeBlock.forward_deepep = wrapped
    Qwen3MoeSparseMoeBlock._h20_capture_installed = True
