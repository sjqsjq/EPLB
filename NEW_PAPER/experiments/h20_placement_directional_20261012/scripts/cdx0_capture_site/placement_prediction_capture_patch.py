"""Route profiling and split real-forward snapshots for prediction study v1.

CD-X0 capture site patch: verbatim copy of placement_prediction_capture_patch.py
(source sha256 9dd82453fa137525994b60b63ad389a6d52893828dee3a5530995b22c02fcc68)
with the single study-id gate widened to h20-placement-cdx0-v1. Recorded in
cdx0_amendment_v1.json (provenance_policy).
"""

from __future__ import annotations

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
_VALIDATED_EXPERT_PATHS: set[pathlib.Path] = set()


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    value = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))
    return int(value)


def _read_trigger() -> dict[str, Any] | None:
    path = pathlib.Path(os.environ["H20_PREDICTION_CAPTURE_TRIGGER"])
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
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


def _save_atomic_no_clobber(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite capture artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale capture temporary exists: {temporary}")
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
    trigger: dict[str, Any],
    capture_id: str,
    rank: int,
    layer_id: int,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    topk_ids: torch.Tensor,
) -> None:
    key = (capture_id, rank)
    if key in _WRITTEN:
        return
    expected_layers = int(trigger["expected_layers"])
    num_experts = int(trigger["num_experts"])
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
            "schema_version": 2,
            "study_id": trigger["study_id"],
            "capture_id": capture_id,
            "mode": "profile",
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
            "expected_layers": expected_layers,
            "num_experts": num_experts,
            "identity_map_required": True,
            "identity_map_sha256": trigger["identity_map_sha256"],
            "capture_context": trigger["capture_context"],
            "layers": layers,
        }
        output_dir = pathlib.Path(trigger["output_dir"])
        _save_atomic_no_clobber(output_dir / f"profile_rank{rank}.pt", payload)
        _WRITTEN.add(key)


def _write_expert_artifact(
    *,
    trigger: dict[str, Any],
    capture_id: str,
    rank: int,
    layer_id: int,
    experts: Any,
    block_shape: list[int],
) -> pathlib.Path:
    weight_dir = pathlib.Path(trigger["weight_output_dir"])
    path = weight_dir / f"layer-{layer_id:03d}" / f"expert_rank{rank}.pt"
    live_tensors = {name: getattr(experts, name) for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    )}
    local_experts = int(live_tensors["w13_weight"].shape[0])
    logical_expert_ids = list(
        range(rank * local_experts, (rank + 1) * local_experts)
    )
    if path.exists():
        if path not in _VALIDATED_EXPERT_PATHS:
            existing = torch.load(path, map_location="cpu", weights_only=False)
            expected_header = {
                "schema_version": 2,
                "study_id": trigger["study_id"],
                "artifact_type": "immutable_layer_expert_weights",
                "rank": rank,
                "layer_id": layer_id,
                "identity_map_required": True,
                "identity_map_sha256": trigger["identity_map_sha256"],
                "logical_expert_ids": logical_expert_ids,
                "weight_block_size": block_shape,
            }
            mismatches = {
                name: (existing.get(name), value)
                for name, value in expected_header.items()
                if existing.get(name) != value
            }
            if mismatches:
                raise ValueError(
                    f"existing expert artifact metadata mismatch at {path}: "
                    f"{mismatches}"
                )
            for name, live in live_tensors.items():
                stored = existing.get(name)
                if not isinstance(stored, torch.Tensor):
                    raise TypeError(
                        f"existing expert artifact lacks tensor {name}: {path}"
                    )
                if stored.shape != live.shape or stored.dtype != live.dtype:
                    raise ValueError(
                        f"existing expert tensor {name} differs in shape/dtype: "
                        f"stored={stored.shape}/{stored.dtype}, "
                        f"live={live.shape}/{live.dtype}"
                    )
            _VALIDATED_EXPERT_PATHS.add(path)
        return path
    tensors = {name: tensor.detach().cpu() for name, tensor in live_tensors.items()}
    payload = {
        "schema_version": 2,
        "study_id": trigger["study_id"],
        "artifact_type": "immutable_layer_expert_weights",
        "first_capture_id": capture_id,
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
        "layer_id": layer_id,
        "identity_map_required": True,
        "identity_map_sha256": trigger["identity_map_sha256"],
        "logical_expert_ids": logical_expert_ids,
        "weight_block_size": block_shape,
        "tensor_descriptions_before_cpu_save": {
            name: _tensor_description(tensor) for name, tensor in tensors.items()
        },
        **tensors,
    }
    _save_atomic_no_clobber(path, payload)
    _VALIDATED_EXPERT_PATHS.add(path)
    return path


def _snapshot_layer(
    *,
    trigger: dict[str, Any],
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

    expert_path = _write_expert_artifact(
        trigger=trigger,
        capture_id=capture_id,
        rank=rank,
        layer_id=layer_id,
        experts=experts,
        block_shape=block_shape,
    )
    tensors = {
        "hidden_states": hidden_snapshot.cpu(),
        "logical_topk_weights": topk_weights.cpu(),
        "logical_topk_ids": topk_ids.to(torch.int32).cpu(),
        "routed_output": routed_output.detach().cpu(),
    }
    payload = {
        "schema_version": 2,
        "study_id": trigger["study_id"],
        "capture_id": capture_id,
        "mode": "snapshot_split",
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
        "layer_id": layer_id,
        "identity_map_required": True,
        "identity_map_sha256": trigger["identity_map_sha256"],
        "capture_context": trigger["capture_context"],
        "num_token_non_padded": _plain_value(
            getattr(forward_batch, "num_token_non_padded", None)
        ),
        "hidden_rows": int(hidden_states.shape[0]),
        "weight_block_size": block_shape,
        "expert_artifact_path": str(expert_path.resolve()),
        "tensor_descriptions_before_cpu_save": {
            name: _tensor_description(tensor) for name, tensor in tensors.items()
        },
        **tensors,
    }
    output_dir = pathlib.Path(trigger["output_dir"])
    _save_atomic_no_clobber(output_dir / f"forward_rank{rank}.pt", payload)


def install() -> None:
    from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    if getattr(Qwen3MoeSparseMoeBlock, "_h20_prediction_capture_installed", False):
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
                trigger.get("study_id") == "h20-placement-cdx0-v1"
                and (capture_id, rank) not in _WRITTEN
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
            hidden_states.detach().clone() if mode == "snapshot_split" else None
        )
        try:
            routed_output = original(self, hidden_states, forward_batch)
        finally:
            handle.remove()
        if "topk_ids" not in captured or "topk_weights" not in captured:
            raise RuntimeError("top-k hook did not observe a selection")

        if mode == "profile":
            _profile_layer(
                trigger=trigger,
                capture_id=capture_id,
                rank=rank,
                layer_id=layer_id,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                topk_ids=captured["topk_ids"],
            )
        elif mode == "snapshot_split":
            _snapshot_layer(
                trigger=trigger,
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
            )
        else:
            raise ValueError(f"unknown prediction-capture mode {mode!r}")
        return routed_output

    Qwen3MoeSparseMoeBlock.forward_deepep = wrapped
    Qwen3MoeSparseMoeBlock._h20_prediction_capture_installed = True
