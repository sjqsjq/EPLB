"""
Cross-Layer Routing Predictor — returns predicted expert IDs for downstream use.
"""
import logging
import torch
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CrossLayerPredictor:
    _instance: Optional["CrossLayerPredictor"] = None

    def __init__(self, num_layers: int, num_experts: int, top_k: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self._prev_gate_input: Optional[torch.Tensor] = None
        self._prev_layer_id: int = -1
        self._total_predictions = 0
        self._total_overlap = 0.0
        self._log_interval = 100

    @classmethod
    def get_instance(cls): return cls._instance

    @classmethod
    def initialize(cls, num_layers, num_experts, top_k):
        if cls._instance is None:
            cls._instance = cls(num_layers, num_experts, top_k)
        return cls._instance

    def on_moe_forward(self, layer_id, gate_input, gate_weight, actual_topk_ids,
                       correction_bias=None) -> Dict:
        result = {}

        if self._prev_gate_input is not None and self._prev_layer_id == layer_id - 1:
            with torch.no_grad():
                predicted_logits = self._prev_gate_input @ gate_weight.T
                predicted_scores = predicted_logits.sigmoid()
                if correction_bias is not None:
                    predicted_scores = predicted_scores + correction_bias.float().unsqueeze(0)
                predicted_topk_ids = predicted_scores.topk(self.top_k, dim=-1).indices

                overlap = self._compute_overlap(predicted_topk_ids, actual_topk_ids)
                self._total_predictions += 1
                self._total_overlap += overlap

                result = {
                    "layer_id": layer_id,
                    "overlap_accuracy": overlap,
                    "num_tokens": gate_input.shape[0],
                    "predicted_topk_ids": predicted_topk_ids,  # Pass downstream
                }

                if self._total_predictions % self._log_interval == 0:
                    avg = self._total_overlap / self._total_predictions
                    logger.info(
                        f"[CrossLayerPredictor] avg overlap: {avg:.4f} "
                        f"over {self._total_predictions} predictions"
                    )

        self._prev_gate_input = gate_input.detach()
        self._prev_layer_id = layer_id
        return result

    def reset_state(self):
        self._prev_gate_input = None
        self._prev_layer_id = -1

    def get_stats(self):
        if self._total_predictions == 0:
            return {"avg_overlap": 0.0, "total_predictions": 0}
        return {
            "avg_overlap": round(self._total_overlap / self._total_predictions, 4),
            "total_predictions": self._total_predictions,
        }

    @staticmethod
    def _compute_overlap(predicted, actual):
        pred_exp = predicted.unsqueeze(2)
        actual_exp = actual.unsqueeze(1)
        matches = (pred_exp == actual_exp).any(dim=2).sum(dim=1).float()
        return matches.mean().item() / predicted.shape[1]
