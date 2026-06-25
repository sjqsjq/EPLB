"""
Cross-Layer Routing Prediction Accuracy Verifier (Libra/Fate Paper)

Verifies: using layer i's gate input (hidden_states) with layer i+1's gate weight
to predict layer i+1's expert routing, then compares with ground truth.

Formula: predicted_logits = gate_input[i] @ gate_weight[i+1].T
         accuracy = |predicted_topk ∩ actual_topk[i+1]| / k

Usage (inside container):
  python3 router/cross_layer_routing_predictor.py \
    --model-path /cpfs01/user/nebula_model/llm_weight/DeepSeek-V3 \
    --dtype bfloat16 --device-map auto
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Core prediction logic
# ============================================================================

def grouped_topk(
    scores: torch.Tensor,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    correction_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepSeek-V3 grouped topk: select topk_group groups, then top_k experts."""
    num_tokens, num_experts = scores.shape

    if correction_bias is not None:
        scores_for_selection = scores + correction_bias.float().unsqueeze(0)
    else:
        scores_for_selection = scores

    if num_expert_group <= 1:
        return scores_for_selection.topk(top_k, dim=-1).indices

    experts_per_group = num_experts // num_expert_group
    grouped = scores_for_selection.view(num_tokens, num_expert_group, experts_per_group)
    group_scores = grouped.max(dim=-1).values
    top_groups = group_scores.topk(topk_group, dim=-1).indices

    mask = torch.zeros(num_tokens, num_expert_group, device=scores.device, dtype=torch.bool)
    mask.scatter_(1, top_groups, True)
    expert_mask = mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(num_tokens, num_experts)

    masked = scores_for_selection.masked_fill(~expert_mask, float("-inf"))
    return masked.topk(top_k, dim=-1).indices


def compute_overlap_accuracy(predicted_ids: torch.Tensor, actual_ids: torch.Tensor) -> Tuple[float, float]:
    """Vectorized overlap accuracy and top1 accuracy."""
    num_tokens, top_k = actual_ids.shape
    pred_expanded = predicted_ids.unsqueeze(2)
    actual_expanded = actual_ids.unsqueeze(1)
    matches = (pred_expanded == actual_expanded)
    per_token_overlap = matches.any(dim=2).sum(dim=1).float()
    overlap_acc = per_token_overlap.mean().item() / top_k
    top1_acc = (predicted_ids[:, 0] == actual_ids[:, 0]).float().mean().item()
    return overlap_acc, top1_acc


# ============================================================================
# Hook-based gate recorder
# ============================================================================

class GateRecorder:
    """Hooks into MoE gates to capture gate_input and router_logits per forward pass."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.hooks = []
        self.gate_inputs: Dict[int, torch.Tensor] = {}
        self.router_logits: Dict[int, torch.Tensor] = {}
        self.gate_weights: Dict[int, torch.Tensor] = {}
        self.correction_biases: Dict[int, Optional[torch.Tensor]] = {}
        self._setup_hooks()

    def _get_moe_layer_indices(self) -> List[int]:
        num_layers = self.config.num_hidden_layers
        first_k_dense = getattr(self.config, "first_k_dense_replace", 1)
        moe_layer_freq = getattr(self.config, "moe_layer_freq", 1)
        return [i for i in range(num_layers) if i >= first_k_dense and i % moe_layer_freq == 0]

    def _setup_hooks(self):
        moe_indices = self._get_moe_layer_indices()
        logger.info(f"MoE layers: {len(moe_indices)} (from {moe_indices[0]} to {moe_indices[-1]})")

        for layer_idx in moe_indices:
            gate = self._find_gate(layer_idx)
            if gate is None:
                continue

            # Store weight and bias references (move to CPU to avoid cross-device issues)
            if hasattr(gate, "weight"):
                self.gate_weights[layer_idx] = gate.weight.detach().cpu()
            if hasattr(gate, "e_score_correction_bias") and gate.e_score_correction_bias is not None:
                self.correction_biases[layer_idx] = gate.e_score_correction_bias.data.detach().cpu()
            else:
                self.correction_biases[layer_idx] = None

            # Register hook
            hook = gate.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _find_gate(self, layer_idx: int):
        try:
            return self.model.model.layers[layer_idx].mlp.gate
        except (AttributeError, IndexError):
            return None

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            gate_input = inputs[0] if isinstance(inputs, tuple) else inputs

            # Flatten to 2D (num_tokens, hidden_size) if needed
            if gate_input.dim() == 3:
                gate_input = gate_input.view(-1, gate_input.shape[-1])

            self.gate_inputs[layer_idx] = gate_input.detach().cpu()

            # Compute raw router_logits ourselves: gate_input @ gate_weight.T
            # This avoids relying on gate's output format which varies across models.
            if layer_idx in self.gate_weights:
                weight = self.gate_weights[layer_idx]
                raw_logits = F.linear(gate_input.detach().cpu().float(), weight.detach().cpu().float())
                self.router_logits[layer_idx] = raw_logits
            else:
                logits = output[0] if isinstance(output, tuple) else output
                if logits.dim() == 3:
                    logits = logits.view(-1, logits.shape[-1])
                self.router_logits[layer_idx] = logits.detach().cpu()
        return hook_fn

    def clear(self):
        self.gate_inputs.clear()
        self.router_logits.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ============================================================================
# Evaluation logic
# ============================================================================

def evaluate_single_forward(
    recorder: GateRecorder,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
) -> Dict[int, Tuple[float, float, float]]:
    """
    After a forward pass, compute cross-layer prediction accuracy.
    Returns: {layer_i: (overlap_acc, top1_acc, cosine_sim)}
    """
    results = {}
    sorted_layers = sorted(recorder.gate_inputs.keys())

    for i in range(len(sorted_layers) - 1):
        curr_layer = sorted_layers[i]
        next_layer = sorted_layers[i + 1]

        curr_input = recorder.gate_inputs.get(curr_layer)
        next_logits = recorder.router_logits.get(next_layer)
        next_input = recorder.gate_inputs.get(next_layer)

        if curr_input is None or next_logits is None:
            continue
        if next_layer not in recorder.gate_weights:
            continue

        next_weight = recorder.gate_weights[next_layer]
        next_bias = recorder.correction_biases.get(next_layer)

        # All tensors are on CPU already (moved in hook). Ensure 2D.
        if curr_input.dim() != 2 or next_logits.dim() != 2:
            continue

        # Align token counts
        min_tokens = min(curr_input.shape[0], next_logits.shape[0])
        curr_input_slice = curr_input[:min_tokens].float()
        next_logits_slice = next_logits[:min_tokens].float()

        # Ground truth topk for next layer
        actual_scores = next_logits_slice.sigmoid()
        actual_topk = grouped_topk(actual_scores, top_k, num_expert_group, topk_group, next_bias)

        # Predicted topk: curr gate_input × next gate_weight
        predicted_logits = F.linear(curr_input_slice, next_weight.float())
        predicted_scores = predicted_logits.sigmoid()
        predicted_topk = grouped_topk(predicted_scores, top_k, num_expert_group, topk_group, next_bias)

        # Accuracy
        overlap_acc, top1_acc = compute_overlap_accuracy(predicted_topk, actual_topk)

        # Cosine similarity between adjacent gate inputs
        cos_sim = 0.0
        if next_input is not None and next_input.dim() == 2:
            next_input_aligned = next_input[:min_tokens].float()
            cos_sim = F.cosine_similarity(curr_input_slice, next_input_aligned, dim=-1).mean().item()

        results[curr_layer] = (overlap_acc, top1_acc, cos_sim)

    return results


def evaluate_single_forward_last_token(
    recorder: GateRecorder,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Same as evaluate_single_forward but only looks at the LAST token position.
    Used for decode-phase evaluation (single-token routing prediction).
    """
    results = {}
    sorted_layers = sorted(recorder.gate_inputs.keys())

    for i in range(len(sorted_layers) - 1):
        curr_layer = sorted_layers[i]
        next_layer = sorted_layers[i + 1]

        curr_input = recorder.gate_inputs.get(curr_layer)
        next_logits = recorder.router_logits.get(next_layer)
        next_input = recorder.gate_inputs.get(next_layer)

        if curr_input is None or next_logits is None:
            continue
        if next_layer not in recorder.gate_weights:
            continue

        next_weight = recorder.gate_weights[next_layer]
        next_bias = recorder.correction_biases.get(next_layer)

        # All tensors are on CPU already. Ensure 2D.
        if curr_input.dim() != 2 or next_logits.dim() != 2:
            continue

        # Only take the last token
        curr_input_last = curr_input[-1:, :].float()
        next_logits_last = next_logits[-1:, :].float()

        # Ground truth topk for next layer (last token only)
        actual_scores = next_logits_last.sigmoid()
        actual_topk = grouped_topk(actual_scores, top_k, num_expert_group, topk_group, next_bias)

        # Predicted topk: curr gate_input (last token) × next gate_weight
        predicted_logits = F.linear(curr_input_last, next_weight.float())
        predicted_scores = predicted_logits.sigmoid()
        predicted_topk = grouped_topk(predicted_scores, top_k, num_expert_group, topk_group, next_bias)

        # Accuracy
        overlap_acc, top1_acc = compute_overlap_accuracy(predicted_topk, actual_topk)

        # Cosine similarity
        cos_sim = 0.0
        if next_input is not None and next_input.dim() == 2:
            next_input_last = next_input[-1:, :].float()
            cos_sim = F.cosine_similarity(curr_input_last, next_input_last, dim=-1).mean().item()

        results[curr_layer] = (overlap_acc, top1_acc, cos_sim)

    return results


def run_evaluation(model, tokenizer, config, prompts, top_k, num_expert_group, topk_group, max_new_tokens, output_path):
    """Main evaluation: prefill + decode, aggregated stats."""
    recorder = GateRecorder(model, config)
    moe_layers = recorder._get_moe_layer_indices()

    # Accumulators: {layer_idx: [list of (overlap, top1, cosine)]}
    prefill_results = defaultdict(list)
    decode_results = defaultdict(list)

    logger.info(f"Evaluating {len(prompts)} prompts | top_k={top_k} | "
                f"groups={num_expert_group} | topk_group={topk_group}")

    for idx, prompt in enumerate(prompts):
        logger.info(f"[{idx+1}/{len(prompts)}] '{prompt[:60]}...'")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        # === Prefill: single forward with all input tokens (no cache) ===
        recorder.clear()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)

        prefill_step = evaluate_single_forward(recorder, top_k, num_expert_group, topk_group)
        for layer_idx, metrics in prefill_step.items():
            prefill_results[layer_idx].append(metrics)
        logger.info(f"  Prefill done ({input_ids.shape[1]} tokens, "
                    f"{len(prefill_step)} layer pairs evaluated)")

        # === Decode: simulate token-by-token by appending to sequence ===
        # Use full-sequence forward (no KV cache) to avoid cache API compatibility issues.
        # Each decode step processes only the last token's routing for accuracy calculation.
        generated_ids = input_ids
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for step in range(max_new_tokens - 1):
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            recorder.clear()
            with torch.no_grad():
                outputs = model(input_ids=generated_ids, use_cache=False)

            # Only evaluate routing for the LAST token position (decode behavior)
            decode_step = evaluate_single_forward_last_token(recorder, top_k, num_expert_group, topk_group)
            for layer_idx, metrics in decode_step.items():
                decode_results[layer_idx].append(metrics)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break

            if (step + 1) % 10 == 0:
                logger.info(f"  Decode step {step+1}/{max_new_tokens-1}")

        logger.info(f"  Decode done ({step+1} steps)")

    recorder.remove_hooks()

    # === Aggregate and print results ===
    print_results(prefill_results, decode_results, config, top_k, num_expert_group, topk_group, len(prompts), output_path)


def print_results(prefill_results, decode_results, config, top_k, num_expert_group, topk_group, num_prompts, output_path):
    """Print and save aggregated results."""
    print("\n" + "=" * 85)
    print("Cross-Layer Routing Prediction Results (Libra/Fate Verification)")
    print("=" * 85)
    print(f"Model: {getattr(config, '_name_or_path', 'unknown')}")
    print(f"Top-K={top_k}, Expert Groups={num_expert_group}, TopK Groups={topk_group}")
    print(f"Prompts evaluated: {num_prompts}")

    output_data = {"model": getattr(config, "_name_or_path", "unknown"),
                   "top_k": top_k, "num_expert_group": num_expert_group,
                   "topk_group": topk_group, "prefill": {}, "decode": {}}

    for phase_name, phase_data in [("PREFILL", prefill_results), ("DECODE", decode_results)]:
        print(f"\n--- {phase_name} Phase ---")
        print(f"{'Layer':<8} {'Overlap%':<11} {'Top1%':<9} {'CosineSim':<11}")
        print("-" * 42)

        all_overlap, all_top1, all_cos = [], [], []
        for layer_idx in sorted(phase_data.keys()):
            metrics_list = phase_data[layer_idx]
            avg_overlap = sum(m[0] for m in metrics_list) / len(metrics_list)
            avg_top1 = sum(m[1] for m in metrics_list) / len(metrics_list)
            avg_cos = sum(m[2] for m in metrics_list) / len(metrics_list)
            print(f"{layer_idx:<8} {avg_overlap*100:>8.2f}%  {avg_top1*100:>6.2f}%  {avg_cos:>9.4f}")
            all_overlap.append(avg_overlap)
            all_top1.append(avg_top1)
            all_cos.append(avg_cos)

            output_data[phase_name.lower()][str(layer_idx)] = {
                "overlap_accuracy": avg_overlap,
                "top1_accuracy": avg_top1,
                "cosine_similarity": avg_cos,
            }

        if all_overlap:
            mean_o = sum(all_overlap) / len(all_overlap)
            mean_t = sum(all_top1) / len(all_top1)
            mean_c = sum(all_cos) / len(all_cos)
            print("-" * 42)
            print(f"{'AVG':<8} {mean_o*100:>8.2f}%  {mean_t*100:>6.2f}%  {mean_c:>9.4f}")
            output_data[phase_name.lower()]["average"] = {
                "overlap_accuracy": mean_o, "top1_accuracy": mean_t, "cosine_similarity": mean_c}

    print("\n" + "=" * 85)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")


# ============================================================================
# Entry point
# ============================================================================

DEFAULT_PROMPTS = [
    # --- Short prompts (diverse topics) ---
    "The future of artificial intelligence lies in",
    "In quantum computing, the major challenge is",
    "Climate change impacts global food production by",
    "The theory of general relativity explains that",
    "Machine learning algorithms can be categorized into",
    # --- Medium prompts (more context for richer hidden states) ---
    "Recent advances in large language models have shown that scaling up model parameters",
    "The human brain processes visual information through a hierarchical system where",
    "In distributed systems, achieving consensus among nodes requires protocols such as",
    "The discovery of CRISPR-Cas9 gene editing technology has revolutionized the field of",
    "Transformer architecture, first introduced in the Attention Is All You Need paper,",
    # --- Longer / more complex prompts ---
    "Explain the difference between supervised learning and unsupervised learning in machine learning. Supervised learning uses labeled data to train models, while unsupervised learning",
    "Write a Python function to compute the Fibonacci sequence. The function should handle edge cases and use dynamic programming for efficiency. Here is the implementation:",
    "The economic implications of artificial intelligence adoption in healthcare include reduced costs, improved diagnostics, and personalized treatment plans. However, challenges remain in",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy. The light-dependent reactions occur in the thylakoid membranes, where water molecules are",
    "In the context of operating systems, virtual memory allows programs to use more memory than physically available by swapping pages between RAM and disk. The page replacement algorithm",
    # --- Code / technical prompts ---
    "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, hidden_size, num_heads):\n        super().__init__()\n        self.attention =",
    "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.created_at > '2024-01-01' GROUP BY",
    # --- Multilingual prompts ---
    "大型语言模型的发展经历了从统计语言模型到神经网络语言模型的转变，其中Transformer架构的提出是一个重要的里程碑。",
    "人工知能の発展において、深層学習は画像認識や自然言語処理などの分野で大きな進歩を遂げました。特に",
    "Les réseaux de neurones profonds ont transformé le domaine de l'intelligence artificielle en permettant des avancées significatives dans",
]


def main():
    parser = argparse.ArgumentParser(description="Cross-layer routing prediction verifier")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Load config
    logger.info(f"[Step 1/4] Loading config from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    top_k = getattr(config, "num_experts_per_tok", 8)
    num_expert_group = getattr(config, "n_group", 1)
    topk_group = getattr(config, "topk_group", 1)
    num_experts = getattr(config, "n_routed_experts", 256)
    num_layers = getattr(config, "num_hidden_layers", 61)
    logger.info(f"  Model config: {num_layers} layers, {num_experts} experts, "
                f"top_k={top_k}, n_group={num_expert_group}, topk_group={topk_group}")

    # Load tokenizer first (fast)
    logger.info(f"[Step 2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"  Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Load model (slow - this is the big one)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    logger.info(f"[Step 3/4] Loading model weights (dtype={args.dtype}, device_map={args.device_map})...")
    logger.info(f"  This may take 5-15 minutes for 671B models. Please wait...")

    import time
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype_map[args.dtype],
        device_map=args.device_map, trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )
    model.eval()
    load_time = time.time() - load_start
    logger.info(f"  Model loaded in {load_time:.1f}s")

    # Show device distribution
    if hasattr(model, "hf_device_map"):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        logger.info(f"  Distributed across devices: {devices_used}")

    prompts = DEFAULT_PROMPTS[:args.num_prompts]
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)

    output_path = args.output or f"result/expert/cross_layer_prediction_{Path(args.model_path).name}.json"

    logger.info(f"[Step 4/4] Running evaluation with {len(prompts)} prompts...")
    run_evaluation(model, tokenizer, config, prompts, top_k, num_expert_group, topk_group, args.max_new_tokens, output_path)


if __name__ == "__main__":
    main()

