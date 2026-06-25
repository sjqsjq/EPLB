#!/usr/bin/env python3
"""
Offline MoE Routing Recorder — Expert Load Balance Analysis

Loads a model offline, runs Prefill on a CSV dataset, and records
how many tokens each expert receives at each MoE layer.

Output: per-layer per-expert token count matrix for load balance analysis.
"""

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Grouped TopK (DeepSeek-V3 style)
# ============================================================================

def grouped_topk(
    scores: torch.Tensor,
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    correction_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Grouped expert selection: pick topk_group groups first, then top_k experts.
    Returns: (num_tokens, top_k) expert indices.
    """
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)

    num_tokens, num_experts = scores.shape

    if num_expert_group == 1 or topk_group == 0:
        if correction_bias is not None:
            scores = scores + correction_bias.float().unsqueeze(0)
        return torch.topk(scores, top_k, dim=-1).indices

    if correction_bias is not None:
        scores_for_selection = scores + correction_bias.float().unsqueeze(0)
    else:
        scores_for_selection = scores

    group_size = num_experts // num_expert_group
    scores_grouped = scores_for_selection.view(num_tokens, num_expert_group, group_size)
    group_scores = scores_grouped.amax(dim=-1)
    top_groups = torch.topk(group_scores, topk_group, dim=-1).indices

    mask = torch.zeros_like(scores_for_selection, dtype=torch.bool)
    for i in range(topk_group):
        group_idx = top_groups[:, i]
        start = group_idx * group_size
        for j in range(group_size):
            col = start + j
            col = col.clamp(max=num_experts - 1)
            mask.scatter_(1, col.unsqueeze(1), True)

    masked_scores = scores_for_selection.masked_fill(~mask, float("-inf"))
    return torch.topk(masked_scores, top_k, dim=-1).indices


# ============================================================================
# Gate Recorder (Hook-based, reused from cross_layer_routing_predictor)
# ============================================================================

class GateRecorder:
    """Hooks into MoE gates to capture router logits per forward pass."""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.hooks = []
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

            if hasattr(gate, "weight"):
                self.gate_weights[layer_idx] = gate.weight.detach().cpu()
            if hasattr(gate, "e_score_correction_bias") and gate.e_score_correction_bias is not None:
                self.correction_biases[layer_idx] = gate.e_score_correction_bias.data.detach().cpu()
            else:
                self.correction_biases[layer_idx] = None

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
            if gate_input.dim() == 3:
                gate_input = gate_input.view(-1, gate_input.shape[-1])

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
        self.router_logits.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ============================================================================
# Dataset loading
# ============================================================================

def load_csv_dataset(csv_path: str, text_column: str, max_prompts: int) -> List[str]:
    """Load text prompts from a CSV file."""
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        available_columns = reader.fieldnames
        if text_column not in available_columns:
            logger.error(f"Column '{text_column}' not found in CSV. Available columns: {available_columns}")
            # Try to auto-detect a text column
            text_candidates = ["text", "prompt", "content", "input", "question", "sentence", "query"]
            for candidate in text_candidates:
                if candidate in available_columns:
                    text_column = candidate
                    logger.info(f"Auto-detected text column: '{text_column}'")
                    break
            else:
                raise ValueError(f"Cannot find text column. Available: {available_columns}")

        for row in reader:
            text = row[text_column].strip()
            if text:
                prompts.append(text)
            if len(prompts) >= max_prompts:
                break

    logger.info(f"Loaded {len(prompts)} prompts from {csv_path} (column: '{text_column}')")
    return prompts


# ============================================================================
# Core: run prefill and count expert assignments
# ============================================================================

def run_prefill_routing(
    model,
    tokenizer,
    recorder: GateRecorder,
    prompts: List[str],
    top_k: int,
    num_expert_group: int,
    topk_group: int,
    num_experts: int,
    max_seq_len: int = 512,
) -> Dict[int, List[int]]:
    """
    Run Prefill on all prompts, accumulate per-layer per-expert token counts.
    Returns: {layer_idx: [count_expert_0, count_expert_1, ..., count_expert_N]}
    """
    moe_layers = sorted(recorder.gate_weights.keys())
    # Initialize counters: {layer_idx: array of size num_experts}
    expert_counts = {layer: [0] * num_experts for layer in moe_layers}
    total_tokens = 0

    if max_seq_len > 0:
        logger.info(f"Max sequence length: {max_seq_len} (longer prompts will be truncated)")
    else:
        logger.info("No sequence length limit — using full prompt length")

    for prompt_idx, prompt_text in enumerate(prompts):
        tokenize_kwargs = {"return_tensors": "pt"}
        if max_seq_len > 0:
            tokenize_kwargs["truncation"] = True
            tokenize_kwargs["max_length"] = max_seq_len
        else:
            tokenize_kwargs["truncation"] = False
        inputs = tokenizer(prompt_text, **tokenize_kwargs)
        input_ids = inputs["input_ids"].to(model.device)
        num_tokens = input_ids.shape[1]
        total_tokens += num_tokens

        recorder.clear()

        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)

        # For each MoE layer, compute top-k expert selection and count
        for layer_idx in moe_layers:
            if layer_idx not in recorder.router_logits:
                continue

            logits = recorder.router_logits[layer_idx].float()
            scores = logits.sigmoid()
            bias = recorder.correction_biases.get(layer_idx)

            selected_experts = grouped_topk(scores, top_k, num_expert_group, topk_group, bias)

            # Count how many tokens each expert received
            for expert_id in selected_experts.flatten().tolist():
                expert_counts[layer_idx][expert_id] += 1

        if (prompt_idx + 1) % 5 == 0 or prompt_idx == 0:
            logger.info(f"  [{prompt_idx + 1}/{len(prompts)}] tokens={num_tokens}, "
                        f"total_tokens_so_far={total_tokens}")

    logger.info(f"Prefill complete: {len(prompts)} prompts, {total_tokens} total tokens")
    return expert_counts, total_tokens


# ============================================================================
# Output: save results
# ============================================================================

def save_results(
    expert_counts: Dict[int, List[int]],
    total_tokens: int,
    num_prompts: int,
    top_k: int,
    num_experts: int,
    model_name: str,
    dataset_path: str,
    output_path: str,
):
    """Save expert load distribution results."""
    output_data = {
        "model": model_name,
        "dataset": dataset_path,
        "num_prompts": num_prompts,
        "total_tokens": total_tokens,
        "top_k": top_k,
        "num_experts": num_experts,
        "total_expert_assignments": total_tokens * top_k,
        "layers": {},
    }

    for layer_idx in sorted(expert_counts.keys()):
        counts = expert_counts[layer_idx]
        total_assignments = sum(counts)
        ideal_per_expert = total_assignments / num_experts if num_experts > 0 else 0

        # Load balance metrics
        max_count = max(counts)
        min_count = min(counts)
        std_count = (sum((c - ideal_per_expert) ** 2 for c in counts) / num_experts) ** 0.5
        coeff_of_variation = std_count / ideal_per_expert if ideal_per_expert > 0 else 0
        # Imbalance ratio: max / ideal (1.0 = perfect balance)
        imbalance_ratio = max_count / ideal_per_expert if ideal_per_expert > 0 else 0
        # Fraction of experts that received 0 tokens
        zero_experts = sum(1 for c in counts if c == 0)

        output_data["layers"][str(layer_idx)] = {
            "expert_counts": counts,
            "total_assignments": total_assignments,
            "ideal_per_expert": round(ideal_per_expert, 2),
            "max_count": max_count,
            "min_count": min_count,
            "std": round(std_count, 2),
            "coeff_of_variation": round(coeff_of_variation, 4),
            "imbalance_ratio": round(imbalance_ratio, 4),
            "zero_experts": zero_experts,
        }

    # Print summary table
    print("\n" + "=" * 90)
    print(f"Expert Load Balance Report — {model_name}")
    print(f"Dataset: {dataset_path} | Prompts: {num_prompts} | Total tokens: {total_tokens}")
    print(f"Top-K: {top_k} | Experts: {num_experts} | Assignments per token: {top_k}")
    print("=" * 90)
    print(f"{'Layer':<8} {'Total':<10} {'Ideal':<10} {'Max':<8} {'Min':<8} "
          f"{'Std':<10} {'CV':<8} {'Imbal.':<8} {'Zero':<6}")
    print("-" * 90)

    all_cv = []
    all_imbal = []
    for layer_idx in sorted(expert_counts.keys()):
        layer_data = output_data["layers"][str(layer_idx)]
        cv = layer_data["coeff_of_variation"]
        imbal = layer_data["imbalance_ratio"]
        all_cv.append(cv)
        all_imbal.append(imbal)
        print(f"{layer_idx:<8} {layer_data['total_assignments']:<10} "
              f"{layer_data['ideal_per_expert']:<10.1f} "
              f"{layer_data['max_count']:<8} {layer_data['min_count']:<8} "
              f"{layer_data['std']:<10.2f} {cv:<8.4f} {imbal:<8.4f} "
              f"{layer_data['zero_experts']:<6}")

    print("-" * 90)
    avg_cv = sum(all_cv) / len(all_cv) if all_cv else 0
    avg_imbal = sum(all_imbal) / len(all_imbal) if all_imbal else 0
    print(f"{'AVG':<8} {'':10} {'':10} {'':8} {'':8} "
          f"{'':10} {avg_cv:<8.4f} {avg_imbal:<8.4f}")
    print("=" * 90)
    print(f"\nCV (Coefficient of Variation): lower = more balanced. Perfect balance = 0.0")
    print(f"Imbalance Ratio: max/ideal. Perfect balance = 1.0")

    # Print per-layer per-expert detailed token distribution
    print("\n\n" + "=" * 120)
    print("Per-Layer Per-Expert Token Distribution")
    print("=" * 120)

    for layer_idx in sorted(expert_counts.keys()):
        counts = expert_counts[layer_idx]
        layer_data = output_data["layers"][str(layer_idx)]
        ideal = layer_data["ideal_per_expert"]

        print(f"\n--- Layer {layer_idx} (Total={layer_data['total_assignments']}, "
              f"Ideal={ideal:.1f}, CV={layer_data['coeff_of_variation']:.4f}) ---")

        # Print expert counts in a compact table (16 experts per row)
        experts_per_row = 16
        for row_start in range(0, num_experts, experts_per_row):
            row_end = min(row_start + experts_per_row, num_experts)
            header_parts = [f"E{i:<6d}" for i in range(row_start, row_end)]
            value_parts = [f"{counts[i]:<7d}" for i in range(row_start, row_end)]
            if row_start == 0:
                print(f"  {'Expert':<8}" + "".join(header_parts))
            print(f"  {'Count':<8}" + "".join(value_parts))

        # Show top-10 hottest and coldest non-zero experts
        indexed_counts = [(i, c) for i, c in enumerate(counts)]
        sorted_desc = sorted(indexed_counts, key=lambda x: x[1], reverse=True)
        non_zero = [(i, c) for i, c in sorted_desc if c > 0]

        top_hot = sorted_desc[:10]
        top_cold = [x for x in reversed(non_zero)][:10] if non_zero else []

        print(f"  Top-10 hottest: " +
              ", ".join(f"E{idx}={cnt}" for idx, cnt in top_hot))
        if top_cold:
            print(f"  Top-10 coldest (non-zero): " +
                  ", ".join(f"E{idx}={cnt}" for idx, cnt in top_cold))

    print("\n" + "=" * 120)

    # Export per-expert CSV for easy analysis
    csv_output_path = str(Path(output_path).with_suffix('.csv'))
    with open(csv_output_path, "w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        # Header: Layer, Expert_0, Expert_1, ..., Expert_N-1
        header_row = ["Layer"] + [f"Expert_{i}" for i in range(num_experts)]
        writer.writerow(header_row)
        for layer_idx in sorted(expert_counts.keys()):
            row = [layer_idx] + expert_counts[layer_idx]
            writer.writerow(row)
    logger.info(f"Per-expert CSV saved to {csv_output_path}")

    # Save JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    return output_data


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Offline MoE routing recorder for load balance analysis")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column name in CSV containing text (default: 'text')")
    parser.add_argument("--max-prompts", type=int, default=100, help="Max prompts to process")
    parser.add_argument("--max-seq-len", type=int, default=0,
                        help="Max token length per prompt. 0 = no truncation (use full prompt). "
                             "Set a positive value to truncate if CUDA OOM (default: 0)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "auto"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # Step 1: Load config
    logger.info(f"[Step 1/4] Loading config from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    top_k = getattr(config, "num_experts_per_tok", 8)
    num_expert_group = getattr(config, "n_group", 1)
    topk_group = getattr(config, "topk_group", 1)
    num_experts = getattr(config, "n_routed_experts", 256)
    num_layers = getattr(config, "num_hidden_layers", 61)
    model_name = Path(args.model_path).name
    logger.info(f"  Model: {model_name} | {num_layers} layers, {num_experts} experts, "
                f"top_k={top_k}, n_group={num_expert_group}, topk_group={topk_group}")

    # Step 2: Load tokenizer
    logger.info(f"[Step 2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"  Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # Step 3: Load model
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": "auto"}
    logger.info(f"[Step 3/4] Loading model (dtype={args.dtype}, device_map={args.device_map})...")
    logger.info(f"  This may take 5-15 minutes for large models...")

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype_map[args.dtype],
        device_map=args.device_map, trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )
    model.eval()
    load_time = time.time() - load_start
    logger.info(f"  Model loaded in {load_time:.1f}s")

    if hasattr(model, "hf_device_map"):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        logger.info(f"  Distributed across devices: {devices_used}")

    # Step 4: Load dataset and run
    logger.info(f"[Step 4/4] Loading dataset and running Prefill...")
    prompts = load_csv_dataset(args.dataset, args.text_column, args.max_prompts)

    recorder = GateRecorder(model, config)
    expert_counts, total_tokens = run_prefill_routing(
        model, tokenizer, recorder, prompts,
        top_k, num_expert_group, topk_group, num_experts,
        max_seq_len=args.max_seq_len,
    )
    recorder.remove_hooks()

    # Save
    output_path = args.output
    if not output_path:
        output_path = f"result/expert/routing_record_{model_name}.json"

    save_results(expert_counts, total_tokens, len(prompts),
                 top_k, num_experts, model_name, args.dataset, output_path)


if __name__ == "__main__":
    main()

