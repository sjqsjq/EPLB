#!/bin/sh
# A100 baseline env (no DeepEP => no NVSHMEM). NCCL over NVLink, offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export MODEL_PATH=/workspace/models/Qwen3-235B-A22B
# placements (hardware-independent, reused per REPRODUCE_BASELINES.md §8.1 / appendix shortest path)
export PLACEMENT_MOETUNER=/data/minghua/sjq/OEPLBdata/experiment_logs/moetuner_baseline_20260916/artifacts/placement_pinned_fair.json
export PLACEMENT_DATAFORE=/data/minghua/sjq/OEPLBdata/experiment_logs/baseline_comparison_20260914/datafore_remap_placement.json
export PLACEMENT_PROVER=/data/minghua/sjq/OEPLBdata/experiment_logs/baseline_comparison_20260914/datafore_prover_placement.json
