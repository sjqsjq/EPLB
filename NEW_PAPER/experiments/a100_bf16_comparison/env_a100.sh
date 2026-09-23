#!/bin/sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B
export MODEL_PATH=/workspace/models/Qwen3-235B-A22B
