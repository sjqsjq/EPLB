# GB200 (sm_100, aarch64) single-node 4-GPU, NVLink. EP=4.
export NVSHMEM_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH}"
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
# CRITICAL: nvidia-smi misreports compute_cap 8.9; torch says (10,0). Force sm_100
# for tvm_ffi JIT (QK-Norm kernel uses griddepcontrol = sm_90+), else ptxas fatal.
export TVM_FFI_CUDA_ARCH_LIST="10.0"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Shadow editable sglang with our patched tree (sitecustomize does meta_path surgery).
export PYTHONPATH=/workspace/sglang-oeplb
export OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B-FP8
