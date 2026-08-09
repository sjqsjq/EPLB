export NVSHMEM_HOME=/tmp/nvshmem_official/root  # 4卡部署改用官方3.7.2完整版(torch自带3.3.20缺静态host lib)
export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:$LD_LIBRARY_PATH"
export NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_IB_ENABLE_IBGDA=0
export NVSHMEM_HCA_LIST=
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_P2P=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- 4×H20 部署补充 (2026-08-09) ---
# torch 2.9.1 自带的 nvidia-nvshmem-cu12(3.3.20, 缺symbol) 会在deep_ep之前抢先
# 被动态链接器加载，导致 soname 冲突覆盖掉apt装的完整版NVSHMEM(3.7.2)。
# 必须用LD_PRELOAD强制优先加载正确版本，否则import deep_ep报undefined symbol。
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/nvshmem/12/libnvshmem_host.so.3
export NVSHMEM_DIR=/tmp/nvshmem_official/root
