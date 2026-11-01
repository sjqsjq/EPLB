# 端到端方案只读盘点（2026-10-18）：SGLang/EPLB/模型/数据/GPU 预算

性质：只读核查（未启动 GPU 实验）。行号以 /opt/conda/lib/python3.11/site-packages/sglang（下称 $SG）为准。

## 1. 服务端 SGLang 版本

- **版本 0.5.6.post2**，pip wheel 安装（dist-info INSTALLER=pip，无 direct_url.json / git commit 元数据——PyPI 轮子，非 git 检出）；Python 3.11（/opt/conda）。
- 本地补丁层（全部有 sha256 锁）：$SG/srt/managers/pb_oeplb/（17 文件，含 legacy/pre-v21 冻结备份目录）；model_runner.py 挂钩 L81-83/L440-444/L2663-2666；server_args.py L441-452（enable_pb_oeplb 及预算旗标，改动前备份 server_args.py.bak_before_pboeplb_fix 在位）；rev1 部署锁 deployed_shas_sp_coupling_rev1.txt（14 路径，SRC=部署位一致）。
- 实际运行 server_args（自 dev_r1 server.log 提取）：tp=8, dp=8, ep_size=8, enable_dp_attention=True, moe_a2a_backend='deepep', moe_runner_backend='deep_gemm', attention_backend='fa3', quantization='fp8', dtype bfloat16, mem_fraction_static=0.8, chunked_prefill_size=1024, max_prefill_tokens=16384, disable_cuda_graph=False, cuda_graph_max_bs=128（bs 表 1..128 共 20 档）, disable_radix_cache, watchdog 600s, ep_dispatch_algorithm='static', init_expert_location='trivial', enable_eplb=False, enable_pb_oeplb=True, ep_num_redundant_experts=0。
- 可执行启动器：$EXP/configs/followup_v1/ea_launchers/launch235b_eacampaign.sh（完整旗标见文件；环境 /workspace/logs/env_235b.sh：NVSHMEM 本地传输、NCCL_IB_DISABLE=1、NCCL_P2P_LEVEL=NVL、HF 离线）。

## 2. 官方 EPLB（$SG/eplb/）

**算法**（eplb_algorithms/__init__.py EplbAlgorithm 枚举）：deepseek、deepseek_hierarchical、deepseek_vec、deepseek_vec_hierarchical、elasticity_aware（弹性 EP 专用）。auto 解析（compute_algorithm）：num_groups 非空且整除 num_nodes → deepseek_hierarchical，否则 deepseek。**Qwen3-235B（128 专家、无专家组、单节点）auto=deepseek**。

**零冗余：支持。** ep_num_redundant_experts 默认 0（server_args L435）；deepseek.py L68-69 `num_redundant = num_phy - num_log; assert >= 0`（0 通过）；deepseek_vec.make_redundant_experts_chunkwise 在 redundancy=0 时 logical_to_physical_map 末维=1，正常退化；expert_location.py L201 physical=logical+redundant。零冗余下 rebalance = 按 token 权重把 128 逻辑专家做 8-rank 均衡划分/置换，无副本。约束：enable_eplb 要求 ep_size>1（server_args L1679 assert）；启用时 expert_distribution_recorder_mode 自动置 "stat"（L1667-1671）；eplb_rebalance_num_iterations（默认 1000）≥ recorder buffer（eplb_manager L27-30 assert）；节奏由 on_forward_pass_end 驱动，每 N 迭代重平衡一次，权重搬运走 expert_location_updater 的同步 batch_isend_irecv（分层 chunk + canary 校验）。

**与 PB 共用后端/CUDA Graph 兼容性：**
- CUDA Graph：两者都在迭代边界同步搬运权重（默认 PG 上的 batch_isend_irecv），权重原地更新、路由经设备侧元数据表读取，**均不需要重捕获图**。PB 路径已在全部封存运行中验证与 CUDA Graph 共存（disable_cuda_graph=False、bs 1..128、deepep+dp-attention）；async_swapper 文档串 L22-24 明确"与官方 ExpertLocationUpdater 同法（同步 batch_isend_irecv + req.wait()），无稳定性问题"。
- 双开：server_args 无互斥 assert（L1673 仅共同设置 ep_dispatch_algorithm="static"），但两者是**各自独立的记录→决策→搬运权限体系**（官方用 expert_distribution_recorder；PB 用自有记录路径，我们运行中 recorder_mode=None），同时启用=两个放置权威写同一份专家权重，**未经测试、语义冲突，配置时二选一**（每臂一个权威）。
- PB 特有 NCCL 约束（async_swapper 文档串，实测教训）：每次交换计划必须单个 batch_isend_irecv（分 chunk 会因 rank 参与不均导致 PG 序列号发散死锁）；内存用 --pb-oeplb-max-total-ops=300 约束；同步阻塞 ~200ms/次（9-92 ops，H20 实测）。官方 updater 按层 chunk（eplb_rebalance_layers_per_chunk）是另一机制，互不共享状态。

## 3. GPU 上可加载的模型

**磁盘上仅一个模型**：Qwen3-235B-A22B-FP8（/root/models/ms_cache/Qwen/Qwen3-235B-A22B-FP8 → models/Qwen--Qwen3-235B-A22B-FP8/snapshots/master，223GB，48 个 safetensors 分片）。config：qwen3_moe，94 层，128 专家/层，top-8，moe_intermediate=1536，hidden=4096，FP8 e4m3 动态激活量化（norm/gate/lm_head 不转换），torch_dtype=bf16。/data/models 为空（bench 脚本旧默认路径，被 OEPLB_MODEL 覆盖）。**作者计划的 8×H20 + Qwen3-235B-FP8 + EP=8 与已验证栈完全一致**——全部封存战役（P1/P2/SMOKE_R1/DEV_R1）都在此栈上运行。

## 4. 可用真实数据池

- **现役工作负载（驱动内 sha256 冻结）** $EXP/inputs/fullservice_v2/workload/：w1_stable_code_4000.jsonl（4000 请求，sha f0391f7c…）、w2_switch_code_chinese_8000.jsonl（8000 请求，域切换 code→chinese，sha 9d852a75…）、smoke_w1_300.jsonl（300，sha 073f79c2…）、calibration.jsonl（8037）/evaluation.jsonl（7963）+ split_manifest.json（源 16000 行 sha256 规则切分，互斥穷尽，selection_uses_performance_timing=false）。
- **上游池** /data/minghua/sjq/OEPLBdata/datasets/multi_domain/：domain_switch_160k.jsonl（701MB）、multidomain.jsonl、fixeddata.jsonl、frozen_requests_{500, hellaswag, in2048, longbench_4k, prefill_heavy, real_production, short_in_long_out}.jsonl（120KB–8.6MB 各档）。
- **捕获池**（专家分布/布局离线材料）：$EXP/inputs/cdx0_v1/（1.6GB：captures×8 + expert_weights/layouts/route_metadata/batches/triggers/capture_index）；$EXP/inputs/live_capture/（2.4GB）。
- **Bench 客户端** /workspace/EPLB/OEPLB/scripts/run_grid_bench.py：流式 HTTP /v1/completions、并发 256（参数）、real requests、无 delay floor；输出 total_time_s/tps/ttft/tpot 全谱。

## 5. GPU 预算（本机实测）

- 硬件：8× NVIDIA H20 96GB（97871MiB）=768GB HBM，当前全空闲（0% util / 4MiB）；主机 178 核 / 480GB RAM / 2.1TB 空闲盘；NVLink P2P（无 IB）。
- 实测单元成本（DEV_R1 12/12 账本）：boot 64–68s；W1 bench（4000req@256）352–357s；W2 bench（8000req@256）708–717s；校验+清场 60–120s → **单运行周期 W1≈8.5min、W2≈14.5min**；12 运行战役全程 2h03m，零基础设施重启。
- 预算参考：3 臂×2 条件×2 block 全战役 ≈ 2h05m（含 SMOKE 门 2 运行 ≈ +20min）；223GB 权重 EP8 每卡 ~28GB + KV（mem_fraction_static=0.8 → ~77GB/卡可用）。
