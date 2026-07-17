# OEPLB — Online Expert Placement Load Balancer

在线事件驱动的 MoE expert placement 微调器，在 SGLang 推理服务运行时动态调整 expert 物理位置，降低 GPU 间负载不均衡。**当前部署版本 v0.11**（本地计数器触发 + 指数衰减统计 + 快速 sync_window，修复了 v0.10 的 double-conversion bug），支持 DP+EP 混合并行、DeepEP low_latency 模式。

## ⚡ 核心结论（2026-07-15 严格验证，请优先看这个，其余章节多为历史过程记录）

**PB-OEPLB 的收益是场景相关的，不是全局正收益：**

| 场景 | 数据集 | TPS delta | 结论 |
|---|---|---|---|
| **prefill-heavy** (4096in/32out) | `frozen_requests_prefill_heavy.jsonl` | **+8.3%** | 真实正收益 |
| **decode-heavy** (domain-clustered, 512out) | `frozen_requests_domain_clustered.jsonl` | **-10.5%** | 真实负收益 |

同一套代码、同一组参数 (`threshold=1.15, sync_window=64`)，只是workload的prefill:decode比例不同，净收益方向就会反转。**结论边界=prefill占比**，不是算法本身有效/无效的问题。两个数据点都是完整跑完全量请求(非采样)得到的，且都用 `--disable-radix-cache` 避免了重复请求命中缓存污染吞吐数字(见下方"已知陷阱")。

逐层逐step不均衡度分析（详见 `scripts/layer_imbalance_analysis.py`，不依赖任何profiler hook，靠kernel按层严格循环发射的时序特征直接还原layer归属）显示：swap **没有改善** dispatch/combine的平均不均衡度（甚至让它们的尾部风险变大——P2P搬权重偶尔跟通信抢NVLink），但**显著压低了expert计算的尾部不均衡**（prefill-heavy场景下 expert imbalance max ratio 2.41→1.74，-27.9%）。在expert计算是真瓶颈的prefill-heavy场景里，这个尾部改善盖过了dispatch/combine变差的代价，净为正；在decode-heavy场景里则相反。

**根因拆解**（record vs allreduce 隔离实验，600-forward-step固定采样，见下方"隔离实验方法论"）：record和allreduce单独打开都会让dispatch变慢约+3.4~3.7%（均超过noise floor 1.9%），两者一起打开(+5.3%)却明显小于简单相加(+7.1%)——是sub-additive，说明它们在竞争同一个瓶颈（很可能是单线程CPU scheduler critical path），不是互相独立叠加开销。

## 已知陷阱（踩过的坑，下次直接抄作业）

1. **循环数据集做长跑测试，必须加 `--disable-radix-cache`**。500条请求循环15次给同一个服务器发，如果不关radix cache，第2轮起会命中缓存直接跳过prefill，TPS会虚高2-3倍（实测从预期~380飙到~1100），完全看不出prefill开销。
2. **`--enable-layerwise-nvtx-marker` 不会产生 `nn.Module: X` 风格的trace事件** ——它用的是 `torch.cuda.nvtx.range_push`，这是给Nsight Systems看的CUDA NVTX marker，不会进 `torch.profiler` 导出的chrome trace。想要逐层归因，别指望这个flag，用kernel发射的时序周期性（见下方方法论）。
3. **profiler `with_stack=True` 在长时间(>60s)+高并发下会OOM/把服务器拖死**——实测71秒的with_stack=True并发采集让4个rank进程各自涨到56GB+还在涨，最后只能kill -9清场。短快照(num_steps≤200，几秒到几十秒)是安全的；长采样一定要 `with_stack=False` 且靠 `num_steps` 的forward_ct自动停止,不要手动算时间去调 `/stop_profile`——后者是历史上"5步vs6步"不一致的真正原因（手动stop的时机取决于客户端请求完成的快慢，跟真实forward_ct脱钩）。
4. **profiler `num_steps` 是"总forward_ct数"不是"prefill窗口数"**：decode在CUDA graph replay下会跳过python层module hook，所以只统计 `nn.Module: DeepEPMoE_X` 出现次数只能拿到prefill次数，不是真实step数。

## 隔离实验方法论（record vs allreduce vs swap 各自贡献多少开销）

对照组 A(纯baseline)/B(只record禁allreduce)/C(只allreduce禁record)/D(record+allreduce都开但禁swap)，用 `OEPLB_EXP_MODE` 环境变量在 `controller.py` 里加了两行判断做隔离（实验完已revert，不在当前代码里，需要复现自己临时加）：
```python
# record_next_layer() 里，MIN_RECORD_TOKENS检查之后：
if os.environ.get("OEPLB_EXP_MODE") == "norecord": return
# _decide_and_begin_swap() 方法体第一行：
if os.environ.get("OEPLB_EXP_MODE") == "noallreduce": return
```
用 `scripts/profile_fixed_steps.py`（num_steps=600固定forward_ct自动停止，无需手动计时）采样，3次重复A测出噪声下限：dispatch/expert~2%，combine~11%，waste_pct~38%（这个指标噪声太大不可用）。B/C/D的dispatch delta分别是+3.7%/+3.4%/+5.3%，全部超过噪声下限，且D远小于B+C简单相加——结论见上方"核心结论"。

## 逐层不均衡度分析方法论

`scripts/layer_imbalance_analysis.py`：不依赖 `--enable-layerwise-nvtx-marker`（不work，见"已知陷阱"）或 `with_modules`（SGLang的`start_profile`没传这个参数，标准PyTorch profiler也拿不到module级别标注）。直接用一个更可靠的不变量：**同一个(category, exact kernel name)在时间上严格按layer 0→47的顺序循环发射，每层每次forward恰好一次**（实测验证：dispatch/combine kernel总数严格是48的整数倍；expert对应2种shape的deep_gemm kernel，各自也严格是48的整数倍）。按时间排序后每48个切一组，组内位置=layer_id，组序号=step index，跨rank对比同一个(step,layer)算imbalance ratio=max_rank/avg_rank。

## 项目结构

```
OEPLB/
├── src/                            # 核心源码 (部署到 sglang/srt/managers/pb_oeplb/)
│   ├── __init__.py                 # 全局 controller 单例
│   ├── config.py                   # PBOEPLBConfig 配置
│   ├── controller.py               # 核心状态机 (v0.11: 本地计数器触发 + 指数衰减)
│   ├── rebalancer.py               # 决策算法 (向量化 + 均衡前后比例诊断日志)
│   ├── async_swapper.py            # 异步 P2P swap 执行器 (独立低优先级 stream)
│   └── fast_metadata.py            # 向量化 metadata 重建 (3.4ms vs 官方 375ms)
├── scripts/
│   ├── run_T1_baseline.sh          # T1: 纯净 baseline (无 EPLB/OEPLB)
│   ├── run_T1_nodeepep.sh          # T1变体: 不用deepep
│   ├── run_T2_oeplb_sparse.sh      # T2: PB-OEPLB 稀疏采样模式
│   ├── run_T3_oeplb_always.sh      # T3: PB-OEPLB always-record 模式
│   ├── run_T4_eplb.sh              # T4: SGLang 官方 EPLB, deepep-mode=normal (对比基线)
│   ├── run_T4_eplb_auto.sh         # T4变体: deepep-mode=auto (注意: 官方EPLB配置在decode阶段可能无法用cuda graph, 见final_report.md)
│   ├── run_bench.py                # 固定请求集压测客户端 (读 FROZEN 常量指定的数据集)
│   ├── long_bench.py               # 长跑压测客户端 (参数化数据集路径, 适合循环数据集做长时长测试; 循环时记得给服务器加--disable-radix-cache)
│   ├── profile_fixed_steps.py      # 固定forward_ct采样(num_steps自动停止), 消除"手动算时间调stop_profile"的样本量漂移
│   ├── layer_imbalance_analysis.py # 逐层逐step不均衡度分析 (dispatch/combine/expert分别算, 见上方方法论)
│   ├── freeze_requests.py          # 生成 frozen_requests_*.jsonl 数据集
│   ├── compare.py                  # 多组结果对比 + 异常检测
│   ├── analyze_expert_heatmap.py   # 分阶段发请求, 观察expert热点随domain切换的迁移
│   ├── sweep_run.py / sweep_one.sh / param_sweep.sh  # 参数扫描 (threshold/sync_window/max_swaps组合)
│   └── test_latency.py             # 单请求TTFT/TPOT精确测量 (streaming)
├── benchmarks/
│   ├── prompts/real_long_unique.jsonl        # 源数据: 134条真实客服对话
│   ├── frozen_requests_prefill_heavy.jsonl   # 500请求, 4096in/32out — PB-OEPLB目标场景
│   ├── frozen_requests_domain_clustered.jsonl # 500请求, ~7.5k字符/域聚类, 512out — decode-heavy对照场景
│   ├── frozen_requests_longbench_4k.jsonl    # 500请求, 长上下文(6k-24k字符)
│   ├── frozen_requests_short_in_long_out.jsonl # 3000请求, 短输入长输出(512out)
│   ├── frozen_requests_{500,in2048,hellaswag}.jsonl  # 早期版本数据集
│   └── results/                    # 历史实验结果 JSON (按标签查找, 详见文件名前缀: T1=baseline, T2=oeplb, A/B/C/D/S=隔离实验/长跑标签)
└── docs/
    ├── final_report.md             # v0.1→v0.9 完整版本演进报告 (DP支持+DeepEP调研+负载均衡验证过程)
    ├── impl_guide.md                # 原始实施指导文档 (v0.1 设计, 历史参考)
    └── experiment_plan.md          # 原始实验方案 (v0.1 设计, 历史参考)
```

## SGLang 集成

需要修改 SGLang 的以下文件（对应 `src/` 下的模块，已部署至 `/opt/conda/.../sglang/srt/managers/pb_oeplb/`，源码与安装路径需手动保持同步——没有symlink，改完`src/`要`cp`过去）：

| 文件 | 改动 |
|------|------|
| `server_args.py` | +7 个 CLI 参数 (`--enable-pb-oeplb`, `--pb-oeplb-sync-window` 等) |
| `model_executor/model_runner.py` | `initialize()` 创建 controller；`forward()` 尾部**无条件**调用 `on_forward_pass_end()`（DP 模式下 IDLE batch 也必须调用，否则各 rank 的 forward 计数器会失去同步） |
| `layers/moe/topk.py` | `select_experts()` 后调用 `controller.record_next_layer(topk_ids)` |
| `managers/scheduler.py` | 无修改 |

此外，本项目额外对 **DeepEP 1.1.0 源码**打了 2 处 patch（详见 `docs/final_report.md` §DeepEP NVLink 适配），使其 low_latency 模式能在单机 NVLink（无 IB/RDMA）拓扑下工作。

## 推荐配置

```bash
python -m sglang.launch_server \
  --model-path <MODEL> \
  --tp 4 --dp 4 --ep-size 4 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.15 \
  --pb-oeplb-min-prefill-tokens 1000 \
  --pb-oeplb-sync-window 64 \
  --pb-oeplb-cooldown-steps 5 \
  --pb-oeplb-max-total-swap-layers 48 \
  --pb-oeplb-max-swaps-per-layer 3
# 循环同一批请求做长跑压测时加: --disable-radix-cache
```

单机纯 NVLink（无 IB）拓扑下使用 DeepEP 需要额外环境变量（见 `scripts/run_T2_oeplb_sparse.sh`）：
```bash
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_HCA_LIST= \
       NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL \
       SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
```

## 历史版本演进 (v0.1→v0.9)

详见 `docs/final_report.md`——完整记录了死锁排查(v0.1→v0.2)、异步swap架构(v0.3)、DP支持踩坑(v0.6→v0.8的-16.6%吞吐回归)、以及对齐官方EPLB本地计数器触发机制(v0.9)的调研过程。v0.10→v0.11在此基础上修复了一个p2l双重转换bug并把sync_window从64调到8又调回64（用指数衰减代替硬清零来平衡响应速度和噪声抗性）。

## 待验证方向

- 中间态workload（输出128-256 token）下净收益转正的具体拐点在哪
- 更保守的threshold(1.3-1.5)是否能在decode-heavy场景下也做到net-positive
- dispatch/combine尾部变差是否真的是P2P抢NVLink带宽导致（需要timestamp级别的时间对齐分析，目前只是相关性证据）
- expert replication（v0.4方向）：swap-only解决不了单一极端热点专家，只能把热点从一个rank挪到另一个rank（layer24案例已验证：baseline下rank1主导热点会轮转到0/2/3，swap后变成rank0持续主导）
