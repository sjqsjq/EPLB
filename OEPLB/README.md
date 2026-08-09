# OEPLB — Online Expert Placement Load Balancer

在线事件驱动的 MoE expert placement 微调器，在 SGLang 推理服务运行时动态调整 expert 物理位置，降低 GPU 间负载不均衡。**当前部署版本 v0.11**（本地计数器触发 + 物理空间直接记录 + 指数衰减代替硬清零），支持 DP+EP 混合并行、DeepEP low_latency 模式。

## ⚡ 核心结论（2026-07-15 严格验证，请优先看这个，其余章节多为历史过程记录）

**PB-OEPLB 在 prefill-heavy 场景下有真实正收益：**

| 场景 | 数据集 | TPS delta | 结论 |
|---|---|---|---|
| **prefill-heavy** (4096in/32out) | `frozen_requests_prefill_heavy.jsonl` | **+8.3%** | 真实正收益 |

该数据点是完整跑完全量请求(非采样)得到的，且用 `--disable-radix-cache` 避免了重复请求命中缓存污染吞吐数字(见下方"已知陷阱")。

逐层逐step不均衡度分析（详见 `scripts/layer_imbalance_analysis.py`，不依赖任何profiler hook，靠kernel按层严格循环发射的时序特征直接还原layer归属）显示：swap **没有改善** dispatch/combine的平均不均衡度（甚至让它们的尾部风险变大——P2P搬权重偶尔跟通信抢NVLink），但**显著压低了expert计算的尾部不均衡**（prefill-heavy场景下 expert imbalance max ratio 2.41→1.74，-27.9%）。在expert计算是真瓶颈的prefill-heavy场景里，这个尾部改善盖过了dispatch/combine变差的代价，净为正。

**根因拆解**（record vs allreduce 隔离实验，600-forward-step固定采样，见下方"隔离实验方法论"）：record和allreduce单独打开都会让dispatch变慢约+3.4~3.7%（均超过noise floor 1.9%），两者一起打开(+5.3%)却明显小于简单相加(+7.1%)——是sub-additive，说明它们在竞争同一个瓶颈（很可能是单线程CPU scheduler critical path），不是互相独立叠加开销。


## 实验示范：L512_O1 单域Placement全面对比

**数据集**: `benchmarks/final_grid/L512_O1.jsonl` (8192条, DeepSeek-Prover-V1数学证明, ~500 token/条, O=1纯prefill)

**配置**: `--pb-oeplb-sync-window 16 --pb-oeplb-threshold-ratio 1.02 --pb-oeplb-max-total-ops 300` (decay_factor=0.5, 默认)

**结果**:

| Placement | total_tps | vs Baseline | 说明 |
|---|---|---|---|
| 最差(ratio=2.61) | 16514.8 | -17.7% | 每rank堆2个热专家 |
| Baseline(trivial round-robin) | 20061.4 | — | SGLang默认 |
| Frozen-EPLB(一次性EPLB+冻结) | 22668.1 | +13.0% | 16冗余专家, auto模式 |
| **OEPLB** | **23363.5** | **+16.5%** | 无冗余, auto模式 |
| EPLB-continuous(官方) | 22908.5 | +14.2% | 16冗余, deepep-mode=normal |
| 最优placement(理论天花板) | 24353.9 | +21.4% | 预计算oracle, 无冗余 |

**OEPLB超越EPLB +2.3个百分点，达到理论天花板的76%。** 且不需要冗余专家、不放弃deepep-mode=auto/CUDA graph。

**复现步骤**:
```bash
source oeplb_env.sh  # H20 NVLink环境变量
# Baseline
python3 -m sglang.launch_server --model-path <MODEL> --tp 8 --dp 8 --ep-size 8 \
  --enable-dp-attention --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm --quantization fp8 --mem-fraction-static 0.8 \
  --cuda-graph-max-bs 128 --disable-radix-cache --watchdog-timeout 600
# OEPLB
# 同上 + 添加:
  --enable-pb-oeplb --pb-oeplb-threshold-ratio 1.02 --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 16 --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64 --pb-oeplb-min-swap-ops 8 --pb-oeplb-max-total-ops 300
```

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
│   ├── frozen_requests_longbench_4k.jsonl    # 500请求, 长上下文(6k-24k字符)
│   ├── frozen_requests_short_in_long_out.jsonl # 3000请求, 短输入长输出(512out)
│   ├── frozen_requests_{500,in2048,hellaswag}.jsonl  # 早期版本数据集
│   └── results/                    # 历史实验结果 JSON (按标签查找, 详见文件名前缀: T1=baseline, T2=oeplb, A/B/C/D/S=隔离实验/长跑标签)
└── docs/                            # (历史设计文档已清理，均为过时的4卡/旧版本记录，以8卡实现为准)
```

## SGLang 集成

需要修改 SGLang 的以下文件（对应 `src/` 下的模块，已部署至 `/opt/conda/.../sglang/srt/managers/pb_oeplb/`，源码与安装路径需手动保持同步——没有symlink，改完`src/`要`cp`过去）：

| 文件 | 改动 |
|------|------|
| `server_args.py` | +7 个 CLI 参数 (`--enable-pb-oeplb`, `--pb-oeplb-sync-window` 等) |
| `model_executor/model_runner.py` | `initialize()` 创建 controller；`forward()` 尾部**无条件**调用 `on_forward_pass_end()`（DP 模式下 IDLE batch 也必须调用，否则各 rank 的 forward 计数器会失去同步） |
| `layers/moe/topk.py` | `select_experts()` 后调用 `controller.record_next_layer(topk_ids)` |
| `managers/scheduler.py` | 无修改 |

此外，本项目额外对 **DeepEP v1.2.1 源码**打了 2 处 patch（详见 `SETUP_GUIDE_H20.md`），使其 low_latency 模式能在单机 NVLink（无 IB/RDMA）拓扑下工作。

## 推荐配置

```bash
python -m sglang.launch_server \
  --model-path <MODEL> \
  --tp 8 --dp 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm \
  --quantization fp8 --mem-fraction-static 0.8 --cuda-graph-max-bs 128 \
  --enable-pb-oeplb \
  --pb-oeplb-threshold-ratio 1.02 \
  --pb-oeplb-min-prefill-tokens 256 \
  --pb-oeplb-sync-window 64 \
  --pb-oeplb-max-total-swap-layers 94 \
  --pb-oeplb-max-swaps-per-layer 64 \
  --pb-oeplb-min-swap-ops 8
# 循环同一批请求做长跑压测时加: --disable-radix-cache
# 注: 曾经文档里写过 --pb-oeplb-cooldown-steps，这个CLI参数从未被注册进argparse，
# 直接照抄会报 unrecognized arguments，已从示例里删掉。

# decay_factor默认0.5(config.py中已更新), 无需CLI传参
# 如果不知道流量的典型长度分布，想让adaptive window自动校准反应灵敏度，额外加：
#   --pb-oeplb-adaptive-window \
#   --pb-oeplb-calibrate-adaptive-sensitivity
```

单机纯 NVLink（无 IB）拓扑下使用 DeepEP 需要额外环境变量（见 `scripts/run_T2_oeplb_sparse.sh`）：
```bash
export NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_IB_ENABLE_IBGDA=0 NVSHMEM_HCA_LIST= \
       NVSHMEM_BOOTSTRAP=UID NVSHMEM_DISABLE_P2P=0 NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL \
       SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
```

## 架构要点回顾

当前 v0.11 架构的几个关键设计点（完整调研过程见 `docs/final_report.md`，仅作历史参考）：

- **异步 P2P swap 执行**：swap 操作在独立低优先级 CUDA stream 上发起，非阻塞；`force_wait` 在每轮决策发起 all_reduce 前强制确认上一轮 P2P 传输已在 GPU 上真正完成，避免跨 rank NCCL op 顺序错位导致的死锁。
- **本地计数器触发，零额外通信**：每个 rank 只看自己的本地 forward 计数器判断"是否到 sync_window"，不需要每次 forward 都做跨 rank 共识——forward pass 本身就是全局同步的，靠这一点隐式对齐，避免了额外的 all_reduce 开销。
- **物理空间直接记录**：`record_next_layer` 直接对 physical slot 做 `scatter_add_`，不做逐次的物理↔逻辑转换；物理转逻辑只在每个 sync_window 做一次向量化转换，而不是每个 prefill batch 都做一次。
- **指数衰减代替硬清零**：每个 sync_window 结束后，load 历史按 `decay_factor` 衰减而不是清零，兼顾对新负载的响应速度和对单批次噪声的抗性。

## Adaptive Window 灵敏度校准（PD比例驱动）

### 初衷

之前的疑问是："能不能根据输入/输出长度，给adaptive window选一个更聪明的初始sync_window？" 用`final_grid`（L=256/512/2048/4096 × O=1/64/256）做了bracketed-baseline（每个测试点前后各夹一次baseline，排除时间漂移干扰）验证后，数据给出的答案跟猜想不一样：

- **收益幅度随输出越长单调递减，是真实、稳健的规律**（L=512上：O=1时+14~22%，O=64时+8~10%，O=256时+3~5%；L=2048/4096上复现了同样的递减趋势）。
- **但"具体该选哪个sync_window"，跟这个比例没有可利用的关系**——同一个(L,O)格子里4个候选窗口(8/16/32/64)之间的差距只有1.5-8个百分点，没有随长度变化的单调走向，属于测量噪声量级。

所以最终落地的不是"选窗口"，是**用运行时可观测的prefill:decode forward-pass比例，去预测这个workload大概能拿到多少收益，再用这个预测去调节adaptive_window现有滞回带机制的反应灵敏度**——预测收益大就让它更敢收缩窗口追不均衡，预测收益小就更保守，避免为小收益买单不必要的all_reduce/P2P开销。不新造一套"选窗口"的机制，只给已有的`window_floor`/`window_shift_confirm_windows`加一层运行时校准。

### 设计

`--pb-oeplb-calibrate-adaptive-sensitivity`（需同时开`--pb-oeplb-adaptive-window`）开启后：

1. 校准阶段（`calibration_forwards`个forward，默认256）：统计本rank的非idle forward里prefill/decode各多少次。
2. 触发时机用`self._forward_id`（本来就跨rank隐式lockstep的本地计数器）判断，不用本地非idle计数达到阈值来判断——**DP模式下不同rank在同一个global step可能真的处于不同阶段**（一个prefill一个decode），本地计数只是这个rank自己的局部视角，不能直接拿来做跨rank必须一致的决策。
3. 触发后对`[prefill_fwd, decode_fwd]`做一次`all_reduce(SUM)`，用全局聚合出的`decode_fraction`统一决策，保证所有rank选到同一个档位。

| decode_fraction | 档位 | window_floor | shift_confirm | 设计意图 |
|---|---|---|---|---|
| < 0.5 | prefill-heavy | 8 | 1 | 收益大，敢收缩，快速反应 |
| 0.5 ~ 0.86 | balanced | 32(默认) | 1(默认) | 维持现状 |
| ≥ 0.86 | decode-heavy | 64(=sync_window，等于禁止收缩) | 3 | 收益小，基本退化成静态窗口 |

边界值是实测校准的：L=512上O=1/64/256分别测出decode_fraction≈0.03/0.78/0.93，这两个高值之间取中点(0.86)做balanced/decode-heavy的分界。**一个验证了设计初衷的发现**：同样是O=64，decode_fraction在L=512时是0.78，L=2048时降到0.49，L=4096时降到0.28——输入越长，prefill在forward-pass总量里占的份额越大，同一个O在不同L下会被正确分到不同档位，不需要提前告诉机制L是多少。这正是选"运行时比例"而不是"静态L/O查表"的原因：真实流量长度混杂时，比例能自动跟着变，查表不能。

### 效果 vs 历史"最优静态窗口"数据——为什么新数字看起来更保守

L2048/L4096网格上实测（bracketed baseline）：O=1时+8.9~10.6%，O=64时+9.9~10.8%，O=256时+5.9~9.1%——方向都对，但比历史`results/final/`里"4个窗口里挑最优"报出来的数字（L=2048: +13.3%/+11.5%/+12.1%；L=4096: +11.1%/+12.7%/+10.7%）低一些。这不是回归，是两件不同的事在对比：

1. **"挑4个里最好的"天然带选择偏差**——4个噪声样本里取最大值，期望上就会比任何单一自动化选择的结果更高，跟这次运气好不好没关系，纯粹是"try 4 times, report the best"这个报告方式本身的数学性质。
2. **测量方法论更严格**：历史数字是单次baseline对单次treatment，这次全部换成"baseline-treatment-baseline"三明治取双侧均值，去掉了单侧baseline运气好或运气差带来的虚高/虚低。
3. **calibration机制的价值主张本来就不是"比人工试4次挑最好的还高"**——它是"不用人工试、不用提前知道L/O，运行时自动给出一个接近最优、且在长度变化时依然合理的选择"。跟"最优静态窗口"打平或小幅落后，是自动化换取泛化能力的合理代价，不是设计缺陷。

## 待验证方向

- dispatch/combine尾部变差是否真的是P2P抢NVLink带宽导致（需要timestamp级别的时间对齐分析，目前只是相关性证据）
- expert replication方向：swap-only解决不了单一极端热点专家，只能把热点从一个rank挪到另一个rank（已验证案例：baseline下某rank主导热点会轮转到其他rank，swap后变成另一个rank持续主导）
- adaptive-sensitivity校准目前只在启动时做一次，不会随流量分布漂移重新校准；真实生产流量如果长度分布随时间变化，这个机制目前不会跟着调整档位——如果需要，可以复用同一套all_reduce校准逻辑，改成每隔N个sync_window重新采样一次
