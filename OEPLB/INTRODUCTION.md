# PB-OEPLB：面向 MoE 推理服务的自适应在线专家负载均衡

## 一、简介

混合专家（MoE）模型在推理服务中存在专家负载不均衡问题：少数热点专家集中在个别 GPU，造成计算瓶颈、尾部延迟与显存压力。现有方案如 SGLang 的 EPLB 需要冗余专家副本、重平衡期间阻塞推理 1.4–4.5 秒、并强制关闭 CUDA graph 导致 decode-heavy 负载退化 62%。

PB-OEPLB（Prefill-Boundary Online Expert Placement Load Balancer）是一种**零冗余、细粒度同步、在线自适应**的专家负载均衡器。它在现有专家预算内操作（不复制专家），仅搬运已有权重的物理位置；每次调整以单次同步 P2P 完成，阻塞代价约为 EPLB 的 1/4；并基于路由稳定性信号自动调整决策频率，跨负载匹配"调好的静态窗口"。在 8×H20 集群服务 Qwen3-235B-A22B-FP8 上实现，已作为 SGLang 0.5.6.post2 的一组补丁落地。

## 二、实现技术

| 技术 | 用途 |
|---|---|
| **PyTorch P2P（`isend`/`irecv`）** | rank 间搬运整块专家权重，不经集体通信 |
| **单次 `batch_isend_irecv` + `req.wait()`** | 把整窗所有交换 op 打成一个同步 collective，默认 PG/流，阻塞到完成 |
| **`empty_cache()` 预留 + 临时缓冲中转** | 为 NCCL P2P channel buffer 让出 raw cudaMalloc 显存；源/目不共用同一块 |
| **`fast_init_by_mapping`** | 向量化反置换重建路由元数据，替代官方双层 Python for（48×128 专家 375ms → ms 级） |
| **复用官方 `ExpertLocationUpdater`** | 按新映射把权重装进正确物理位置，不重造这层 |
| **指数衰减累加器（α=0.5）** | 以 `M=W/(1−α)` 统一窗口与衰减为单一 bias-variance 自由度 |
| **cos_sim 稳定性信号** | 替代 noise-chasing 的 ratio-delta 驱动自适应窗口 |
| **物理空间 scatter_add_ 记录** | 热路径零逐次转换，CUDA graph capture 期间自动跳过 |

落地形态：新增 `sglang/srt/managers/pb_oeplb/` package（controller / rebalancer / async_swapper / config / fast_metadata），并覆盖 3 个自带文件——`server_args.py`（注册 `--pb-oeplb-*` 参数 + 互斥校验 + 强制 `ep_dispatch_algorithm="static"`）、`model_runner.py`（controller 生命周期 + forward 末钩子）、`topk.py`（`select_experts` 后插记录钩子）。

## 三、算法流程

```
[记录]  每层 select_experts 产出 physical topk_ids
        → scatter_add_ 进 self.load[layer, phys_slot]
        → CUDA graph capture 期间 (decode) 自动跳过；只记 prefill

[决策]  每 sync_window 个 prefill forward 末 (model_runner.on_forward_pass_end):
        1. self.load.clone() → all_reduce(SUM) 聚合全局负载
        2. 算不均衡度 r = max(GPU负载)/mean
        3. 三道闸门: dead-zone(r≤r_k 不动) / bias 校正(小 token 噪声) / bias_gate / swap 预算
        4. try_build_swap_plan: 贪心成对规划
           - 选 r 最高的层, hot rank(降序) × cold rank(升序) 遍历候选槽
           - 大 gap → max-delta 贪心; 小 gap → 目标 gap/2 防 overshoot
           - 模拟交换, 仅 new_ratio < ratio-0.0005 才接受; 否则回滚标 tried
           - 全局预算 max_total_ops=300 / max_total_swap_layers=94

[执行]  AsyncSwapExecutor.begin(plan):
        单次 batch_isend_irecv + req.wait() (默认 PG, 同步阻塞 ~200ms)
        + verify checksum 校验权重确实搬运

[收尾]  try_finish:
        1. p2l 映射对调 → fast_init_by_mapping 重建元数据
        2. 喂官方 ExpertLocationUpdater.update 装载权重
        3. self.load 历史按新布局重映射 (让衰减历史跟随专家, 否则决策不收敛)

[自适应] cos_sim 驱动窗口:
        cos_sim<0.85 changepoint → adaptive_decay→0 清陈旧历史 (不 shrink)
        cos_sim≥0.95 连续2窗 → grow window 翻倍向 ceiling=M*
        窗口末: self.load *= α
```

核心设计点：本地步数计数器触发（零额外通信共识）、只记 prefill（提前纠偏）、all_reduce 必须 clone（防复合膨胀）、历史必须重映射（防决策卡死）。

## 四、实测效果

环境：8×H20，Qwen3-235B-A22B-FP8（TP=DP=EP=8），SGLang 0.5.6.post2 + DeepEP v1.2.1。

| 指标 | 结果 |
|---|---|
| prefill 密集负载吞吐（vs identity） | **+17.5%**（n=2） |
| vs EPLB | 高 15.7 个百分点 |
| 多域漂移负载 | +9.76% |
| 收敛性 | 3 个决策窗口内不均衡度 1.74 → 1.02 |
| 稳态每次调整阻塞 | 0.34–0.41s（首 swap 冷启动 4.21s），EPLB 1.55s，**降低 4 倍** |
| swap 阻塞占比 | ~3.42% 服务时间 |
| record 热路径开销 | 235B 约 2.3%，30B 约 1.6% |
| 自适应窗口 | 3/4 workload 匹配或超越 tuned static（+2.1%~+4.6%，长 decode −1.4%） |
| 校准最优 | M*=128（decode 内部峰） |

关键对比：PB-OEPLB 是唯一同时实现零冗余、CUDA graph 兼容、在线自适应、架构通用的系统——代价是每次调整 ~0.37s 有界阻塞，但以单次同步 `batch_isend_irecv` 完成、不破坏 SLA 一致性。
