# OEPLB V2 设计计划

## 一、V1的核心问题（有数据支撑）

### 1.1 开销过高
- **record**: 108μs/call × 48层/forward × ~300 forward/benchmark = 1.7s, 占4.9%
- **allreduce**: 每sync_window一次, 累计220ms
- **swap P2P**: 每次swap含跨rank权重传输, 占NVLink带宽, 导致dispatch +12.4%
- conc=512满载下净效果 = **-1.75%**（开销 > 收益）

### 1.2 swap策略过于粗暴
- V1: 每sync_window检查一次, 对48层各自独立做max_swaps_per_layer轮swap
- threshold=1.15时, ~30%的(layer,step)会超阈值 → 一次决策可能触发48层×3轮 = 144个swap
- 实测sw128单次决策平均触发~32个swap, 整个benchmark累计180次swap done
- **过度治疗**: 每层mean imbalance只有1.13-1.17, 天花板效应严重; 大量swap只从1.17降到1.15, 收益极小但P2P开销全额发生

### 1.3 MoE在总推理中占64%
- expert占36%, dispatch占41%, combine占23%
- expert的per-step imbalance mean=1.139, 意味着每个forward pass里最慢rank的expert计算比平均慢14%
- 理论最大TPS收益 = 14% × 64% × (expert占比/moe占比) ≈ 14% × 36% ≈ 5%
- 但这5%是理论上限, 实际要扣除开销

## 二、V2核心设计（你的思路 + 数据驱动的参数选择）

### 2.1 弹性阈值 + 全局预算
**不再按"每层最多N轮swap"限制, 改为:**
- 每个sync_window, 计算所有48层的imbalance ratio
- 只对ratio > `high_threshold`(如1.30)的层做swap（真正严重不均衡才动）
- 每层降到`target_ratio`(如1.15)就停, 不必降到1.0
- **全局预算**: 每次决策所有层加起来最多做`max_total_swaps`次swap（如8-12次）
- 按ratio从高到低排序, 把预算优先分给最不均衡的层

### 2.2 降低record频率
- V1: 每个eligible prefill batch都记录所有48层 → 108μs × 48 = 5.2ms/batch
- V2: 只记录每N个forward pass的一个batch（`record_interval`参数）
  - 用累计的指数衰减统计代替"每次都记录+定期清零"
  - 目标: record开销降低4-8倍

### 2.3 更高效的allreduce
- V1: all_reduce一个[48, 128]的int64 tensor = 48×128×8 = 49152 bytes
- V2: 只all_reduce有效信息——可以先本地筛选出ratio > low_threshold(如1.10)的层, 只allreduce这些层的load
  - 或者更简单: 减小allreduce频率(已经通过sync_window控制), 保持tensor不变

### 2.4 同一层多window连续调整
- 如果某层在window#N被swap了但下一个window#N+1仍然超阈值, 说明一次swap不够
- V2: 保留该层的历史ratio, 如果连续K个window都超阈值, 自动加大该层的swap力度(在全局预算内)
- 但如果该层的ratio在改善(比如从1.5降到1.3), 就维持当前力度不加码

## 三、V2参数初始值（基于prefill_heavy数据）

| 参数 | V1值 | V2初始值 | 理由 |
|---|---|---|---|
| sync_window | 128 | 128 | V1实测最优, 不改 |
| threshold_ratio | 1.15 | **废弃** | 用high/target两级替代 |
| high_threshold | — | **1.30** | 只在>30%不均衡时才swap(数据显示pct>1.3只有3-13%) |
| target_ratio | — | **1.15** | swap到1.15就停 |
| max_total_swaps | 48层×3=144(V1) | **8** | 全局预算, 大幅减少P2P次数 |
| record_interval | 每batch | **4** | 每4个forward pass记录一次 |
| cooldown_steps | 5 | 5 | 不改 |

## 四、实施步骤

### Step 1: 创建OEPLB_V2目录结构
```
/workspace/EPLB/OEPLB_V2/
├── src/
│   ├── __init__.py
│   ├── config.py          # V2参数
│   ├── controller.py      # V2控制器(精简版)
│   ├── rebalancer.py      # V2决策算法(弹性阈值+全局预算)
│   ├── async_swapper.py   # 复用V1修复后的版本
│   └── fast_metadata.py   # 复用V1
├── scripts/
│   ├── run_baseline.sh
│   ├── run_v2.sh
│   └── long_bench.py      # 复用
└── README.md
```

### Step 2: 实现V2 rebalancer
- `try_build_swap_plan_v2()`:
  1. 计算所有48层的ratio
  2. 筛选ratio > high_threshold的层, 按ratio降序排列
  3. 对每个选中层, 贪心做1-2轮swap(每轮交换最热rank的最热expert和最冷rank的最冷expert), 直到ratio降到target_ratio或用完全局预算
  4. 返回plan(总swap数 <= max_total_swaps)

### Step 3: 实现V2 controller
- 降频record: `_forward_id % record_interval != 0` 时skip record_next_layer
- 其余控制逻辑跟V1一致(sync_window触发, async swap执行)

### Step 4: 在prefill_heavy×15 conc=512上验证
- 先测baseline(已有: 604.3 TPS)
- 跑V2(high_threshold=1.30, max_total_swaps=8, record_interval=4)
- 如果正收益, 微调high_threshold(1.25? 1.35?)和max_total_swaps(6? 12?)
- 目标: TPS > 604.3 × 1.10 = 664.7

### Step 5: 如果Step 4收益不够, 进一步优化
- 分析V2的开销breakdown(record/allreduce/swap各占多少)
- 可能需要更激进地降record频率(record_interval=8或16)
- 或者只记录top-K最不均衡的层(不是全部48层)

## 五、10%收益目标的可行性分析

理论上限:
- expert imbalance mean=1.139 → 最慢rank比平均慢14%
- 完全消除不均衡 → expert时间减少14% × (瓶颈rank权重)
- expert占MoE的36%, MoE占总推理的64% → expert占总推理23%
- expert完全均衡 → 总推理提速 14% × 23% / (1 - 14%×23%) ≈ 3.3%
- 加上combine的间接改善(~15%) → 额外 15% × 16% ≈ 2.4%
- **理论天花板 ≈ 5.7%**

要达到10%, 仅靠消除expert不均衡理论上不够——除非:
1. 在更高并发(conc=512+)下MoE占比更高(排队更长→MoE瓶颈效应放大)
2. 或者找到除了expert之外的其他优化点(比如dispatch的不均衡1.43→也可以通过更好的token路由来优化, 但这不在当前swap框架的能力范围内)

诚实预期: V2目标设为 **+3~6%** 比较现实, +10%需要突破当前框架(比如加expert replication)。但先实现V2, 看数据再说。
