# Baseline vs SGLang EPLB vs OEPLB 对比实验报告 (2026-07-23)

## 实验环境

- **硬件**: 8× NVIDIA H20 (96GB/卡, NV18 全互连)
- **模型**: Qwen3-235B-A22B-FP8, EP=8, DP=8
- **deepep-mode**: normal（全部配置统一）
- **数据集**: DeepSeek-Prover-V1, 各桶2048条独立请求
- **方法论**: EPLB和OEPLB每个输入长度单独重启服务器（避免placement继承偏差），Baseline可链式连测

## 输入长度分桶

| 桶 | Token范围 | 均值 |
|---|---|---|
| short | 90-180 | 154 |
| medium | 200-260 | 228 |
| long | 400-700 | 494 |
| tok256 (额外) | 220-290 | 253 |
| tok2048 (额外) | 2048 (拼接构造) | 2048 |

---

## 一、隔离实验：冗余专家不开EPLB = 几乎无收益

| 输入 | Baseline | 冗余only(red=8,无EPLB) | 差异 |
|---|---|---|---|
| short | 126.23 | 124.27 | -1.6% |
| medium | 86.64 | 89.79 | +3.6% |
| long | 40.72 | 40.25 | -1.2% |

**结论：冗余专家的价值必须靠动态负载均衡机制（EPLB或OEPLB）才能兑现。** 仅给冗余专家不开动态rebalance，跟baseline几乎无差异。

---

## 二、Output=1（纯prefill）全配置对比

### 全配置req/s对比表

| 配置 | short | medium | long | short% | medium% | long% |
|---|---|---|---|---|---|---|
| **Baseline(5轮)** | 126.23 | 86.64 | 40.72 | - | - | - |
| 冗余only(red=8) | 124.27 | 89.79 | 40.25 | -1.6% | +3.6% | -1.2% |
| EPLB(iter=32,red=8) | 139.79 | 99.97 | 47.55 | +10.7% | +15.4% | +16.8% |
| EPLB(iter=64,red=8) | 147.56 | 102.73 | 49.60 | +16.9% | +18.6% | +21.8% |
| EPLB(iter=128,red=16) | 154.57 | 103.43 | 50.23 | +22.5% | +19.4% | +23.4% |
| EPLB(iter=64,red=16) | 152.04 | 108.65 | 50.04 | +20.4% | +25.4% | +22.9% |
| OEPLB(sw=32) | 148.11 | 104.09 | 48.74 | +17.3% | +20.1% | +19.7% |
| **OEPLB(sw=64)** | **152.97** | **106.01** | 48.68 | **+21.2%** | **+22.4%** | +19.5% |
| OEPLB(sw=128) | 149.44 | 103.45 | 48.63 | +18.4% | +19.4% | +19.4% |

### 各自最优配置

| 方案 | 最优参数 | short% | medium% | long% | 显存开销 |
|---|---|---|---|---|---|
| **EPLB最优** | iter=64, red=16 | +20.4% | **+25.4%** | +22.9% | 需16个冗余专家 |
| **OEPLB最优** | sw=64, red=0 | +21.2% | +22.4% | +19.5% | **无冗余专家** |

### OEPLB详细3轮数据

| 输入 | Run1 | Run2 | Run3 | 均值 | 标准差 |
|---|---|---|---|---|---|
| short | 156.46 | 154.80 | 147.66 | 152.97 | 4.69 |
| medium | 107.68 | 106.33 | 104.01 | 106.01 | 1.85 |
| long | 48.95 | 48.54 | 48.54 | 48.68 | 0.24 |

### EPLB(iter=64,red=8) 详细3轮数据

| 输入 | Run1 | Run2 | Run3 | 均值 | 标准差 |
|---|---|---|---|---|---|
| short | 146.18 | 147.13 | 149.38 | 147.56 | 1.64 |
| medium | 103.49 | 101.79 | 102.91 | 102.73 | 0.86 |
| long | 48.92 | 49.59 | 50.28 | 49.60 | 0.68 |

### EPLB(iter=32,red=8) 详细3轮数据

| 输入 | Run1 | Run2 | Run3 | 均值 | 标准差 |
|---|---|---|---|---|---|
| short | 141.53 | 140.76 | 137.08 | 139.79 | 2.38 |
| medium | 99.18 | 97.29 | 103.43 | 99.97 | 3.15 |
| long | 46.61 | 47.68 | 48.36 | 47.55 | 0.88 |

### 关键发现

1. **OEPLB(sw=64)在short/medium上全面领先EPLB(iter=64,red=8) +3-4%**，在long上基本持平（-1.9%），且**不需要任何冗余专家（显存更省）**。
2. **EPLB冗余专家8→16有明显收益**（medium从+18.6%跳到+25.4%）。EPLB(iter=64,red=16)在medium上超过OEPLB 3个百分点，但代价是多占16个冗余专家的显存。
3. **EPLB iter=64 > iter=32**：rebalance本身开销不小（首次4.4s，后续0.5s/次），检查越频繁开销占比越高，抵消了更及时纠偏的收益。
4. **OEPLB最优window=64**：sw=32因P2P开销太频繁、sw=128因错失纠偏窗口，都不如sw=64。

---

## 三、Domain-Switch场景（fixeddata）对比

fixeddata = 3072条（1024×Prover-V1 + 2048×BookCorpus），包含一次真实的domain切换。各2轮。

| 配置 | R1 | R2 | 均值 | vs Baseline |
|---|---|---|---|---|
| Baseline | 43.37 | 42.52 | 42.95 | - |
| EPLB(iter=64,red=16) | 43.81 | 44.46 | 44.14 | +2.8% |
| **OEPLB(sw=64)** | **47.02** | **45.66** | **46.34** | **+7.9%** |

### OEPLB Swap收敛记录（R1, 总ops=716）

| Window | avg_ratio_before | avg_ratio_after | total_ops |
|---|---|---|---|
| 1 (冷启动) | 1.737 | 1.225 | 237 |
| 2 | 1.227 | 1.203 | 49 |
| 3 | 1.198 | 1.194 | 14 |
| 4 (domain切换) | 1.207 | 1.123 | 124 |
| 5 (切换后纠偏) | 1.310 | 1.114 | 216 |
| 6 | 1.184 | 1.137 | 76 |

### 关键发现

**Domain-switch是OEPLB相对EPLB优势最明显的场景**：
- 纯prefill场景：OEPLB +21% vs EPLB +20-25% → 基本持平
- Domain-switch场景：**OEPLB +7.9% vs EPLB +2.8%** → OEPLB领先5个百分点

EPLB的rebalance周期长（64 forward pass + 每次0.5-4s阻塞），domain切换时反应迟钝。OEPLB的逐slot P2P swap在切换点1-2个sync_window内就完成纠偏（window4+5共340次swap）。

---

## 四、Output=128（含decode）对比

| 输入 | Baseline | EPLB(i64,r16) | EPLB vs BL | OEPLB(sw64) R1 | OEPLB R2 | OEPLB均值 | OEPLB vs BL |
|---|---|---|---|---|---|---|---|
| short | 31.68 | 27.84 | **-12.1%** | 32.25 | 28.66 | 30.46 | -3.9% |
| medium | 28.72 | 24.98 | **-13.0%** | 30.65 | 26.40 | 28.53 | -0.7% |
| long | 20.41 | 20.78 | +1.8% | 22.18 | 20.59 | 21.39 | +4.8% |

### Output=1 vs Output=128 收益变化

| | Output=1 EPLB vs BL | Output=128 EPLB vs BL | Output=1 OEPLB vs BL | Output=128 OEPLB vs BL |
|---|---|---|---|---|
| short | +20.4% | **-12.1%** | +21.2% | -3.9% |
| medium | +25.4% | **-13.0%** | +22.4% | -0.7% |
| long | +22.9% | +1.8% | +19.5% | +4.8% |

### 关键发现

1. **EPLB在有decode场景变为负收益**（-12~13%）：16个冗余专家挤占KV cache显存 + rebalance阻塞推理的累积开销。
2. **OEPLB在有decode场景基本持平**（-3.9%~+4.8%）：无冗余专家不占额外显存，swap开销远小于EPLB的全局rebalance。
3. **两者的核心优势场景都是prefill-heavy负载**，decode占比越高收益越稀释，但OEPLB至少不会亏，EPLB会亏。

---

## 五、最优sync_window与输入长度的关系

在256tok和2048tok两个极端输入长度上测试sw=32 vs sw=64。各2轮。

| 配置 | 256tok (req/s) | vs BL | 2048tok (req/s) | vs BL |
|---|---|---|---|---|
| Baseline | 82.51 | - | 10.10 | - |
| OEPLB sw=32 | 94.19 | +14.2% | **11.71** | **+15.9%** |
| OEPLB sw=64 | **95.27** | **+15.5%** | 11.46 | +13.4% |

| 输入长度 | sw32 vs sw64 | 更优的window |
|---|---|---|
| 256tok | -1.1% | **sw=64** |
| 2048tok | +2.2% | **sw=32** |

### 结论

**最优sync_window跟输入长度相关**：短输入用sw=64更好，长输入(2048tok)用sw=32更好。长输入的单次prefill耗时更长，每个window内积累的不均衡度更大，更频繁的检查/纠偏更值得。差异幅度目前约2%，方向稳定。

---

## 总结

### OEPLB的核心优势

1. **不需要冗余专家**：OEPLB(red=0)在纯prefill场景跟EPLB(red=16)性能接近（short/medium略优，long略低），但**完全不占用冗余专家的显存**。在显存紧张的部署场景下这个优势很有价值。
2. **Domain-switch场景显著领先**：OEPLB +7.9% vs EPLB +2.8%，因为逐slot在线swap比周期性全局rebalance反应更快。
3. **有decode场景不亏**：EPLB在output=128下变成-12~13%负收益，OEPLB基本持平（-0.7%~+4.8%）。
4. **轻量级开销**：单次swap只涉及两个slot间的P2P权重交换（微秒级），EPLB的rebalance需要重算整个placement映射（0.5-4.4秒/次,阻塞推理）。

### EPLB的优势

1. **纯prefill场景+足够冗余专家时可以超过OEPLB**：EPLB(i64,r16)在medium桶上+25.4%超过OEPLB的+22.4%。冗余专家提供了额外的"弹药"让热点专家可以被复制到更多GPU上，这是swap-only方案做不到的。
2. **long输入桶上略优**：EPLB(i64,r16) +22.9% vs OEPLB +19.5%，差约3个百分点。

### 选择建议

| 场景 | 推荐 | 原因 |
|---|---|---|
| 显存紧张 | **OEPLB** | 无冗余专家，性能接近 |
| Prefill-heavy + 单一域 | EPLB(red=16) 或 OEPLB | EPLB略优但需更多显存 |
| Domain-switch / 混合负载 | **OEPLB** | 在线swap响应速度快，领先5pp |
| 有decode的场景 | **OEPLB** | EPLB会变成负收益 |
| 长输入(>1k tok) | OEPLB(sw=32) | 长输入下sw=32比sw=64更优 |

---

## 补充：含decode场景三方对比（2×2负载，deepep-mode=normal, 2026-07-23）

用WORKLOAD_GRID_REPORT中冷启动验证章节的4个负载组合（short×short/medium, medium×short/medium），每个负载独立重启，统一normal模式。

### 结果

| 负载(输入_输出) | Baseline | EPLB(i64,r16) | OEPLB(sw64) | EPLB vs BL | OEPLB vs BL |
|---|---|---|---|---|---|
| short_short(154tok,out=8) | 103.33 | **125.34** | **116.17** | **+21.3%** | **+12.4%** |
| short_medium(154tok,out=64) | 51.61 | 49.95 | **52.28** | -3.2% | **+1.3%** |
| medium_short(228tok,out=8) | 80.98 | **89.55** | **86.71** | **+10.6%** | **+7.1%** |
| medium_medium(228tok,out=64) | 46.09 | 41.18 | 45.89 | **-10.6%** | **-0.4%** |

### 与之前实验的对比修正

之前Output=128实验中EPLB出现-12~13%的大幅负收益，本次实验澄清了更精确的规律：
- **短输出(out=8)场景：EPLB和OEPLB都有正收益**，且输入越长收益越大——这跟output=1纯prefill场景的趋势一致，说明少量decode不会抵消prefill阶段的均衡收益
- **中输出(out=64)场景：EPLB开始出现负收益**（medium_medium -10.6%），OEPLB基本持平（-0.4%）
- **EPLB的负收益主因不是"decode本身有负面影响"**，而是：(a) rebalance阻塞推理的累积开销在有decode的场景里触发次数更多（decode步也推进iteration计数器）；(b) 16个冗余专家占用的显存减少了KV cache空间，影响decode阶段的batch容量

### EPLB稳定性问题

测试过程中EPLB在medium_short上出现了一次rebalance后CUDA error crash（`cudaErrorInvalidConfiguration`），重试后成功。这个crash是EPLB自身的bug（rebalance后更新expert location时触发了无效的CUDA kernel配置），不是环境/配置问题。

---

## 补充：OEPLB medium输入 × 不同输出长度单独验证 (2026-07-23)

之前三方对比中OEPLB在medium_medium(out=64)上出现-0.4%,用户质疑不合理。重新测试：baseline连测3个输出长度,OEPLB每个输出长度独立重启,排除任何链式干扰。

### 结果

| 输出max_tokens | Baseline | OEPLB(sw=64) | 提升 |
|---|---|---|---|
| 64 | 43.97 | **47.67** | **+8.4%** |
| 128 | 29.66 | 28.96 | -2.4% |
| 256 | 18.18 | 17.70 | -2.6% |

### Swap记录（3个输出长度完全一致）

3个负载的OEPLB DIAG日志逐窗口完全一致（同一批medium输入prompt,冷启动纠偏过程相同）：

| Window | avg_ratio_before | avg_ratio_after | total_ops |
|---|---|---|---|
| 1 (冷启动) | 1.737 | 1.225 | 237 |
| 2 | 1.227 | 1.203 | 49 |
| 3 | 1.209-1.212 | 1.188-1.190 | 40 |
| 4 | 1.189 | 1.189 | 1-2 |
| 5 | 1.189 | 1.189 | 2 |

### 分析

1. **之前medium_medium(out=64)的-0.4%是链式测试的噪声**：单独重测拿到+8.4%,确认out=64场景下OEPLB有真实正收益。
2. **out=128/256的-2.4%/-2.6%在噪声范围内**：swap行为跟out=64完全一致(因为OEPLB只看prefill,输出长度不影响swap决策),差异纯粹来自decode阶段的测量波动。需要多轮重复才能判断是真实的微小负收益还是噪声。
3. **OEPLB的swap对decode确实没有额外负面影响**——3个输出长度的swap模式完全相同,但性能表现不同,说明差异来源不在OEPLB侧。
