# PAPER_zh.md 评审（面向 ASPLOS/EuroSys）— 2026-08-11 实测复核

全部结论均在本机 8×H20 + SGLang 0.5.6.post2 实测验证，数据集
`comprehensive_grid/L256_O1_realprover_n16384.jsonl`（16384 req, conc=256, 独立重启交替）。

## A. 硬伤（不改会被拒）

### A1. "异步非阻塞"描述的是已删除的代码
`src/async_swapper.py` 类 docstring: *"Previous async design (dedicated stream +
separate PG) caused NCCL hangs... New design: swap is performed SYNCHRONOUSLY"*。
`try_finish()` 现在直接返回 pending plan，没有 `event.query()`。
故摘要"不阻塞推理"、§1 贡献"专用低优先级 CUDA stream 异步 P2P"、§3.4 整节均与 artifact 矛盾。

实测（235B, oeplb_r2, 171.25s wall）：7 次决策，926 次 swap，
每 rank 阻塞合计 **4.22s = wall 的 2.47%**，首次决策 298 op 阻塞 1.82s。
§5.6 表7 开销 0.67%（Finalize 仅 62ms）**未计入同步 swap 阻塞**，低估约 4×。

改法：改成"每次决策阻塞 ~170–600ms，vs 官方 EPLB 每次 rebalance 1.95s（实测）"，
把阻塞列进开销表。这仍是强 claim，且诚实。

### A2. 头条数字**可复现**，但 EPLB 那一行不可复现；内部三处互相矛盾（已实测更新）
我在 §5.3 表1 用的同一数据集 `L512_O1_realprover_n8192` 上重跑了三配置 × 2 轮
（8192 req, conc=256, 独立重启, 0 错误）：

| 配置 | req/s (r1, r2) | 均值 | wall | 换算输入 tok/s (×512) | 论文表1 |
|---|---|---|---|---|---|
| baseline | 40.1 / 40.1 | 40.10 | 204.3s | 20531 | 20167.8 |
| **OEPLB** | 46.8 / 47.4 | **47.10 (+17.5%)** | 174.1s | **24115** | 23870.5 (+18.4%) |
| EPLB(16红/周期64) | 40.5 / 41.1 | 40.80 (**+1.75%**) | 200.8s | 20890 | 21992.2 (+9.0%) |

结论修正（比我先前的判断对论文更有利）：
- **baseline 与 OEPLB 复现**：差 1.8% / 1.0%，方差极小（两轮 baseline 完全相同）。
  头条 +18.4% 站得住，我实测 +17.5%。
- **EPLB 那一行复现不了**：论文给 +9.0%，实测只有 **+1.75%**；README 记的 +14.2%
  更远。三者两两矛盾，且都无原始 json（全库搜不到 21992/22908）。
- **§5.8 的 22603.6/22885.2/22850.8 是冷启动 outlier**，不能用来质疑表1
  （我先前据此推出的"重算只有 +12.9%、结论会反转"作废）。
- 仍需修的：(a) 表1/README/§5.8 三处数字必须统一到同一批 run 并给 mean±std；
  (b) EPLB 行必须给出周期 sweep（见 B1）——现在这个值既高于我实测的最优
  也低于 README，无法判断是哪个配置；(c) 附原始 json。

### A3. "理论加速上界"是恒等式，不是上界
β_c 由同一组 trace 反推：β_c = (ΔT_c/T_c)/ratio_imp。代回后
Δ_max = ratio_imp·Σβ_c f_c ≡ Σ ΔT_c / T_total = 实测节省。
数值验证：公式给 0.1588，实测节省/总时间 0.1585（三位小数一致）。
故"系统效率 105%/116%"只是舍入误差，不是发现；"旧公式误差 89%、新公式 5.7%"
是自证。且 β_dispatch=-0.78 把 OEPLB 自身 all_reduce 开销塞进"敏感度"，
与随后再减 c_overhead 有重复计数风险。

改法：β 必须先验推导（如 straggler 同步给出 β_combine=1）或在 held-out
配置（不同 r、不同模型）上验证；术语从"上界"改为"一阶模型"。

### A4. 贡献点5 已在上游实现
`sglang/srt/models/qwen3_moe.py:1131` 已有 `if not hasattr(self,
"routed_experts_weights_of_layer")` + 遍历 `get_moe_weights()`，与论文 §4 自称的
"跨架构 fallback" 同一逻辑。`qwen2_moe.py` 中 0 处 → 限制2 **仅对 Qwen2-MoE 成立**。
故"对 Qwen2-MoE 和 Qwen3-MoE 都会 AttributeError"、"main 分支未修复"为错。
贡献降级为"扩展到 Qwen2-MoE"。（我们今天在 Qwen3-235B 上跑 EPLB 全程无崩溃。）

## B. 方法论（会被要求补实验）

### B1. baseline 未调优 → strawman 风险
EPLB 只测一个 rebalance 周期。实测（16 冗余、周期 64、L256_O1）：
**+1.8%**（83.62 vs 82.12 rps），8 次 rebalance × 1.95s 全卡阻塞吃掉收益，
ttft_p99 3878→6170ms (+59%)。论文给 +9.0%、README 给 +14.2% → EPLB 对周期极敏感，
必须 sweep 周期（64/256/1000）取其最优再比。

### B2. n=1~2 且无显著性检验
今天的反例：57B 8卡 base r1=175.34 vs r2=189.61（差 8.1%，r1 被 DeepGEMM JIT
冷启动污染）。只跑一轮会得出 +10.4% 的假结论，两轮才看出真实 ~+1%。
主结果必须 ≥2–3 轮交替 + mean±std，并显式弃掉冷启动轮。

### B3. 历史数据与本次实验混用、硬件从 8 卡换 4 卡；§5.3 自标"历史数据"。
顶会要求单一受控 campaign + artifact 可复现原始 json。

## C. 内容错误（实测纠正）

### C1. 附录 D 缩放律被实测否证
D.1 称 r 本质由每卡专家数 n 决定（n↓→r↑）。实测 avg_ratio_before：

| 配置 | n | avg_before | max_before | after | 吞吐 |
|---|---|---|---|---|---|
| 235B 8卡 | 16 | **1.721** | 2.486 | 1.05 | +17.9% |
| 57B 8卡 | **8** | **1.216** | 2.760 | 1.015 | ~+1.0% |
| 57B 4卡 | 16 | **1.113** | 1.741 | 1.011 | +1.85% |

n=8 的 avg 反而低于 n=16 的 1.721 → 主导项是 router 偏斜 σ/μ（模型属性），非 n。
且表 D.2 给 235B 与 57B 同填 "1.74"，实为 235B 的 **avg** 与 57B 4卡的 **max**
——同一列混了两种统计量。

### C2. 显存三个互相矛盾的数
§2.2 限制3 "12.5% 额外显存"、§2.5 "δ=8.1%"、§5.7 表9 "KV 48.8→46.3GB=-5.1%"。
实测 KV pool：baseline **227269** tok → EPLB **208750** tok = **-8.15%**，
§2.5 正确，限制3 与表9 需改。表9 还有 EPLB 总量 79.8 < baseline 88.7 的反常。

### C3. §2.5 排队论模型未经推导/验证
ρ'=ρ/(1-δ) 与 M/M/1 的 W_q∝1/(1-ρ) 对 continuous-batching 推理不成立。
建议删模型、保留实测（KV -8.15% 与 L4096_O256 上 EPLB -3.2% vs OEPLB +16.0%）。

### C4. 命题1 的 NP-hard 措辞
G 为常数时问题规模有界；3-PARTITION 归约要求 G 随输入增长。需把 G 写成输入参数。

### C5. §5.2 数据集表与磁盘不符（L256_O1 标 8192，实际用 n16384）。

## D. 正面：实测确认成立的 claim
- OEPLB 235B 单域 prefill **+17.9%**（82.12→96.84 rps，交替 2 轮，0 错误），接近论文 +18.4%。
- EPLB 强制 `deepep-mode=normal`（auto 直接 NotImplementedError）确认。
- EPLB 冗余专家挤压 KV：227269→208750 tok（-8.15%），与 §2.5 完全一致。
- OEPLB 收敛快：926 次 swap 全部在开跑后 35s 内完成，此后 135s 零 swap。
- swap 真实搬权重：`VERIFY-WEIGHT-MOVE` 显示 rank4 换后 checksum
  (-2997826.25) 精确等于 rank7 换前值 → 跨 rank P2P 真实传输（非仅翻路由表）。
  但该检查只覆盖 plan[0] 的第一个张量；建议补功能等价性检验（贪心输出 swap 前后一致）。

## E. 建议的顶会框架
1. 核心 insight 改为"零冗余 + 保留 CUDA graph"，而非"异步非阻塞"。
2. 主结果统一 mean±std over ≥2–3 轮交替独立重启，弃冷启动轮。
3. 补 sensitivity：EPLB rebalance 周期 sweep、sync_window sweep、并发度 sweep。
4. β 模型改为先验推导 + held-out 验证，或降级为经验分解（诚实呈现）。
5. tail latency 单列一节（EPLB p99 +59% vs OEPLB 收敛后无影响）。
6. artifact: 一键复现脚本 + 全部原始 json。
