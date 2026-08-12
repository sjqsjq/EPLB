# 复现脚本 / Reproduction scripts

本目录是论文中每一条实测断言的脚本与原始记数。每个 `driverNN.sh` 都是一次独立的实验，
日志与结果分别落在 `/workspace/logs/driverNN.log` 与 `OEPLB/benchmarks/results/_dNN_*.json`。

This directory holds the driver scripts and raw routing counts behind every measured
claim in the paper. Each `driverNN.sh` is one self-contained experiment; results land in
`OEPLB/benchmarks/results/_dNN_*.json`.

## 实验索引 / Experiment index

| 脚本 | 论文位置 | 测什么 | 结论 |
|---|---|---|---|
| `driver12.sh` | 附录G.2 | 57B/EP8 的 T(r) 扫描，7 布局 ×2 轮 | 铰链 r_k=1.099, β=0.285, RSS 优 12.1× |
| `driver13.sh` | 附录G.2(f) | 57B/EP4 的 T(r) 扫描 | r_k=1.032, β=0.342 → r_k 随 EP 移动 |
| `driver14.sh` | 附录G.2(h) | 235B/EP8 的 T(r) 扫描 | r_k=1.093, β=0.352, f_sens=0.496 → 关闭 109% 矛盾 |
| `driver15.sh` | 附录E.2 | 等价性：GSM8K + 语料 logprob，含 baseline 对照臂 | baseline 逐 bit 确定；精度不变 |
| `driver16.sh` | 附录G.2(i) | 负载依赖：conc=64/512 | B 几乎不变，f_sens 随并发升，r_k 基本不动 |
| `driver17.sh` | 附录D.2 | 逐数据集录制路由计数 | 四数据集 r_before 差 <1.5% |
| `driver18.sh` | §5.4 | 4卡 L256 同源 OEPLB 臂 | +2.70%，η=105% |
| `driver19.sh` | 附录E.2(d) | 静态布局扰动（零 swap） | 复现 256-swap 的扰动 → swap 机制清白 |
| `driver20.sh` | §5.4 | 4卡 L512 重测 | +2.39%（原 +4.7% 的数据集已不存在） |
| `driver21.sh` | 附录E.2(d) | `--init-expert-location` 惰性对照 | 传恒等映射 = 不传，逐 bit 相同 |
| `driver22.sh` | §5.4 | 4卡 多域/ShareGPT 重测 | −0.24% / −0.15%（不同数据集，不可与原表比） |
| `driver23.sh` | 附录G.3 | profiling 验 β = routed-GEMM 墙钟占比 | 进行中 |
| `driver24.sh` | 附录G.3 | 57B/EP2 扫描（含预注册预测） | 进行中 |
| `driver26.sh` | 附录D.2 | 提高 min-prefill-tokens / 阈值取 r_k | 进行中 |

## 分析脚本 / Analysis

| 脚本 | 作用 |
|---|---|
| `gen_placement.py` | 按目标 r 构造 `--init-expert-location` 布局（**逐层**置换 + deficit-greedy；单一共享置换无法制造不均衡） |
| `fit_f3.py` | 同时拟合仿射与铰链（对 r_k 网格搜索 + 闭式 LSQ），报 R²/RSS/held-out 误差/上界 |
| `r_avg.py` | 从录制计数离线算 identity r_avg 与 LPT 下界（已四次与运行时 DIAG 交叉校验） |
| `r_window.py` | 三种 r 口径（聚合 / 逐窗口 token 加权 / 逐 forward），用于定位 DIAG 的小样本偏差 |
| `two_ceilings.py` | 摆放最优 vs 完全均衡路由两个天花板，判据 r_LPT ≤ r_k（§2.5） |
| `bound_curve.py` | 画上界折线图 Δ=β·max(0, r−r_k)，输出 `../fig_bound.png` |
| `parse_trace.py` | 把 torch profiler trace 按 GEMM/dispatch/combine/attn/nccl 分桶 |

## 两个踩过的坑 / Two pitfalls

1. **`gen_placement.py` 必须逐层生成置换。** 所有层共用一个置换时，任何目标 r 都会塌缩到
   r_agg≈1.05——因为每层路由到不同的逻辑专家，单一置换既不能制造也不能消除逐层不均衡。
   若未察觉，整条 T(r) 曲线会埋在噪声里并给出假的 "f_sens≈0"。
2. **`--expert-distribution-recorder-mode stat` 与 `--deepep-mode auto` 不兼容**
   （`expert_distribution.py:314` 抛 `NotImplementedError`）。录制服务器须用 `--deepep-mode normal
   --disable-cuda-graph`；这不影响结论，因为路由计数由 router argmax 决定、与 dispatch kernel 无关，
   而所有**计时**运行仍用 `auto`，与 baseline 逐字相同。
