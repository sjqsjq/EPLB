# §5.3.1 Baseline 复现 · A100 跨硬件移植(2026-09-26)

> 依照 `OEPLB/baselines/REPRODUCE_BASELINES.md`(附录"最短复现路径")在
> 8×A100-SXM4-80GB 上复现两张表的 5 个 baseline:identity / EPLB静态 / EPLB动态 /
> DataForest-Remap / MoETuner。

## 1. 环境与口径

- 硬件: 8×A100-SXM4-80GB NVLink(无 IB) · 模型: Qwen3-235B-A22B **BF16**(438GB, 94 层, 128 专家, top-8)
- 软件: SGLang 0.5.6.post2 · torch 2.9.1+cu128 · Triton MoE runner · NCCL a2a(无 DeepEP)
- 并行: TP8 + EP8, dp=1(**不开 dp-attention**)
- bench: `bench_0914.py`(同域 N=256 一次并发) / `freq6_bench_json.py`(跨域 conc=32, N=1800)
- 聚合(§7.3): 同域丢 r1 取 r2/r3 中位;跨域 3-run 中位;**同 session identity 归一**

## 2. 相对文档的移植偏差(逐条记录)

| # | 偏差 | 原因 |
|---|---|---|
| 1 | 跳过 Phase P/S(profile+Gurobi ILP),复用 H20 归档 placement | 本机无 gurobipy/license;§8.1 明确 placement 为 94×128 纯整数、硬件无关;附录最短路径认可 |
| 2 | 去 `--enable-dp-attention --dp 8`(文档 §3.2 示例仍保留) | A100 无 DeepEP,dp-attention 桥接无高效 kernel、与 `--disable-overlap-schedule` 叠加不稳定(详见 quickstart_a100/REPRODUCE.md §2) |
| 3 | 加 `--moe-runner-backend triton --disable-overlap-schedule`(§3.2 示例漏写) | A100 无 DeepGEMM;不禁 overlap 会 NCCL 死锁(quickstart_a100 已验证) |
| 4 | `--mem-fraction-static 0.88`(文档 0.78) | §8.2#3 允许按显存调整:BF16 权重 55GB/卡(FP8 仅 27.6GB),0.78 下 KV 池不足以容纳 freq6(book 实测最长 ~5668 tok) |
| 5 | `--context-length 8192`(文档未设,用默认 40960) | 覆盖 book_4438tok 实际最长 ~5668 tok(+10 输出),同时约束 KV 预留 |
| 6 | EPLB静态/动态 `--ep-num-redundant-experts 16` 维持文档值 | 冗余仅 +~6.4GB/卡,0.88 下可容纳(实测无 OOM 则维持;若 OOM 降为 0 并记录) |
| 7 | 不加 `--skip-server-warmup`(文档 identity/moetuner 有加、其余没有) | 统一 5 方法口径;warmup 把 Triton JIT 挪到 boot 阶段,r1 仍按 §7.3 丢弃 |
| 8 | 模型 checkpoint: BF16 原始权重(非 FP8) | A100 sm80 无 FP8 张量核(§8.2#2) |
| 9 | forward_normal dispatch-info 补丁必打(已打) | §1.3 警告:不打则所有放置全部退化为 identity |

## 3. 使用的 placement(H20 归档,直接复用)

| 方法 | 文件 | 键 | 形状 |
|---|---|---|---|
| MoETuner | `/data/minghua/sjq/OEPLBdata/experiment_logs/moetuner_baseline_20260916/artifacts/placement_pinned_fair.json` | `physical_to_logical_map` | 94×128 |
| DataForest-Remap | `/data/minghua/sjq/OEPLBdata/experiment_logs/baseline_comparison_20260914/datafore_remap_placement.json` | `logical_count` | 94×128 |
| EPLB静态 | `.../baseline_comparison_20260914/datafore_prover_placement.json` | `logical_count` | 94×128 |
| EPLB动态 | (无,在线录制+周期重排, iter=100, buffer=32, redundant=16) | — | — |

两份 datafore 文件确认不同(12032 元素中仅 298 相同)→ EPLB静态(prover 全量录制)与 DataForest(pinned fair-split 录制)是不同放置,臂间可区分。

## 4. 结果

**已全部完成(含 OEPLB 三配置补测)。完整记录与全部原始 run 见:
`NEW_PAPER/experiments/a100_baselines_repro_20260926/EXPERIMENT_LOG.md`**

摘要(同域中位/跨域中位, vs 同session identity):
DataForest +21.8%/+3.8% · MoETuner +18.9%/+3.1% · EPLB静态 −10.2%/−4.7% · EPLB动态 −3.2%/−13.8% · **PB-OEPLB 稳态最高 +20.2%/+13.6%**

### 表 1 · 同域(prover_256tok_out1, N=256)

| 方法 | r1 | r2 | r3 | 中位(r2/r3) | vs identity |
|---|---|---|---|---|---|
| identity | | | | | — |
| DataForest-Remap | | | | | |
| MoETuner | | | | | |
| EPLB静态 | | | | | |
| EPLB动态 | | | | | |

### 表 2 · 跨域(freq6, N=1800, conc=32)

| 方法 | r1 | r2 | r3 | 中位 | vs identity |
|---|---|---|---|---|---|
| identity | | | | | — |
| DataForest-Remap | | | | | |
| MoETuner | | | | | |
| EPLB静态 | | | | | |
| EPLB动态 | | | | | |

### H20 参考(同表口径,来自 h20_archive/)

identity 同域中位 55.65 req/s(r2=55.2, r3=56.1);跨域中位 4.50 tps。
A100 绝对值低于 H20 属预期(BF16/Triton/无 DeepEP vs FP8/DeepGEMM/DeepEP),核对点是**相对排序**(§8.3)。

## 5. §10 验证清单核对

(跑完后逐项填)

- [ ] 补丁生效: identity 与 DataForest tps 显著不同
- [ ] placement 格式: MoETuner=physical_to_logical_map / DataForest=logical_count(加载日志确认分支)
- [ ] 同域丢 r1
- [ ] 同 session identity 归一
- [ ] 相对排序: DataForest ≳ MoETuner > EPLB静态 > EPLB动态(同域);跨域离线放置 ≤ identity
- [ ] 绝对 gain 方向与 §8.3 A100 预测一致(Δ_max≈18.3%)

## 6. 文件产物

- 启动/driver: `launch_a100.sh`(参数化 5 方法)、`driver_baselines_a100.sh`、`run_all_a100.sh`、`env_a100.sh`
- 结果: `benchmarks/results/_0914_a100_<m>_r<N>.json`、`_freq6_a100_<m>_r<N>.json`
- H20 原始结果备份: `benchmarks/results/h20_archive/`(12 个文件,git HEAD 同源)
- server 日志: `/workspace/logs/server_a100_<table>_<method>.log`
- bench 日志: `baselines/a100/logs/`
