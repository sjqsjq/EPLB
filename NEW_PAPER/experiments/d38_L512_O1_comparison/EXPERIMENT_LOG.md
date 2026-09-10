# 实验: L512_O1_realprover 三方对比 (identity / OEPLB-adaptive / EPLB)

**日期**: 2026-09-10
**目的**: 在 prefill-dense 负载上对比 identity baseline / OEPLB-adaptive / SGLang EPLB
**模型**: Qwen3-235B-A22B-FP8, TP=DP=EP=8, 8×H20

## 数据集
- `/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl`
- 512tok prompt (realprover 数学证明), O=1 (prefill-dense), 8192 req, conc=256

## 配置

| arm | launch 脚本 | 关键参数 |
|---|---|---|
| identity | launch_identity.sh | cuda-graph-max-bs 128, deepep-mode auto, mem 0.8, 无 OEPLB/EPLB |
| OEPLB-adaptive | launch_oeplb_adaptive.sh | + --enable-pb-oeplb + adaptive-window + adaptive-decay, 其余同 identity |
| EPLB | launch_eplb_warmup.sh | deepep-mode normal, --enable-eplb, 16 redundant, mem 0.8, cuda-graph-max-bs 128, **无 --skip-server-warmup** |

## 结果

| arm | tps | time(s) | ok | gain vs identity | crash |
|---|---|---|---|---|---|
| identity | 40.1 | 204.35 | 8192 | — | ✅ |
| OEPLB-adaptive | 46.3 | 176.81 | 8192 | **+15.5%** | ✅ |
| EPLB (normal+warmup) | 39.4 | 208.02 | 8192 | **−1.7%** | ✅ |

## 关键发现

1. **OEPLB +15.5%**:增量 swap + cuda-graph,在 prefill-dense 上大正收益。
2. **EPLB −1.7%**:全量重平衡+16冗余,开销 > 收益,略负。
3. **EPLB 崩溃问题**:用 `--skip-server-warmup` 时 EPLB 崩(DeepGEMM JIT 在 client forward 内编译→崩);去掉 `--skip-server-warmup`(warmup 预编译 JIT)→不崩。**崩溃根因是 --skip-server-warmup,不是数据集或 EPLB 本身。**
4. **OEPLB 用 cuda-graph(auto 模式)**;EPLB 必须用 normal 模式(禁 cuda-graph)→ EPLB 的 decode 更慢,这也是 EPLB 负收益的原因之一。

## 文件清单
- `identity_baseline.json` / `oeplb_adaptive.json` / `eplb_normal_warmup.json` — run_grid_bench 结果
- `launch_*.sh` — 启动脚本(可复现)
- `*_keylines.txt` — server log 关键行(DIAG/JIT/crash/rebalance)
- `EXPERIMENT_LOG.md` — 本文件
