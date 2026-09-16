# 实验: L512_O1_realprover 复现 (post-bugfix, 2026-09-15)

**日期**: 2026-09-15
**目的**: 修复 `server_args.py` `ep_dispatch_algorithm` bug 后，重跑论文 §1 头条数据
**模型**: Qwen3-235B-A22B-FP8, TP=DP=EP=8, 8×H20

## 背景：修复的 bug

`/opt/conda/lib/python3.11/site-packages/sglang/srt/server_args.py` L1673 原条件缺少 `enable_pb_oeplb` 分支，导致 `--enable-pb-oeplb` 单独启用时 `ep_dispatch_algorithm` 保持 `None`：

- `ExpertLocationDispatchInfo.init_new()` 返回 None
- `topk_ids_logical_to_physical()` 原样返回 topk_ids
- Rebalancer/AsyncSwapper 真的做了物理权重 P2P，但 token 路由继续按逻辑 ID 送到原 GPU
- 每次 window 前 imbalance 立刻回弹（1.55+ 反复出现）

修复：条件加 `or self.enable_pb_oeplb`。修复后 window N+1 的 `avg_ratio_before` ≈ window N 的 `avg_ratio_after`，回弹消失。

## 数据集
`/data/minghua/sjq/OEPLBdata/datasets/grid_benchmarks/comprehensive_grid/L512_O1_realprover_n8192.jsonl`
512tok prompt (realprover 数学证明), O=1 (prefill-dense), 8192 req, conc=256

## 配置

| arm | launch 脚本 |
|---|---|
| identity | launch_identity.sh (与论文一致 + `--dtype bfloat16`) |
| OEPLB-adaptive | launch_oeplb_adaptive.sh (论文全部参数 + **对齐后新增** `--pb-oeplb-window-floor 8 --pb-oeplb-window-shift-confirm 2`) |

## 结果

### Round 1: 修复 bug 但未加新对齐参数

| arm | tps | time(s) | ok |
|---|---|---|---|
| identity r1 | 39.80 | 205.71 | 8192 |
| identity r2 | 39.90 | 205.42 | 8192 |
| identity 平均 | **39.85** | 205.57 | — |
| OEPLB r1 | 46.90 | 174.51 | 8192 |
| OEPLB r2 | 47.60 | 171.97 | 8192 |
| OEPLB 平均 | **47.25** | 173.24 | — |
| **gain** | **+18.6%** | — | — |

### Round 2: 补齐 `--pb-oeplb-window-floor 8` + `--pb-oeplb-window-shift-confirm 2`

| arm | tps | time(s) | ok |
|---|---|---|---|
| align_oe_r1 | 47.30 | 173.12 | 8192 |
| align_oe_r2 | 47.90 | 171.08 | 8192 |
| OEPLB 平均 | **47.60** | 172.10 | — |
| **gain (对 identity 39.85)** | **+19.4%** | — | — |

## DIAG 输出（无回弹验证）

`oe_r1`:
```
Window 1: layers_touched=93 total_ops=298 avg_before=1.743 avg_after=1.182 max_before=2.606 max_after=1.425
Window 2: layers_touched=90 total_ops=225 avg_before=1.186 avg_after=1.073 max_before=1.418 max_after=1.150
```
Window 2 `avg_before` = 1.186 ≈ Window 1 `avg_after` = 1.182 → **物理/逻辑映射同步生效，无回弹**（修复前 `avg_before` 会回弹到 1.55+）。

## 对比论文 EXPERIMENT_LOG.md 原记录

| Arm | 论文原记录 (2026-09-10) | 本次复现 (2026-09-15) |
|---|---|---|
| identity | 40.1 | 39.85 |
| OEPLB-adaptive | 46.3 | 47.60 |
| gain | +15.5% | **+19.4%** |

本次复现结果**超过**论文原记录，原因是原记录时的 `ep_dispatch_algorithm` bug 让 OEPLB 只发挥了部分能力。

## 文件清单

- `_d38repro_{bl,oe}_r{1,2}.json` — 原 4 轮结果
- `_align_oe_r{1,2}.json` — 补齐 floor/confirm 后的 2 轮
- `launch_identity.sh` / `launch_oeplb_adaptive.sh` — 复用启动脚本
- `driver_alt_repro.sh` — 交替 BL/OE 4 轮 driver
- `driver_align.sh` — 追加 2 轮 align driver
- `keylines_*.txt` — 每轮 server log 的 DIAG/ready 行
