# A100 快速复现（8×A100-80GB / BF16 / 非-DeepEP 路径）

在 A100 上复现 OEPLB 的跨硬件三方对比（Baseline / 官方 EPLB / OEPLB）。**A100（sm80）无 FP8 张量核**，
走 `forward_normal` 路径：**BF16 + Triton MoE runner + NCCL**，**不用编译 DeepEP/DeepGEMM**——环境比 H20 简单。

> 论文主线（FP8/DeepEP）的复现在 `../quickstart_h20/`。

## 本文件夹内容

| 文件 | 作用 |
|---|---|
| `ENVIRONMENT.md` | **第一步**：A100 环境搭建（装栈、下 BF16 模型、`deploy_oeplb.sh` + `patch_sglang.py` 两个补丁、启动自检） |
| `REPRODUCE.md` | **第二步**：跑 Baseline/EPLB/OEPLB 三方对比，含 A100↔H20 差异、为什么不开 dp-attention、根因 bug、结果表 |
| `patch_sglang.py` | 幂等补丁器：补丁 A（server_args static dispatch）+ 补丁 B（qwen3_moe `forward_normal` 补传 dispatch info，A100 关键 bug 修复） |
| `deploy_oeplb.sh` | 把 `OEPLB/src/*.py` 复制进 SGLang 并 diff 三个 patch 文件 |
| `env_a100.sh` | A100 环境变量（NCCL over NVLink、offline、模型路径） |
| `launch_baseline_a100.sh` | identity 基线服务器 |
| `launch_eplb_a100.sh` | 官方在线 EPLB 服务器（0 冗余专家，每 50 步全模型重排） |
| `launch_oeplb_a100.sh` | OEPLB 服务器（自适应窗口 + 变点重置） |
| `launch_oeplb_lowovh_a100.sh` | OEPLB 低开销变体（可选） |

正确顺序：**先 `ENVIRONMENT.md` 把服务跑通并自检 → 再照 `REPRODUCE.md` 复现结论。**

## 最快复现路径

数据集在仓库内、无需下载（`max_tokens=1`，纯 prefill）：

```
单域: OEPLB/benchmarks/final_grid/L512_O1.jsonl        # Lean-4 数学域, 8192 条
多域: OEPLB/benchmarks/multidomain/MD_L512_O1.jsonl    # 数学↔英文小说 M/E/M/E, 3 次域切换, 8192 条
```

一条命令级（细节见 `REPRODUCE.md` §5）：

```bash
cd /workspace/EPLB/OEPLB/quickstart_a100
python3 patch_sglang.py                                              # 打两个补丁（幂等）
sh launch_oeplb_a100.sh /workspace/logs/server_oeplb.log &           # 等 "fired up"
OEPLB_MODEL=/workspace/models/Qwen3-235B-A22B \
  python3 ../scripts/run_grid_bench.py oeplb_fixed ../benchmarks/final_grid/L512_O1.jsonl 1024
```

## 结果一览

| 场景 | Baseline | 官方 EPLB | OEPLB（修复后） |
|---|---|---|---|
| 单域 L512_O1 | 20.9 tps | 23.5 tps (+12.4%) | **25.1 tps (+20.1%)** |
| 多域 MD_L512_O1 | 23.5 tps | 25.3 tps (+7.7%) | **26.6 tps (+13.2%)** |

排名两场景一致：**OEPLB > EPLB > Baseline**。

## 关键前提（否则 OEPLB 静默不生效）

先跑 `python3 patch_sglang.py` 打上**两个**补丁：
- 补丁 A：`ep_dispatch_algorithm='static'`（同 H20）。
- 补丁 B（**A100 专有关键 bug**）：`forward_normal` 补传 `expert_location_dispatch_info`，否则搬了权重但路由不跟 → 不均衡度回弹、吞吐反降。EPLB 和 OEPLB 都依赖它。
