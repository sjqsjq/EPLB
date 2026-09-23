# H20 快速复现（8×H20 / FP8 / DeepEP 主线）

这是论文主线实验的复现入口。**H20 走 `forward_deepep` 路径：FP8 权重 + DeepGEMM 分组 GEMM + DeepEP all-to-all + dp-attention**，是论文正文所有主结果（§5）的产出环境。

> 想在 **A100 / BF16 / 非-DeepEP** 上复现？请改用 `../quickstart_a100/`（环境更简单，不用编译 DeepEP/DeepGEMM）。两套硬件的差异见 `../quickstart_a100/REPRODUCE.md` §1。

## 本文件夹三份文档

| 文档 | 回答的问题 | 什么时候读 |
|---|---|---|
| `ENVIRONMENT.md` | 怎么从裸机把环境搭起来 | **第一步**：装 CUDA/DeepEP/DeepGEMM、patch SGLang、下模型 |
| `REPRODUCE.md` | 环境好了跑哪些实验复现论文 | **第二步**：逐个复现论文 §1~§8 的图表 |
| `REPRODUCE_EN.md` | 同上，英文精简版 | 给英文读者/审稿人 |

正确顺序：**先 `ENVIRONMENT.md` 把服务跑通并自检 → 再照 `REPRODUCE.md` 复现结论。**

## 最快复现路径（1 条命令级）

只想快速看到头条收益，按下面走（细节见 `REPRODUCE.md` §1）：

1. 按 `ENVIRONMENT.md` 装好环境，启动 OEPLB 服务器（`ENVIRONMENT.md` 第五节「OEPLB 推荐配置」）。
2. **数据集（仓库内直接可用，无需下载）**：
   ```
   OEPLB/benchmarks/final_grid/L512_O1.jsonl     # 8192 条 Prover-V1 单域, ~500tok 输入, max_tokens=1（纯 prefill）
   ```
   更短更快可用 `L256_O1.jsonl`（~250tok 输入）。
3. 压测：
   ```bash
   cd OEPLB/scripts
   python3 run_grid_bench.py oeplb ../benchmarks/final_grid/L512_O1.jsonl 256
   ```
4. **预期**：baseline(identity) ≈201s → OEPLB ≈169s，**+19.4%**（`REPRODUCE.md` §1）。

## 硬件与关键版本

- 8× NVIDIA H20 96GB，NVLink-only（无 IB/RDMA）
- SGLang 0.5.6.post2 + PB-OEPLB patch，sgl-kernel 0.3.19，torch 2.9.1+cu128，DeepEP v1.2.1，DeepGEMM
- 模型：`Qwen3-235B-A22B-FP8`（ModelScope 下载，223GB）

## 自检（避免 OEPLB 静默不生效）

启动后必看两处（详见 `ENVIRONMENT.md` §5.1）：

- server log 里 `ep_dispatch_algorithm='static'`（不能是 `None`）——否则 `ENVIRONMENT.md` §3.1 的 bug 修复没做，OEPLB 无效。
- `[PB-OEPLB-DIAG]` 行里 window N+1 的 `avg_before` ≈ window N 的 `avg_after`（不均衡度**不回弹**）。
