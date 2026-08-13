# 复现指南（Reproduction Guide）

本文档是论文《面向MoE推理服务的自适应在线专家负载均衡》的**核心复现实验**，
按论文主张的依赖关系组织。每个实验都给出：要验证的论文声明、命令、预期结果、
以及失败时如何判断是环境问题还是结果不符。

**所有结果文件**保存在 `benchmarks/results/`，按 `_dNN_*` 前缀对应本指南中的实验编号。
**所有驱动脚本**保存在 `repro/`，原始计数保存在 `repro/counts*.json`。

---

## 0. 环境

- 硬件：8×NVIDIA H20 96GB，NVLink-only（无 IB）
- 软件：SGLang 0.5.6.post2 + PB-OEPLB patch，sgl-kernel 0.3.19，torch 2.9.1+cu128，DeepEP v1.2.1，DeepGEMM
- 模型：`Qwen3-235B-A22B-FP8`、`Qwen2-57B-A14B-Instruct`、`Qwen3-30B-A3B-FP8`
- 数据：`/data/minghua/sjq/OEPLBdata/datasets/` 下的 jsonl

```bash
# 验证 patch 已生效
grep -q "pb_oeplb" $(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))")/srt/server_args.py && echo OK
```

---

## 1. 头条收益：prover 单域 235B（论文 §1、§5.3）

**论文声明**：PB-OEPLB 在 235B prefill 密集负载上提升吞吐 +17.5%（复测均值）。

这是论文最核心的单一结果。`driver38.sh`（`repro/driver38.sh`）。

```bash
# identity baseline（不传 --init-expert-location，默认 trivial 布局）
# OEPLB 用默认参数（threshold 1.02, sync_window 16, decay 0.5）
bash repro/driver38.sh
```

**预期**（2 轮独立重启）：
| 臂 | 时间(s) | CV |
|---|---|---|
| baseline(identity) | ~201 | <0.1% |
| OEPLB | ~169 | ~1% |
| **收益** | **+17~+20%** | |

实测（本次）：baseline 201.2s CV0.02%，OEPLB 168.5s CV1.18%，**+19.43%**。
对应 `_d38_bl_r{1,2}.json`、`_d38_oe_r{1,2}.json`。

**失败排查**：
- 若 baseline 时间 ≈165s 而非 ~200s，说明误用了静态最优布局做 baseline（那是 oracle，不是 identity），收益会算成负。确保 baseline **不传** `--init-expert-location`。
- 若 OEPLB 收益为负且日志 swap 次数=0，检查 `--enable-pb-oeplb` 是否生效。

---

## 1b. 多域漂移负载：两个不同的量（论文 §5.3）

**论文声明**：漂移多域负载下，OEPLB 相对 identity 的**头条收益 +9.76%**，相对静态最优的
**adaptation benefit +5.80%**。这两个量**必须分开**，参照基线不同。`driver39.sh`（头条）与
`driver35.sh`（adaptation）。

```bash
bash repro/driver39.sh   # identity 基线 -> 头条收益
bash repro/driver35.sh   # 静态最优(bal)基线 -> adaptation benefit
```

**预期**（各 2 轮独立重启）：
| 实验 | 基线 | OEPLB | 量 |
|---|---|---|---|
| driver39 | identity 824.7 s | 751.4 s | **头条 +9.76%** |
| driver35 | 静态最优 801.2 s | 757.2 s | **adaptation +5.80%** |

**失败排查**：
- 若把 driver35 的 +5.80% 当成头条收益，是混淆了两个量——它的基线是静态最优，不是 identity。
- OEPLB 比静态最优还快（757.2 < 801.2）只在负载随时间漂移时出现：静态最优只对聚合分布最优，
  OEPLB 逐域跟随。若负载不漂移，OEPLB 不应超过静态最优。

## 2. T(r) 扫描：上界模型的核心证据（论文 §2.4、附录 G）

**论文声明**：$T(r)=T_{flat}+\beta T_{flat}\max(0,r-r_k)$（铰链，死区），拟合 $R^2>0.996$，
且拟合上界与经验天花板（identity→bal）吻合到 0.4pp 内。

这是建模的基石。`driver12.sh`（57B/EP8）、`driver13.sh`（57B/EP4）、`driver14.sh`（235B/EP8）、`driver28.sh`（30B/EP4）。

```bash
# 以 57B/EP8 为例（7 个布局点 ×2 轮）
python3 repro/gen_placement.py repro/counts57b.json 8 plc57b 1.10 1.22 1.35 1.50
bash repro/driver12.sh
python3 repro/fit_f3.py 1.218 1.04 id plc57b.txt _d12_
```

**预期**：
- 铰链 RSS 比直线低 7~25×（57B/EP8 低 12.1×）
- `r_k` ≈ 1.099（57B/EP8）、1.032（EP4）、1.093（235B/EP8）、1.031（30B/EP4）
- `β` ≈ 0.285 / 0.342 / 0.352 / 0.207
- held-out identity 点误差 <0.5%

**关键自检**：把 identity（$r=1.218$）与 bal（$r=1.010$）两个静态布局直接对比，
得到的经验天花板应 ≈ 铰链上界（57B/EP8 上 +3.82% vs +3.40%，差 0.42pp）。
这是不依赖拟合的独立检验，若两者差 >1pp 说明测量有问题。

---

## 3. r_k 的 EP 幂律（论文 §2.4）

**论文声明**：$r_k-1 = 0.00408\cdot EP^{1.52}$，235B/EP8 跨模型盲测误差 3.8%。

用第 2 节的 4 个 `r_k` 值拟合验证：

```python
import numpy as np
EP=np.array([2,4,8]); rk=np.array([1.012,1.032,1.099])
p=np.polyfit(np.log(EP),np.log(rk-1),1)   # 斜率≈1.52
pred=np.exp(p[1])*np.array([8])**p[0]+1   # EP=8 预测≈1.096, 实测 235B=1.093
```

**预期**：斜率 ≈1.5，235B/EP8 预测与实测误差 <5%。

---

## 4. 死区阈值把 η 从 26% 提到 100%（论文 §3.2、D.3）

**论文声明**：默认 threshold 1.02 落在死区内导致狂 swap；改为 $r_k$+swap预算后 η 26%→100%。
`driver26.sh` / `driver27.sh`。

```bash
bash repro/driver26.sh   # 57B/EP8, 三臂: (1.02,256) / (1.02,1e5) / (1.099+预算)
```

**预期**（57B/EP8 L256，经验天花板 +3.82%）：
| 臂 | 收益 | η |
|---|---|---|
| threshold=1.02 | +0.98% | 26% |
| threshold=1.099+budget | +3.81% | **100%** |

**失败排查**：若 threshold=1.099 臂的决策次数没降到个位数，检查 `--pb-oeplb-threshold-ratio` 是否传入。

---

## 5. 自适应窗口（论文 §3.5、附录 H）

**论文声明**：$M=W/(1-\alpha)$ 是近似充分统计量（$M\ge32$ 时组内差<5%）；
自适应衰减+gate 比静态更快且 swap 少 44%。`driver31.sh`。

```bash
bash repro/driver31.sh   # 235B, segp_L1000, (W,α) 网格 + adaptive 臂
```

**预期**：
- M=32 组（W32α0 / W16α0.5 / W8α0.75）三者吞吐差 <5%
- `adw+adaptive_decay` 臂是全表最快（本次 67.0s，比同起点静态 69.5s 快 3.6%，swap 10 vs 18）
- (W=8, α=0.5) 是病态角（慢 >5×），应避开

---

## 6. 数值等价性（论文附录 E）

**论文声明**：swap 是恒等变换，扰动来自 FP8 归约顺序而非 swap 机制；零 swap 的静态布局
复现同等扰动。`driver15.sh` / `driver19.sh` / `driver21.sh`。

```bash
bash repro/driver15.sh   # baseline vs OEPLB, GSM8K + 语料logprob, 前后各一次
bash repro/driver19.sh   # 静态布局扰动(零swap)
```

**预期**：
- baseline 逐 bit 相同（50385/50385），即噪声底为零
- OEPLB 前后扰动 ≈ 静态布局扰动（9.9e-2 vs 9.3~9.6e-2）
- GSM8K 准确率无显著变化（±2pp 内，n=200 的 1σ=2.6pp）

---

## 7. 30B 负收益的正确归因（论文 §2.4、30B 脚注）

**论文声明**：30B β=+0.207（正，不均衡有害），但默认配置负收益源于固定开销 > 上界；
swap预算可止损。`driver28.sh`（T(r) 扫描）+ `driver30.sh`（三臂）。

```bash
bash repro/driver30.sh   # 默认1.02 / 死区1.031 / 死区+预算
```

**预期**：
| 臂 | 收益 |
|---|---|
| threshold=1.02 | −3.8% |
| threshold=1.031 | −2.66% |
| threshold=1.031+budget | **+0.53%** |

---

## 8. predict_gain.py：零测量上界估算（论文 §2.4 三维度）

```bash
python3 tools/predict_gain.py --model-config /path/to/config.json --ep 8
```

**预期**：给出 $r_k$（EP幂律）、$\beta$（FLOP/1.6 粗估）、上界 $\Delta_{max}$。
误差：$r_k$ ±5%，$\beta$ ±30%。**定量上界仍需第 2 节的 T(r) 扫描**（β 是唯一无法
从 profiling 预测的环节，见论文 §2.4 与 `driver23b.sh`）。

---

## 已知限制（诚实声明）

1. **β 无法从单次 profiling 预测**：需 T(r) 扫描（~2h/配置）。profiling 只能粗估（±30%）。
2. **多域 235B 的 +14.0%（历史）不可复现**：数据集已删除。可复现的多域数据集
   （`prefill_heavy_universal.jsonl`，16000请求）上测得 **+5.80%**（`driver35.sh`）。
3. **M\* 闭式的数值系数未定标**：d31/d34 确认方向（短段最优 M 小、长段最优 M 大）
   但未见峰，$\sqrt{L_{seg}}$ 的系数待定。
4. **所有实验在 H20 NVLink 上**：$r_k$ 幂律未在其他硬件验证。
5. **r 对 30B 是弱充分统计量**（held-out 误差 3.2% vs 57B/235B 的 <0.4%），
   因 30B 层间方差大。

## 复现优先级建议（给评委）

若时间有限，按此顺序最有说服力：
1. **第 1 节**（prover 单域 +17.5%）——论文头条，~1 小时
2. **第 2 节 + 第 3 节**（T(r) 扫描 + r_k 幂律）——建模核心，~3 小时
3. **第 4 节**（死区阈值 η 26%→100%）——算法改进的可验证证据，~1 小时

