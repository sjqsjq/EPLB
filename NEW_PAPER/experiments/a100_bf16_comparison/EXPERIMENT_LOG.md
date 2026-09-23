# 实验: A100 跨硬件三方对比 (identity / OEPLB-adaptive / EPLB)

**日期**: 2026-09-23
**目的**: 在**非-Hopper、非-DeepEP** 硬件路径上验证 OEPLB 的跨硬件泛化——H20 主实验(§5)全部基于 FP8/DeepEP/DeepGEMM (`forward_deepep`),此实验切到 A100/BF16/Triton (`forward_normal`) 复现单域+多域三方对比。
**模型**: Qwen3-235B-A22B (BF16), TP=EP=8, 8×A100-80GB, NVLink

> **重要 (2026-09-23)**: A100 的 `forward_normal` 路径存在一个致命 bug——调用 `self.topk(...)` 时**漏传 `expert_location_dispatch_info`**(而 `forward_deepep` 正确传了),导致 `topk_ids_logical_to_physical()` 变 no-op、topk_ids 停留 logical → **OEPLB/EPLB 对 `physical_to_logical_map` 的一切修改被路由完全忽略**。症状=**不均衡度窗口间回弹**、开 OEPLB 反而 −3.5%。这与 H20 主实验早期的 bug(`enable_pb_oeplb→ep_dispatch_algorithm='static'` 强制分支缺失,修复后 +19.4%)**同类不同根因**——都需让 logical→physical 重映射真正对路由生效。修复(镜像 `forward_deepep` 补传 dispatch info,见 `patch_sglang.py`)后 OEPLB/EPLB 全部生效。**以 post-bugfix 数值为准。**

## 数据集
- 单域: `OEPLB/benchmarks/final_grid/L512_O1.jsonl` — 512tok prompt (Lean-4 数学证明), O=1 (prefill-dense), 8192 req, conc=1024
- 多域: `OEPLB/benchmarks/multidomain/MD_L512_O1.jsonl` — 数学↔英文小说 4 段交替 (M/E/M/E, 各 2048), 3 次域切换, O=1, 8192 req, conc=1024。数学 p50=301词 / 英文 p50=288词 (长度可比)

## 配置

| arm | launch 脚本 | 关键参数 |
|---|---|---|
| identity | launch_identity.sh | moe-runner-backend **triton** (BF16), cuda-graph-max-bs 64, **--disable-overlap-schedule**, 无 dp-attention, 无 OEPLB/EPLB |
| OEPLB-adaptive | launch_oeplb.sh | + --enable-pb-oeplb + --pb-oeplb-adaptive-window + OEPLB_ADAPTIVE_DECAY=1, sync-window 16, 其余同 identity |
| EPLB | launch_eplb.sh | + --enable-eplb, **0 冗余专家**, --eplb-rebalance-num-iterations 50, 其余同 identity |

> **A100 与 H20 配置差异**: BF16(非 FP8) · Triton runner(非 DeepGEMM) · NCCL a2a(非 DeepEP) · **必须 --disable-overlap-schedule**(否则 NCCL 死锁) · **无 dp-attention**(见 paper §5.9 说明) · P2P 权重搬运同步(非异步)。EPLB 用 0 冗余专家以与 OEPLB 公平对比(两者都做纯放置优化、无复制),且 A100 路径**不受 H20 上 EPLB 被迫禁 CUDA graph 的拖累**。

## 结果

### 单域 (L512_O1, Lean-4 数学)

| arm | tps | time(s) | ok | gain vs identity | crash |
|---|---|---|---|---|---|
| identity | 20.9 | 392.3 | 8192 | — | ✅ |
| OEPLB (bug 前) | 20.2 | 405.9 | 8192 | −3.5% | ✅ |
| **EPLB** (在线, 0冗余, 每50步全局重排) | 23.5 | 348.4 | 8192 | **+12.4%** | ✅ |
| **OEPLB-adaptive** (修复后) | **25.1** | **327.0** | 8192 | **+20.1%** | ✅ |

### 多域 (MD_L512_O1, 数学↔英文, 3 次域切换)

| arm | tps | time(s) | ok | gain vs identity | crash |
|---|---|---|---|---|---|
| identity | 23.5 | 348.0 | 8192 | — | ✅ |
| **EPLB** (在线) | 25.3 | 324.3 | 8192 | **+7.7%** | ✅ |
| **OEPLB-adaptive** (修复后) | **26.6** | **307.4** | 8192 | **+13.2%** | ✅ |

## 关键发现

1. **两场景排名一致: OEPLB > EPLB > identity。** OEPLB 单域超 EPLB +6.8%、多域超 EPLB +5.1%。
2. **跨硬件泛化成立**: OEPLB 在 A100/BF16/Triton 上单域 +20.1% / 多域 +13.2%,与 H20/FP8/DeepEP 上 +19.4%/+9.76% 同量级。
3. **A100 上 EPLB 转正 (+12.4%/+7.7%)**,与 H20 上 EPLB 几乎无收益(§5.8 全负、§5.3 +1.75%)形成对照。原因: A100 路径无 DeepEP 的 `deepep_mode=normal` 约束,EPLB 可**保留 CUDA graph**,不受"CG 禁用抵消均衡收益"退化——但 OEPLB 的增量 swap 仍稳定胜出。
4. **多域自适应机制被日志证实生效**(见 `oeplb_MD_keylines.txt`): 3 次 `PB-OEPLB-RESET` 检测到域切换、窗口自适应减半 32→16→8、ratio 每次跳升后由 swap 压下且 ops 递减收敛 (300→49),不再回弹。
5. **Qwen3-235B 支持官方在线 EPLB**: 与 §5.3 提到官方 EPLB 在 Qwen2-MoE 抛 AttributeError 不同,Qwen3-235B 暴露了 `routed_experts_weights_of_layer` (qwen3_moe.py:1141 懒创建),EPLB 全模型重排 (每次 ~2s) 无崩溃、跑通 (见 `eplb_keylines.txt`)。

## 文件清单
- `identity_L512.json` / `oeplb_L512.json` / `eplb_L512.json` / `oeplb_L512_prebugfix.json` — 单域 run_grid_bench 结果
- `identity_MD.json` / `oeplb_MD.json` / `eplb_MD.json` — 多域结果
- `launch_*.sh` / `env_a100.sh` — 启动脚本(可复现)
- `patch_sglang.py` — **携带真正的代码修复**(forward_normal dispatch-info);clone 后需重跑
- `oeplb_MD_keylines.txt` / `eplb_keylines.txt` — server log 关键行(RESET/WINDOW/DIAG / rebalance)
- `EXPERIMENT_LOG.md` — 本文件
