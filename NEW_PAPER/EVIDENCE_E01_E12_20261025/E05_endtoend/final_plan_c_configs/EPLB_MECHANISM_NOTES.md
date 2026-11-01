# EPLB 机制说明（final_plan_20261019，事前规则记录）

安装版 sglang 0.5.6.post2（实测 import 版本；/opt/conda/lib/python3.11/site-packages/sglang），语义均从安装源码核对，未做任何搜索/调参；全部默认值 = 该版本事前规则，仅记录。

## 1. 旗标语义（E 臂使用的三个显式旗标）
- `--eplb-rebalance-num-iterations N`：rebalance 周期。请求生成器每 yield N 次触发一次 rebalance（默认 1000，即 e6 的"默认周期参照"）。断言 `period >= expert_distribution_recorder_buffer_size`；buffer 默认 None → 自动设为 period。
- `--eplb-algorithm A`：A ∈ {auto, deepseek, deepseek_hierarchical, deepseek_vec, deepseek_vec_hierarchical, elasticity_aware}。auto → compute_algorithm：`num_groups % num_nodes == 0` 时 deepseek_hierarchical，否则 deepseek。Qwen3-235B-A22B（8 ep nodes）→ auto 解析为 deepseek（与总方案 "auto→deepseek" 预期一致；F0 smoke 以启动日志实际回显为准）。
- `--ep-num-redundant-experts R`：冗余专家数；默认 0，R=0 合法（e0）。

## 2. 随 enable_eplb 自动生效的默认（记录，不搜索）
- `ep_dispatch_algorithm`: None → 自动置 "static"
- `init_expert_location`: "trivial"
- `eplb_rebalance_layers_per_chunk`: None（整层一次）
- `eplb_min_rebalancing_utilization_threshold`: 1.0
- `expert_distribution_recorder_buffer_size`: None → = period

## 3. 校准候选 → CLI 映射（CALIB1，22 cells，run_list_calib1.csv sha 锁）
| 候选 | algorithm | R | period | 含义 |
|---|---|---|---|---|
| e0 | auto | 0 | 64 | 无冗余基线 |
| e1 | auto | 16 | 64 | 标准冗余 |
| e2 | auto | 32 | 64 | 高冗余 |
| e3 | deepseek_vec | 16 | 64 | 向量化算法对照 |
| e4 | auto | 16 | 16 | 高频 rebalance |
| e5 | auto | 16 | 256 | 低频 |
| e6 | auto | 16 | 1000 | 默认周期参照 |
F 候选（PB-fixed）：(S,H,W) ∈ {(1,16,16),(2,16,32),(4,16,64),(1,8,8)}，每候选 ×2 重复。

## 4. 完整性检查所用日志标记
- 启动横幅：`[EPLBManager] system started, will rebalance per {period} iterations.`
- 动作证据：`[EPLBManager] rebalance start`（integrity_check_fp.py 要求 ≥1 次）
- I 臂要求两者均不出现，且无任何 `[PB-OEPLB` 行。

## 5. F0 smoke 中 E 臂参数确认
smoke 用 e1（auto 16 64）验证 algo/R/period 与配置一致后，上述映射才视为冻结（总方案 §7 要求）。

## 6. E 臂 deepep-mode 预注册偏差（2026-10-18，F0 smoke 发现）
**事实**：本 sglang 版本 `--enable-eplb` 自动置 `expert_distribution_recorder_mode='stat'`（server_args.py L1667-1668，buffer 自动=period）。'stat' 模式的 `_SinglePassGatherer.init_new` 仅实现 deepep_mode ∈ {normal→_SelectExpertsSinglePassGatherer, low_latency→_DeepepLowLatencySinglePassGatherer}，`auto` 落入 `raise NotImplementedError`（eplb/expert_distribution.py L314）。F0 smoke 中 f0_E_e1 因此秒败（两次，零数据，INFRA 类）。
**先例**：本机已封存的 d38_L512_O1_comparison EPLB 基线（2026-09-10，同模型 Qwen3-235B-A22B-FP8、同 tp8/dp8/ep8/deepep/deep_gemm/fp8 栈）即以 `--deepep-mode normal` 成功运行（keylines 存 EPLBManager 横幅，recorder='stat'）；gb200_ep4_repro 与 a100 系列 EPLB 脚本同为 normal。
**决定（在任何 E 臂计时前预注册）**：E 臂基座将 `--deepep-mode auto` 替换为 `normal`（launch_arm.sh E 分支内注释同步）；其余全部臂（I/P/F/A1-A4/P1）保持 rev1 封存基座 `auto` 不变。这是该版本 EPLB 机制的实现约束+封存先例沿用，不是调参。
**公平性 caveat（必须进冻结文件与最终报告 limitations）**：E 臂 decode 阶段无 low-latency buffer，其墙钟可能因栈差异（而非放置策略）偏慢；P-vs-E 对比解释时须带此保留。P-vs-I 主对比不受影响（两臂同基座）。
