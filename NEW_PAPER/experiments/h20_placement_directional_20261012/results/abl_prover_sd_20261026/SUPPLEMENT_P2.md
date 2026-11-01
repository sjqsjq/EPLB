# SUPPLEMENT_P2 — 作者只读补充项(判定 6.5% 的机制解释方向)

只读提取自 boots_p2/ 已落盘工件;预注册结果与冻结件未动;块内配对区间为描述性补充,不替代 Holm 主检验。

## 1. 真实 forward 驻留(每臂 8 boot 合计)

- P: 按配置 {"(1, 16, 16)": 62.5, "(4, 16, 64)": 18.8, "(1, 8, 8)": 18.8} %;按状态 {"TRACK": 1280, "QUIET": 384, "RECOVER": 384}
- F_1_16_16: 按配置 {"(1, 16, 16)": 100.0} %;按状态 {"FIXED": 2048}
- F_4_16_64: 按配置 {"(4, 16, 64)": 100.0} %;按状态 {"FIXED": 2048}

## 2. 转换原因(sp_reasons 原文,跨 boot 去重)

- P: {"(6, 'TRACK', 'QUIET')": [["quiet_streak>=3: z<z_lo & h_rate<theta_h_low & coverage & no pending/accepted swaps for 3 consecutive valid H cycles"]], "(6, 'QUIET', 'RECOVER')": [["P1c/P1d confirmed change evidence: event='confirmed_grow' swap_pending=False swaps_started=0"]], "(8, 'RECOVER', 'TRACK')": [["recover_min_cycles>=2 complete clean H8 cycles (stable+low-margin+coverage); no timeout convergence"]], "(11, 'TRACK', 'QUIET')": [["quiet_streak>=3: z<z_lo & h_rate<theta_h_low & coverage & no pending/accepted swaps for 3 consecutive valid H cycles"]], "(11, 'QUIET', 'RECOVER')": [["P1c/P1d confirmed change evidence: event='confirmed_grow' swap_pending=False swaps_started=0"]], "(13, 'RECOVER', 'TRACK')": [["recover_min_cycles>=2 complete clean H8 cycles (stable+low-margin+coverage); no timeout convergence"]], "(16, 'TRACK', 'QUIET')": [["quiet_streak>=3: z<z_lo & h_rate<theta_h_low & coverage & no pending/accepted swaps for 3 consecutive valid H cycles"]]}
- F_1_16_16: {}
- F_4_16_64: {}

## 3. 首次迁移时机

- P: 首 swap 窗 [2] @true_steps [32];首 PLANHASH 时间 − bench 开始 = [4.32, 6.62, 8.13, 10.84, 5.28, 4.6, 6.8, 4.74] s
- F_1_16_16: 首 swap 窗 [2] @true_steps [32];首 PLANHASH 时间 − bench 开始 = [4.68, 5.38, 6.81, 5.21, 4.98, 5.37, 7.23, 8.2] s
- F_4_16_64: 首 swap 窗 [1] @true_steps [80];首 PLANHASH 时间 − bench 开始 = [23.97, 30.29, 23.53, 24.52, 24.47, 32.0, 28.55, 23.31] s

## 4. 计划与布局指纹(跨臂比较)

- P: plan_hash 按窗 {"1": ["b49737229d96aeca"], "2": ["57e07f5f842e25f7"]}; n_ops {"1": [299], "2": [230]}; p2l_fp8 终值集 [""]; p2l_version ["2"]; det_map 轨迹 [[0, 1, 2]]
- F_1_16_16: plan_hash 按窗 {"1": ["bc3572a137c51e90"], "2": ["7076aef150b8aea7"]}; n_ops {"1": [300], "2": [224]}; p2l_fp8 终值集 [""]; p2l_version ["2"]; det_map 轨迹 [[0, 1, 2]]
- F_4_16_64: plan_hash 按窗 {"1": ["737c2f03f2f80d93"], "2": ["64f6b3742a71b933"]}; n_ops {"1": [300], "2": [223]}; p2l_fp8 终值集 [""]; p2l_version ["2"]; det_map 轨迹 [[0, 1, 2]]

## 5. 精确墙钟

- P: bench wall_s mean±sd [85.717, 3.544] (list [84.21, 84.76, 88.32, 93.33, 84.85, 82.96, 84.91, 82.4]);整 boot 墙钟 mean±sd [209.75, 14.21];控制器 Σc_window_ms mean±sd [101.725, 41.849];Σbcost_us mean±sd [107727.0, 36698.617]
- F_1_16_16: bench wall_s mean±sd [86.159, 2.536] (list [84.5, 85.1, 87.74, 90.55, 82.41, 84.71, 87.94, 86.32]);整 boot 墙钟 mean±sd [205.125, 12.654];控制器 Σc_window_ms mean±sd [87.938, 28.571];Σbcost_us mean±sd [101936.125, 25549.751]
- F_4_16_64: bench wall_s mean±sd [91.331, 3.425] (list [88.59, 94.75, 92.91, 87.9, 85.9, 95.28, 92.72, 92.6]);整 boot 墙钟 mean±sd [206.875, 9.062];控制器 Σc_window_ms mean±sd [1871.562, 1943.611];Σbcost_us mean±sd [1867157.75, 1944462.439]

## 6. 差值区间(tps)

- welch_P_minus_F1: {"diff_mean": 0.537, "ci95": [-3.031, 4.106], "df": 12.88, "welch_p": 0.74989}
- welch_P_minus_F4: {"diff_mean": 5.875, "ci95": [2.021, 9.729], "df": 13.88, "welch_p": 0.00562}
- welch_F1_minus_F4: {"diff_mean": 5.338, "ci95": [1.988, 8.687], "df": 13.43, "welch_p": 0.00428}
- paired_by_block_P_minus_F1: {"paired_diffs": [0.3, 0.3, -0.6, -2.7, -2.9, 2.1, 3.3, 4.5], "mean": 0.537, "ci95": [-1.685, 2.76], "paired_p": 0.58533, "note": "DESCRIPTIVE supplement (block-paired), not part of preregistered Holm set"}
- paired_by_block_P_minus_F4: {"paired_diffs": [4.8, 10.1, 4.6, -5.4, 1.1, 12.8, 8.1, 10.9], "mean": 5.875, "ci95": [0.892, 10.858], "paired_p": 0.027, "note": "DESCRIPTIVE supplement (block-paired), not part of preregistered Holm set"}
- paired_by_block_F1_minus_F4: {"paired_diffs": [4.5, 9.8, 5.2, -2.7, 4.0, 10.7, 4.8, 6.4], "mean": 5.338, "ci95": [1.917, 8.758], "paired_p": 0.00776, "note": "DESCRIPTIVE supplement (block-paired), not part of preregistered Holm set"}

## 7. 分时代吞吐(首迁移边界 = 首个 PLANHASH 墙钟时刻;完成请求/s,见 §9 单位勘正)

- P: 迁移前 [50.83, 13.05] (时代长 [6.42, 2.22] s, 完成请求均值 301.6) → 迁移后 [99.54, 1.93] (时代长 [79.3, 1.59] s)
- F_1_16_16: 迁移前 [53.39, 9.47] (时代长 [5.98, 1.26] s, 完成请求均值 309.8) → 迁移后 [98.39, 2.78] (时代长 [80.17, 2.4] s)
- F_4_16_64: 迁移前 [71.33, 8.3] (时代长 [26.33, 3.42] s, 完成请求均值 1853.6) → 迁移后 [97.68, 4.13] (时代长 [65.0, 2.86] s)

## 8. 证据判定(6.5% 更接近哪种机制解释)

- 排除『自适应逻辑自身开销』:P−F1 = +0.54 tps 配对 CI95[−1.69,+2.76] NS;P 每 boot 7 次转换的 Σc_window_ms 101.7±41.8 vs F1 87.9±28.6,量级相同。
- 支持『稀疏固定配置本身的代价』:F1−F4 = +5.34 tps CI95[+1.99,+8.69] p=0.0043,与 P−F4(+5.88 [+2.02,+9.73])同量级同方向——差距由配置携带,与是否自适应无关。
- 实测通道 (i) 首迁移延迟:F4 首 swap @true_steps 80(流 31%)/墙钟 23–32s;P、F1 @32(12.5%)/4–11s;pinned 10× 失衡下多暴露 ~20s 初始布局 ≈ bench 墙钟的 ~22%。
- 实测通道 (ii) 稀疏窗采集开销:F4 Σbcost_us 1.87s±1.94/boot(init 窗单窗 0.83s)vs P 0.108s±0.037、F1 0.102s±0.026,~18×;折合墙钟 ~2%。
- 未排除通道 (iii) 计划内容差异:跨臂 plan_hash 不同值(P 299/230 ops,F1 300/224,F4 300/223),各臂内 8-boot 单一 hash、8-rank 全同;布局非逐字节相同,小幅布局差对差距的贡献无法由指纹判定,登记为未定。
- 驻留佐证:P 按 forward 驻留 62.5% (1,16,16) + 18.8% (4,16,64) + 18.8% (1,8,8),与 P≈F1 一致。
- MISSING 登记:p2l_fp8 全为空串(不可用作指纹);p2l_version 全臂=2。分时代吞吐为只读派生量,边界取计划发出时刻,异步 swap 完成略晚,后时代速率偏保守。

## 9. 单位勘正与异常登记(诚实修正)

- 勘正:§7 数值为 Σcomp_tokens/s;本流每请求 comp_tokens≡1(max_tokens=1),故单位是完成请求/s,不是 prompt-tokens/s。换算 prompt 吞吐 ≈ ×252。原始 perreq 数据未动,仅标签修正。
- 分时代事实(完成请求/s):迁移前 P 50.8±13.1 (6.4s) < F1 53.4±9.5 (6.0s) < F4 71.3±8.3 (26.3s);迁移后 P 99.5±1.9 (79.3s) > F1 98.4±2.8 (80.2s) > F4 97.7±4.1 (65.0s)。
- 即:F4 的早期并不慢(稀疏采集负担轻),它输在 (a) 首计划按 true_steps 迟到(80 vs 32)⇒ 达到同等累计完成数耗时更长(约 26.3s 完成 1854 条;P 约 22.0s 完成同量),(b) 迁移后稳态率略低(−1.8% vs P),(c) 直接采集开销 18×(Σbcost 1.87s±1.94/boot)。(a)+(b) 合计与 5.6s 总墙钟差同量级。
- 异常登记:b4 为唯一负配对差(P_b4=87.8 为 P 臂最差 boot,其迁移前速率 26.7 req/s 异常低、边界 10.8s;F4_b4 稳态 101.1 为全场最高)。该 block 漂移同时影响三臂,预注册分析未做校正,仅登记。
- 稳态率差异 (b) 的归因在 plan_hash 跨臂不同值的事实下无法进一步分解(布局内容差 vs 稀疏窗稳态开销),维持『未定』登记;不宣称单一机制。
