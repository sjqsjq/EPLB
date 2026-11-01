# FREEZE record — ADW v2.1 六参数（r3 预注册规则，一次盲拟合）

冻结时间：2026-10-17T07:25:26+00:00（生成器 gen_freeze_v2p1.py，全部数值取自 JSON 工件，无手填）。
规则来源：calibration_plan_v2p1.md（r3）§2/§2.1/§2.2/§3；公式逐字沿用
calibration_plan_v2.md §3。数据：仅 4 次 W0 固定窗口 logging run
（r3_logging/W{8,16,32,64}/），无任何正式比较数据参与。

## 1. 冻结的 6 个 knob（OEPLB_V21_*）

| knob | 值 | 规则 |
|---|---|---|
| KAPPA (κ) | 3875.1768798828125 | median(Z_s pooled) |
| Z_LO | 22.643858727592757 | p90(Z_s/κ)，numpy percentile linear |
| Z_HI | 314.14010272527486 | p99(Z_s/κ) |
| Z_URGENT | 942.4203081758246 | max(3·Z_HI, p50(Z_c/κ)) |
| THETA_R_LOW | 0.010095552687949321 | p30(H_rate_s pooled) |
| THETA_R_ACT | 0.011395643112507271 | p60(H_rate_s pooled) |

沿用不拟合：N_MIN=12320768、M_SHRINK=2、M_GROW=3、DWELL 2/2、U_MAX=4、
W_UNCERTAIN_TARGET=16、URGENT_EPISODE_CAP=3、BETA=0.5、W_init=16。
分位数方法：numpy.percentile(method="linear")；合并池每窗等权；
有效样本数 = 池长度（下表）。

## 2. 池与门（§2.2）

| 池 | n | 门 | 结果 |
|---|---|---|---|
| Z_s / H_rate_s（合格静止窗） | 182 | ≥150 | PASS |
| Z_c（合格伪切换窗） | 61 | ≥20 | PASS |

逐 run 合格窗实数 vs 预期（§2.2 表）：

| W | 记录窗 | 合格 | 合格静止（预期） | 合格变化 | 排除原因 |
|---|---|---|---|---|---|
| 8 | 132 | 131 | 101（104-107） | 30 | {"excluded_event:init": 1} |
| 16 | 66 | 65 | 49（51-53） | 16 | {"excluded_event:init": 1} |
| 32 | 33 | 32 | 23（25-26） | 9 | {"excluded_event:init": 1} |
| 64 | 16 | 15 | 9（12） | 6 | {"excluded_event:init": 1} |

窗清单 sha256：6087350edec8bf3639ed64b3dbb96a3b164b5e3dd9430baf5f760dd0d1ea9fd7

## 3. 验收门（§4，冻结前全部重跑）

- **PT1（离线机制级，最终源码重跑）**：test_window_v2p1_logic.py →
  ALL_V2P1_TESTS_PASS ✓（PASS 计数 39，含 T5/T6b/T7 成对缩放、T2/T3 逻辑身份）。
- **PT2 A/B（数据驱动成对缩放，pre-fit）**：窗数 182，精确有理恒等
  182/182（门=100%：PASS），float64 逐位相同 180，最大偏差 1 ulp
  （纯舍入，分类记录）；θ 布尔不变扫描失败 0。
  范围声明：不声称复制计数为新增独立样本；不声称重算 z / UNCERTAIN
  天然不变（n_min 为绝对计数门，单独报告）。
- **PT2-C（冻结 knob 决策恒等，post-fit pre-freeze；harness v2 (fixes B1 knob mismatch, B2 state off-by-one, B3 z basis; v1 archived as pt2c_results_v1_harnessbug.json)）**：
  成对缩放窗 243（静止 182 = 注册门，变化区 61 为补充覆盖）；决策不一致 1
  （其中 n_min 绝对门类 1，F2 缺陷类 0）；h_rate 逐位相同 241，最大绝对差
  1.73472e-18；状态重构保真：243/243 窗复现 live 事件 + W_after + 全部 post-state 字段
  （0 失配）；n_min 侧：orig_below=1 scaled_below=0 flips=1。
  v1 harness 缺陷（跨 knob 重放 / 状态错位一窗 / z 基准未换算）已于冻结前
  修复并归档：pt2c_harness_fix_note.md；v1 输出存
  pt2c_results_v1_harnessbug.json。零 knob/参数/拟合改动。
- **G2 分离度**（报告项）：p50(Z_c/κ)=0.69114517800646 vs Z_HI=314.14010272527486 → NOT separable（如实报告）。
- **G3 重尾**（报告项）：p90/p50=22.643858727592757 → heavy tail（报告，不自动处理）。
- **G5 κ 跨 run 稳定性**（报告项）：{"8": 0.8656882440062511, "16": 0.996293547462504, "32": 2.2998746056481987, "64": 91.58818223046711}（W64 单拟小样本仅列示）。
- **G1-D（诊断，解释带）**：h_rate 中位数逐 W {"8": 0.011013, "16": 0.010846, "32": 0.009962, "64": 0.01257}；max/min 比 1.2617478814961764 →
  带判定：<=1.5 compatible。raw h_w 中位数比 9.110803487059567（F2 修复前后对照存档）。
- **D1（诊断，解释带）**：R_swap=6.1716980936337595（n_swap_stat=53, n_nonswap_stat=129）→
  带判定：>2 STOP-before-freeze: offline F1 path reconstruction required。带超限 → 已按计划 §4 于冻结前完成排查并归档
  investigation_d1_rswap.md（sha256 `2ee4ce6cebdac58eba47f1b686097e465f47971d270ec15a07f62c50b59a3577`）：结论为 planner 选择效应 +
  真实漂移随窗长积累 + 跨 W 组成效应；F1 泄漏的三条判别签名全部排除
  （z 峰值先于映射变化、量级超出 n_layer 缩放、衰减非 β^k）；解释带
  未移动，任何 knob 未因排查改动。
- **D2（分层报告，非判据）**：{"n_swap_windows": 62, "recovered_overall": 0.9032258064516129, "by_zone": {"stationary": {"n": 53, "recovered": 50}, "change": {"n": 9, "recovered": 6}}, "note": "layered report; pseudo-switch-zone swaps naturally carry high z; NOT a fail criterion"}。

## 4. 代码锁（sha256，src == site-packages 部署实测）

| 文件 | sha256 |
|---|---|
| window_v2p1.py | `d20d340f9d44ae997bbb8f2c311bed40e84673173eba90b9da1658374353fdb8` |
| v2p1_signals.py | `98a13608ac56f0f132daa8d189c9aebfee4e62b6633f009f28c3ca2b66e2c436` |
| controller.py | `24ccf3048544ed06cbc4f5342c2c418264688c8c9996cb5deb31ab3347ed7c86` |
| config.py | `66b0760d3e7169bd17ebe3f71818826b9eaf19e9f1e059266279e44e1bb09b16` |
| window_v2.py（冻结 v2，未触碰） | 810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74 |

git：/workspace/EPLB main @ 51b83e6（logging 启动前锁定，见
LOCK_record_r3_logging.md；本冻结记录与战役工件将随后续本地 commit 归档）。

## 5. 冻结判定

**全部验收门 PASS → 六参数冻结。** 行为验证 run（第 5 次启动）按 §5.2 执行。
