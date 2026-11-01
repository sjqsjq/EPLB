# REPORT — calibration M1→M2→M3 (route-B v1.1, development calibration)

- STATUS: COMPLETE (15/15)
- spec_rev: v1.1-calib；治理：LOCAL ONLY、描述性统计（PRE_REG §E）、留空项保持留空、MECHANISM 运行（cal_m2_w1_abort）永不混入性能统计、跨 rank 时间只用 MAX 或 per-rank（永不求和）、不扣除任何实测成本、不挑选有利重复。
- 估计器：`observational_ABA_diff_trend_corrected`；`causal_isolation_claimed=False`；残留假设 A1 drift near-linear over the span; A2 no regime event inside the probe; A3 stream-slice heterogeneity slow; A4 layout stable in segment or flagged（随结果报告，不称已隔离因果）。

## 1. 全部启动结果（作者第 6 点：全部 15 次）

| seq | stage | cond | mode | label | verdict | boot_s | total_time_s | tps | ok/req | errors | ttft_p99_ms | lockstep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | M1 | W1 | off | cal_m1_w1_a | VALID | 92 | 354.99 | 11.30 | 4000/4000 | 0 | 25957.3 | True |
| 2 | M1 | W1 | off | cal_m1_w1_b | VALID | 68 | 359.19 | 11.10 | 4000/4000 | 0 | 29272.4 | True |
| 3 | M1 | W2 | off | cal_m1_w2_a | VALID | 72 | 711.85 | 11.20 | 8000/8000 | 0 | 25220.5 | True |
| 4 | M1 | W2 | off | cal_m1_w2_b | VALID | 64 | 706.97 | 11.30 | 8000/8000 | 0 | 26135.9 | True |
| 5 | M2 | W1 | gate | cal_m2_w1_norm | VALID | 68 | 352.19 | 11.40 | 4000/4000 | 0 | 24954.8 | True |
| 6 | M2 | W1 | gate | cal_m2_w1_abort **(MECHANISM)** | VALID | 72 | 352.45 | 11.30 | 4000/4000 | 0 | 24783.6 | True |
| 7 | M2 | W2 | gate | cal_m2_w2_obs | VALID | 68 | 706.15 | 11.30 | 8000/8000 | 0 | 23786.0 | True |
| 8 | M3 | W1 | aaa | cal_m3_b1_aaa | VALID | 64 | 351.58 | 11.40 | 4000/4000 | 0 | 24747.4 | True |
| 9 | M3 | W1 | gate | cal_m3_b1_aba | VALID | 72 | 352.54 | 11.30 | 4000/4000 | 0 | 25088.6 | True |
| 10 | M3 | W1 | aaa | cal_m3_b2_aaa | VALID | 68 | 352.15 | 11.40 | 4000/4000 | 0 | 25299.2 | True |
| 11 | M3 | W1 | gate | cal_m3_b2_aba | VALID | 64 | 356.82 | 11.20 | 4000/4000 | 0 | 27310.9 | True |
| 12 | M3 | W2 | aaa | cal_m3_b3_aaa | VALID | 68 | 708.22 | 11.30 | 8000/8000 | 0 | 24061.1 | True |
| 13 | M3 | W2 | gate | cal_m3_b3_aba | VALID | 64 | 711.45 | 11.20 | 8000/8000 | 0 | 26153.5 | True |
| 14 | M3 | W2 | aaa | cal_m3_b4_aaa | VALID | 68 | 707.39 | 11.30 | 8000/8000 | 0 | 23680.2 | True |
| 15 | M3 | W2 | gate | cal_m3_b4_aba | VALID | 64 | 708.41 | 11.30 | 8000/8000 | 0 | 24152.3 | True |

预算：BOOT_OK=15（计时启动，上限 15）；VERDICT VALID=15；RESULT_RETIRE=0；INFRA_MISSING=0（上限 3，不占计时预算）；MECH_RETIRE=0；INTEGRITY_FAIL=0；STAGE_END=['M1', 'M2', 'M3']；HALT=[]

## 2. 配置校验（meta.txt vs 冻结 run list）

| label | MODE | EPS_RISK_US | FORCE_ABORT_AT | COMMIT | CALIB_TRIGGER | R | NREF/NCAND/NFB | EPS_PRAC_FRAC | EA_W | knobs |
|---|---|---|---|---|---|---|---|---|---|---|
| cal_m1_w1_a | off | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m1_w1_b | off | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m1_w2_a | off | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m1_w2_b | off | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m2_w1_norm | gate | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m2_w1_abort | gate | 1000 | 1 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m2_w2_obs | gate | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b1_aaa | aaa | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b1_aba | gate | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b2_aaa | aaa | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b2_aba | gate | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b3_aaa | aaa | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b3_aba | gate | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b4_aaa | aaa | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |
| cal_m3_b4_aba | gate | 2000000 | 0 | 0 | 1 | 12 | 4/3/4 | 0.01 | 16 | FROZEN-OK |

## 3. M1 gate（决定论复原，m1_gate.json）

**PASS = True**
- cal_m1_w1_a: checks=15 failed=无 z_max|Δz|=0.2815 z_min_margin=21.559 c_window_Δ=0.4621ms record_us=121.07 cap=127.09
- cal_m1_w1_b: checks=15 failed=无 z_max|Δz|=0.1681 z_min_margin=21.673 c_window_Δ=0.9123ms record_us=120.275 cap=127.09
- cal_m1_w2_a: checks=15 failed=无 z_max|Δz|=37.7221 z_min_margin=14.713 c_window_Δ=0.5412ms record_us=117.14500000000001 cap=126.14
- cal_m1_w2_b: checks=15 failed=无 z_max|Δz|=17.2962 z_min_margin=13.907 c_window_Δ=0.5519ms record_us=113.365 cap=126.14
- 封存参照 W1: windows=20 sum_steps=992 swaps=2 record_us_median=115.53999999999999 (n=8)
- 封存参照 W2: windows=37 sum_steps=1952 swaps=4 record_us_median=114.67 (n=8)

## 4. M2 gate（机制正确性，m2_gate.json）

**PASS = True**
- cal_m2_w1_norm: failed=无
- cal_m2_w1_abort: failed=无
- cal_m2_w2_obs: failed=无

## 5. 探测运行时间线与估计（M2 机制 + M3 配对）

| label | stage | mode | final_state | trigger | REF/CAND/FB 窗数 | classification | Δ_net µs/step | Δ_net rel | ε_prac | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| cal_m2_w1_norm | M2 | gate | DONE | 12 | 4/3/4 | favorable | -4678.85 | -0.01332 | 3511.39 | — |
| cal_m2_w1_abort **(MECHANISM，不入性能统计)** | M2 | gate | DONE | 12 | 4/2/4 | UNKNOWN | 826.91 | 0.00238 | 3476.81 | abort:force_abort_at |
| cal_m2_w2_obs | M2 | gate | DONE | 12 | 4/3/4 | UNKNOWN | -349.81 | -0.00100 | 3485.31 | — |
| cal_m3_b1_aaa | M3 | aaa | DONE | 12 | 4/6/4 | UNKNOWN | -2272.32 | -0.00649 | 3501.44 | — |
| cal_m3_b1_aba | M3 | gate | DONE | 12 | 4/3/4 | UNKNOWN | -1846.33 | -0.00526 | 3507.48 | — |
| cal_m3_b2_aaa | M3 | aaa | DONE | 12 | 4/6/4 | UNKNOWN | -169.36 | -0.00049 | 3477.35 | — |
| cal_m3_b2_aba | M3 | gate | DONE | 12 | 4/3/4 | UNKNOWN | -9874.58 | -0.02803 | 3523.16 | FB:layout_change |
| cal_m3_b3_aaa | M3 | aaa | DONE | 12 | 4/6/4 | UNKNOWN | 3370.15 | 0.00967 | 3484.20 | — |
| cal_m3_b3_aba | M3 | gate | DONE | 12 | 4/3/4 | UNKNOWN | -449.44 | -0.00129 | 3490.09 | — |
| cal_m3_b4_aaa | M3 | aaa | DONE | 12 | 4/6/4 | favorable | -3761.77 | -0.01079 | 3485.66 | — |
| cal_m3_b4_aba | M3 | gate | DONE | 12 | 4/3/4 | UNKNOWN | 1543.76 | 0.00444 | 3477.73 | — |

## 6. AAA 误判表现（零假设组：真 Δ=0，任何 favorable/unfavorable 即误判）

| label | cond | Δ_net µs/step | Δ_net rel | ε_prac | classification | 误判? |
|---|---|---|---|---|---|---|
| cal_m3_b1_aaa | W1 | -2272.32 | -0.00649 | 3501.44 | UNKNOWN | 否 |
| cal_m3_b2_aaa | W1 | -169.36 | -0.00049 | 3477.35 | UNKNOWN | 否 |
| cal_m3_b3_aaa | W2 | 3370.15 | 0.00967 | 3484.20 | UNKNOWN | 否 |
| cal_m3_b4_aaa | W2 | -3761.77 | -0.01079 | 3485.66 | favorable | **是** |

AAA 误判数 = 1 / 4：[('cal_m3_b4_aaa', -3761.7734375, 'favorable')]

## 7. ABA 估计稳定性（M3 正式配对组；M2 观测运行单列不合并）

**W1**（M3 ABA：cal_m3_b1_aba, cal_m3_b2_aba）
- cal_m3_b1_aba: rate_ref=350748.31 rate_cand=348074.05 rate_fb=349092.45 g=-10.349121 pred=349920.38 Δ_net=-1846.33 rel=-0.00526 ε_prac=3507.48 cls=UNKNOWN midpoints={'ref': 224.0, 'cand': 304.0, 'fb': 384.0} n_steps={'REF': 64, 'CAND': 96, 'FB': 64}
- cal_m3_b2_aba: rate_ref=352315.66 rate_cand=347791.40 rate_fb=363016.30 g=66.879004 pred=357665.98 Δ_net=-9874.58 rel=-0.02803 ε_prac=3523.16 cls=UNKNOWN midpoints={'ref': 224.0, 'cand': 304.0, 'fb': 384.0} n_steps={'REF': 64, 'CAND': 96, 'FB': 64}
- 描述：ABA Δ_net 范围 [-9874.58, -1846.33]，极差 8028.25 µs/step；符号一致
- AAA（零假设组）Δ_net 范围 [-2272.32, -169.36]，极差 2102.96 µs/step

**W2**（M3 ABA：cal_m3_b3_aba, cal_m3_b4_aba）
- cal_m3_b3_aba: rate_ref=349009.45 rate_cand=347855.72 rate_fb=347600.86 g=-8.803711 pred=348305.16 Δ_net=-449.44 rel=-0.00129 ε_prac=3490.09 cls=UNKNOWN midpoints={'ref': 224.0, 'cand': 304.0, 'fb': 384.0} n_steps={'REF': 64, 'CAND': 96, 'FB': 64}
- cal_m3_b4_aba: rate_ref=347772.84 rate_cand=348567.75 rate_fb=346275.14 g=-9.360645 pred=347023.99 Δ_net=1543.76 rel=0.00444 ε_prac=3477.73 cls=UNKNOWN midpoints={'ref': 224.0, 'cand': 304.0, 'fb': 384.0} n_steps={'REF': 64, 'CAND': 96, 'FB': 64}
- 描述：ABA Δ_net 范围 [-449.44, 1543.76]，极差 1993.20 µs/step；符号不一致
- AAA（零假设组）Δ_net 范围 [-3761.77, 3370.15]，极差 7131.93 µs/step

M2 gate 模式运行（机制阶段，单列，不与 M3 合并）：
- cal_m2_w1_norm: Δ_net=-4678.85 cls=favorable excess_cum=0 overshoot=0 abort_reason=None
- cal_m2_w1_abort **(MECHANISM)**: Δ_net=826.91 cls=UNKNOWN excess_cum=0 overshoot=0 abort_reason=force_abort_at
- cal_m2_w2_obs: Δ_net=-349.81 cls=UNKNOWN excess_cum=0 overshoot=0 abort_reason=None

## 8. 完整服务差（每 block：ABA − AAA，total_time_s；探测成本不扣除、全部保留在内）

| block | cond | AAA total_time_s | ABA total_time_s | 差 (s) | 差 (%) |
|---|---|---|---|---|---|
| b1 | W1 | 351.58 | 352.54 | 0.96 | 0.273% |
| b2 | W1 | 352.15 | 356.82 | 4.67 | 1.326% |
| b3 | W2 | 708.22 | 711.45 | 3.23 | 0.456% |
| b4 | W2 | 707.39 | 708.41 | 1.02 | 0.144% |

注：差值包含探测窗自身的全部成本（W32 CAND 段 + 边界开销），不做任何扣除；这是完整服务口径。

## 9. 中止/污染/异常事件全记录（保留，不筛选）

- cal_m2_w1_abort: state=DONE contaminated=False abort_reason=force_abort_at excess_cum=0 overshoot=0
  - 事件：[[17, 'risk_abort_set', 'force_abort_at excess=0 overshoot=0 ncand=1'], [18, 'abort_fallback', 'force_abort_at']]
  - lag 窗：[('CAND', 18)]
- cal_m3_b2_aba: state=DONE contaminated=False abort_reason=None excess_cum=0 overshoot=0
  - 段 flags：{'FB': ['layout_change']}

率（分母 = 全部 11 次探测运行，含 MECHANISM）：污染 0/11；中止（abort_reason 非空）1/11；终态 ABORTED 0/11。MECHANISM 运行（cal_m2_w1_abort）的中止是设计行为，单列说明，不解读为环境风险。

## 10. 新增成本（日志自成本账本 vs ε_log：边界新增中位 ≤ 1 ms/窗；每 forward 对封存参照 ≤ +10%）

M1-off 基线（4 runs）：c_window_ms 中位=2.3141；bcost_partial_us_max 中位=2285.8；t_confirm_us 中位=373.2

| label | mode | c_window_ms 中位 (Δ vs off) | bcost_partial 中位 | t_confirm 中位 | record_us 末块中位 (封存参照, Δ%) | prof_probe_cum_ms 末值 | avg_probe_us max | last_bcost_us max |
|---|---|---|---|---|---|---|---|---|
| cal_m1_w1_a | off | 1.8075 (-0.5067) | 2069.0 | 377.5 | 121.07 (115.54, 4.79%) | — | 0.00 | 25406 |
| cal_m1_w1_b | off | 2.2577 (-0.0565) | 2415.5 | 340.0 | 120.28 (115.54, 4.10%) | — | 0.00 | 19015 |
| cal_m1_w2_a | off | 2.3706 (0.0565) | 3467 | 382 | 117.15 (114.67, 2.16%) | — | 0.00 | 19960 |
| cal_m1_w2_b | off | 2.3813 (0.0671) | 2156 | 369 | 113.36 (114.67, -1.14%) | — | 0.00 | 18634 |
| cal_m2_w1_norm | gate | 2.3968 (0.0827) | 1995 | 378 | 113.68 (115.54, -1.61%) | 8.468401 | 323.73 | 19025 |
| cal_m2_w1_abort | gate | 1.3997 (-0.9144) | 2824.5 | 336.5 | 113.16 (115.54, -2.06%) | 7.248924 | 322.40 | 20050 |
| cal_m2_w2_obs | gate | 1.7784 (-0.5357) | 2136.0 | 346.0 | 112.64 (114.67, -1.77%) | 7.613663 | 325.16 | 19554 |
| cal_m3_b1_aaa | aaa | 2.0275 (-0.2867) | 1894.5 | 359.5 | 114.29 (115.54, -1.08%) | 8.474206 | 319.34 | 18769 |
| cal_m3_b1_aba | gate | 1.4798 (-0.8343) | 2903 | 348 | 113.12 (115.54, -2.09%) | 8.112017 | 325.76 | 19496 |
| cal_m3_b2_aaa | aaa | 2.6640 (0.3498) | 3132.0 | 393.0 | 113.38 (115.54, -1.87%) | 9.358411 | 334.32 | 21153 |
| cal_m3_b2_aba | gate | 2.8620 (0.5479) | 3374 | 354 | 113.59 (115.54, -1.69%) | 8.295352 | 358.62 | 19629 |
| cal_m3_b3_aaa | aaa | 1.7443 (-0.5698) | 3080 | 359 | 115.98 (114.67, 1.14%) | 8.437504 | 320.36 | 21707 |
| cal_m3_b3_aba | gate | 3.0684 (0.7543) | 3295.0 | 415.5 | 114.11 (114.67, -0.49%) | 7.131171 | 304.55 | 20074 |
| cal_m3_b4_aaa | aaa | 2.3106 (-0.0035) | 2131 | 399 | 115.19 (114.67, 0.45%) | 7.652778 | 302.95 | 19701 |
| cal_m3_b4_aba | gate | 2.1640 (-0.1501) | 2185.5 | 347.5 | 116.66 (114.67, 1.73%) | 7.383728 | 330.48 | 20461 |

ε_log 判定（描述性）：边界项 = c_window_ms 中位相对 M1-off 基线之差 ≤ +1.0 ms/窗；每 forward 项 = avg_record_us 末块 8-rank 中位对封存 <cond>_A__r1 参照变化 ≤ +10%（运行时导出参照，PRE_REG §F #5 更正）。M1 gate G4 为正式判定；此处为全部 15 run 的同法描述。prof_probe_cum_ms = 新增仪表（p2l 指纹）累计账目，如实报告不扣除。

## 11. 留空项（不支持的统计保证——明确留空，不用假设值补齐）

- **BLANK**：coverage guarantee of Delta_net (no confidence/interval claim -- withdrawn)
- **BLANK**：production admission thresholds (probe_commit=0 everywhere this campaign)
- **BLANK**：T-b production trigger specification (calibration uses deterministic T-c)
- **BLANK**：TTFT delta endpoint (not measured against a sealed TTFT budget here)
- **BLANK**：formal eps_prac justification (0.01 x rate_ref is provisional)
- **BLANK**：W64 behavior (out of v1 scope by author directive)
- **BLANK**：MDE / power analysis (no inferential claim made)

全部 eval dict 携带 `causal_isolation_claimed=False`、`coverage_guarantee=None`（撤回“完成探测自动获得条件置信”表述）。

## 12. 可辨识性陈述（规则化，不调参）

- W1: **在本样本量下无法把 ABA 估计与 AAA 零假设分布可靠区分**（ABA 分类不一致（['UNKNOWN', 'UNKNOWN']）; ABA Δ_net 区间 [-9874.58,-1846.33] 与 AAA 零假设组 [-2272.32,-169.36] 重叠）。如实报告该限制：现有校准数据不足以支持净效果方向判定；不反复调到通过。
- W2: **在本样本量下无法把 ABA 估计与 AAA 零假设分布可靠区分**（ABA 分类不一致（['UNKNOWN', 'UNKNOWN']）; ABA Δ_net 区间 [-449.44,1543.76] 与 AAA 零假设组 [-3761.77,3370.15] 重叠）。如实报告该限制：现有校准数据不足以支持净效果方向判定；不反复调到通过。

本轮成功标准 = 测量可信 + 机制正确 + 估计器能够接受检验；**不以是否出现正收益判定实验成功**。正式部署收益另行评价。M3 数据仅作开发校准：不回填封存结论、不预挑正式战役参数。

## 13. 工件索引

- analysis_calibration.json（本报告机器可读源）；m1_gate.json；m2_gate.json；各 run 目录：verdict.json / bench.json / v2p1_rank0.jsonl / probe_run_end.json / server.log / meta.txt；calib_state.tsv（预算与事件账本）；driver.log
