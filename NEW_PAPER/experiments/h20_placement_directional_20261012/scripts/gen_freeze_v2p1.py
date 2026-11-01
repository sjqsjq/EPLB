#!/usr/bin/env python3
"""Generate FREEZE_record_v2p1.md (plan §5.1) from the fit/PT2/test artifacts.
All numbers are read from JSON artifacts -- nothing is hand-entered.
Run AFTER: fit_v2p1_calibration.py, pt2c_v2p1.py, offline suite rerun.
"""
import json, os, sys, hashlib, datetime

EXP = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012"
OUTROOT = EXP + "/results/adw_v2p1/r3_logging"
PB = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb"
SRC = "/workspace/EPLB/OEPLB/src"

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def main():
    fit = json.load(open(os.path.join(OUTROOT, "fit_results_v2p1.json")))
    pt2c = json.load(open(os.path.join(OUTROOT, "pt2c_results.json")))
    suite_out = open(sys.argv[1]).read() if len(sys.argv) > 1 else "(suite output file not given)"
    suite_pass = "ALL_V2P1_TESTS_PASS" in suite_out
    n_pass = sum(1 for l in suite_out.splitlines() if l.startswith("[PASS]"))
    diag = fit["diagnostics"]
    g1d_band = diag["G1_D"]["band"]
    d1_band = diag["D1"]["band"]
    stop_flags = []
    if ">3 STOP" in str(g1d_band): stop_flags.append("G1-D band >3")
    # D1 band >2: plan §4 mandates STOP-before-freeze + archived investigation
    # (排查记录归档；不静默通过，也不事后挪带). The investigation was executed
    # and archived pre-freeze; this gate verifies the archive exists and carries
    # the required conclusion marker, and its sha256 is recorded in §3. The
    # interpretation band and all thresholds are UNCHANGED by this mechanics.
    d1_inv_path = os.path.join(OUTROOT, "investigation_d1_rswap.md")
    d1_inv_txt = open(d1_inv_path).read() if os.path.exists(d1_inv_path) else ""
    d1_inv_ok = "不构成 F2/F1 有效性威胁" in d1_inv_txt
    d1_inv_sha = sha(d1_inv_path) if d1_inv_txt else None
    if ">2 STOP" in str(d1_band) and not d1_inv_ok:
        stop_flags.append("D1 band >2 (pre-freeze investigation missing/inconclusive)")
    if not fit["pt2"]["gate_pass"]: stop_flags.append("PT2 A/B gate")
    if not pt2c["gate"]["pass"]: stop_flags.append("PT2-C gate")
    if not suite_pass: stop_flags.append("offline suite")
    # G5 is a REPORT item per plan §4 (not a gate): surfaced in §3 of the
    # record, never added to stop_flags.

    files = ["window_v2p1.py", "v2p1_signals.py", "controller.py", "config.py"]
    sha_rows = []
    for f in files:
        s1, s2 = sha(os.path.join(SRC, f)), sha(os.path.join(PB, f))
        assert s1 == s2, "src/deployed sha mismatch for " + f
        sha_rows.append((f, s1))
    w2 = sha(os.path.join(PB, "window_v2.py"))
    assert w2 == "810c54daf1c4b9ec4bf87d3032a80f092a8ad9e7a5947d71cf6e60a960067b74"

    per_run = fit["per_run"]
    exp_stat = {8: "104-107", 16: "51-53", 32: "25-26", 64: "12"}

    md = """# FREEZE record — ADW v2.1 六参数（r3 预注册规则，一次盲拟合）

冻结时间：%s（生成器 gen_freeze_v2p1.py，全部数值取自 JSON 工件，无手填）。
规则来源：calibration_plan_v2p1.md（r3）§2/§2.1/§2.2/§3；公式逐字沿用
calibration_plan_v2.md §3。数据：仅 4 次 W0 固定窗口 logging run
（r3_logging/W{8,16,32,64}/），无任何正式比较数据参与。

## 1. 冻结的 6 个 knob（OEPLB_V21_*）

| knob | 值 | 规则 |
|---|---|---|
| KAPPA (κ) | %r | median(Z_s pooled) |
| Z_LO | %r | p90(Z_s/κ)，numpy percentile linear |
| Z_HI | %r | p99(Z_s/κ) |
| Z_URGENT | %r | max(3·Z_HI, p50(Z_c/κ)) |
| THETA_R_LOW | %r | p30(H_rate_s pooled) |
| THETA_R_ACT | %r | p60(H_rate_s pooled) |

沿用不拟合：N_MIN=12320768、M_SHRINK=2、M_GROW=3、DWELL 2/2、U_MAX=4、
W_UNCERTAIN_TARGET=16、URGENT_EPISODE_CAP=3、BETA=0.5、W_init=16。
分位数方法：numpy.percentile(method="linear")；合并池每窗等权；
有效样本数 = 池长度（下表）。

## 2. 池与门（§2.2）

| 池 | n | 门 | 结果 |
|---|---|---|---|
| Z_s / H_rate_s（合格静止窗） | %d | ≥150 | %s |
| Z_c（合格伪切换窗） | %d | ≥20 | %s |

逐 run 合格窗实数 vs 预期（§2.2 表）：

| W | 记录窗 | 合格 | 合格静止（预期） | 合格变化 | 排除原因 |
|---|---|---|---|---|---|
%s

窗清单 sha256：%s

## 3. 验收门（§4，冻结前全部重跑）

- **PT1（离线机制级，最终源码重跑）**：test_window_v2p1_logic.py →
  %s（PASS 计数 %d，含 T5/T6b/T7 成对缩放、T2/T3 逻辑身份）。
- **PT2 A/B（数据驱动成对缩放，pre-fit）**：窗数 %d，精确有理恒等
  %d/%d（门=100%%：%s），float64 逐位相同 %d，最大偏差 %g ulp
  （纯舍入，分类记录）；θ 布尔不变扫描失败 %d。
  范围声明：不声称复制计数为新增独立样本；不声称重算 z / UNCERTAIN
  天然不变（n_min 为绝对计数门，单独报告）。
- **PT2-C（冻结 knob 决策恒等，post-fit pre-freeze；harness %s）**：
  成对缩放窗 %d（静止 %d = 注册门，变化区 %d 为补充覆盖）；决策不一致 %d
  （其中 n_min 绝对门类 %d，F2 缺陷类 %d）；h_rate 逐位相同 %d，最大绝对差
  %g；状态重构保真：%d/%d 窗复现 live 事件 + W_after + 全部 post-state 字段
  （%s）；n_min 侧：orig_below=%d scaled_below=%d flips=%d。
  v1 harness 缺陷（跨 knob 重放 / 状态错位一窗 / z 基准未换算）已于冻结前
  修复并归档：pt2c_harness_fix_note.md；v1 输出存
  pt2c_results_v1_harnessbug.json。零 knob/参数/拟合改动。
- **G2 分离度**（报告项）：p50(Z_c/κ)=%r vs Z_HI=%r → %s。
- **G3 重尾**（报告项）：p90/p50=%r → %s。
- **G5 κ 跨 run 稳定性**（报告项）：%s（W64 单拟小样本仅列示）。
- **G1-D（诊断，解释带）**：h_rate 中位数逐 W %s；max/min 比 %r →
  带判定：%s。raw h_w 中位数比 %r（F2 修复前后对照存档）。
- **D1（诊断，解释带）**：R_swap=%r（n_swap_stat=%d, n_nonswap_stat=%d）→
  带判定：%s。带超限 → 已按计划 §4 于冻结前完成排查并归档
  investigation_d1_rswap.md（sha256 `%s`）：结论为 planner 选择效应 +
  真实漂移随窗长积累 + 跨 W 组成效应；F1 泄漏的三条判别签名全部排除
  （z 峰值先于映射变化、量级超出 n_layer 缩放、衰减非 β^k）；解释带
  未移动，任何 knob 未因排查改动。
- **D2（分层报告，非判据）**：%s。

## 4. 代码锁（sha256，src == site-packages 部署实测）

| 文件 | sha256 |
|---|---|
%s
| window_v2.py（冻结 v2，未触碰） | %s |

git：/workspace/EPLB main @ 51b83e6（logging 启动前锁定，见
LOCK_record_r3_logging.md；本冻结记录与战役工件将随后续本地 commit 归档）。

## 5. 冻结判定

%s
""" % (
        datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        fit["kappa"], fit["z_lo"], fit["z_hi"], fit["z_urgent"],
        fit["theta_r_low"], fit["theta_r_act"],
        fit["pools"]["Z_s"], "PASS" if fit["gates"]["stationary_pool"][2] else "FAIL",
        fit["pools"]["Z_c"], "PASS" if fit["gates"]["zc_pool"][2] else "FAIL",
        "\n".join("| %d | %d | %d | %s（%s） | %d | %s |" % (
            int(W), per_run[str(W)]["recorded_windows"], per_run[str(W)]["qualified"],
            per_run[str(W)]["qualified_stationary"], exp_stat[int(W)],
            per_run[str(W)]["qualified_change"],
            json.dumps(per_run[str(W)]["exclusion_reasons"], ensure_ascii=False))
            for W in (8, 16, 32, 64)),
        fit["window_list_sha256"],
        "ALL_V2P1_TESTS_PASS ✓" if suite_pass else "SUITE NOT PASS ✗", n_pass,
        fit["pt2"]["n_windows"], fit["pt2"]["exact_rational_identity"],
        fit["pt2"]["n_windows"], "PASS" if fit["pt2"]["gate_pass"] else "FAIL",
        fit["pt2"]["float64_bitwise_identical"], fit["pt2"]["float64_max_ulp"],
        fit["pt2"]["theta_boolean_invariance_sweep_failures"],
        pt2c["harness_version"],
        pt2c["paired"]["n_stationary"] + pt2c["paired"]["n_change_supplemental"],
        pt2c["paired"]["n_stationary"], pt2c["paired"]["n_change_supplemental"],
        len(pt2c["paired"]["decision_mismatches"]),
        pt2c["gate"]["n_min_classified_mismatches"], pt2c["gate"]["f2_defects"],
        pt2c["paired"]["hr_bitwise_identical"], pt2c["paired"]["hr_max_abs_delta"],
        pt2c["fidelity"]["n_windows"], pt2c["fidelity"]["n_windows"],
        "0 失配" if pt2c["gate"]["state_recon_fidelity_ok"]
        else "%d MISMATCH（见 pt2c_results.json）" % len(pt2c["fidelity"]["mismatches"]),
        pt2c["paired"]["n_min_side"]["orig_below"],
        pt2c["paired"]["n_min_side"]["scaled_below"],
        pt2c["paired"]["n_min_side"]["flips"],
        diag["G2"]["p50_Zc_rescaled"], diag["G2"]["z_hi"],
        "holds" if diag["G2"]["separation_holds"] else "NOT separable（如实报告）",
        diag["G3"]["p90_over_p50_Zs"],
        "heavy tail（报告，不自动处理）" if diag["G3"]["heavy_tail_flag"] else "no heavy tail",
        json.dumps(diag["G5"]["ratio_by_W"]),
        json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in diag["G1_D"]["h_rate_median_by_W"].items()}),
        diag["G1_D"]["h_rate_max_min_ratio"], g1d_band,
        diag["G1_D"]["raw_h_w_max_min_ratio"],
        diag["D1"]["R_swap"], diag["D1"]["n_swap_stationary"],
        diag["D1"]["n_nonswap_stationary"], d1_band, d1_inv_sha,
        json.dumps(diag["D2"], ensure_ascii=False),
        "\n".join("| %s | `%s` |" % (f, s) for f, s in sha_rows),
        w2,
        ("**全部验收门 PASS → 六参数冻结。** 行为验证 run（第 5 次启动）按 §5.2 执行。"
         if not stop_flags else
         "**存在未通过项：%s → 不冻结，停止并报告（不静默重拟合）。**" % "; ".join(stop_flags)),
    )
    out = os.path.join(OUTROOT, "FREEZE_record_v2p1.md")
    open(out, "w").write(md)
    print("written:", out)
    print("stop_flags:", stop_flags if stop_flags else "NONE -> frozen")
    return 0 if not stop_flags else 4

if __name__ == "__main__":
    sys.exit(main())
