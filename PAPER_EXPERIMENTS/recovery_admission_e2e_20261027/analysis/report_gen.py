# report_gen.py -- render analysis/report.md from analysis/summary.json
# Usage: python3 report_gen.py   (run after analyze_rge2e.py)
# Chinese report, zero single quotes (runner-safe), facts + pre-registered stats.
import json, os
A2 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027"
S = json.load(open(A2 + "/analysis/summary.json"))
ARMS = ["I", "ES", "ED", "MT", "DF", "P0", "P1"]
NAME = {"I": "Identity", "ES": "静态EPLB", "ED": "动态EPLB", "MT": "MoETuner风格", "DF": "DataFore风格", "P0": "AERO-base", "P1": "AERO-RG"}
L = []
L.append("# recovery_admission_e2e_20261027 端到端比较报告")
L.append("")
L.append("完成度: " + S["boots_ok"] + " (目标 28/28)。本报告由 report_gen.py 从 summary.json 渲染, 全部为实测事实; 统计口径见 PROTOCOL.md 第7节 (主指标 total_time_s 全服务墙钟, 同报 tps, 按轮配对, 均值差 + 95% 区间 paired t df=3, 双侧配对 t 检验, P1 对六臂 Holm 校正)。禁止与任何历史均值拼接。")
L.append("")
L.append("## 1. 全部 boot 原始值 (主指标 total_time_s, 秒)")
L.append("")
L.append("| 臂 | R1 | R2 | R3 | R4 | 均值 |")
L.append("|---|---|---|---|---|---|")
ser = S["tables"]["per_arm_series"]
for a in ARMS:
    tt = ser[a]["total_time_s"]
    cells = ["%.1f" % v if v is not None else "--" for v in tt]
    have = [v for v in tt if v is not None]
    mean = ("%.1f" % (sum(have) / len(have))) if have else "--"
    L.append("| " + a + " " + NAME[a] + " | " + " | ".join(cells) + " | " + mean + " |")
L.append("")
L.append("## 2. 同报: 实际输出吞吐 tps (ok/total_time_s)")
L.append("")
L.append("| 臂 | R1 | R2 | R3 | R4 |")
L.append("|---|---|---|---|---|")
for a in ARMS:
    tp = ser[a]["tps"]
    L.append("| " + a + " | " + " | ".join(["%.1f" % v if v is not None else "--" for v in tp]) + " |")
L.append("")
L.append("## 3. 配对统计 (P1 对各臂, 差值 = P1 - X, 负值表示 P1 更快)")
L.append("")
L.append("| 对比 | n对 | 均值差(s) | 95%CI | t | p(双侧) | p(Holm) | 含义 |")
L.append("|---|---|---|---|---|---|---|---|")
MEAN = {"P1_vs_P0": "机制效应 (恢复准入)", "P1_vs_I": "全系统 vs Identity", "P1_vs_ES": "全系统 vs 静态EPLB",
        "P1_vs_ED": "全系统 vs 动态EPLB", "P1_vs_MT": "全系统 vs MoETuner风格", "P1_vs_DF": "全系统 vs DataFore风格"}
holm = S["stats"].get("holm_adjusted_p") or {}
for k in ["P1_vs_P0", "P1_vs_I", "P1_vs_ES", "P1_vs_ED", "P1_vs_MT", "P1_vs_DF"]:
    c = S["stats"]["comparisons"][k]
    md = ("%.2f" % c["mean_diff"]) if "mean_diff" in c else "--"
    ci = ("[%.2f, %.2f]" % (c["ci95"][0], c["ci95"][1])) if "ci95" in c else ("n<4" if c["n_pairs"] else "--")
    t = ("%.3f" % c["t_stat"]) if c.get("t_stat") is not None else "--"
    p = ("%.4f" % c["p_two_sided"]) if c.get("p_two_sided") is not None else "--"
    ph = ("%.4f" % holm[k]) if k in holm else "--"
    L.append("| " + k.replace("_vs_", " vs ") + " | " + str(c["n_pairs"]) + " | " + md + " | " + ci + " | " + t + " | " + p + " | " + ph + " | " + MEAN[k] + " |")
L.append("")
L.append("检验方法(命名登记): 双侧配对 t 检验 (round-paired, df=3, tcrit=3.182445); P1 对六臂 p 值统一 Holm 步进校正; scipy=" + str(S["stats"]["scipy"]) + "。")
L.append("")
L.append("## 4. 机制记录 (P0/P1 全部 boot)")
L.append("")
for a in ["P0", "P1"]:
    L.append("### " + a + " " + NAME[a])
    for r in [1, 2, 3, 4]:
        rec = S["boots"].get("rge2e_" + a + "_r" + str(r)) or {}
        if rec.get("status") != "ok":
            L.append("- r" + str(r) + ": " + rec.get("status", "missing")); continue
        m = rec.get("mech") or {}
        sre = rec.get("sp_run_end") or {}
        mig = m.get("migration") or {}
        L.append("- r" + str(r) + ": 驻留=" + json.dumps(m.get("residency_final"), ensure_ascii=False)
                 + " 采样窗口=" + str(m.get("n_rows")) + " committed=" + str(m.get("detection_cycles_committed"))
                 + " recorded_steps=" + str(sre.get("recorded_steps"))
                 + " 转移数=" + str(len(sre.get("transitions") or []))
                 + " RG注释行=" + str(m.get("rg_annotation_rows")) + " RG策略转移=" + str(m.get("rg_policy_tagged_transitions"))
                 + " 首次迁移窗口=" + str(rec.get("first_migration_window"))
                 + " 迁移批(decode_record_batches)=" + str(sre.get("decode_record_batches"))
                 + " swaps_started=" + str(mig.get("swaps_started_sum"))
                 + " issued窗口=" + json.dumps(mig.get("serverlog_issued_windows"))
                 + " planhash8秩=" + str(mig.get("planhash_8rank_identical"))
                 + " 计划ops=" + str(sre.get("nocommit_planned_ops_total")))
    L.append("")
L.append("## 5. 轮1 机制时间线案例 (预选)")
L.append("")
for a in ["P0", "P1"]:
    rec = S["boots"].get("rge2e_" + a + "_r1") or {}
    tl = rec.get("timeline") or []
    if not tl:
        L.append("### " + a + ": pending"); L.append(""); continue
    L.append("### " + a + " " + NAME[a] + " r1 逐窗口 (w=窗口, st=状态, ev=事件, tr=转移)")
    L.append("")
    L.append("| w | st_before | st_after | ev | tr | z | h_rate |")
    L.append("|---|---|---|---|---|---|---|")
    for x in tl:
        L.append("| " + str(x.get("w")) + " | " + str(x.get("st_before")) + " | " + str(x.get("st_after")) + " | " + str(x.get("ev"))
                 + " | " + ("Y" if x.get("tr") else "") + " | " + ("%.3f" % x["z"] if isinstance(x.get("z"), (int, float)) else "--")
                 + " | " + ("%.5f" % x["h_rate"] if isinstance(x.get("h_rate"), (int, float)) else "--") + " |")
    L.append("")
L.append("## 6. 基线臂机制事实 (ED 触发次数 / ES 注入 / 完整性)")
L.append("")
L.append("| boot | completeness | ED rebalance次数 | banner | 注入证据条数 |")
L.append("|---|---|---|---|---|")
for v in S["tables"]["per_boot_raw"]:
    if v["arm"] in ("ED", "ES"):
        rec = S["boots"][v["label"]]
        bf = rec.get("baseline_facts") or {}
        L.append("| " + v["label"] + " | " + str(v["completeness_pass"]) + " | " + str(bf.get("eplb_rebalance_starts")) + " | " + str(bf.get("eplb_banner")) + " | " + str(len(bf.get("injection_evidence") or [])) + " |")
L.append("")
L.append("## 7. 数据完整性")
L.append("")
nok = sum(1 for v in S["tables"]["per_boot_raw"] if v["status"] == "ok" and v["completeness_pass"] is True and v["bench_rc"] == 0)
L.append("- status=ok 且 completeness_pass=True 且 bench_rc=0 的 boot 数: " + str(nok) + "/" + str(len(S["tables"]["per_boot_raw"])))
L.append("- 全部 28 boot 原始值见 analysis/per_boot_raw.tsv 与 analysis/summary.json; 逐 boot 现场在 boots/<label>/。")
open(A2 + "/analysis/report.md", "w").write("\n".join(L) + "\n")
print("REPORT_OK lines=" + str(len(L)) + " boots_ok=" + S["boots_ok"])
