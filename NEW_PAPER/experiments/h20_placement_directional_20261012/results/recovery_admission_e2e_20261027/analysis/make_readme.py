# make_readme.py -- render GitHub-side archive README.md (Chinese, zero single quotes)
import json, os
A2 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027"
GH = "/workspace/EPLB/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027"
S = json.load(open(A2 + "/analysis/summary.json"))
st = S["stats"]["comparisons"]; holm = S["stats"].get("holm_adjusted_p") or {}
ser = S["tables"]["per_arm_series"]
NAME = {"I": "Identity", "ES": "静态EPLB", "ED": "动态EPLB(auto/16/256, E*=e5冻结)", "MT": "MoETuner风格(Gurobi ILP1)", "DF": "DataFore风格(贪心重映射)", "P0": "AERO-base(现有版本)", "P1": "AERO-RG(新版恢复准入)"}
def mean(a):
    v = [x for x in ser[a]["total_time_s"] if x is not None]
    return (sum(v) / len(v)) if v else None
L = []
L.append("# recovery_admission_e2e_20261027 归档 README")
L.append("")
L.append("恢复准入策略修改(AERO-RG) + 七臂多baseline端到端比较。本目录为 GitHub 精选档; 全档(含逐boot server.log/perreq 与流文件)在 /data/minghua/sjq/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/(清单 ARCHIVE_SHA256SUMS.txt); 工作现场 /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027/。")
L.append("")
L.append("## 结论速览 (完成度 " + S["boots_ok"] + ")")
L.append("")
L.append("- 主指标: 全服务墙钟 total_time_s (保留全部在线成本); 同报 tps。4轮x7臂, 同一冻结流 prover_256tok_out1_x4 (8192请求, sha b312ba50...), O=1, conc=256, 统一 --disable-cuda-graph, mem 0.8。")
for a in ["I", "ES", "ED", "MT", "DF", "P0", "P1"]:
    m = mean(a)
    L.append("- " + a + " " + NAME[a] + ": 均值墙钟 " + ("%.1fs" % m if m else "--") + " 逐轮 " + str(["%.1f" % v if v is not None else None for v in ser[a]["total_time_s"]]))
L.append("")
L.append("## 配对统计 (差值 = P1 - X, 负值表示 P1 更快; 双侧配对t df=3; Holm 校正)")
L.append("")
for k in ["P1_vs_P0", "P1_vs_I", "P1_vs_ES", "P1_vs_ED", "P1_vs_MT", "P1_vs_DF"]:
    c = st[k]
    L.append("- " + k.replace("_vs_", " vs ") + ": n=" + str(c["n_pairs"])
             + " 均值差=" + ("%.2fs" % c["mean_diff"] if "mean_diff" in c else "--")
             + " CI95=" + (("[%.2f, %.2f]" % (c["ci95"][0], c["ci95"][1])) if "ci95" in c else "--")
             + " p=" + ("%.4f" % c["p_two_sided"] if c.get("p_two_sided") is not None else "--")
             + " Holm=" + ("%.4f" % holm[k] if k in holm else "--"))
L.append("")
L.append("## 机制对照 (恢复准入)")
L.append("")
L.append("- P0 AERO-base: 驻留 TRACK/QUIET/RECOVER=12/2/4 模式跨boot稳定, 7转移(窗口6,6,8,11,11,13,16), 其中 w6/w11 的 QUIET->RECOVER 由孤立 confirmed_grow 触发(P1c)。")
L.append("- P1 AERO-RG: 唯一改动=孤立 confirmed_grow 不再作为变化证据; 实测驻留 6/10/0 RECOVER, 仅 w6 TRACK->QUIET, [AERO-RG] 注释行3-4/boot, 策略标签转移1, 8秩 planhash 全一致; 离线: 30/30分支测试 + apsd2_P_b1 回放(orig精确/new0逐字节同/rg1反事实与实测吻合)。")
L.append("- 基线臂事实: ED 每boot实际 rebalance 次数与 ES 注入证据见 analysis/report.md 第6节。")
L.append("")
L.append("## 目录")
L.append("")
L.append("- PROTOCOL.md 预注册协议(0-9节) / LEDGER_rge2e.tsv append-only 治理日志")
L.append("- code_lock/: sp_state.py.orig_ba572f93(原版备份) + AERO_RG.patch.diff(79行权威diff)")
L.append("- order/: 锁定顺序表(种子20261027, 轮1-3循环移位, 轮4反转)")
L.append("- scripts/: 七臂启动器+runner+mech_check+三个queue+归档/git脚本(均sha登记于LEDGER)")
L.append("- tests_rgv/: 离线验证(30分支检查+回放) / placements/: 三份sha锁定静态素材")
L.append("- analysis/: analyze+report 渲染脚本, summary.json, report.md(28原始值+全部统计), per_boot_raw.tsv")
L.append("- boots_curated/: 31 boots 小件(bench json, MECH_CHECK, boot_meta, sp_run_end, v2p1, bench.log); server.log/perreq 大件在 /data 全档")
open(GH + "/README.md", "w").write("\n".join(L) + "\n")
print("README_OK lines=" + str(len(L)))
