# update_map.py -- append recovery_admission_e2e_20261027 row + summary to EXPERIMENT_MAP.md
# Run AFTER final analyze_rge2e.py --final + report_gen.py. Idempotence guard: skip if row exists.
import json
A2 = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027"
MAP = "/workspace/EPLB/PAPER_EXPERIMENTS/EXPERIMENT_MAP.md"
S = json.load(open(A2 + "/analysis/summary.json"))
assert S["boots_ok"] == "28/28", S["boots_ok"]
st = S["stats"]["comparisons"]
def fmt(k):
    c = st[k]
    md = c.get("mean_diff"); ci = c.get("ci95"); p = c.get("p_two_sided")
    ph = (S["stats"].get("holm_adjusted_p") or {}).get(k)
    s = k.replace("_vs_", "-") + ": 差=" + ("%.2fs" % md if md is not None else "--")
    if ci: s += " CI95[%.2f,%.2f]" % (ci[0], ci[1])
    if p is not None: s += " p=%.4f" % p
    if ph is not None: s += " Holm=%.4f" % ph
    return s
ser = S["tables"]["per_arm_series"]
def mean(a):
    v = [x for x in ser[a]["total_time_s"] if x is not None]
    return sum(v) / len(v)
txt = open(MAP).read()
if "recovery_admission_e2e_20261027" in txt:
    print("MAP_ALREADY_HAS_ROW skip"); raise SystemExit
ROW = "| recovery_admission_e2e_20261027(恢复准入策略 AERO-RG + 七臂多baseline端到端比较) | 第 5.3.1 节「多baseline端到端比较」;第 5.4 节「恢复准入新旧对照」 | /data/minghua/sjq/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/(全档: PROTOCOL.md、LEDGER_rge2e.tsv、analysis/report.md、boots/ 逐boot现场、ARCHIVE_SHA256SUMS.txt);GitHub 精选档 PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/(README.md、协议、顺序表、统计、代码锁与79行diff、boots_curated 小件) | /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027/(工具:scripts/ 与 analysis/;流:同级 streams/ 及 /data/minghua/sjq/OEPLBdata/datasets/single_domain/) |\n"
lines = txt.split("\n")
out = []
inserted = False
for l in lines:
    out.append(l)
    if (not inserted) and l.startswith("| adaptive_prover_sd_20261026"):
        out.append(ROW.rstrip("\n")); inserted = True
assert inserted
SUM = "\n结果概括(recovery_admission_e2e_20261027): 主指标为全服务墙钟 total_time_s(保留全部在线成本), 同报 tps; 4轮x7臂=28 boots 全部 ok=8192、按轮配对、双侧配对t(df=3)、P1对六臂Holm校正。七臂均值墙钟: " + ", ".join(a + "=" + ("%.1fs" % mean(a)) for a in ["I", "ES", "ED", "MT", "DF", "P0", "P1"]) + "。机制效应(P1-P0): " + fmt("P1_vs_P0") + "。全系统对比: " + "; ".join(fmt(k) for k in ["P1_vs_I", "P1_vs_ES", "P1_vs_ED", "P1_vs_MT", "P1_vs_DF"]) + "。机制记录: P0 各boot驻留 TRACK/QUIET/RECOVER=12/2/4 模式稳定重现(7转移窗口 6,6,8,11,11,13,16), P1 驻留 6/10/0(仅 w6 TRACK->QUIET, 孤立confirmed_grow注释行3-4条, 策略标签转移1), 8秩planhash全一致; ED 实际rebalance次数见 analysis/report.md 第6节。详见 analysis/report.md 与 PROTOCOL.md 第7节。"
out.append(SUM)
open(MAP, "w").write("\n".join(out))
print("MAP_UPDATED rows_now=" + str(sum(1 for l in out if l.startswith("| "))))
