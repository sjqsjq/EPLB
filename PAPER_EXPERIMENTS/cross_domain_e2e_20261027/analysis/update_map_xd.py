# update_map_xd.py -- append cross_domain_e2e_20261027 row + summary to EXPERIMENT_MAP.md (idempotent)
import json, sys
X = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027"
MAP = "/workspace/EPLB/PAPER_EXPERIMENTS/EXPERIMENT_MAP.md"
S = json.load(open(X + "/analysis/summary.json"))
assert S["boots_ok"] == "24/24", S["boots_ok"]
st = S["stats"]["comparisons"]; holm = S["stats"]["holm_adjusted_p"]
ser = S["tables"]["per_arm_series"]
def mean(a):
    v = [x for x in ser[a]["total_time_s"] if x is not None]; return sum(v) / len(v)
txt = open(MAP).read()
if "cross_domain_e2e_20261027" in txt:
    print("MAP_ALREADY_HAS_ROW skip"); sys.exit(0)
ROW = "| cross_domain_e2e_20261027(跨域切换多baseline端到端: 六臂x4轮 E3四域相位流) | 第 5.3.1 节「跨域主比较」;第 5.5 节「域边界时间线案例」 | /data/minghua/sjq/PAPER_EXPERIMENTS/cross_domain_e2e_20261027/(全档: PROTOCOL.md+AMENDMENT-1、LEDGER_xd.tsv、analysis/REPORT_xd.md、timeline_5_5.{json,md}、boots/ 逐boot现场、prep/ 录制与画像、ARCHIVE_SHA256SUMS.txt);GitHub 精选档 PAPER_EXPERIMENTS/cross_domain_e2e_20261027/(README.md、协议、顺序表、统计、代码锁、boots_curated 小件) | /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027/(工具:scripts/ 与 analysis/;流:final_plan_20261019/B_data/streams/e3_multidomain_switch.jsonl) |\n"
SUM = ("\n\n结果概括(cross_domain_e2e_20261027): 主指标全服务墙钟, 同报 tps(=req/s, O1流); 24/24 boots ok=8192/0err, 轮配对 t(df=3) 双侧, P0 对五臂 Holm x5。"
       "六臂均值墙钟: I=%.1fs, ES=%.1fs, ED=%.1fs, MT=%.1fs, DF=%.1fs, P0=%.1fs。" % (mean("I"), mean("ES"), mean("ED"), mean("MT"), mean("DF"), mean("P0")) +
       " Holm 后六臂无显著差异 (最小 raw p=0.0499 P0_vs_DF +11.78s, Holm 0.2494); 与历史同流结论一致 (prefill 主导跨域流上放置策略收益有限)。"
       " §5.5 机制: P0 在三个域边界后第 1 窗 (~6s) 均发起 swap (4/4 boots), prover 段 req/s 全臂最高 40.3, code 段 33.3 最低 (持续迁移成本); ED 事实更正: 2 独立重平衡事件/boot (16 行=2事件x8rank), ev1 恰落边界2 ±0.8s, ev2 流末, prover 段 ttft p95 13.26s 停顿, wall 最慢 208.4s; 静态臂布局 t=0 冻结 runtime rebalance=0。"
       " limitations 必带: O1 tps=req/s 差距结构性压缩; 混合域摊平倾斜 2.379 vs 4.093; 静态臂 oracle 校准偏向; ES/ED deepep normal 混杂。详见 analysis/REPORT_xd.md 与 notes/E3_historical_crosscheck.md。\n")
lines = txt.split("\n")
out = []; inserted = False
for i in range(len(lines) - 1, -1, -1):
    if lines[i].startswith("| recovery_admission_e2e_20261027"):
        lines.insert(i + 1, ROW.rstrip("\n")); inserted = True; break
assert inserted, "anchor row not found"
open(MAP, "w").write("\n".join(lines) + SUM)
print("MAP_UPDATED row+summary appended")
