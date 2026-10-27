# make_readme_xd.py -- GitHub curated README for cross_domain_e2e_20261027
import json, os, math, datetime
X = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027"
GH = "/workspace/EPLB/PAPER_EXPERIMENTS/cross_domain_e2e_20261027"
s = json.load(open(X + "/analysis/summary.json"))
assert s["boots_ok"] == "24/24", s["boots_ok"]
ser = s["tables"]["per_arm_series"]
st = s["stats"]["comparisons"]; holm = s["stats"]["holm_adjusted_p"]
def mean(a, k):
    v = [x for x in ser[a][k] if x is not None]; return sum(v) / len(v)
L = []
L.append("# cross_domain_e2e_20261027 -- 跨域切换多 baseline 端到端 (论文 §5.3.1 主比较 + §5.5 时间线案例)")
L.append("")
L.append("六臂 (I / 静态EPLB ES / 动态EPLB ED / MoETuner-style MT / DataFore-style DF / AERO-base P0) x 4 轮 = 24 boots,")
L.append("冻结 E3 四域相位切换流 (e3_multidomain_switch.jsonl sha d55fe2f7..., 8192 req = sharegpt/prover/book/code x 2048, in512/out1, conc 256), 六臂同期同窗测量, 未拼接历史数值。")
L.append("")
L.append("## 主结果 (primary = 全服务墙钟 s, n=4)")
L.append("| 臂 | wall mean | tps mean |")
L.append("|---|---|---|")
for a in ["I", "ES", "ED", "MT", "DF", "P0"]:
    L.append("| %s | %.2f | %.2f |" % (a, mean(a, "total_time_s"), mean(a, "tps")))
L.append("")
L.append("P0 对五臂轮配对 t(df=3) 双侧 + Holm x5: **全部 n.s.** (最小 raw p=0.0499 P0_vs_DF +11.78s, Holm 0.2494)。")
for k in ["P0_vs_I", "P0_vs_ES", "P0_vs_ED", "P0_vs_MT", "P0_vs_DF"]:
    v = st[k]
    L.append("- %s: diff=%+.2fs p=%.4f Holm=%.4f" % (k, v["mean_diff"], v["p_two_sided"], holm[k]))
L.append("")
L.append("## §5.5 时间线要点")
L.append("- P0: 4/4 boots 在域边界后第 1 窗 (w9/17/25, ~6s) 发起 swap; P0_r4 有 4 次 TRACK<->QUIET 迁移, w9 z=133 检测尖峰; prover 段 req/s 全臂最高 (40.3), code 段最低 (33.3, 持续迁移成本)。")
L.append("- ED 更正: 旧口径 16 次/boot 为日志行数, 实为 2 独立事件 x 8 DP rank (period256 x 512 步); ev1 恰在边界2 (±0.8s), ev2 落流末; prover 段 ttft p95 13.26s 边界停顿; wall 最慢 208.4s。")
L.append("- 静态臂 (ES/MT/DF) 与 I: runtime rebalance 行 = 0, 布局 t=0 冻结, 相位 ttft p95 平稳。")
L.append("")
L.append("## Limitations (必带)")
L.append("1. O1 prefill-only: tps=req/s, 差距结构性压缩 (~±5-7%), 历史同流旧战役结论一致 (E1/E3 四臂互不显著)。")
L.append("2. 混合域摊平倾斜 (layer0 2.379 vs prover 单域 4.093), 每相位 ~48s, 在线适应回报小。")
L.append("3. 静态臂画像录自评测流本身 (oracle 校准, 作者设计, 偏向静态臂)。")
L.append("4. ES/ED 被迫 deepep normal (官方路径 auto 下 NotImplementedError), 与其余四臂 a2a 口径不同, 机制归因需带此混杂。")
L.append("")
L.append("## 目录")
L.append("- PROTOCOL.md (+AMENDMENT-1) / LEDGER_xd.tsv / ORDER_TABLE.json -- 预注册、台账、轮转顺序")
L.append("- analysis/ -- analyze_xd.py, timeline_xd.py, report_gen_xd.py, summary.json, per_boot_raw.tsv, timeline_5_5.{json,md}, costs_profile_solve.md, REPORT_xd.md")
L.append("- scripts/ -- 启动器、boot runner、mech 检查、队列、录制与看门狗脚本")
L.append("- code_lock/placement_shas.env + placements/ -- 布局文件与只读 sha 锁")
L.append("- notes/E3_historical_crosscheck.md -- 作者质询核对备忘 (历史同流交叉验证)")
L.append("- boots_curated/ -- 24 boots 小件 (bench json/MECH_CHECK/boot_meta/bench.log; P0 含 v2p1+sp_run_end)")
L.append("- prep_curated/ -- 录制 boot 证据 (不含 48MB .pt) + profile_build (npz/ILP 日志/sanity)")
L.append("")
L.append("全档 (含 server.log、recorder .pt 48MB、bench perreq): /data/minghua/sjq/PAPER_EXPERIMENTS/cross_domain_e2e_20261027/ (清单 ARCHIVE_SHA256SUMS.txt)。")
L.append("原始来源: /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/cross_domain_e2e_20261027/")
open(GH + "/README.md", "w").write("\n".join(L) + "\n")
print("README_OK lines=" + str(len(L)))
