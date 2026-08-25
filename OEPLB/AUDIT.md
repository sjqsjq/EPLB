# AUDIT — 证据链审计合集（4轮，2026-08-12/13）

> 4轮审计(AUDIT_findings/AUDIT2/AUDIT3/AUDIT_FINAL)合并。历史记录：审查论文数字与原始数据的一致性，发现的CRITICAL/MAJOR/MINOR问题已陆续修复。保留作审稿历史。

---

# Audit of PAPER_zh.md / PAPER_en.md — evidence-chain review

Date: 2026-08-12 (audit window ~11:00–12:00 UTC).
Method: full read of both papers; every load-bearing number traced to /workspace/logs/ (server logs, driver logs, benchmarks/results/*.json), repro/ scripts + EXPERIMENT_NOTES.md, /workspace/logs/NOTES.md, COMPREHENSIVE_EXPERIMENT_LOG.md, WORKLOAD_GRID_REPORT.md, PAPER.md (old draft), and on-disk datasets; key fits re-computed independently from raw data.

**⚠ Meta-warning: the paper files were being edited DURING this audit** (PAPER_en.md mtime 11:15, PAPER_zh.md mtime 11:35). Several fixes landed mid-audit (zh §1 retraction note, zh G.3 rewrite, zh §2.5 open-item rewrite). Findings below are against the file state at ~11:40. The English version currently LAGS the Chinese fixes (see M14). Line numbers will drift; quotes are exact.

Verified infrastructure facts used below:
- driver28.sh (30B T(r) sweep) was started 11:08 today; at audit time round 1 is partially complete, **no fit/results exist**.
- /tmp/exp_data/ is gone (confirmed); the 30B/57B/ShareGPT historical runs all reference it (see result JSONs).
- driver27/28/29.sh live in /workspace/logs/, not in repro/; repro/README.md indexes only driver12–26.
- Reproducible re-measurement datasets (*_realprover_*, multidomain_v2_out1, sharegpt_natural_20k) exist ✓.

---

## CRITICAL

### C1. The EPLB baseline comparison is self-contradictory; the paper presents a number it itself declares unreproducible
- zh L14 (abstract): “相比基线提升吞吐+17.5%（复测均值，n=2），**相比EPLB高出15.7个百分点**” — built on the re-test pair OEPLB +17.5% / EPLB +1.75% (raw data verified: L512_{baseline,eplb,oeplb}_r{1,2}.json, Aug 11; EPLB recomputes to +1.8%, fine).
- zh §5.3 表1 / §5.5 表6 (L~422, L~492): “EPLB（连续） | 21992.2 | **+9.0%**” — historical, presented with no caveat.
- zh L186 (§2.4): “实测EPLB为+1.75%（L512，2轮独立重启，**见§5.3**）…而非早期草稿中‘净上界17.0%、实测+9.0%’的版本（**两个数均无法复现**）。”
  Problems: (i) §5.3 contains no +1.75% — the pointer is wrong (type d); (ii) the paper states +9.0% is unreproducible, yet 表1/表6 still display it as a result (type f); (iii) the 5× discrepancy between historical EPLB +9.0% and re-test +1.75% is never explained anywhere. The headline “margin over EPLB” is 15.7 pp in the abstract and 9.4 pp in §5.5 — two different stories sold simultaneously. (types c, d, f) — also en L14/L256/L584.

### C2. All 30B “β<0 / negative upper bound” claims are currently UNMEASURED, yet are counted as validated predictions
- zh L141–142 (§2.4 table): “30B L512（4卡） | 1.70 | 未测* | 0.388 | **<0** | 负** | -2.6%”; footnote L149: “**30B的dispatch占比39%（vs 235B的10%），β_dispatch f_dispatch的负项超过combine改善→f_sens<0→上界为负**”.
- zh L174 (§2.4 three dimensions): “dispatch主导的模型（如30B）→ β<0，即使r_b很高也为负”.
- zh L755 (D.3): “4卡30B | 1.70 | 未测 | 0.388 | **<0**（dispatch占比39%） | 负 | … | -3.9% ✅” and L758: “修正后的模型对五个配置的收益**符号全部预测正确**”.
- zh L591 (§8): “32专家/卡时负收益（−2.6%~−3.9%），**与理论预测的符号一致**”.
Status: the T(r) sweep that would measure β(30B) is driver28.sh, started 11:08 today with **no results** (log shows round 1 still incomplete). Until it finishes, “f_sens<0” is an extrapolation from the nsys β-decomposition that the paper itself **falsified** for 235B (§2.4/G.2(h): 0.384 vs 0.496). Supporting numbers are also shaky: “39%” reproduces only as dispatch/kernel = 2997/7674 in Table 8, but with the same denominator 235B gives 7323/25546 = 28.7%, not “10%” — “10%” cannot be reproduced from any table in the paper (type a). The −2.6%/−3.9% throughput measurements themselves exist (qwen30b_ab3_*.json) but were run on /tmp/exp_data/* (verified in the JSONs) — so even the measured negative gains are on datasets that no longer exist, and Table 4 carries no reproducibility caveat (unlike Table 3’s †). (types a, b, e, d)

### C3. The 30B row’s r_before = 1.70 contradicts the paper’s own counts dump and its own avg/max rule
- zh L141/L755 use r_before = 1.70 for 30B (x_eff→0.388).
- The only 30B routing dump in existence, /workspace/logs/counts30b.json (recorded for driver28, dataset L512_O1_realprover_n8192), gives via the paper’s own r_avg.py: **identity r_avg = 1.338**, per-layer max = 1.723. The old draft (PAPER.md L579) also lists 30B at 1.34.
- 1.70 is recognizable as the per-layer **max**, i.e. exactly the avg/max conflation the paper condemns in D.1/D.2 (“两者不可混用：时间模型用avg”; the retracted “草稿把avg与max并列” episode). Every quantity derived from it (x_eff=0.388, “r_b很高也为负” narrative) inherits the inflation. (type c)

---

## MAJOR

### M1. §3.5 claims Appendix H “verifies” results that do not exist (experiment not run)
- zh L325: “§附录H用235B实验验证M确为充分统计量（不同(W,α)但相同M的吞吐落在同一条曲线上）”; zh L343 / en L343: “附录H用235B分段负载实验**验证**M★的闭式预测，**并对比**自适应与‘每段用最优M’的oracle”.
- Appendix H itself (zh L~1085–1090 / en L1088): “（driver29.sh，**进行中**）… *结果待driver29.sh完成后填入。* 本附录当前给出设计与预测”. No driver29.log exists; nothing was measured. “Verifies/compares” must be “will test”. (type d, also f)

### M2. Decision-count story contradicts the cited driver27 logs and itself (72 vs 21 vs 15.5; η attribution wrong)
- zh L288 (§3.3): “driver27.sh的g8_base臂，**21次决策、343个op**”.
- zh L299 (§3.3) & L780 (D.3): “决策数从**72**降到8，η从26%升到100%”; “8卡57B…做了**72次决策/轮**”.
- zh L768 (D.3 gate table): “57B/EP8 L256 | **15.5** | 3.68 | 2.92 | 1.26 | 29%”.
- Actual server57b_g8_base_r1.log: **9** decision windows issued swaps (139+25+15+15+14+17+16+12+8 = 261 ops/rank; blocking 3942ms r1 / 4436ms r2). driver27.log’s “decisions=72” = 9 windows × 8 ranks of log lines — an 8× overcount; 21 and 15.5 have no trace anywhere. So the causal narrative “72 decisions/round, anomalously many” is quantitatively wrong.
- Same passage misattributes η: D.3’s own A/B table shows threshold=r_k alone gives η=**59%** (+2.26%); **100% (+3.81%) requires the budget knob too** — yet L299 and §8 (L591, “启用…死区感知阈值后…从29%跃升到100%”) credit the threshold alone. (Arm gains themselves verified from _d27_g8_* jsons: +0.98/+2.26/+3.81/+2.20 ✓; the 3.94s swap blocking ✓ = r1 log.) (type c)

### M3. 4-GPU 57B L256 row still “待测” in three places although it has been measured (+2.70%, η=105%)
- zh L140 (§2.4), L754 (D.3), L902 (F.5): “57B L256（4卡）… 2.29% | **待测†**”.
- §5.4 Table 3 (L~448) reports the measurement: +2.70%, upper bound 2.57%, η=105% (verified from _d18_oe_r{1,2}.json + G-sweep identity 139.06s). Worse, the stale rows’ bound 2.29% is **below** the measured gain (+2.70% = 118% of it); only §5.4 uses the recomputed bound (r_after measured 1.011, not assumed 1.04). Stale rows make the model look falsified by its own data. (types c, f)

### M4. Two incompatible multi-domain (235B) comparisons coexist without reconciliation
- zh L47 (§1 contrib. 4): “多域负载**+10.6%**（vs EPLB **+6.3%**)”.
- zh L137 (§2.4) and 表6 (L~493): multi-domain **+14.0%**, EPLB **+12.0%**, η=118%⚠.
- Sources: +10.6/+6.3 = COMPREHENSIVE_EXPERIMENT_LOG “全场景最终汇总表” (22611/21713/20436); +14.0/+12.0 = old-draft PAPER.md Table 3 (23372.2/22941.7/20493.4), a measurement round absent from the experiment log. The same log also contains rounds where EPLB beats OEPLB on multi-domain (+8.1% vs +7.8%, decay-final table) and +9.1%/+7.3%. The paper never says which multi-domain measurement is canonical. (types c, a)

### M5. Re-measured ShareGPT gain: −0.15% (F.2) contradicts +0.14% (§5.4) in the same paper
- zh L870 (F.2): “重测得到的收益为+2.70%/+2.39%/−0.24%/−0.15%（§5.4表3下方）”; en L865 identical.
- §5.4 Table 3 (L~451): ShareGPT **+0.14%** (verified from _d22_share_* jsons: mean 4665.4→4671.9). −0.15% is round-1 only (4651.5/4658.5). The F.2 sentence cites its own §5.4 table and contradicts it, and cherry-picks one round. (type c)

### M6. §3.6 “四个数据集…吻合到≤1%” contradicts D.2’s ShareGPT result (97% discrepancy)
- zh §3.6 (L~349): “附录D.2中四个数据集的r_before全部用纯prefill录制的计数算出，与DIAG…自报值吻合到**≤1%**”.
- D.2 (zh L~712): “四次交叉校验吻合到1%以内…**但ShareGPT上两者相差97%**（DIAG首窗2.161 vs 离线1.0965)”. The ≤1% claim is false for exactly the dataset D.2 highlights as the failure case. (type c)

### M7. §5.8 reproducibility runs are mislabeled “L512_O1” — they are the multi-domain runs
- zh L552 / en L545: “8卡235B上3次独立冷启动（**L512_O1**）：22603.6 / 22885.2 / 22850.8，均值22780±156”.
- COMPREHENSIVE_EXPERIMENT_LOG §8 实验4 shows these exact three runs are on the **multi-domain 16K** dataset (baseline 20714.8, “+10.0%±0.7%”). They cannot be L512_O1: 表1’s L512_O1 OEPLB value is 23870.5, ~7σ away from 22780±156. (type d/c)

### M8. Observation 1’s cosine numbers have no measurement trail; the one logged cross-domain cos_sim contradicts “0.16”
- zh L88: “域内…余弦相似度**>0.95**…（235B上1.20→1.39；**57B上1.03→1.07**）。**跨域余弦相似度仅0.16**”.
- The only logged cross-domain cos_sim measurement is **0.8603** (COMPREHENSIVE_EXPERIMENT_LOG L229, code↔chinese large-scale test); “0.16” appears in studypaper/03_routing_distribution_model.md only as a *hypothetical* input (“当 cos_sim = 0.16（强域切换）”). “57B 1.03→1.07” appears nowhere outside the paper. “>0.95” is a threshold constant (window_stable_cos), not clearly a measurement. A headline “observation” is built on them. (type a)

### M9. “−68.2% (表5)” cites a table that does not contain the number; original measurement unarchived
- zh L74 (§2.2 限制1): “在decode密集型负载（O=256）上造成**-68.2%吞吐退化**（表5）”.
- Current 表5 is the 4-GPU 57B multi-domain O=1 comparison (EPLB +0.4%). No −68.2% measurement exists in any log/report (the experiment log only asserts “−50%~−68%” for O≥64 without data). The only archived reproduction is 57B O=256: −62.4% (F.3/F.4, backed by F.3 table). Fix pointer + cite the 57B reproduction as the evidence. (type d)

### M10. “占用约12.5%额外GPU显存” contradicts the paper’s own Table 9
- zh L80 (§2.2) & L208 (§2.5): “12.5%的显存”.
- 12.5% = redundant slot ratio (144/128, old draft PAPER.md L58 says exactly this). But GPU memory: Table 9 gives weights +2.5 GB on 88.7 GB total (+2.8%) / 28.0 GB weights (+8.9%). The downstream KV claim (8.1%, 227K→209K) is backed (log §13 ✓), but “12.5% additional GPU memory” as written is wrong and is reused in §2.5’s design argument. (type c)

### M11. Abstract “调整代价比周期性重平衡低一个数量级” is unsupported and contradicted by the paper’s own numbers
- zh L14/en L14: “调整代价…低**一个数量级**”.
- Measured: per steady-state adjustment 0.37s vs 1.55s = 4.2×; cumulative blocking 5.95s vs 14.7s = 2.5× — and §1 contrib. 4 (L47) itself says “阻塞总量为EPLB的**1/2.5**”. 2.5–4× ≠ an order of magnitude. (types a, c)

### M12. “减少约50%开销” (abstract/§1/§3.6) has no derivation and the two given rationales contradict each other
- zh L14/L37/L351: “减少约50%记录开销”; §3.6 rationale (i): decode steps outnumber prefill 10:1 ⇒ skipping decode should remove ~91% of record calls, not 50%; rationale (ii): decode recording already returns immediately under CUDA-graph capture (“零开销”) ⇒ skipping it saves ~nothing. Both cannot justify 50%. The only logged measurement (PAPER_EXPERIMENTS L149–153: record-all 68.77 vs prefill-only 67.97 req/s) shows a 1.2% difference declared noise. (types a, b)

### M13. §2.6 KV-pressure “实测验证”: config mix-up + unmeasured mechanism + arithmetic slip
- zh L234: “EPLB **-3.2%**（KV cache压力导致排队暴增），OEPLB +16.0%，差距19.2pp（L4096_O256 conc=512，ρ≈0.9）”.
- The −3.2% run in the log (COMPREHENSIVE_EXPERIMENT_LOG §13) is **Frozen-EPLB (auto mode)**, whose KV loss was **−13.9%**, not the δ=8.1% (continuous EPLB) used in the queueing model right above; “ρ≈0.9” and “KV cache压力导致排队暴增” are assertions (no utilization/queue measurement). Also the table doesn’t match its own formula W_q∝1/(1−ρ′), ρ′=ρ/(1−δ): for ρ=0.85 the multiplier is 2.00×, not 2.13×; for ρ=0.90 it is 4.83×, not 4.76×. (types b, c)

### M14. zh and en versions have diverged (en lags the 11:35 zh fixes)
- en L35 still asserts the retracted claim: “and it is **no longer triggered after convergence**” (zh L35 now carries the retraction; both revision-notes blocks list “0 swaps after convergence” as retracted). (type f)
- en L220 (§2.5 open item) still says “**three points are not enough** to determine the form (§2.4‡)” while zh §2.5 now cites the 4-point power law; en thus contradicts en §2.4 L147 (its own power-law paragraph) and en G.3 L1061 (“now predictable… Four sweeps”). (type c)
- en L584 conclusion: “+5.3% to **+18.4%**” vs zh L591 “+5.3%到**+17.5%**”. (type c)

### M15. §2.4 main-table caption contradicts its own table (stale)
- zh L132/en L132: “235B用**β校准值f_sens=0.384**，57B 8卡用附录G实测值” — but the table immediately below uses **0.496** (235B, Appendix G sweep) and Appendix G values for every row. The caption describes the pre-revision table. (type c)

### M16. D.3 “符号全部预测正确 / 幅度误差0.4个百分点” rests on untraceable numbers
- zh L755–758/en L751: five-config sign claim includes (i) 30B (unmeasured β<0, see C2), (ii) “4卡DS-V2-Lite | … | -4.5% ✅”: the only trace of −4.5% is old-draft PAPER.md L580; no experiment log; model weights no longer on the machine (NOTES 04:10: “30B/DS-V2-Lite已不在/data/models上”). (iii) “235B的幅度误差0.4个百分点” matches nothing: predicted net +21.3% vs measured +17.5% is 3.8 pp; the only “0.04pp” in the paper is bound-vs-ceiling (G.2(h)), a different quantity. (types a, d, c)

### M17. §3.2 α-table rows are cross-round comparisons (different baselines), undisclosed
- zh L266–270: α=0 → +2.5%, α=0.5 → **+10.6%**, α=0.9 → +6.9% (multi-domain). In the log: +2.5% (21516.5) and +6.9% (22174.6) come from the decay-comparison table (baselines ≈20992/≈20743), while +10.6% is the final-summary number (baseline 20436); the same-day decay=0.5 measurement was **+7.8%** (22780.4 vs 21132.6). Rows are not like-for-like; the table presents them as a controlled comparison. (types c, a)

### M18. Table 5/Table 6 reuse /tmp-era numbers without flags; Table 5’s own EPLB % doesn’t match its own row
- 表6 (zh L~495–497): “L512单域（4卡57B）+4.3%”、“多域（4卡57B）+3.0% | +0.4%”、30B rows −2.6%/−3.9% — all from /tmp/exp_data (verified in result JSONs), flagged with † in §5.4 but **unflagged in 表5/表6**. (type d/f)
- 表5 row: baseline 26.9, EPLB 27.2 → “+0.4%”; but 27.2/26.9−1 = **+1.1%**. +0.4% only works against a *different* baseline run (27.1, the tri_57b/bl_auto runs). A table row whose percentage contradicts its own cells. (type c)

### M19. §3.5 worked examples don’t reproduce from the stated formula
- zh L331: M_min = c²/((γ(r−r_k))²·t̄); “8卡57B（c=5.2，r−r_k=0.119，t̄≈2000/层）：**M_min≈0.5**” — computed: 27.04/(0.00354×2000) = **3.8**. “ShareGPT/4卡（c=2.6，r−r_k=0.065，t̄≈110/层）：**M_min≈22**” — computed: **58** (22 would require t̄≈290). The qualitative point (default W=16 is too small for ShareGPT) survives, but both numbers are arithmetically wrong. The previous (11:00) version of §3.5 had the same defect (sw_min=0.4 claimed vs 3.75 computed). (type c)

### M20. Appendix C promises a 20-cell grid that exists in neither version
- zh L696: “（完整20格见英文版PAPER_en.md）”; en L689: “(For all 20 cells, see the English version PAPER_en.md)” — self-referential; en Appendix C also shows only the 5 O=1 rows. The full 5×4 grid exists only in COMPREHENSIVE_EXPERIMENT_LOG.md (and its raw logs are not archived). (type d)

---

## MINOR

1. **zh L160/en L160**: “三个β…一律低于…FLOP占比…**1.65–1.93倍**” — actual ratios: 0.469/0.285=1.65, 0.469/0.342=**1.37**, 0.679/0.352=1.93; the middle pair is outside the stated range. Same paragraph “β…极差18%”: (0.352−0.285)/0.352 = 19% (20.5% vs mean), not 18%. (type c)
2. **zh L177/en L177**: “β=0.352×(1.737−1.093)=22.7%，实测+17.5%（η=79%）” — 17.5/22.7 = **77%** (as in NOTES/bound_curve); 79% belongs to the f_sens-based 22.09% bound. The two “algebraically equivalent” forms give different η because one uses r_b=1.737, the other 1.721. (type c)
3. **zh L151**: “f_sens需达**0.470**才能解释+14.0%” — solving Δ=0.14 with x_eff=0.214 gives f=**0.57** (with naive x=0.252: 0.49). 0.470 reproduces neither. (type c)
4. **§5.4 Table 3 L256 row**: tps 118.0→121.0 implies +2.54%, stated Delta is +2.70% (time-based: 139.06→135.40 ✓); baseline tps mean from _d18/base jsons is 117.5, not 118.0. CV line “L256 0.24%/0.10%” swaps the arm order vs the L512 convention (measured: baseline CV ≈0.10–0.16%, OEPLB CV ≈0.24–0.34%). (type c)
5. **zh L292 (§3.4) vs L288 (§3.3)**: first decision “**132**个op × 27.5 MB ≈ 1.2–1.9 GB” vs §3.3’s “#1 **139** ops” vs EXPERIMENT_NOTES driver10 “**135** ops”; and 132×27.5 MB = 3.6 GB, not 1.2–1.9 GB. Three unreconciled first-decision sizes; memory estimate inconsistent with its own factors. (type c)
6. **zh L587 (§7)**: “record开销在30B上达**1.6%**（vs 235B的0.34%）” — no derivation in paper or logs; Table 8 doesn’t yield 1.6% under any obvious denominator. (type a)
7. **G.3 item 3 (zh L1069/en L1063)**: “235B‘多域’配置系统效率**131%**的异常” — §2.4 table says **118%⚠** for the same row. Stale figure in both versions. (type c)
8. **Abstract “稳态每次调整0.37s vs 1.55s”**: 1.55s matches no statistic of the verified logs exactly (steady-state means: 1.62s r1 / 1.54s r2, pooled 1.58s). Close, but cite the actual statistic. (type a, borderline)
9. **zh L86 Observation 2**: “短prompt需要更大的同步窗口（sw=32-64）” — the grid’s best static windows for O=1 rows are sw=8/16; ShareGPT’s winner was *adaptive*. Loose support only. (type a)
10. **zh L286**: “638个有效swap配对存在” — no source anywhere. (type a)
11. **§2.4 β_c table + “证明” (zh L121–127)**: the component decomposition (Expert 0.08/Combine 1.33/Dispatch −0.78 → 0.384) is still presented with a “Proof. □” immediately after being labeled falsified (“已否证，低估26%” two rows up; G.2(h)3). Needs an in-place caveat that the β_c values are not quantitatively reliable (they also underpin the 30B footnote, see C2). (type f)
12. **Power-law wording (zh L147/en L147)**: “用57B的三个EP点拟合…57B/EP2预测…EP4预测…EP8预测…” — EP2/4/8 are the fit points; calling them “predicted” (with errors) is in-sample; only 235B/EP8 is a genuine test (the text does say so at the end). Also EP2 measured 1.012 falls slightly *outside* the pre-registered range “1.00–1.01” (NOTES 04:10) without comment. Wording/clarity. (type e, borderline)
13. **repro/ completeness**: driver27/28/29.sh and their results are in /workspace/logs/, not in repro/; repro/README.md (claims to index “每一条实测断言的脚本”) omits them. Reproducibility packaging gap, incl. the driver behind the conclusion’s η=100% claim. (type d)
14. **en §3.6 is a different (shorter) text than zh §3.6** — en drops the sufficiency argument and the (problematic, see M6) ≤1% validation claim entirely; the versions are not translations of each other here. (type c)

---

## Checked and found SUPPORTED (evidence chain intact)

- Hinge fits & sweeps: G.2(a–i) numbers match EXPERIMENT_NOTES/NOTES.md and driver12/13/14 logs (57B/EP8 r_k=1.099, β=0.285, f_sens=0.335, +3.40%/+3.82%; EP4 r_k=1.032, f_sens=0.369, +2.29%/+2.63%; 235B r_k=1.093, f_sens=0.496, +22.08%/+22.12%; conc-sweep table; DeepGEMM crash at r=4.686).
- **r_k power law**: independently re-fit from raw d24 timings + placement files: hinge gives r_k(EP2)=1.012; 3-point fit reproduces **0.00408·EP^1.522** with the paper’s exact error figures (−2.4/+5.1/−2.4/+3.8%). Backed. (Note: supersedes NOTES.md 06:40’s 0.00278·EP^1.73, which was fit to partial d24 data.)
- Re-test gains: d18 +2.70% (η=105% vs 2.57% bound), d20 +2.39%, d22 −0.24%/+0.14% — all reproduce from _d18/_d20/_d22 jsons.
- d27 four-arm gains (+0.98/+2.26/+3.81/+2.20%, η vs ceiling 3.82%: 26/59/100/57%) reproduce from _d27_* jsons; g4s decision cut 110→8 ✓.
- Table 7b / §3.4 / abstract blocking numbers: verified against server_L512_{eplb,oeplb}_r{1,2}.log (EPLB 4.465s first, 1.43–1.82s steady, totals 15.84/13.61s; OEPLB 989/1002 ops, 4.21s/2.55s first, 337–407ms steady, 6.74/5.16s; wall-clock % ✓).
- Re-test headline: OEPLB +17.4–17.5%, EPLB +1.8% from L512_*_r{1,2}.json ✓ (the values; their *use* is the problem, see C1).
- Observation 3 TPOT range −3.0%~−12.5% over 9 workloads: WORKLOAD_GRID_REPORT table ✓ (raw logs were in /tmp — note retention risk).
- Appendix E numbers (50385/50385; 9.92e-2 vs 9.59e-2; 256 swap logs; GSM8K flips) match NOTES d15/d19/d21 ✓.
- KV 8.1% (227,269→208,750), Table 9 pattern, grid O=1 rows, Table 1 values, F.3/F.4 (−62.4%, 2502.3 vs 6652.9), §2.4 x_eff/Δmax arithmetic for the 235B/57B rows, §2.5 r_place/LPT values (counts jsons + two_ceilings.py present), “1/2.5” blocking ratio, predict_gain.py existence.
- Dataset availability: re-measurement datasets exist; /tmp/exp_data confirmed gone (hedges in §5.4/F.2 are accurate).

---

# AUDIT2 — PAPER_zh.md 证据链审计（2026-08-12）

范围：针对 driver28（30B β=+0.20 已测）、r_k 幂律（4点、235B 完成）、摘要-正文一致性、撤回项残留的定点审计。
结论：**7 处 CRITICAL，6 处 MAJOR，4 处 MINOR**。核心问题：driver28 已测得 30B β=+0.20（正），但全文仍按"30B f_sens<0、上界为负"组织叙述（§2.4 表、§2.4 控制维、附录 F.4 验证表、结论），共 6 处互相呼应的断言全部与新测量矛盾。

---

## CRITICAL

### C1. L141 — §2.4 效率表 30B 行断言 f_sens<0
> `| 30B L512（4卡） | 1.70 | 未测* | 0.388 | <0 | 负** | -2.6% | — |`
问题：f_sens "<0"、Δmax "负" 与 driver28 实测 β=+0.20（正）直接矛盾。且 0.388 本身无出处（nsys 分解推断，未标注脚本）。
类型：与测量矛盾 + 数字无依据。

### C2. L149 — 30B 负上界的推导脚注
> `**30B的dispatch占比39%（vs 235B的10%），β_dispatch·f_dispatch的负项超过combine改善→f_sens<0→上界为负。`
问题：该推导的结论 f_sens<0 被 driver28（β=+0.20）否证。推导路线（nsys β分解）在 L116 已被标记"已否证，低估26%"，此处却仍作为定论脚注。
类型：与测量矛盾。

### C3. L174 — "30B → β<0" 作为普适规律
> `dispatch主导的模型（如30B）→ β<0，即使r_b很高也为负`
问题："三个控制维度"表把 30B 当作 β<0 的代表案例；driver28 实测 β=+0.20>0。
类型：与测量矛盾。

### C4. L755 — 附录 F.4 验证表：30B 行标 "✅ 符号预测正确"
> `| 4卡30B | 1.70 | 未测 | 0.388 | **<0**（dispatch占比39%） | 负 | ~6%(record) | **负** | -3.9% ✅ |`
问题：同 C1；更严重的是 ✅ 声称"理论符号预测正确"。β=+0.20 下模型预测正上界，实测 -3.9%——符号预测实际是错的（负收益应归因于 ~6% 开销项，而非 f_sens<0）。
类型：与测量矛盾 + 验证结论失效。

### C5. L758 — "修正史"叙述与"五个配置符号全部预测正确"
> `修正后的模型对五个配置的收益**符号全部预测正确**…早期草稿…把30B的f_sens当成正值，对30B给出"+4%理论、实测-3.9%"的矛盾。`
问题：该"修正"（把 30B f_sens 从正改负）恰被 driver28 推翻——直接测量 β=+0.20 为正。"符号全部预测正确"不再成立。
类型：与测量矛盾。

### C6. L591 — 结论："32专家/卡负收益与理论预测的符号一致"
> `32专家/卡时负收益（−2.6%~−3.9%），与理论预测的符号一致。`
问题：理论（β=+0.20>0）现在预测正上界，"符号一致"为假。负收益应由开销解释（L536/L680 的 7.1s 分析已在），但结论句仍挂在已被否证的 f_sens<0 上。
类型：与测量矛盾。

### C7. L35 — 已撤回的"收敛后 0 swap"仍在 §1 断言
> `稳态每次决策阻塞约0.37s，为EPLB单次全局重平衡（1.55s）的1/4；且**收敛后不再触发**。`
问题："收敛后不再触发"正是 L9/L523 明确撤回的说法（E.2 实测稳态仍有 256 条 swap 日志）。撤回在 §3/附录，§1 贡献列表未同步。
类型：撤回但仍断言。

---

## MAJOR

### M1. L47 vs L14/L186/L424/L490 — 摘要与正文的 235B 主结果两套数字
- 摘要 L14：`+17.5%（复测均值，n=2），相比EPLB高出15.7个百分点`（隐含 EPLB≈+1.8%，与 L186 "实测EPLB为+1.75%" 一致）。
- L47 贡献4：`+18.4%（vs EPLB +9.0%）`；表1 L424：`+18.4%`；L426：`超越EPLB（+9.4个百分点）`；L490：`+18.4% | +9.0% | +9.4 pp`。
问题：正文表格与贡献列表仍是旧的 +18.4%/+9.0%/+9.4pp，摘要是复测的 +17.5%/+1.75%/15.7pp。"vs EPLB 高多少"两处相差 6pp 以上。
类型：摘要-正文数字不一致。

### M2. L47 多域数字过时
> `多域负载+10.6%（vs EPLB +6.3%）`
问题：§5 表 L493 为 `+14.0% | +12.0% | +2.0 pp`，§2.4 L137/L148/L151 也用 +14.0%。L47 的 +10.6%/+6.3% 无正文对应。
类型：内部数字矛盾。

### M3. L220 — "r_k 三个点不足以定形式"已过时
> `r_k随EP上升（1.032@EP4 → 1.099@EP8），三个点不足以定形式（§2.4 ‡）。因此本节只给出判据与两侧的极限行为，不给出交叉点位置。`
问题：§2.4 ‡（L146-147）现有 4 点幂律 r_k−1=0.00408·EP^1.52（含 235B 盲测）。此处"开放项"引用的交叉引用指向已闭合内容。
类型：过时表述（r_k 幂律已建立）。

### M4. L1024 — G.2 结论1 与 §2.4 L144 直接矛盾
> L1024：`这否证了"r_k是配置无关常数"，因此§2.4标星行不得借用1.10。`
> L144：`235B实测r_k=1.093，与57B 8卡的1.099几乎相同（**借用本身没错**），真正的错误在f_sens`
问题：driver14 后 §2.4 已改判"跨模型借用没错"，附录 G.2 结论1 仍写"不得借用1.10"。同页结论2（L1026）与 L146 一致，唯结论1 未更新。
类型：内部矛盾（driver14 后未同步）。

### M5. L899-902 / L908 — 附录 F.5 仍把已撤回的 +4.7%/+2.6%/+3.1% 当活分析对象
> `| 57B L512_O1（4卡） | 1.113（借用*） | 0.084 | 待测 | +4.7% | 未定 |`（另两行 +2.6%、+3.1%）
> L908：`+4.7%就需要f_sens=0.54…三种出路…都需要新数据。`
问题：此三行即 L9/L870 明言"数据集已不存在、不可复现、重测−0.24%~+2.70%（L512 行重测 +2.39%）"的 F.2 三行。F.2（L870）已声明"以可复现的重测为准，本表仅作历史记录"，但 F.5"修正后的口径"表重新将其列为待解张力并做定量分析。基于已撤回数字的开放问题应删除或改用重测值。
类型：撤回但仍作为依据使用。

### M6. L787/L789 vs L139/L464/L768/L591 — η=26% 还是 29%
- L787（driver26/27 四臂表默认臂）：`threshold=1.02 | +0.98% | **26%** | 72`
- L139/L464/L768：57B/EP8 L256 默认 η=**29%**（+1.0%）；L591：`从**29%**跃升到100%`
- L789：`η从**26%**升到100%`
问题：同一配置两处给 26%、四处给 29%。按 §2.4 口径 0.98/3.40=28.8%≈29%，26% 算不出来。另注：η=100%（+3.81%）是相对经验天花板 3.82% 的口径，而 §2.4 定义 η=实测/Δmax（L136：17.5/22.09）；按后者 +3.81%/3.40%=112%。分母切换未声明。
类型：内部数字矛盾 + η 分母口径不一致。

---

## MINOR

### m1. L132 — 表头与表体不一致
表头：`235B用β校准值f_sens=0.384`；表体 L136 用 **0.496**。表头是 driver14 之前的旧口径（0.384 在 L116 已标"已否证，低估26%"）。

### m2. L105 / L638 / L181 — 算例仍用已否证的 0.384
一阶截断示例（L105、L638）与 EPLB 上界重算（L181，得 19.2%）均用 0.384；主表已是 0.496/22.09%。作为算例可保留但需注明是旧 nsys 值。

### m3. L143 — "30B行的死区仍未测量"
driver28 的 30B 扫描已完成；脚注需引用 driver28 并更新 r_k/β 列，"未测*"标记过时。

### m4. L177 与 L136 的 r_b 口径并存（1.737 vs 1.721）
L177 `β=0.352×(1.737−1.093)=22.7%`，L136 用 1.721→22.09%。差 0.9%（L721 有交叉校验），不构成矛盾，但同节两个 r_b 并存未加注。

---

## 通过项（抽查无误）

- **235B 主链一致**：L136（§2.4 表）= L144 = L177 = L752 = L717：r_before=1.721、r_k=1.093、x=0.365、f_sens=0.496、Δmax=22.09%、+17.5%、η=79%（17.5/22.09=79.2% 算术正确）。β=0.352/0.285/0.342 链（L159/L169）与 G.2 一致。
- **r_k 幂律叙述闭合**：L147 = L1066（0.00408·EP^1.52，4点，235B 盲测 +3.8%），driver13/driver14 均有脚本名可溯。未发现"r_k 无法预测/235B 待测"的残留断言（L220 除外，见 M3）。
- **5 个数字溯源抽查**：f_sens=0.496（driver14，L144）✓；β=0.352（G.2 扫描，L159）✓；r_k=1.093（driver14 12 runs，L144）✓；+3.81%（driver26/27，L785-786）✓；**f_sens=0.388（30B 行）✗ 无出处，见 C1**。
- **7.7×、17.0%、0-swap 的撤回声明本身**（L9/L120/L186/L523）措辞到位，未发现其他地方仍把 7.7× 或 17.0% 当作事实；唯一残留是 L35（见 C7）。
- 摘要的阻塞数字 5.95s/14.7s（1/2.5，与 L47 一致）、0.37s/1.55s（与 L35/L514/L795 一致）。

## 修复优先级建议
1. 按 driver28（β=+0.20）重写 30B 证据链：L141/L143/L149/L174/L755/L758/L591（C1-C6）；负收益改由开销项（L536/L680 的 7.1s）解释，符号预测如实标注失败。
2. 删除 L35 "且收敛后不再触发"（C7）。
3. 统一 235B 主结果到复测口径（+17.5% / EPLB +1.75% / 15.7pp）或明确两套口径对应关系；修正 L47 多域 +10.6%→+14.0%（M1/M2）。
4. 更新 L220、L1024 至 driver14/幂律之后的结论（M3/M4）。
5. F.5 撤回三行改为历史记录或删除 L908 三出路分析（M5）；统一 η=29% 并声明 100% 的分母口径（M6）。

---

# AUDIT3 — PAPER_zh.md 完整证据链审计（2026-08-12，第三轮）

范围：(A) 验证前两轮审计后声称已修复的 10 项是否真正落地；(B) 在前两轮覆盖较少的部分（§3、§4、§5.1–5.3、§6、§7、附录 A/B/C 及 D–H 的改写段落）寻找**新**问题。
方法：先通读 AUDIT_findings.md（AUDIT1）与 AUDIT2.md，再对 PAPER_zh.md（1098 行）分段全文读取 + 定点 grep + 对每处关键代数独立重算。行号以当前文件为准；未逐行核实处标 `~`。

---

# PART A — 声称已修复项的逐条验证

**总评：10 项中 6 项完全落地（PASS），4 项部分落地（PARTIAL）。没有完全未修复项；但 PARTIAL 的 4 项中有 2 项在修复时引入了新错误（见 N1、N9）。**

### A1. 30B：β=+0.207、r_b=1.338、r_k=1.031、+6.36%、η<0 归因、三臂 — **PARTIAL（叙述全部落地，但表格修复时引入列错标，见 N1）**
- §2.4 主表行（~L141）：`30B L512（4卡） | 1.338 | 1.031 | 0.230 | 0.207 | 6.36% | −3.8%~+0.5%‖ | ≤8%` ✓（1.70 已消失；1.338 与 counts30b.json 一致，AUDIT1 C3 关闭）。
- 脚注 ‖（L149）：明确"早期草稿推断 f_sens<0…扫描（15次运行）否证了这个推断：β=+0.207（正），铰链上界+6.36%…负收益的真相是 η<0"，三臂 −3.80%/−2.66%/+0.53% 齐备，归因"开销吃掉正上界" ✓。
- 三维表（~L175）：30B 改写为"β>0但死区窄+record开销大→η<0（可用swap预算止损）" ✓；"39% vs 10%"推导脚注已删除（grep 无残留）✓。
- D.3（~L759）：30B 行 `+0.207（实测）| +6.36% | ~7s固定 | 净负（η<0）| −3.8%~+0.5% ✅幅度` ✓；"五个配置符号全部预测正确"已改为"四个235B/57B配置" ✓。
- §8（~L591）："32专家/卡…T(r)上界实为正（β=+0.207）——负收益源于固定开销超过上界（η<0）…止损至+0.5%" ✓。
- 全文 grep：`f_sens<0`/`上界为负` 仅出现在明确标注"已撤回/已否证"的历史叙述中 ✓。
- **但**：修复把 β 值填进了 `f_sens` 列，导致该行代数不自洽（详见 PART B N1，MAJOR）；另"≤8%"与 0.53/6.36=8.3% 轻微不符（MINOR，见 N1 附注）。

### A2. EPLB 基线 +9.0% vs +1.75% 的 5× 差距告诫 — **PASS**
- 表1（L427）⚠ 注：明示"历史单次批次…EPLB相差5倍（9.0% vs 1.75%）…最可能来源是EPLB收益对 eplb_rebalance_num_iterations 与首次重平衡时机高度敏感…历史批次该参数未被记录…凡涉及与EPLB的定量比较一律以复测值为准" ✓（AUDIT1 C1(iii) 的"从未解释"已补上解释性假设，并明确标注为假设）。
- 表6（L490）：表前注"以复现值计…优势为15.7pp（摘要采用此值），而非本表的9.4pp" ✓，EPLB 列带 ⚠。
- §2.4 指针"见§5.3"（~L186）现在有效：§5.3 的 ⚠ 注本身含 +1.75%（AUDIT1 C1(i) 的坏指针已愈合）。

### A3. 4卡57B L256：+2.70% / 2.57% / η=105% — **PARTIAL（主位置全部落地；F.5 两个脚注未同步，见 N9）**
- §2.4 表行 + 脚注 †（L140–145）：`+2.70% | 105%`，说明 driver18、r_after=1.011、2.57%、天花板+2.63% ✓；"待测"仅作为历史说明出现（"此前此行标'待测'是因为…"）✓。
- §5.4 表3 重测表（~L456）：`L256 | 118.0 | 121.0 | +2.70% | 2.57% | 105%` ✓。
- D.3（~L758）：`4卡57B | 1.107 | 1.032（实测）| 0.068 | 0.369（实测）| 2.57% | ≤0.4%（反推）| +2.2% | +2.70% ✅` ✓。
- **残留**：F.5 末行单元格虽已填 `+2.57%‡ | +2.70% | 105%`，但其脚注 †（"缺OEPLB对照臂，driver18.sh正在补"）与脚注 ‡（按 2.29%/r_after=1.04 口径计算）均未更新，与单元格直接矛盾（N9）。

### A4. 决策数 "21次决策/轮（9次发出swap）"、"21→1"，不得有 72 / 15.5 — **PARTIAL（正文口径落地；D.3 四臂表仍残留 72，且 8 vs 1 口径未统一）**
- L784：`做了21次决策/轮（其中9次实际发出swap）` ✓；L288：`g8_base臂，21次决策、343个op`（139+204=343 ✓）；L299：`决策数从21降到1（发出swap 9→1）` ✓。
- grep：`15.5` 仅出现在附录C网格的吞吐数字中 ✓；"72次决策/轮"类表述已删 ✓。
- **但 L788 四臂 A/B 表默认臂仍写 `| 72 |`（决策/轮）**，与同节 L784 的"21次决策/轮"、L299 的"21降到1"直接矛盾；且 +死区臂写 `8` 而正文说降到 `1`（若 8=1×8 ranks，则该表是 rank 级计数而正文是全局计数——口径未声明）。这正是 AUDIT1 M2 原始病灶（72=9窗×8rank）的残留。**MAJOR，见 PART B N2/N3 之外的独立残留，建议把 L788 改为 21 并给全表注明计数口径。**

### A5. M_min = 3.8 与 58，M=16/0.5=32<58 — **PASS（且算术经重算核实）**
- ~L331–333：`8卡57B（c=5.2，γ=0.5，r−r_k=0.119，t̄≈2000/层）：M_min≈3.8`；`ShareGPT/4卡（c=2.6，r−r_k=0.065，t̄≈110/层）：M_min≈58`；`default M=W/(1−α)=16/0.5=32<58` ✓。
- 重算：5.2²/((0.5×0.119)²×2000) = 27.04/7.080 = **3.82** ✓；2.6²/((0.5×0.065)²×110) = 6.76/0.11619 = **58.2** ✓。c=0.65·EP 与 5.2/2.6 一致 ✓。（AUDIT1 M19 关闭。）

### A6. 跨域 cos_sim ≈0.86 实测、0.16 标注为早期草稿非实测 — **PASS**
- L86：`实测的跨域余弦相似度最低约0.86（同域内>0.999）…早期草稿引用的0.16是理论假设值，非实测，已更正` ✓；全文再无把 0.16 当实测的用法（grep 仅 L86 一处）；"57B上1.03→1.07"已删除 ✓。
- 小注：新引入的"同域内>0.999"无出处（MINOR，可补脚本名）。

### A7. 附录H "designed to test" 而非 "verified" — **PASS**
- §3.5（~L325）：`附录H设计了235B实验来检验M是否为充分统计量（…driver29.sh进行中）` ✓；（~L343）`附录H给出…设计与预注册预测（driver29.sh进行中）` ✓。
- H.3（L1091–1097）：预注册 P1–P4 + `结果待driver29.sh完成后填入。本附录当前给出设计与预测` ✓。H.1 "验证两个可证伪命题"为意图表述，可接受。

### A8. r_k 幂律处处一致；无 "无法预测"/"235B待测" 残留 — **PASS**
- grep `无法预测`/`不足以定形式`/`不得借用`/`235B…待测`：全部为空 ✓。
- §2.5 开放项（~L220）已改写为引用 4 点幂律并只声明外推不确定度（AUDIT2 M3 关闭）；G.2(f) 结论1 已改写，不再与 §2.4 L144 矛盾（AUDIT2 M4 关闭）；L147 与 G.3 项1 的幂律叙述一致（0.00408·EP^1.52，盲测+3.8%）✓。
- 附带发现：H.3 引用"附录G.2(f)对EP=2的预注册处理（4条全部命中）"存在指针与事实问题——见 N12。

### A9. §3.6 充分统计量声明承认 ShareGPT 例外 — **PASS**
- L349：`在同质负载上吻合到≤1%…**唯一的例外是ShareGPT（1.0965 vs DIAG首窗2.161，差97%），但该偏差的成因是DIAG对窗口不加权…而非prefill采样不充分**…失效的是DIAG的统计口径` ✓（AUDIT1 M6 关闭，且给出了例外的归因与 token 加权复算 1.1000/0.3%）。

### A10. 12.5% 改为专家槽位比、显存实测 +2.8% — **PARTIAL（§2.2 已修；§2.5 仍写"12.5%的显存"）**
- §2.2（L80）：`专家槽位增加12.5%（144/128）。注意这不等于GPU显存增加12.5%：实测显存占用为+2.5GB（88.7GB基数上+2.8%，表9）` ✓（AUDIT1 M10 的 L80 半边关闭；表9 分列相加 88.7/79.8 复核无误）。
- **但 §2.5（L208）仍写**：`冗余专家的额外收益低于测量分辨率——而它要付12.5%的显存和8.1%的并发量（§2.2）`——在 §2.2 刚否定了"12.5%=显存"的读法之后，§2.5 的设计论证仍按旧读法引用，内部矛盾仍在（AUDIT1 M10 的 L208 半边未修）。建议改为"12.5%的专家槽位（实测显存+2.8%）"。

### 附带观察（前两轮已报告、本轮顺带确认状态，非 PART A 清单内）
- AUDIT2 C7（L35"收敛后不再触发"）：已加撤回括注 ✓。AUDIT2 M1：贡献4 已加"单次gross vs 复测+17.5%"披露 ✓。AUDIT2 M6（η 26/29 分母）：D.3 已加分母口径说明（~L790），基本关闭（§3.3 L299 仍写 26%，由该说明覆盖）。AUDIT2 M2（L47 多域 +10.6% vs +14.0%）：**仍未修**（见残留清单 R10）。AUDIT2 M5（F.5 三出路）：**部分**（三行已标"未定"并加借用说明，但"+4.7%需要f_sens=0.54"的三出路分析仍把已撤回数字当活对象，~L910）。

---

# PART B — 新发现问题（均不在前两轮审计中）

### N1. §2.4 主表 30B 行：`f_sens` 列填的是 β，行内代数不自洽 — **MAJOR**（内部矛盾 + 定义错标；A1 修复引入）
- L141：`30B L512（4卡）| 1.338 | 1.031 | 0.230 | **0.207** | **6.36%** | …`
- 问题：本表 `f_sens` 定义为 B·r_b/T(r_b)（§2.4 定理段与 G.1 第4步一致），且其余各行都满足换算 f_sens=β·r_b/(1+β(r_b−r_k))（逐行验证：235B 0.352→0.496 ✓；57B/EP8 0.285→0.335 ✓；57B/EP4 0.342→0.369 ✓）。对 30B：β=0.207 应换算为 **f_sens=0.260**，而表中直接把 β=0.207 填进 f_sens 列。
- 后果：用表中印刷值套本文自己的 Amdahl 公式 Δmax=f·x_eff/(1−f·x_eff)：0.207×0.230/(1−0.0476)=**5.0%**，得不到 6.36%。6.36% 只在把 0.207 当 β 用线性式 β(r_b−r_k)=0.207×0.307=6.35% 时成立。**两种读法必居其一，f_sens 列与 Δmax 列不能同时成立。**
- 建议：f_sens 列改 0.260（保留 Δmax 6.36%），或在脚注声明该行填的是 β。
- 附注（MINOR）：系统效率列"≤8%"与最好臂 0.53%/6.36%=8.3% 不符（若对 5.0% 口径则 10.6%），需随口径一并澄清。

### N2. §3.3："阻塞是可用空间的1.4倍，注定净负" — 与同配置实测 +0.98%/+1.0% 直接矛盾 — **MAJOR**（定量断言被自家测量否定）
- L292–294：`默认臂的累计swap阻塞为3.94s，而该配置的headroom（β(r_b−r_k)T_flat=0.285×0.119×82.86）只有2.81s——阻塞是可用空间的1.4倍，注定净负。`
- 问题：默认臂（threshold=1.02）实测收益 **+0.98%**（D.3 A/B 表 L788；§2.4 主表 +1.0%），是正的端到端吞吐增益，且 benchmark 墙钟本身已含 swap 阻塞。"注定净负"与实测正收益矛盾。若 3.94s>2.81s 核算成立，应解释为何实测不是约 −1.3% 而是 +1.0%（这恰是 D.3 自称的"主要开放问题"），而非断言"注定净负"。
- 建议：改为"按此核算应为净负，实测却为+0.98%——核算与实测间约2s缺口未解释，与 η=29% 问题同源"。

### N3. D.3 条件A："两个 swap/headroom≥1 的配置（1.26 与 1.09）η都塌到零…净负" — 与同表 1.26 行 η=29% 矛盾 — **MAJOR**（内部矛盾）
- ~L775：`两个swap/headroom≥1的配置（1.26与1.09）的η都塌到零——均衡器花在搬专家上的墙钟时间超过了不均衡本身的代价。这是净负，不是调参能救的。`
- 问题：门控表（~L768）中 swap/headroom=1.26 的行是 **57B/EP8 L256，η=29%、实测+0.98%（正收益）**；仅 1.09 行（多域，η≈0，−0.24%）符合描述。"都塌到零/净负"对 1.26 行不成立。
- 建议：把条件A限定到 η≈0 的行，或解释 29% 行为何例外（如决策#1 一次走完有效行程）。

### N4. §3.5 联合最优闭式 M★ 不从所写目标函数推出 — **MAJOR**（数学推导：结论与前提不一致）
- ~L336–339：目标写作 min_M [a·bias(M)² + b·β(r−r_k)·M·ln2/L_seg]，其中 bias(M)=c/√N、N=M·t̄；给出的闭式却是 M★=(a·c²·L_seg/(b·β·t̄·γ²·(r−r_k)³·ln2))^{1/2}。
- 重算：对所写目标求导，FOC 给 M★² = a·c²·L_seg/(b·β·t̄·(r−r_k)·ln2) —— **没有 γ²，(r−r_k) 是 1 次而非 3 次**。所写闭式只有在方差项写成 a·(bias/(γ(r−r_k)))²（噪声相对信噪比的平方）时才成立，而正文并未这样写。由此"随信噪比 (r−r_k)^{3/2} 下降"的标度律也按所写目标应为 (r−r_k)^{1/2}。
- 建议：把目标函数显式写成信噪比归一形式，或修正闭式与标度律表述。

### N5. 附录 B.1 表：总计行 ≠ 四个分类之和（每边差约 1.8 ms/步，~7%），且无"其他"行 — **MAJOR**（表内算术不闭合；实验出处缺失）
- ~L664–670：基线列 7323+5479+6214+4731 = **23747**，总计却写 **25546**（差 1799）；OEPLB 列 6300+4015+6179+4695 = **21189**，总计写 **23082**（差 1893）。各单项 Δ% 与总计 Δ%（−9.6%）各自内部自洽，但表内不闭合。
- 问题：若存在未列出的 kernel 类别，应加"其他"行；否则总计行与分类行至少有一组错。另：B.1 仅写"在8卡235B上测量（GPU利用率~62%，~795 forward步）"，无数据集/脚本/日志指针，−14.0%/−26.7% 不可追溯。
- （若确系漏列"其他"类，可降为 MINOR；按现表呈现是硬矛盾。）

### N6. 对 EPLB 的"领先幅度"仍用历史值，违反论文自设规则 — **MAJOR**（规则自违反）
- L427 规则：`本文凡涉及与EPLB的定量比较，一律以可复现的复测值（EPLB +1.75%）为准`。
- 违反点1：L429（紧接表1 ⚠注之下）：`PB-OEPLB达到oracle最优的97.6%…显著超越EPLB（**+9.4个百分点**）` —— 按复测值应为 **15.7pp**。
- 违反点2：§8（~L591）：`持续超越SGLang的EPLB **2-10个百分点**` —— 该区间来自表6 历史行（9.4/7.8/7.0/2.0/10.3pp）；按复测口径仅 L512 一点即 15.7pp。
- 建议：两处改引 15.7pp，或如 L490 那样双口径并列。

### N7. 表1 Frozen-EPLB 行百分比与自身单元格不符 — **MINOR**（表内算术）
- ~L424：`Frozen-EPLB | ~1.00 | 22668.1 | +13.0%`；22668.1/20167.8−1 = **+12.4%**。其余各行（−20.4/+9.0/+18.4/+21.3）复核与基线单元格吻合，唯此行差 0.6pp。

### N8. §5.6 "ratio从1.72降到1.05" 与其余各处的 ~1.02 不一致 — **MINOR**
- L518：`它换来的是ratio从1.72降到1.05，净收益为+17.5%`；表1（~1.02）、表2（稳态~1.02）、§1（1.02）、§3.3（1.03↔1.01）均为 1.02。

### N9. F.5 脚注 †/‡ 未随 A3 修复更新，与单元格矛盾 — **MAJOR**（内部矛盾；A3 修复残留）
- ~L905 行：`57B L256（4卡）| 1.107（实测）| 0.079 | **+2.57%‡** | **+2.70%** | **105%**`。
- 脚注 †（~L906）：`末行…缺OEPLB对照臂，driver18.sh**正在补**` —— driver18 已完成、+2.70% 已填入（§2.4 脚注 † L145 明确"已由driver18.sh补测"），此处过时。
- 脚注 ‡（~L908）：`Δmax=**2.29%**按§2.4的口径算，取…r_after=1.04…x_eff=0.061` —— 单元格写的是 **2.57%**（r_after=1.011 口径，L145）。脚注解释的是旧值，与单元格直接冲突。

### N10. G.2(h) 算例分母与印刷拟合式不一致 — **MINOR**（算术）
- ~L1047：`f_sens = 58.78×1.721/203.11 = 0.496`。按同节印刷拟合式 T=167.07+58.78·max(0,r−1.093)，T(1.721)=167.07+58.78×0.628=**203.98**，代入得 101.16/203.98=**0.498**。印刷的 203.11 疑为旧拟合残留（差 0.002 不改结论，但算例应可按印刷公式复现）。

### N11. §2.4 "残差平方和低13×" vs G.2(b) "低12.1×" — **MINOR**（同一量两个版本）
- ~L100：`残差平方和比纯线性形式低**13×**`；G.2(b)（~L984）：`铰链的残差平方和低**12.1×**`（1.866/0.154=12.12 ✓）。应统一为 12.1×。

### N12. H.3 "（4条全部命中）" 与 "附录G.2(f)对EP=2的预注册处理" — **MINOR**（交叉引用错误 + 预注册声明存疑）
- L1097：`与附录G.2(f)对EP=2的预注册处理一致（4条全部命中）`。
- 问题：(i) G.2(f) 是 **EP=4** 的扫描，EP=2 在附录 G 中没有实验小节（r_k=1.012 仅见于 §2.4 L147/三维表/G.3），指针错误；(ii) AUDIT1 核对过 NOTES 04:10 预注册：EP2 预注册区间 **1.00–1.01**，实测 **1.012 在区间外**（§2.4 只报"误差−2.4%"），"全部命中"与预注册记录不符；(iii) 四个点中 EP4/EP8 为拟合样本内点（AUDIT1 minor 12），称"命中"需限定为 EP2+235B 两个真预测点。

### N13. 附录 A.1 定理1：无证明，2/N_G 因子与单 swap 力学不符 — **MINOR**（附录数学）
- ~L615–619：`Δr ≥ 2(L[a]−L[b])/(N_G·L̄)·(1−(L[a]−L[b])/(2·gap))`。
- 问题：一次 swap 把负载 d=L[a]−L[b] 从最热 GPU 移出，若最热 GPU 仍为最大则 Δr=d/L̄；定理含无解释的 2/N_G 因子且全文无证明。定理2 及其 tight example 复核自洽（1+(G−1)/G·G=G ✓），定理1 目前不可验证。建议补证明或降级为引理。

### N14. §2.4 EPLB 净预测中 "0.68×(1−0.77)" 为无定义魔数 — **MINOR**（数字无出处）
- ~L183：`CUDA graph禁用代价：1−0.157（0.68×(1−0.77)，见§2.2）`。0.157=0.68×0.23 算术 ✓，但 0.68 与 0.77 在 §2.2 及全文他处均无定义（0.77 仅在 L181 作为"早期草稿的 f_MoE"出现）。"见§2.2"指针落空。

### N15. D.3 部署建议4："本文8卡57B为0.37s/rank/决策，占比0.2%" — **MINOR**（数字无法从同表追溯）
- ~L797：`稳态swap开销应远小于Δmax（本文8卡57B为0.37s/rank/决策，占比0.2%）`。
- 问题：门控表给 8卡57B 默认臂 swap=3.68 s/rank/轮、21 决策/轮 → **0.175 s/决策**；0.37s 恰是 235B 的稳态每调整值（§3.4/表7b，337–407ms），疑张冠李戴。"占比0.2%"分母未给（3.68s/86s≈4.3%；0.175/86≈0.2%，若指单决策占 benchmark 应写明）。

### N16. §2.4 定理段 "7个布局点×2轮，14次运行0错误" 过度泛化 — **MINOR**
- ~L97：该句作为覆盖全部 T(r) 扫描的表述出现，但 235B 扫描实为 **6 点×2轮=12 次运行**，第 7 点 r=4.686 无法启动（E.3；G.2(h) 明写"6个布局点各2轮共12次运行"）。7点/14次仅对 57B 两个 EP 配置成立。建议注明"（57B；235B为6点12次，见E.3）"。

---

# 附：前两轮已报告、本轮确认仍然存在（未声称修复或声称而未改）— 简要清单（非新发现）

供修订对照，均已见于 AUDIT1/AUDIT2，不重复论证：
- R1. 摘要"调整代价低**一个数量级**"（实测 2.5–4.2×）— AUDIT1 M11，L14 仍在。
- R2. "减少约50%记录开销"无推导（10:1⇒~91%；CUDA-graph 下 decode 记录零开销⇒~0）— AUDIT1 M12，L14/L37/~L350 仍在。
- R3. §2.6 排队表与公式 W_q∝1/(1−ρ′) 不符（ρ=0.85 应 2.00× 而非 2.13×；ρ=0.90 应 4.83× 而非 4.76×）；−3.2% 行配置混用与"ρ≈0.9"无测量 — AUDIT1 M13，仍在。
- R4. §5.8 三次冷启动标"L512_O1"，但 22780±156 与表1 的 L512_O1 值 23870.5 相差 ~7σ（实验日志记载为多域数据集）— AUDIT1 M7，仍在。
- R5. §2.2 "−68.2%（表5）"：表5 无此数，可追溯复现值为 F.3/F.4 的 −62.4%（F.4 自己写"论文报告−68%，本次复现−62.4%"）— AUDIT1 M9，L74/L28 仍在。
- R6. §3.2 α 表三行为跨轮次、跨基线对比（+2.5/+10.6/+6.9）未披露 — AUDIT1 M17，L266–269 仍在。
- R7. 表5 行内矛盾：27.2/26.9−1=+1.1% 而表写 +0.4% — AUDIT1 M18，仍在。
- R8. 附录C "完整20格见英文版"自指、两版均无 20 格 — AUDIT1 M20，仍在。
- R9. §2.4 表头"235B用β校准值 f_sens=0.384"与表体 0.496 不符（表头亦未覆盖 30B/4卡行来源）— AUDIT1 M15 / AUDIT2 m1，L132 仍在。
- R10. L47 多域 "+10.6%（vs EPLB +6.3%）" 与 §5.5/§2.4 的 +14.0%/+12.0% 矛盾 — AUDIT2 M2，仍在（§3.2 α 表亦用 +10.6%）。
- R11. F.5 对已撤回 +4.7% 的"三出路"定量分析 — AUDIT2 M5，~L910 仍在（三行已标"未定/历史记录"，部分改进）。
- R12. D.3 DS-V2-Lite 行 −4.5% ✅ 仍无实验出处（权重已不在机器上）；"五配置"声明已缩为四配置，但该行 ✅ 未撤 — AUDIT1 M16 残留，~L760。
- R13. 杂项（均 AUDIT1 已报）：L286 "638个有效swap配对"无出处；~L307 "132个op×27.5MB≈1.2–1.9GB"（132×27.5MB=3.6GB，且与 §3.3 的 139 ops 不一致）；L587 及现 L149 "record开销1.6%"无推导（且被新 30B 脚注引用而扩散）；~L158 "β极差18%"（实为19%）；表3 L256 tps 118.0→121.0=+2.54% 与 +2.70%（时间口径）并存、CV 行臂序；L105/L181/L642 算例仍用已否证的 0.384 未标"旧值"（AUDIT2 m2）；表4（30B）源自 /tmp/exp_data 而无 † 类告诫；G.3 项3 "131%" vs §2.4 表 "118%⚠"（L1073）；观察1 ">0.95" 的阈值/实测双重身份（M8 半边）。

# 修复优先级建议（本轮新增项）
1. **N1**（30B 行 f_sens/β 列错标）与 **N9**（F.5 脚注）——本轮修复自身引入/遗留的矛盾，优先改。
2. **N2/N3**（"注定净负"/"都塌到零"）——与实测 +0.98%/η=29% 的正面矛盾，属结论级表述，必须软化或给出核算-实测对账。
3. **N4**（M★ 闭式与目标函数不一致）——改目标函数写法或改闭式。
4. **N6**（9.4pp/2-10pp 违反自设复测口径）与 **A4 残留的 L788 "72"**——数字一致性。
5. **N5**（B.1 总计不闭合）补"其他"行或更总数；**A10**（§2.5 L208 "12.5%的显存"）按 §2.2 新口径改写。
6. 其余 MINOR（N7/N8/N10–N16）按清单顺手修正。

---

# AUDIT_FINAL — 第三轮（终审）证据链审计（2026-08-13）

范围：PAPER_zh.md（1140行，08-13 09:54）与 PAPER_en.md（1130行，08-13 10:00）。
方法：先读 AUDIT_findings.md / AUDIT2.md / AUDIT3.md 了解已修项，再全文分段通读两版，
对全部关键数字做 grep 对账，并对本轮新主张直接回查原始数据：
`_d38/_d39/_d35/_d30/_d31/_d34/_d28` 结果 JSON（benchmarks/results/）、driver28 布局文件 +
counts30b.json（独立重算 r_avg 与铰链拟合）、L512_{baseline,eplb,oeplb}_r{1,2}.json、
/workspace/logs/NOTES.md（d31 预注册 17:25、d38/d35 记录）、repro/EXPERIMENT_NOTES.md、
repro/two_ceilings.py（重跑）、repro/ 与 /workspace/logs/ 的脚本清单。

**总评：5 处 CRITICAL、15 处 MAJOR、27 处 MINOR。**
核心科学内容（30B β=+0.207 修正、d38/d39/d35 新结果、铰链模型）经独立重算全部站得住；
问题集中在 **(a) 英文版对最近一批修复严重滞后（5 个 CRITICAL 全在 en），
(b) 中文版残留的过时脚注/表格（72、F.5/D.3 脚注、§2.5 表混数据集），
(c) 复现指南与 repro/ 不同步**。没有任何问题需要新实验，全部是传播/脚注/算术修订。

---

# CRITICAL

### CF1. en §5.3：表1的整段⚠告诫缺失，且仍把 +9.4pp 当作现状断言
- en L417–425：表1（EPLB +9.0%、OEPLB +18.4%）下方**没有**zh L427–434 的⚠注（历史单次批次、
  复测 EPLB +1.75%/OEPLB +17.5%、5×差距解释、"凡涉及与EPLB的定量比较一律以复测值为准"规则）。
- en L425：`PB-OEPLB reaches 97.6% of the oracle optimum (23870/24460), significantly outperforming EPLB (by +9.4 percentage points).`
  直接违反论文自设的复测口径规则（zh L438 已改为"历史口径9.4pp vs 复测口径15.7pp，正文一律采用后者"），
  并与 en 自己的摘要（L14：15.7pp）矛盾。
- 类型：zh/en 失同步 + 规则自违反（AUDIT3 N6 仅在 zh 修复）。
- 修：把 zh L427–438 的⚠注与双口径句完整译入 en。

### CF2. en §8 结论仍断言"持续超越EPLB 2–10个百分点"
- en L593：`consistently outperforming SGLang's EPLB by 2-10 percentage points`。
- zh L606 已改为：`按可复现的复测口径（EPLB +1.75%）在L512上超越EPLB 15.7个百分点（历史单次批次给出的2–10个百分点区间已加⚠标注）`。
  en 把历史区间当作现状结论，无 ⚠、无 15.7pp。
- 类型：zh/en 失同步 + 过时数字当现状（AUDIT3 N6 违规点2 仅 zh 修复）。
- 修：en 结论镜像 zh。

### CF3. +14.0% 多域是否已被 −1.1% 否证：zh/en 立场相反，en 自相矛盾
- zh L152：v2 数据集测得−1.1%，但"**不构成同条件重测**，不能用来否证原值"；⚠ = "无法确认也无法否证，不计入定量主张"。zh G.3 L1086 同样说"跨负载外推需重新测量"（未解决）。
- en L493（表6行）：`+14.0%⚠→**-1.1%**(重测)`（把 −1.1% 当作替代值，且英文表里残留未翻译的中文"重测"）；en G.3 L1076：`has been **resolved by re-measurement**: the actual gain is −1.1% (η=−5%), not +14.0%`。
- 但 en L152 与 zh 一致（"cannot falsify the original"）——en 内部 L152 vs L493/L1076 直接矛盾。
- 附带：en 的 `η=−5%` 口径不明：−1.1/22.09=−5.0%（借用 L512 行上界），若用多域行自己的 Δmax=11.86% 应为 −9%。
- 类型：科学结论级 zh/en 矛盾 + en 内部矛盾。
- 修：两版统一到一个立场。若维持 zh 立场：en 表6行删"→−1.1%(重测)"、G.3 项3 改回"未解决"；若采用 en 立场：zh L152/L1086 必须同步改写并解释为何 −1.1% 能否证 +14.0%。

### CF4. en 的 4卡57B 行整体未落地 driver18 结果（+2.70%/2.57%/105%），与 en 自己的 §5.4 矛盾
- en L140（§2.4 主表）：`57B L256 (4-GPU) | … | 0.061 | 0.369 | 2.29% | to be measured† | —`（zh L141 已是 `0.068† | 2.57%† | +2.70% | 105%`）。
- en L756（D.3 表）：`x_eff 0.061 | 2.29% | ~2% (estimated) | +0.3% | TBD†`（zh L769：`0.068 | 2.57% | ≤0.4%（反推）| +2.2% | +2.70% ✅`）。
- en L910（F.5 末行）：`+2.29%‡ | TBD† | undetermined`（zh L919：`+2.57%‡ | +2.70% | 105%`）。
- en 脚注 L145/L766/L913 均仍写 `driver18.sh is filling that gap`（driver18 早已完成，+2.70% 已进 en §5.4 表 L455）。
- 类型：zh/en 失同步 + en 内部矛盾（§5.4 有 +2.70%，§2.4/D.3/F.5 说 TBD）。
- 修：把 AUDIT3 A3 的修复（r_after=1.011→x_eff=0.068、2.57%、+2.70%、η=105%、脚注†改写）传播到 en 三处。

### CF5. en §2.4 残留脚注"30B 死区仍未测量、上界是乐观值"——与 30B 核心修正直接矛盾
- en L143：`*The dead zone for the 30B row remains unmeasured; x=1−r_after/r_before is used in place of x_eff, giving an optimistic upper bound.`
- zh L143 已改为：`*所有标星行的死区现均已测量。`
- en L143 与 en 自己的 L141（表中 r_k=1.031）、L150（driver28 扫描 15 runs 测得 β=+0.207）、L757（D.3 "1.031（实测）"）矛盾——是"30B 未测"旧叙述的最后残留。
- 类型：撤回但仍断言（30B 故事一致性，任务检查点2）。
- 修：en L143 换成 zh 的句子。

---

# MAJOR

### MF1. 贡献4（两版）的多域叙述过时：只字未提可复现的 d39/d35 结果
- zh L47 / en L47：`多域负载+10.6%⚠（vs EPLB +6.3%⚠；…均依赖已删除的/tmp/exp_data/数据集，不可复现）`。
- §5.3（zh L436/en L427）现已有**可复现**多域结果：头条 **+9.76%**（d39，vs identity，824.7→751.4 s，已核 JSON）与 adaptation benefit **+5.80%**（d35，vs 静态最优，801.2→757.2 s，已核 JSON）。贡献列表却宣称多域结果全部不可复现——d39/d35 之后这是错的。
- 类型：过时数字 + 两个量（headline vs adaptation）在贡献层未呈现（任务检查点1）。
- 修：贡献4 改为引用 d39 +9.76%（vs identity）与 d35 +5.80%（vs 静态最优）作为可复现多域结果；+10.6/+6.3/+14.0/+12.0 明确降为历史值。

### MF2. 同一数据集两个数：§2.4 说 v2 测得 −1.1%，§5.4 说 −0.24%
- zh L152：`在…multidomain_v2_out1.jsonl（4400请求）上测得−1.1%`；zh L468（§5.4 表3）：`多域（4.4K）| multidomain_v2_out1 | 41.00 | 40.90 | −0.24%`。
- 同一数据集、两个不同 Delta，且两处都没写是哪个模型/配置（§5.4 是 57B；§2.4 的 −1.1% 若是 235B 应注明）。
- 修：补模型/配置标签并核对原始 JSON，统一到实测值。

### MF3. §5.3 排序句混用不同 run、且拿 r1 冒充均值
- zh L436 / en L427：`identity(824.7) > 静态最优(806.7) > OEPLB(751.4)`。
- 核查 JSON：806.72 = **d35 的 bal r1**（_d35_bl_r1.json）；d35 均值是 801.2（本段前文自己引用的）；d39 没有 bal 臂（只有 _d39_bl/_d39_oe）。即该三元组把 d39 均值（824.7、751.4）与 d35 的单轮 bal 拼在一起，紧跟在"两个量分别用各自同run的数据计算"之后自相矛盾。
- 修：改成两个同 run 对比（d39：identity 824.7→OEPLB 751.4；d35：bal 801.2→OEPLB 757.2），或明确标注 806.7 为 d35-r1 并给出 d39 无 bal 臂。

### MF4. §2.5 两天花板表：57B 两行把 L256 的 r_native 与 L512 的上界拼在同一行
- zh L201–202 / en 同位：`57B EP=4 | r_native 1.107 | … | 摆放上界 2.75%`、`57B EP=8 | 1.218 | … | 3.70%`。
- 重算：按本行 r_native 与 β(r_native−r_k) 应为 **2.57%**（0.342×0.075）与 **3.39%**（0.285×0.119）；印刷的 2.75%/3.70% = β(r_b−r_k) 取 **L512** 的 r_b=1.1125/1.2288（溯源：EXPERIMENT_NOTES L279/L283 即按 L512 计算）。235B 行 22.66% 与本行自洽（0.352×0.644）。
- 类型：表内列来自不同数据集（AUDIT3 未发现）。注意 §2.4 三维表（zh L172）已披露"L256→3.4%、L512→3.7%"，但 §2.5 表未披露且行内混用。
- 修：每行统一到一个数据集（推荐 L256：上界改 2.57%/3.39%；或 r_native 改 1.1125/1.2288 并保留 2.75%/3.70%），并注明数据集。

### MF5. zh D.3 四臂 A/B 表仍是 72/8/8/8（en 已修为 21/1/1/1 且加了口径注）
- zh L799–804：`| 默认 | threshold=1.02 | +0.98% | 26% | 72 |`（其余三臂 8/8/8）。
- 同文 zh L291 `21次决策、343个op`、L302 `决策数从21降到1（发出swap 9→1）`、L786 门控表 `21`——zh 内部矛盾。
- en L786–791 已改为 21/1/1/1，并加注 `(Decision counts are global, i.e. DP0 DIAG lines; an earlier version counted all 8 ranks' log lines, inflating by 8×.)`；zh 无此注。
- 类型：AUDIT3 A4 残留（仅 en 落地）。72=9个swap窗口×8 rank 的日志行。
- 修：zh 表改 21/1/1/1 并同步口径注。

### MF6. en §3.3 仍断言"阻塞是可用空间的1.4倍，注定净负"（zh 已按 AUDIT3 N2 修复）
- en L296：`the blocking is 1.4× the available space, guaranteeing a net loss.`
- zh L292–294 已补：实测 **+0.98%（正）而非净负**、前置阻塞解释、~2s 缺口列为开放项。en 未同步，与本文实测（默认臂 +0.98%，en §2.4/D.3）正面矛盾。
- 修：en 镜像 zh 的限定说明。

### MF7. 附录 H.2 两处错：`(driver29.sh，进行中)`
- zh L1102 / en L1092。实测结果已存在且全部对上（_d31_g_*：69.88/340.4/69.11/69.49/69.17/72.19/69.05/68.87；_d31_a_adwd 67.05；_d31_b_base 62.9），网格是 **driver31** 跑的（结果文件名即 `_d31_g_M*`；NOTES.md 17:25 预注册、19:35 "d31 完成"）。§3.5（zh L330/en L328）正确引用 driver31。
- 修：H.2 改 `driver31.sh`、删"进行中"。

### MF8. H.3 P2：zh 方向写反 + markdown 损坏；en 整段滞后（缺 d34 结果）
- zh L1134：`吞吐随M单调**上升**（65.8→67.9 s，最优M=16）`——d34 原始数据（_d34_*：65.77→65.92→66.37→67.15→67.94 s）是**时间随 M 增加**，即吞吐随 M **下降**（否则最优是 M=256 而非 16）。同句还有损坏的 ``与`segp_L1000$上的``。
- en L1112 仍是 d34 之前的文本（"the latency side needs an experiment with a shorter L_seg"），与 zh §3.5/H.3（方向已由 driver34 确认）矛盾；en §3.5 也因此仍写"延迟侧尚未检验"。
- 修：zh 改"时间随 M 单调上升（吞吐下降）"、修 markdown；en 补 d34 段并同步 §3.5 末句。

### MF9. F.5 脚注‡/† 与单元格矛盾（两版；en 连单元格也是旧的）
- zh L919 单元格 `+2.57%‡ | +2.70% | 105%`；脚注‡（L920）却解释 `Δmax=2.29%、x_eff=0.061`，且断言"压到1.02时 x_eff 仍是 0.061"——算术错：r_after=1.02<r_k=1.032 时 x_eff=(1.107−1.032)/1.107=**0.068**（正是 §2.4 主表值）。脚注†"driver18.sh 正在补"过时。
- en L910 单元格本身仍是 `+2.29%‡ | TBD† | undetermined`（见 CF4）。
- 类型：AUDIT3 N9 未修。修：‡ 按 2.57%/0.068 重写；删过时†。

### MF10. D.3 悬浮的过时†脚注（两版）
- zh L779 / en L766：`该行的"实测"仍空缺…driver18.sh正在…补测…若同源实测收益显著高于+0.3%…`——而 D.3 本表该行已填 `+2.70% ✅`、开销 `≤0.4%（反推）`。脚注已无锚点且全段过时。
- 修：删除或改写为对 +2.70% 的说明。

### MF11. 表4（30B）与表6的30B行缺复现性告诫
- zh L477–483 / en L466+：−2.6%/−3.9%/+0.9% 三行来自 /tmp/exp_data（已核 qwen30b_ab3_*.json 的 dataset 字段）；表3 对同类数据有†告诫，表4 没有；表6 的 30B 两行也没有。也不指向可复现的 driver30 三臂。
- 类型：AUDIT1 C2/M18 残留。修：加†式告诫 + 指向 driver30/d28。

### MF12. §3.2 α 表：zh 被告诫段拦腰截断；en 完全没有跨轮次告诫
- zh L266–272：告诫段插在 α=0.5 行与 α=0.9 行**之间**，markdown 表格被劈成两段（α=0.9 行脱离表头）。
- en L267–271：表格完整但**没有**该告诫——+2.5/+10.6/+6.9 三行被呈现为受控对比（AUDIT1 M17 的问题在 en 原样保留）。且这三行依赖的多域16K 是已删除的 /tmp 数据集，两版均未标注。
- 修：zh 把告诫移到表后；en 补告诫；加数据集不可获取标注。

### MF13. REPRODUCE.md / REPRODUCE_EN.md：引用的脚本不在 repro/，且漏掉 d39 头条
- 两版开头声称"所有驱动脚本保存在 repro/"，但 repro/ 只有 driver12–24/26/28–31（已核 ls）；**driver27/32/34/35/38/39 只在 /workspace/logs/**。被点名的 `repro/driver38.sh`（头条实验）、driver27（§4）、driver35（已知限制2）均不存在于 repro/。
- 两版指南的多域条目只给 **+5.80%（driver35）**，完全不提 d39 的 **+9.76% 头条**——正是论文要求严格区分的两个量，在复现指南里被漏掉一半。
- 修：把缺失脚本拷入 repro/（或改指针）；新增 d39 小节并写清 identity-vs-identity（头条）与 bal-vs-OEPLB（adaptation）的区别。

### MF14. en G.3 项1 把幂律拟合集写成四点（含 235B），与"盲测"矛盾
- en L1066：`Four sweeps (57B/EP2, EP4, EP8; 235B/EP8) give r_k−1=0.00408·EP^1.52, with a cross-model blind-test error of 3.8%.`——235B 既在拟合集内就不是盲测。
- zh G.3 L1081 与两版 §2.4 均为"用57B的三个EP点拟合，235B/EP8 是跨模型盲测（拟合从未见过235B）"。
- 修：en 改为 3 点拟合 + 235B 盲测的表述。

### MF15. 附录 B.1 总计行不闭合（AUDIT3 N5 未修，两版）
- zh L671–679 / en L659–666：基线 7323+5479+6214+4731=**23747**≠总计 **25546**（差1799）；OEPLB 6300+4015+6179+4695=**21189**≠**23082**（差1893）。各分类 Δ% 与总计 Δ%（−9.6%）各自自洽，但表内不闭合；且无"其他"行、无数据集/脚本指针。
- 修：加"其他"类行（若 25546 含未列 kernel）或更正总计；补测量出处。

---

# MINOR

1. **zh L98 / en L98**："残差平方和低**13×**" vs G.2(b) zh L994/en L999 "低**12.1×**"（1.866/0.154=12.1）。应统一 12.1×。（AUDIT3 N11 未修）
2. **zh L94 / en L94**："附录G的T(r)扫描（7个布局点×2轮，14次运行0错误）"只对 57B 两个扫描成立；235B 是 6点×2轮=12次且第7点 r=4.686 无法启动（G.2(h) zh L1047、E.3 zh L867 "只有5个构造点"）；30B 是 15 次（脚注‖）。建议按配置分述。（AUDIT3 N16 未修）
3. **zh L1056 / en L1047**：G.2(h) 算例 `f_sens=58.78×1.721/203.11=0.496`；按印刷拟合式 T(1.721)=203.98，101.16/203.98=0.498。203.11 疑为旧拟合残留。（AUDIT3 N10 未修）
4. **zh L231–233 / en L232–233**：§2.6 排队表 ρ=0.85 应 2.00×（非 2.13×），ρ=0.90 应 4.83×（非 4.76×）；ρ=0.70 行 1.26× ✓。（AUDIT1 M13 未修）
5. **zh L489–494**：表5 行内矛盾 27.2/26.9−1=**+1.1%** 而表写 +0.4%（"超越+2.6pp"继承该错）。（AUDIT1 M18 未修）
6. **zh L423 / en L421**：Frozen-EPLB 22668.1/20167.8−1=**+12.4%**，表写 +13.0%。（AUDIT3 N7 未修）
7. **zh L527**："ratio从1.72降到**1.05**"——其余各处（表1/表2/§1/§3.3）均为 ~1.02。（AUDIT3 N8 未修）
8. **zh L159**：β"极差18%"——(0.352−0.285)/0.352=19%。（AUDIT1 minor1 未修）
9. **zh L120 / en L120**："FLOP占比高估 f_sens 约 **1.3–1.9×**"、A.3（zh L652）与附录G引言（zh L935）作 "**1.4–1.9×**"：§2.4 自己的对照表给 1.4/1.4/1.3×，实际范围 1.3–1.4×；1.9 属于 β-vs-FLOP 比值（1.37–1.93）。上界数字不一致。
10. **zh L184 / en L184**：EPLB 净预测中 `0.68×(1−0.77)，见§2.2`——0.68/0.77 在 §2.2 及全文无定义，指针落空。（AUDIT3 N14 未修）
11. **zh L358 / en L37、L352**："减少约50%记录开销"无推导，两个理由互斥（10:1⇒~91%；CUDA graph 下 decode 记录零开销⇒~0）。（AUDIT1 M12 未修）
12. **zh L600（§7）/ 脚注‖ zh L150**："record开销在30B上达1.6%"无推导（表8 不能直接给出），且已被 30B 新脚注引用而扩散。（AUDIT1 minor6 未修）
13. **zh L812 / en L803**：部署建议4 "0.37s/rank/决策，占比0.2%"——门控表 3.68s/21决策=0.175s/决策；0.37s 是 235B 稳态值，疑张冠李戴；0.2% 分母未给。（AUDIT3 N15 未修）
14. **zh L209 / en L209**："要付 **12.5%的显存**和8.1%的并发量"——与 zh §2.2 自己的更正（12.5% 是专家槽位，实测显存 +2.8%）矛盾；**en §2.2 L80 更仍是旧版**（"occupying about 12.5% additional GPU memory"）。（AUDIT3 A10 未修 + en §2.2 未修）
15. **zh L132 / en L132**：主表表头只交代 235B/57B 8卡的 f_sens 来源，未覆盖 4卡行与 30B 行（虽有♦脚注）。（AUDIT3 R9 未修）
16. **zh L786 vs L292**：门控表 headroom=**2.92s**，§3.3 按定义 β(r_b−r_k)T_flat=0.285×0.119×82.86=**2.81s**（2.92=Δmax×T(r_b)，口径不同）；swap/headroom 随之 1.26 vs 1.31。统一口径。
17. **zh L314 vs L291/293（en L312 vs L289）**：首次决策 "132个op×27.5MB≈3.6GB" vs §3.3 表 "#1 139 ops"。算术已修（132×27.5MB=3.6GB ✓）但 op 数两处不一。（AUDIT3 R13 部分未修）
18. **zh L466 / en L455**：表3 L256 行 tps 118.0→121.0 隐含 +2.54%，Delta 列写 +2.70%（时间口径 139.06→135.40 ✓）；CV 行 "L256 0.24%/0.10%" 臂序与 L512 行约定相反。（AUDIT1 minor4 未修）
19. **zh L750/L921/L925（en L760/L911/L916）**："driver17.sh 正在录制/在测"——driver17 已完成（driver17.log 末尾 D17_DONE，含各数据集 r_identity）。F.5 对已撤回 +4.7% 的"三种出路"定量分析仍把它当活对象（AUDIT2 M5 残留）。
20. **zh L121–130 / en 同位**：β_c 组件分解仍以"*证明*…□"呈现并导出 f_sens=0.384，证明处无就地告诫（0.384 已否证、低估26% 只在邻近文本）。（AUDIT1 minor11 未修；任务检查点5——属"部分告诫"，建议证明后补一句定量不可靠声明）
21. **zh L615–619 / en A.1**：定理1 无证明、2/N_G 因子无解释。（AUDIT3 N13 未修）
22. **en L152**：数学排版损坏——"(, 16000 requests)"、"DIAG's {\text{before}}$"、"{\text{sens}}$"、"(r)$ sweep"（行内公式被截断）。
23. **zh L606 / en L593**：§8 "+5.3%到+17.5%"区间低于 d38 确认的 +19.43%（同一数据集）；两个结论对多域头条 +9.76% 也只字未提。建议扩区间或加注，并考虑在结论提及 d38/d39。
24. **en L493**：表6 多域行残留未翻译中文"(重测)"；OEPLB优势列空（zh 为 +2.0 pp）。
25. **en L697**：附录C 仍写 "(For all 20 cells, see the English version PAPER_en.md)"——在英文版里自指；zh L712 已改指 gridL*.json。（AUDIT1 M20 部分未修）
26. **zh L150**："30B自己的T(r)扫描（15次运行）"——d28 实为 7布局×2轮=14 次 bench + 1 次录制 run（_d28_rec.json）。建议写"14次布局运行（另1次录制）"。
27. **zh L565–571 vs L436**：同一数据集（L512_O1_realprover）OEPLB 时间跨 campaign 为 174.3±2.2%（d32）vs 168.5（d38）、baseline 204.4（复测）vs 201.2（d38），差 ~1.5–3.3%，大于各自报告的 CV；"+17.5% 与 +19.43% 在 run 间噪声内一致"可辩护，但建议加一句对账（两版同）。

---

# 复核通过项（本轮独立重算/回查原始数据确认）

- **30B β=+0.207 全链**：用 counts30b.json + plc30b_*.json 独立重算各布局 r_avg（bal 1.0001 / r110 1.1000 / r120 1.2000 / id 1.3376 / r140 1.3999 / r160 1.5997 / conc 3.1225），对 _d28_* 时间（63.98/65.98/65.72/65.91/68.16/71.51/91.83 s）做网格搜索铰链拟合：**T_flat=63.98、B=13.25、r_k=1.031、β=0.2071**，与论文♦脚注完全一致；f_sens=0.260、Δmax=6.36%（f·x/(1−f·x) 与 β(r_b−r_k) 两式互差<0.01pp）✓。全文 grep 无残留"30B f_sens<0/上界为负"的活跃断言（仅存于明确标注已撤回/已否证的历史叙述，zh/en L150）✓。
- **d38/d39/d35 原始 JSON**：+19.43%（201.24→168.49 s；r1 18.43/r2 20.46；CV 0.02%/1.18%）、+9.76%（824.70→751.37 s；r1 8.08/r2 11.49；identity CV 0.13%）、+5.80%（801.18→757.25 s）全部与 §5.3 文字一致 ✓。driver39.sh 注释明确 d35=bal 基线（adaptation benefit）、d39=identity 基线（headline），两量的区分在 §5.3（两版）表述正确 ✓。
- **d30 三臂**：67.11→69.76=**−3.80%**、→68.95=**−2.66%**、→66.76=**+0.53%** ✓。
- **d31（附录H表）**：全部 9 臂 + 参照（62.9 s）与 _d31_* JSON 逐一吻合（340.4/69.9/69.1/69.5/69.2/72.2/69.0/68.9/67.0）；P1' 极差 387%/0.5%/4.8%、P3 的 3.6%/44% 复核无误；与 NOTES.md 17:25 预注册及 19:35 记录一致 ✓。
- **L512 复测**：baseline 204.28/204.50、EPLB 202.44/199.08（+1.8%）、OEPLB 175.21/172.92（+17.4%）✓ 支持 +17.5%/+1.75%/15.7pp。
- **重测表上界**：2.57%/2.86%/2.26%/2.21% 溯源至 EXPERIMENT_NOTES（x_eff=0.0753、r_b=1.116/1.0980/1.0965）并重算 ✓；η 列（105/84/≈0/6%）算术 ✓。
- **公式抽查**：Δmax=f·x_eff/(1−f·x_eff) 六行（235B 22.09、多域 11.86、57B EP8 3.40、EP4 2.57、30B 6.36、ShareGPT 22.09）✓；β=B/T_flat 四配置（0.285/0.342/0.352/0.207）✓；Δ_ceiling=β(r_b−r_k) 与 Amdahl 式代数等价 ✓；M_min 算例 3.8/58 ✓；M★ 闭式与（N4 修正后的）目标函数 FOC 一致 ✓；r_k 幂律 0.00408·EP^1.52 四点预测/实测 ✓；§2.5 r_place 值与 two_ceilings.py 重跑一致（1.0039/1.0100/1.0003/1.0345/1.1709/1.8725/1.3710/10.8251）✓。
- **杂项**：摘要与 §5 数字全部对账一致（17.5/15.7pp/5.95s/14.7s/3.4%/7.3%/0.37s/1.55s/+5.3%）；"低一个数量级"已改"约4倍"（两版）✓；"收敛后不再触发"撤回注两版齐备 ✓；"638个swap配对"已删 ✓；附录C 网格值与 COMPREHENSIVE_EXPERIMENT_LOG L22 抽查吻合 ✓；表1 其余行（−20.4/+9.0/+18.4/+21.3）与基线单元格重算吻合 ✓；−62%~−68% 口径两版一致 ✓。

---

# 总体结论

论文的科学主干在本轮审计中经受住了独立重算：30B 的 β=+0.207/正上界/η<0 归因链条完整且与 driver28/30 原始数据逐一吻合，全文不再有活跃的"30B 负上界"断言；d38 对单域 +17.5% 的确认（+19.43%）和 d39/d35 的多域两个量（+9.76% 头条 vs identity、+5.80% 自适应收益 vs 静态最优）在 §5.3 的表述正确且可溯源。**但论文尚不能宣布完成**，阻塞项有二。其一是英文版对最近一轮修复系统性滞后——表1 ⚠注缺失、结论仍写 2–10pp、+14.0% 是否被 −1.1% 否证的立场与中文版相反且 en 自相矛盾、driver18 结果在 en 三处仍是 TBD、30B"死区未测"旧脚注仍在——使 en 版当前自相矛盾、不可单独发表。其二是中文版仍有六处 MAJOR 残留：D.3 四臂表的 72、F.5/D.3 的过时脚注、§2.5 表行内混用两个数据集、v2 数据集 −1.1% vs −0.24% 不一致、§5.3 排序句跨 run 拼数、α 表被告诫段截断；外加 REPRODUCE.md 点名不存在的脚本且漏掉 d39 头条。所有问题均属传播、脚注与算术修订，无需新实验；按 CRITICAL→MAJOR 顺序修复后，两版可望达到内部一致的发表标准。
