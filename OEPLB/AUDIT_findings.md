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
