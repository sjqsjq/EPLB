# PictureGuide — 每张图的目的、数据集配置、数据集路径、trace 路径

模型:Qwen3-235B-A22B-FP8(TP=DP=EP=8, 94 MoE层/128专家/top-8),SGLang+DeepEP+DeepGEMM FP8,8×H20。
数据集根:`/data/minghua/sjq/shortdomain/`。trace 根:`/data/minghua/sjq/paperpicturetrace/`。

---

## 一、动机/跨域图(Fig 0,1,1b,2,2b,3,4,7,9,13)

**Fig 0 放置过程方法论**
- 目的:讲清"不均衡度怎么算、identity/optimal 放置是什么"。
- 数据集+配置:book,layer 62,identity 放置,离线录制。
- 数据集路径:`/data/minghua/sjq/shortdomain/`(bookcorpus 见 bookcorpus_grid)。
- trace:`/data/minghua/sjq/paperpicturetrace/fig1_2_3_4_7_9_13_motivation_crossdomain/counts235b.json`(94×128,book)。

**Fig 1 逐层负载不均衡度**
- 目的:identity 下逐层 max/min 负载比(2.26-4.38×,最极端 11.79×)。
- 数据集+配置:book 4438tok,identity,离线录制。
- 数据集路径:`/data/minghua/sjq/shortdomain/`(book)。
- trace:`counts235b.json`(同上,max/min 10.35≈论文 11.79×)。

**Fig 1b per-forward 不均衡度分布**
- 目的:per-forward ratio(median 3.7-6.4×,聚合的 1.5-1.7×)。
- 数据集+配置:MMLU/prover/book,conc=256,O=10,identity+routing_tracer。
- 数据集路径:`mmlu_O10_8k.jsonl`/`prover_short_O10_8k.jsonl`/book。
- trace:`paperpicturetrace/fig5_6_8_14_PD_correlation/rt2_<dataset>/rank*_fwd_chunk*.npz`。

**Fig 2 identity vs 最优放置**
- 目的:LPT 最优把 ratio 降到 ~1.0(降 53-73%)。
- 数据集+配置:book 4438tok,identity vs LPT 模拟。
- 数据集路径:book。
- trace:`counts235b.json`(+ gen_placement.py 模拟)。

**Fig 2b per-forward identity vs LPT**
- 目的:per-forward 粒度 identity vs LPT 对比。
- 数据集+配置:3 数据集,conc=256,O=10。
- 数据集路径:`mmlu/prover/book O10 jsonl`。
- trace:`rt2_<dataset>/*.npz`。

**Fig 3 per-GPU 负载**
- 目的:identity 下各 GPU 负载(最重 GPU3 7.7M,最轻 GPU2 0.66M)。
- 数据集+配置:book。
- 数据集路径:book。
- trace:`counts235b.json`。

**Fig 4 跨域路由相似度矩阵**
- 目的:跨域 Spearman ρ≈0(MMLU vs prover 0.054,vs book −0.038)。
- 数据集+配置:MMLU/prover/book 三域,identity 离线录制。
- 数据集路径:`mmlu_O10_8k.jsonl`/`prover_short_O10_8k.jsonl`/book。
- trace:`counts235b.json`(book)+ rt2_MMLU/prover 重建。

**Fig 7 专家激活频率热力图**
- 目的:94×128 热力图,3 域热点不重叠。
- 数据集+配置:MMLU/prover/book。
- 数据集路径:同 Fig 4。
- trace:`counts235b.json`+ rt2 重建。

**Fig 9 跨域放置迁移矩阵**
- 目的:为 A 优化的 LPT 放置迁移到 B 后 ratio 2.1-4.1(MMLU→prover 3.67>identity 3.51)。
- 数据集+配置:MMLU/prover/book 跨域 LPT 迁移。
- 数据集路径:同 Fig 4。
- trace:`counts235b.json`+ rt2 重建。

**Fig 13 多粒度不均衡度**
- 目的:per-forward(4.5×)是聚合(2.3×)的 2×。
- 数据集+配置:3 数据集,conc=256。
- 数据集路径:`mmlu/prover/book O10`。
- trace:`rt2_<dataset>/*.npz`。

---

## 二、PD 相关性图(Fig 5,6,8,14)

**Fig 5 MMLU 逐层 prefill→decode ρ**
- 目的:MMLU ρ=0.833,94/94 层强。
- 数据集+配置:MMLU 25tok,conc=256,O=10,identity+SGLANG_OEPLB_ROUTING_TRACE。
- 数据集路径:`mmlu_O10_8k.jsonl`。
- trace:`paperpicturetrace/fig5_6_8_14_PD_correlation/rt2_MMLU_25tok_QA/rank*_fwd_chunk*.npz`;ρ 脚本 `/workspace/EPLB/benchmark/analysis/compute_pd_rho.py`。

**Fig 6 top-K 专家 overlap**
- 目的:prefill vs decode top-K 重叠(K=5→49%,K=40→75%)。
- 数据集+配置:MMLU,conc=256,O=10。
- 数据集路径:`mmlu_O10_8k.jsonl`。
- trace:`rt2_MMLU_25tok_QA/*.npz`。

**Fig 8 ρ vs prompt 长度**
- 目的:ρ 随长度变化(MMLU 25tok 0.833,prover 1253tok 0.98,book 4438tok 0.967)。
- 数据集+配置:7 数据集,conc=256,O=10。
- 数据集路径:`mmlu/prover/book` 等 O10 jsonl。
- trace:`rt2_<dataset>/*.npz`。

**Fig 14 9 数据集 PD 相关性 bar**
- 目的:QA/推理强(0.78-0.85)、中文多语言 0.616、数学 0.44-0.69、代码 0.485。
- 数据集+配置:9 数据集(MMLU/ARC/ARC-E/CSQA/OBQA/GSM8K/prover/HumanEval/CMMLU),conc=256,O=10。
- 数据集路径:9 个 `*_O10*.jsonl`(shortdomain 下)。
- trace:`rt2_<9 datasets>/*.npz`;ρ 结果 `paperpicturetrace/fig5_6_8_14_PD_correlation/short_pd_9dataset_samesession.json`;重画脚本 `/tmp/plot_fig14.py`。

---

## 三、热点 GPU/路由原型图(Fig 12,12b,12c,12d)

**Fig 12 域切换时间线**
- 目的:MMLU→prover→book 热点 GPU 瞬间切换(GPU4→GPU5→GPU0)。
- 数据集+配置:3 域拼接,per-forward 热点 GPU。
- 数据集路径:`mmlu/prover/book O10`。
- trace:`rt2_MMLU/prover` + `counts235b`(book)。

**Fig 12b identity vs LPT 热点 GPU(7 域)**
- 目的:identity 6/7 域热点=GPU4,LPT 后 entropy 2.7-2.9。
- 数据集+配置:7 域,各 100 forward。
- 数据集路径:`mmlu/arc/csqa/obqa/gsm8k/prover/arc-e O10`。
- trace:`rt2_<7 datasets>/*.npz`。

**Fig 12c 9 数据集热点 GPU + entropy/switch_rate**
- 目的:pinned(entropy<1,prover 0)vs volatile(≥1,book 1.89)。
- 数据集+配置:9 数据集。
- 数据集路径:9 个 O10 jsonl。
- trace:`rt2_<9>/*.npz`。

**Fig 12d pinned vs volatile**
- 目的:prover(全 GPU5,entropy 0)vs book(散布,entropy 1.89)。
- 数据集+配置:prover + book,100 forward。
- 数据集路径:`prover_short_O10_8k.jsonl`/book。
- trace:`rt2_prover_107tok_math/` + `counts235b.json`。

---

## 四、OEPLB 在线运行图(Fig 10,11,15,15b,16,17,17b)

**Fig 10 swap timeline / Fig 11 remap effect**
- 目的:swap 决策时间线 + remap 效果。
- 数据集+配置:crossdomain_freq6,conc=32,O=10,--enable-pb-oeplb。
- 数据集路径:`crossdomain_freq6_O10.jsonl`。
- trace:`paperpicturetrace/fig15_15b_16_17_17b_oeplb_online/rt_churn_A_adpt_rank0/` + `server_churn_A_adpt__DIAG_ADW_TIMING.log`。

**Fig 15 OEPLB 真实在线时间线**
- 目的:97 次决策,域切换 spike(1.35-1.72)→swap→稳态(1.01-1.05)。
- 数据集+配置:crossdomain_freq6,4200 req,conc=32,O=10。
- 数据集路径:`crossdomain_freq6_O10.jsonl`。
- trace:同 Fig 10。

**Fig 15b entropy 对比 / Fig 16 逐域收敛 / Fig 17 ratio / Fig 17b per-domain**
- 目的:entropy(identity 0-1.9→OEPLB 2.1-2.9)、逐域首决策降幅-24~-33%、prover 1.166→1.006(−14%)。
- 数据集+配置:同 Fig 15。
- 数据集路径:同上。
- trace:同 Fig 10/15。

---

## 五、对比/开销图(Fig A,B,C,D,E)

**Fig A 放置谱系**
- 目的:Worst→identity→EPLB→PB-OEPLB→Oracle。
- 数据集+配置:crossdomain_freq6/per-dataset,本 session。
- 数据集路径:`crossdomain_freq6_O10.jsonl`+9 数据集。
- trace:`paperpicturetrace/figA_B_C_D_E_placement_overhead_eplb/perds_gain.json` + `server_churn_A_adpt__DIAG_TIMING_overhead.log`。

**Fig B ratio 收敛**
- 目的:max-delta 停滞 1.26 vs gap-targeting 3 窗到 1.02。
- 数据集+配置:crossdomain_freq6。
- 数据集路径:`crossdomain_freq6_O10.jsonl`。
- trace:`server_churn_A_adpt.log`(DIAG 时序)。

**Fig C EPLB vs OEPLB 全场景**
- 目的:OEPLB 每 dataset 超 EPLB。
- 数据集+配置:6 数据集,identity/OEPLB/EPLB 三方,conc=256,O=10。
- 数据集路径:`mmlu/arc/cmmlu/prover/humaneval/book` O10。
- trace:`perds_gain.json` + `server_eplb_MMLU__EPLB_comparison.log`。

**Fig D 开销分解 / Fig E 迁移阻塞**
- 目的:swap 3.42% 主导;稳态 0.37s vs EPLB 1.55s。
- 数据集+配置:crossdomain。
- 数据集路径:`crossdomain_freq6_O10.jsonl`。
- trace:`server_churn_A_adpt.log`(TIMING begin())。

---

## 六、死区/增益上界图(Fig G,H,I,J,K,L)

**Fig G 两个天花板 / Fig H 跨模型效率 / Fig I 铰链 / Fig J 边际 swap / Fig K r_k 幂律 / Fig L 30B 验证**
- 目的:r_place≤r_k 零冗余;Δ_max×η 两段(235B η79%/57B 84%/30B ≈0);T(r)铰链 R²=0.998;第1次 swap 100%有用;r_k−1=0.00408·EP^1.52;30B Δ_max 正但 η≈0。
- 数据集+配置:57B+235B+30B,T(r)扫描(7 布局×2 轮),离线 counts。
- 数据集路径:57b/235b/30b 离线录制。
- trace:`paperpicturetrace/figG_H_I_J_K_L_bound_theory/counts235b/30b/57b*.json` + `bound_curve.py`。

---

## 七、消融图(Fig F,F2,M,N)

**Fig F 衰减系数 / Fig F2 α-sweep**
- 目的:α=0.5 最优;α-sweep 显示固定 α 非全负载最优(A α0.9、C α0)。
- 数据集+配置:crossdomain_freq6 4438tok conc32 / universal_16k conc256;α=0/0.5/0.9。
- 数据集路径:`crossdomain_freq6_O10.jsonl`/`crossdomain_universal_16k_O10.jsonl`。
- trace:`paperpicturetrace/figF_F2_M_N_ablation/figF2_alpha_sweep_throughput_reconstructed.json`(d3/d4 重建)+ `figF2_decay_sweep_hotGPU_arrays.json`;**重跑中**:`/workspace/logs/rerun_f2_alphasweep.runlog` + `figF2_alphasweep_insession.json`。

**Fig M KV cache 压力**
- 目的:EPLB 16 副本→12.5% 显存→KV -8.1%→排队 2-4.8×。
- 数据集+配置:EPLB 配置分析(ad-hoc 计算)。
- 数据集路径:—(分析,非数据集)。
- trace:ad-hoc(无单一文件,从 EPLB 配置 16 副本×专家显存算)。

**Fig N M 收敛/统计充分性**
- 目的:同 M=128 不同(W,α)聚簇(4.8%)→M 充分。
- 数据集+配置:segp_L1000(纯 prefill 变点),(W,α)同 M 网格。
- 数据集路径:`/workspace/logs/segp_L1000.jsonl`(d31 driver)。
- trace:`paperpicturetrace/figF_F2_M_N_ablation/_d31_g_M<W>a<α>.json`(8 个,带 tps)+ `_d34_d34_M<a>.json`;重画脚本 `/tmp/redraw_ablation2.py`。

---

## 八、架构图

**system_architecture**
- 目的:五组件(routing tracer→controller→rebalancer→async executor→p2l map)+ 四观察标注。
- 数据集+配置:—(示意图)。
- 数据集路径:—。
- trace:`OEPLB/figure1/system_architecture.png`(已复制到 NEW_PAPER/figures/)。
