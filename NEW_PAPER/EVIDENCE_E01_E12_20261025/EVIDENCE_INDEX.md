# EVIDENCE_INDEX — E01–E12 原始材料清点（只读审计，2026-10-25）
Verdicts: FOUND = 清单要求的原始材料在档且已入包/已登记 path+sha; PARTIAL = 主体在档但有登记在案的字段缺口; MISSING = 未找到（如实登记，见 GAPS_AND_RETEST_DESIGN.md 的 G# 与 D#）。
Manifest: manifest.tsv / manifest.json（13,171 条：e, tag, rel, bytes, mtime, sha256, packaged, src, note）。
Build: 13,157 文件 + 6 sharegpt 增补 + 8 空目录登记；拷入 11,286 文件 ≈772MB；哈希 13,025；ROOT ERROR=0；SOURCE_MISSING=0；core 28 个（547GB）只登记不哈希不移动。

## 总表
| E | 主题 | 判定 | 包内目录（tag 前缀） | 主要缺口 |
|---|------|------|----------------------|----------|
| E01 | 计数均衡/padding/执行时间 | FOUND | E01_count_padding/ | G10(e0 仅日志) |
| E02 | 就绪依赖/局部延迟暴露 | FOUND | E02_readiness_delay/ | 无（不利结果已保留） |
| E03 | 收益上限跨硬件 | PARTIAL | E03_bound_crosshw/ | G1(H800 e2e 原表) |
| E04 | PD 代表性 | PARTIAL | E04_pd_representativeness/ | G2(分段脚本), G3(book_rank0) |
| E05 | 最终版端到端 | FOUND | E05_endtoend/ | G8(空目录), G9(超限大件按政策) |
| E06 | SLO/Goodput | FOUND | E06_slo_goodput/ | G11(GOOD 为 DERIVED，D6 可独立复核) |
| E07 | 阶段感知价值 | FOUND | E07_phase_aware/ | 无 |
| E08 | 差额引导规划器 | PARTIAL | E08_planner_gap/ | G4(逐规划 plan_time/迁移字节) |
| E09 | 自适应vs固定/采样消融 | FOUND | E09_adaptive_ablation/ | G9(inputs 超限件) |
| E10 | 正确性/可靠性/成本 | FOUND | E10_reliability_cost/ | 无 |
| E11 | 基线/冗余压力 | FOUND | E11_baselines_redundancy/ | 无 |
| E12 | 泛化/参数范围 | PARTIAL | E12_generalization/ | G5/G6(DSV4 尾部行与 old-sglang 原件), G1, G7 |

## E01 计数均衡、padding 与执行时间 — FOUND
- 12 干预捕获系列（观察一真实布局测量）→ E02_readiness_delay/deadzone_results/（calibration_N{256,4096}.json ×6 变体已拷入；delay/load/matching/readiness 大件 path+sha）——E01/E02 共用原始树，一份入包双向引用
- 48 次启动（12 臂×4 start）逐次计时 → E01/confirmatory_v1/（layout_effects.csv, summary.json, raw/, environment/, logs/ 全拷）
- 布局前缀/逐专家有效行数与对齐形状 → E01/h20_e0e1_files/（e1_layout_effects.csv, e1_summary.json, e0_pilot.log, e0_smoke.log, e1_pilot.log）+ E01/h20_raw/（e1_pilot.json 613KB, correctness_all_layouts_pilot.json 331KB, raw_index.json 含逐文件 sha）
- DeepGEMM 阶梯微基准原表 → E01/microbench_deepgemm/（bench_flat_clean/dense/duplication/p2p_cost.py + deepgemm_flat_0_256_clean.json + deepgemm_flat_dense.json + deepgemm_duplication.json + p2p_cost.json + deepgemm_key.png）+ E01/microbench_hooks/（microbench_hook{,2}.py, gemm_profiler_hook.py）+ E01/deepgemm_precompile_log/
- 计数改善但执行变慢的案例（不利结果保留）→ deadzone REPORT r1.02 −4.848µs（E02 tag）；gb200 RESULTS.md 附9 OEPLB ratio 1.002=171.5tps vs MoETuner 1.000=181.6tps、EPLB-static r_eff 1.517 vs accounted 1.011（E03/gb200_top_docs）；d38 原件 +15.5% 作废（E05/d38_L512_O1_comparison）
- 缺口 G10：e0 只有日志无 json（e1 全）

## E02 就绪依赖与局部延迟暴露 — FOUND
- original/strict/all-rank 干预原表 → deadzone_results/：readiness_original_N4096.json(164MB)、readiness_strict_N4096.json(163MB)、delay_dense_N{256,4096}(117/121MB)、delay_scan_N{256,4096}+pre_floor_interleave(106MB×3)、matching_route_delay_N{256,4096}(48MB×2)、route_ready_N4096(23MB)、load_scan_N{256,4096}(12MB×2) —— 全部 path+sha256；对应 *_plan_*.json 与全部 analysis_* 目录（≤4MB）已拷入
- 注入位置与实际延迟 → delay_scan/delay_dense + timeline-{barrier,original}-{inside,outside,zero}_N4096.json ×6（已拷入）
- 各 rank 就绪与层结束时间 → readiness_analysis_N4096/（已拷入）；关键数值在 deadzone_top_docs/REPORT.md：H original 16.729/18.075µs、strict 20.181/18.711µs、per-rank strict H_i 26.876–42.907µs
- 原始时间线 → timeline_analysis_N4096/ + timeline_plans_N4096/ + deadzone_traces/（nsys 66MB path+sha）
- 不利结果保留：跨 rank 冻结 H_i 未验证（Pearson 0.086 / RMSE 27.425µs）；ΔR_global 直接 RMSE 6.734（主）vs legacy 11.895 vs frozen-H_i 21.628（失败）；smoke_N256_cleanup_failed.json 已拷入
- 版本分离：calibration/delay_scan 的 pre_floor_interleave、pre_uniform_kernel 变体与终版同 tag 并存（文件名即版本）

## E03 硬件、模型与负载相关的收益上限 — PARTIAL（G1）
- H20 → E01/E02 树 + h20 顶层 ENVIRONMENT.md（_COMMON）；A100 → a100_theory_validation_newpaper/（counts235b_a100.json sha f8ca2858… + fits/）与 theory_a100_oeplb/（launch 脚本+data/）——互补双副本，THEORY_VALIDATION_A100.md 两侧字节一致（sha 02e1efca…）；a100_baselines_repro/（EXPERIMENT_LOG.md, METRICS_MATRIX_A100.md, results 113 json）；a100_bf16_comparison/
- GB200 → gb200_results/(394 json)、gb200_profiles/(27 csv：prof_{eplb,identity,oeplb}_kern + util_*)、gb200_placements/(5)、gb200_scripts/(39)、gb200_top_docs/(RESULTS.md 43KB, TABLES.txt, paper_GB200.md, paper_with_5.10.md)、gb200_logs/(≤2MB 拷入)
- H800 → sjq_H800_*（5 个 Dec-2025 网格 json 全拷 + failure_report 原件）+ sjq_H800profiler_result_HASHONLY(626MB)/sjq_H800lama70B_profiler_result_HASHONLY(178MB)/H800profiler_result.tar.gz(626MB, sha 已登记)——为时间参数剖析来源；**§5.11 自旋吸收 11 次端到端原表 MISSING（G1，从未归档）**
- 跨硬件网格补充 → sjq_A100_*/sjq_H20_*（11 目录 3.8MB 全拷）+ 4 棵 A100/H20 profiler 树（629/201/593/179MB path+sha）
- 时间参数拟合/校准与独立验证 → oeplb_repro/（counts235b/30b/57b{,_l512,_multi,_share}.json ×6 + fit_f3.py + two_ceilings.py + bound_curve.py + r_avg/r_window.py + drivers）；prediction_v1/（calibration/ + evaluation/ 分离，≤2MB 拷入、大件 path+sha）；prediction_capture_failed_attempt01/（失败原件整目录）；nearscan model_holdout.json（E08 tag）
- 逐项预测与实测 → prediction_v1/evaluation + gb200 profiles util_* 系列
- r_k/β/f_sens/η 表 → THEORY_VALIDATION_A100.md、gb200 RESULTS.md、studypaper/06+08（E12 tag）、oeplb_fig_bound.png

## E04 Prefill–Decode 代表性 — PARTIAL（G2/G3）
- 九任务 PD 原迹 → rt2_nine_task_npz_HASHONLY/：fig5_6_8_14_PD_correlation 下 9 个 rt2_* 任务 282 个 npz（181MB，逐个 path+sha256）；九任务数据集原件 gb200_nine_task_datasets/（arc_easy, cmmlu, csqa, gsm8k, humaneval, mmlu, obqa, prover256, sd_prover512_O32 全拷 9.6MB）
- 94 层相关性/请求与 forward 覆盖 → oeplb_pd_docs/（PREFILL_DECODE_CORRELATION_STUDY.md, PREFILLBOUNDARY_DATA.md, short_pd_{9dataset_samesession,all7,conc256_O10}.json）+ capture_and_pd_scripts/analysis/compute_pd_rho.py
- 3 任务相关研究原迹 → pdcorr_3task/（README+jsonl ≤1MB 拷入；traces/{book_4438tok_O10,mmlu_25tok_QA_O10,prover_1253tok_math_O10} 每任务 32 npz（rank×chunk）path+sha）
- 采集配置 → capture_and_pd_scripts/（run_routing_record.sh＝topk_ids GPU scatter_add_ 记录器、run_expert_dist.sh＝SGLang 原生记录器、bench_expert_swap.py、run_pd_benchmark.sh）
- 固定映射证据 → h20_placements_fixed_map/（manifest.json＋identity_service_map.json＋followup_v1/）+ gb200_placements/（E03 tag）
- ShareGPT：分段产物 sharegpt_datasets_HASHONLY/（6 jsonl 182MB path+sha：O1_100k/O256_20k/long_O1/long_O256/natural_20k/real_4000）；**分段脚本本体 MISSING（G2）**，邻近构建脚本 freeze_segmented_scripts/freeze_requests.py 与 oeplb_repro/make_segmented.py 已入包
- fig12 rt2_book_rank0/ EMPTY（G3 登记）

## E05 最终版本端到端比较 — FOUND
- 修复后 64 次（版本包 phase4_v2_fixed/，全量拷贝 604 文件）：boots/64 × {bench.log, s_*.json, s_*_perreq.jsonl(全量), server.log(全量), assert_result.json, boot_meta.json(+.end), server.pid}；P 臂另含 v2p1_rank0.jsonl + sp_run_end.json + assert_r2_extra.json（E 臂无 window 日志＝静态臂预期行为）；supplement_matrix_locked.csv、LEDGER_phase4.tsv、ANALYSIS_PLAN_SUPP.md、PHASE4_CODE_VERSION_ADDENDUM.md、PHASE5_GATE_GAP.md、queue_phase4.sh(+run.log)、run_supp_boot.sh（内含冻结 v2 双 sha 守卫）、gen_supplement_csv.py、analysis/{SUPP_RESULTS.json(sha 2b3f7a57…), RESULTS_SUPP.md, analyze_supp.py}
- 旧 168＋32（版本包 final_plan_*/）：main 168 格（per-cell bench.json/m_*.json/meta*.txt/verdict.txt/sp_run_end.json/v2p1_rank0.jsonl/bench.log ≤128KB 全拷；perreq 163 份 + server.log 168 份全部 path+sha256；**163 vs 168 差额＝5 个 hang 格，与 INCIDENT_E3P_b3/E4F_b7/E3F_b3/E3F_b4/E3F_b8 一一对应**）；p1 32 格同规；analysis/unified 全拷（REPORT_final.md, main/p1_table.{csv,json}, stats_unified.py, unified_stats.json, f6/f7 csv, fig_f6/f7/f8 png, build_tables.py, figures.py）；scripts/ 全拷（bench_client_v2.py, launch_arm.sh 等）；C_configs/（ARM_FLAGS.md, EPLB_MECHANISM_NOTES.md, FREEZE_final_plan.md, bench_client_v2.sha256, deployed_shas_final_plan.txt, integrity_check*.py）；D_matrix/（rotation_{main,p1}.json, run_list_*×5, runlist_locks.json）；calib/（FINAL_SELECTION.json, STAGE1_SELECTION.json, stage1/24, stage2/9, INCIDENT_e6.md）；B_data/（registry{,_p1}.json, splits.json, pool_inventory.json, doc_manifest.json, domain_capacity.json, index/ ≤1MB 拷入；7 条 stream jsonl 4.5–30MB path+sha；tokcache.pkl path+sha）；smoke_f0/ + p1_smoke/ 全拷；logs/；顶层 PROGRESS.md；main 内 INCIDENT_*.md ×7 已拷
- 版本锁与代码 → code_lock_final/（SHA256SUMS_code_lock.txt + deployed_pb_oeplb/ + oeplb_src/）；h20_configs_followup_v1/（48 个协议/配置/发射器文件 ≤2MB 拷入，含 eacampaign_code_lock.json、ea_v1_code_lock.json、各战役 protocol json；>200MB 载荷 path+size 登记，G9）；h20_scripts/（analyze_*.py 等 ≤512KB 拷入）
- d38_L512_O1_comparison/（第三版本包，全拷）：原件（identity_baseline.json tps 40.1、oeplb_adaptive.json 46.3、eplb_normal_warmup.json 39.4、launch ×3、server keylines ×3、EXPERIMENT_LOG.md+bak）+ repro_20260915/（_d38repro_bl_r{1,2} 39.8/39.9、_d38repro_oe_r{1,2} 46.9/47.6、_align_oe_r{1,2} 47.3/47.9、keylines ×6、drivers ×2、EXPERIMENT_LOG_repro.md＝server_args L1673 bug 全记录）——头条增益溯源未决（G7）
- 空目录登记（G8）：A_gap_table/E_p1_protocol/F_package/freeze、H20 traces/logs

## E06 SLO 与真正 Goodput — FOUND
- 冻结 SLO 参考与资格 → phase3_prep/（SLO_REFERENCE_FROZEN.json（2026-10-20T09:24:53Z 冻结：E3 ref_p95_TTFT=6.7419s；E4 7.3127s + TPOT 0.08313s）、SAMPLING_ELIGIBILITY.json（NOT ELIGIBLE：S>1 share 0.0<0.10，如实保留）、PROTOCOL.md、LEDGER_prep.tsv、assert_prep_boot.py、extract_slo.py、extract_eligibility{,_resume}.py、queue_prep{,_resume}.sh(+log)、run_prep_boot.sh；B1-hang 4 个 core 原位登记）
- 校准请求原件 → phase3_prep/boots/ 9 个（prep3_{A1_Enative_E3,A2_Enative_E4,B1_P_E3,B1r_P_E3,B2_P_E3,B3_P_E4,B4_P_E4,C1_I_E3,C2_I_E4}，18MB 全拷含 perreq/server.log）
- 客户端全文 → bench_client_v2_copy/bench_client_v2.py（E05 final_plan_scripts 内同源；sha 见 C_configs/bench_client_v2.sha256）；语义备忘：t0 在 Semaphore(256) 获取之后（每请求 TTFT/elapsed 不含客户端排队）；TTFT=首个非 [DONE] data: 帧；comp_tokens 以服务端 usage 为准；每请求仅 3 个时间锚点（无逐 token 时间戳）
- 每次启动的整数 GOOD、N、完整 T → DERIVED_good_counts/（DERIVED__good_counts_per_boot.tsv：64 行 × {label,stream,arm,boot,n=8192(E3)/2048(E4),ok,errors,total_time_s,ttft_p95_ms,tpot_p95_ms,good_rate@{0.8,1.0,1.2},GOOD_int@k_DERIVED,goodput_req_s@1.0,ttft_p95_s_perreq}；64/64 精确整数、0 个非精确转换；DERIVED__README.md 载明源 sha 2b3f7a57… 与冻结 GOOD 定义；derive_good.py 同目录供复核）——DERIVED 标注明确，独立重数设计见 D6
- 原始请求日志 → E05 phase4_v2_fixed/boots/*/*_perreq.jsonl 全量在包（327,680 请求）

## E07 阶段感知统计的实际价值 — FOUND
- P 与 all-phase A1 完整数据 → E05 final_plan_main A1 格（decode_record_batches 1020–1054 vs P=0）+ phase4 P 臂 boots（全量）
- 实际 Prefill/Decode 记录数与成本 → sp_coupling_v1/（EXPORT_evidence_readonly、INVENTORY_endtoend_stack、MAPPING、PROGRESS、PROTOCOL_dev_r1、REPORT_sp_coupling_dev_P2（F_sparse 记录归约 4.03–4.14×）、REPORT_sp_mech_halt_seq2、RESULTS_dev_r1、REV1_exit_branch_supplement、REV1_quiet_exit_revision——v1 vs rev1 版本对照同包：裸阈值退出 26/26→0、转换 −56/−25/−45%；gate CPU 5–31ms、控制面 ≤0.06% wall）
- 阶段统计生成布局后评价后续执行 → fullservice_v1/v2/v3（analysis.json, audit_nine_starts.json, audit_online_mechanisms.json, audit_report_step1.json, bench/, diagnostics/, campaign_state.json）+ cdx0_v1/（cdx0_analysis.json, cdx0_summary.md, t_quantile_audit_note, analyzer_setup_fixes_note, capture_responses/, execution/, smoke/）+ crosstab_v1/ + et_v1/analysis.json + evidence_addendum_et_v1.json（h20_e0e1_files tag）
- 采集器源码 → tracer_src/（per_request_tracer.py, routing_tracer.py）；机制图数据 → perds_gain/（MMLU −16.6%, CMMLU −19.4%, prover +12.7%, book +13.7%）+ figA_E_overhead_npz_LIST（path+size）
- 不利结果保留：sp_coupling v1 裸阈值退出 26/26（rev1 修复至 0）；cdx0 analyzer_setup_fixes_note

## E08 差额引导规划器比较 — PARTIAL（G4）
- PB-gap/max-delta/best-improvement 同输入快照 → ea_v1/（fivearm_campaign/、fivearm_campaign_analysis.json（A11−A00 +10.40s [5.00,15.80]；A11−P11 −0.44ns）、fivearm_campaign_summary.md、calibration_grid/、gating_G_decision.json、grid_selection.json、switch_independence_static{,_postp11}.json、c5_dynamic_verification.json）+ eacampaign_code_lock.json/ea_v1_code_lock.json（E05 h20_configs tag）
- 规划器离线质量 → ep_v1/planner_offline_quality.json（全拷）
- 接受轨迹/单位成本 → nearscan_v1/（cost_benefit.json：单 swap ≈1360.8–1407.6µs、model_holdout.json、analysis.json、**analysis_attempt1_retired_mechanical_region_label_bug.json + analyzer_run_attempt1_retired.log 不利原件保留**、execution/）
- 旧 A2 完整数据 → E05 final_plan_main A2 格（+8.21% p=0.0102、swaps 26–28 vs 12–14）+ phase4 无 A2（矩阵仅 I/P/E/F）
- 规划器源码 → rebalancer_src/（rebalancer.py, probe_admission.py）+ predict_gain_tool/predict_gain.py
- 初始映射/预算 → E05 D_matrix + C_configs ARM_FLAGS + h20_placements（E04 tag）
- 缺口 G4：逐规划 plan_time_s/migration_bytes 显式字段未见 → 设计 D3（CPU-only 离线审计）

## E09 自适应相对固定日程、采样消融 — FOUND
- P/F/A3 逐次结果 → E05 final_plan_main（P/F/A3 格 per-cell 小件全拷 + perreq/server path+sha）+ phase4_v2_fixed（P/F 臂 64 次全量）
- S/H/W 轨迹 → adw_v2p1/（calibration_m1m2m3/、formal_eval/、r2/、r3/、next_version/、acceptance_results.md、config_v21.diff、controller_v21.diff、minimal_diff.md、calibration_plan_v2p1.md；W0 拟合 8037/8037、FREEZE knobs）+ adw_v2/（v2 冻结对应件）+ 各格 v2p1_rank0.jsonl（phase4 P 臂全拷；final_plan 格 path+sha）
- 实际记录次数/forward 加权驻留/转换原因/迁移 → step6/（deliverable_F_analysis.json + deliverable_F_report.md（36/36、时间零假设如实、网格余量 ≤2.3%）、phaseA/B/C + logs）+ stopping_v1/（analysis.json、migration/、execution/）
- rev1 完整版本对照 → E07 sp_coupling_v1（REV1 两文档 + v1 原件同包）
- 采样消融原始 cohort → E12 oeplb_benchmarks_results/（eadw/ea*/fs300v2/v3 69 件 + adaptive_opt/comprehensive/final/h20_archive，2,374 文件全拷）
- 工作流输入 → h20_inputs_fullservice/（499 文件：≤1MB 拷入、≤200MB path+sha、超限 path+size，G9）
- 不利结果保留 → oeplb_adaptive_results/（oeplb_fixed_adaptive_results.json 修复前 −6.0%、reproduce_d38_adaptive.json 4/4 boot_fail、oeplb_adaptive_both_constructions.json A +9.0%/B +8.1%）；phase-4 E4 P−I tps −226.3、G −0.148376、E 臂 E4 Goodput 全 0（tpot_p95 0.307s=3.7×ref）；figF/fig15-17 npz 清单（path+size）

## E10 统计正确性、可靠性与成本 — FOUND
- 重映射/覆盖/共同提交测试 → correctness/（ACCEPTANCE_EVIDENCE{,_v2}.md、ROOT_CAUSE_ROUND2.md、cpu_test_results.json、cpu_test_full.log、cpu_test_run.log、run_cpu_test.sh、test_swap_race_cpu.py、gpu_harness/、gpu_jobs/）+ correctness_logs_h20/（all_layouts_pilot、identity_smoke、tolerance_validation 三日志）
- 故障注入原件 → forensics/（ROOT_CAUSE_migration_race.md、MINIMAL_DIFF_async_swapper.patch、MINIMAL_DIFF_controller.patch、evidence_rank_split.csv）+ phase3_prep B1-hang cores（原位）+ INCIDENT ×8（E05 tag）
- 修复版服务验收/冻结 → freeze_code_fixed/（SHA256SUMS_code_fixed{,_v2}.txt + code_fixed/ + code_fixed_v2/）+ locked_v2_frozen/（MANIFEST.sha256：config 180fa41e…、controller 2a4dedb2…、window_v2 810c54da…）+ code_lock_final（E05 tag）
- canary/镜像序列 → stopping_v1/migration/（perpeer 40/40、offline 17/17、live R2×4 1527 对）+ adw_v2p1/formal_eval + phase4 64 boots server.log（5706 镜像对的原始日志，全量在包）
- record/归约/规划/同步迁移成本原件 → cost_originals_scripts/（bench_nvl72_p2p.py、bench_ce_bandwidth.py、bench_cross_node_ce{,_v2,_v3}.py）+ E01 microbench p2p_cost.json（22.4–26.4ms/300ops v2 vs 20.1 v1）+ cpu_audit/（ablation_adjustment.csv、completion_decomposition.csv、config_table.csv；CPU 台账 3.13–4.05%）+ sp_coupling gate CPU 5–31ms（E07 tag）
- 当前源码快照 → src_snapshot_current/（OE/src 全部 py ≤1MB：controller.py 139KB、async_swapper、window_v2{,p1}、sp_state、config、v2p1_signals、fast_metadata 等）

## E11 官方与学术基线、冗余资源压力 — FOUND
- E*/F* 校准选择记录 → E05 final_plan_calib/（FINAL_SELECTION.json、STAGE1_SELECTION.json、stage1/24、stage2/9：c2_e0_r{1,2}、c2_e5_r{1,2}、c2_f_1_16_16_r{1,2}…）+ ERRATA_addendum_20261020.md（E5 每臂配置表：E 臂 CG DISABLED + deepep normal；κ=3875.1768798828125）（_COMMON tag）
- 实际 Graph 与后端路径 → C_configs/ARM_FLAGS.md + EPLB_MECHANISM_NOTES.md + 各格 meta_env.txt/boot_meta.json + server.log 启动命令行（path+sha）
- MoETuner 真实实现与映射来源 → moetuner/（solve_ilp.py、placements 94×128、≤4MB 拷入；15GB cores 原位登记）+ baselines_top_docs/REPRODUCE_BASELINES.md(27KB)
- DataFore → datafore_repro/（placement_algo.py、5 placements、+14.5% vs 论文 +15.5% 记录，全拷）
- A100 基线与 KV 容量 → baselines_a100/（EXPERIMENT_LOG.md（KV pool 297K→149K）、latency/lat_freq6.jsonl 22.1MB path+sha）+ quickstart_a100/ + quickstart_h20/
- 网格原表 → bm_bookcorpus_grid(238MB)/bm_comprehensive_grid(107MB)/bm_final_grid(281MB)（≤1MB 拷入、≤100MB path+sha、其余 path+size）+ bm_multidomain(13MB 全拷) + bm_prompts(872KB 全拷)
- 旧 32 次副本压力与 KV 原表 → E05 final_plan_p1/（32 格 R∈{0,16,32}，KV 227.2k→208.6k→190.2k 在 REPORT_final/p1_table）+ sjq_multi_domain/（frozen_requests_* 7 件 ≤10MB 拷入；domain_switch_160k.jsonl 735MB 已哈希；prefill_heavy_universal 73MB 已哈希）+ sjq_single_domain/（79MB 同规）

## E12 泛化与参数范围 — PARTIAL（G5/G6/G1/G7）
- 跨模型计数 → E03 oeplb_repro/counts{235b,30b,57b,57b_l512,57b_multi,57b_share}.json（η 235B 79%/57B 84%/30B ≈0% 见 outline.md，_COMMON）
- 预算/有效阈值/采样敏感性原表及实际生效值 → oeplb_benchmarks_results/（2,374 文件全拷：eadw_W{8,16,32,64}_A0p{0,5,9}_s{1,2} 24 件 + eacR1-5 25 件 + fs300v2/v3 + adaptive_opt/comprehensive/final/h20_archive）+ studypaper/（8 篇：01 planner 近似…08 理论上限）+ oeplb_scripts/（run_grid_bench.py、run_final_sweep.py、run_adaptive_optimal.py、freeze_requests.py、compare.py、analyze_* 等 13 件）+ adw knobs 冻结值（E09 adw_v2/v2p1 + ERRATA E5 + locked MANIFEST）+ sp_run_end.json（各格内）
- 跨硬件 → E03 全部（A100/GB200/H800-部分/H20）+ author_diag/（adhoc_{I,P}_{gh,orig,v2}_E3 6 boots ≤1MB 拷入 + LEDGER_adhoc.tsv + README×5（DS_MATRIX/F6_COMPARISON/GH_COMPARISON/RESEARCH_DIRECTION）+ aggregate_diag.py；6 cores 原位）
- DSV4 负结果（作者令删除后恢复）→ dsv4_recovered_from_transcript/ 41 文件：**溯源更正＝git 对象字节级恢复**（/workspace/EPLB/.git HEAD 5df703d6a951b8a42fe86d2dd99dc03faf76d399，branch sp_coupling_dev，commit 2026-10-22T18:05:34Z，只读 cat-file，无 checkout/commit；_RECOVERY_MANIFEST.tsv 35 件 sha256 复验通过）：LEDGER_dsv4 217/249 行逐字、fixab_comparison.md 全量（e2 40.9 −3.8%/e3 41.4 −7.8%/e4 2763.1 −14.1%；P_fix −6.4/−11.7/−14.3%；P_broken −18.8/−17.1/−18.4%；0 errors）、STATS_PATH_TRACE.md 全量（κ=3875.1768798828125、N_MIN=12320768、E4 61/75=53 空窗+8 参考归零级联）、bench_matrix 终表（P-vs-I：tps −19.1/−17.1/−18.4%、ttft_p95 +39.6/+37.1/+65.9%；分解 static-dispatch ~26%/instrumentation ~20%/swap ~54%）、fixab_gate.json（FP PASS、tokens_identical_vs_I=True）、24 个 bench cell json、planb_assessment.md（INFEASIBLE）、ENV_NOTES、patches_OVERHEAD_ANALYSIS、fix 表 ×3；UNRECOVERED__ ×2（rows 218–249 与 old-sglang EFFECT_REPORT：客户端主机转录提取指引内嵌）+ SUPPLEMENT__old_sglang_wall_ratios_parent_context.md（父会话删除前亲读数字：wall-ratio e2 4.3→4.1 −6.0%、e3 4.4→4.1 −6.7%、e4 229.6→203.5 −11.4%，UNVERIFIED 标注）
- 缺口：G1（H800 e2e）、G3（fig12 book_rank0 空）、G5/G6（DSV4 尾部）、G7（头条增益溯源未决：+15.5 作废/+18.6 R1/+19.4 R2/+16.5 README/+17.5 outline/+17.46 paper——候选原始值全部在包，待作者裁定基准表）

## 完整性核验（本审计内执行）
- 拷贝保真：随机 4 件（cdx0 smoke 1.5MB 级/author_diag server.log ×2/nearscan raw）重算 sha256 与 manifest 全匹配（4/4）
- phase4：64/64 boots、592 文件全拷（all_copied=True）
- final_plan main：perreq 163/163 哈希、server.log 168/168 哈希；5 缺口＝5 hang（INCIDENT 对应）
- 冻结部署一致性：run_supp_boot.sh 守卫内嵌 async_swapper 2e4652a0…/controller 4964cfc8…（与 /opt/conda 冻结版一致，本会话早先已字节复核）
- DERIVED GOOD：64/64 精确整数（0 非精确转换）
- THEORY_VALIDATION_A100.md 双副本 sha 相同（02e1efca…）
- 构建：ERRORS=0、SOURCE_MISSING=0；list_only 登记 118 件（fig npz 目录 path+size）；core 28 件 547GB 原位未动

## 版本锁汇总（供论文引用）
| 层 | 关键 sha |
|---|---|
| 冻结部署 /opt/conda v2（race-fix） | async_swapper 2e4652a0306cb315…、controller 4964cfc8e8af397a…、sp_state ba572f93…、config 52f7f4d7… |
| OEPLB/src/locked_v2_frozen_20261016 | config 180fa41e…、controller 2a4dedb2…、window_v2 810c54da… |
| final_plan deployed_shas v3 | async_swapper 93fa4ccd…、controller c6d2f764…、window_v2p1 d20d340f…、model_runner 6580c364… |
| phase4 矩阵 | d3ff2b09…（PHASE4_CODE_VERSION_ADDENDUM.md） |
| SUPP_RESULTS.json | 2b3f7a57b5bd7c5303c318f4a3f9177efc9e124f9b951c1466cee94a961572bc |
| κ（ERRATA E5） | 3875.1768798828125 |
| DSV4 git 恢复源 | HEAD 5df703d6a951b8a42fe86d2dd99dc03faf76d399（sp_coupling_dev，2026-10-22T18:05:34Z） |
