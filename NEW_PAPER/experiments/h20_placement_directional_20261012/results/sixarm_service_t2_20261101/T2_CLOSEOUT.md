# T2 CLOSEOUT — sixarm_service_t2_20261101 (closed 2026-11-02T00:35:51Z)

## 0. 执行账目
- smoke 2/2 通过 (PROV_I_s0, PROV_P0_s0, 严格门). 正式 36/36 boots 有数据且 mech all_ok=True, bench errors=0, 零 rc=2 停线.
- 队列内 26 boots 有数据 + 10 boots 因 DEFECT_FIX (launcher line19 PL= 前缀在字节拼接时丢失, 影响 ES/MT/DF 三臂) 零数据, 修复后按冻结顺序表重跑, 首次尝试证据归档 boots_fail1/. 协议/顺序/口径零变更, 无臂被丢弃.
- 墙钟: MASTER_END 20:24:57Z (rc_t1=0 rc_t2smoke=0 rc_t2formal=0), RERUN_QUEUE_END 21:04:06Z; 自 14:18Z 点火共 6.8h < 7.5h 硬上限.
- Path A 全程实证: 每 boot deepep_mode=normal + disable_cuda_graph=True; 冗余账目 ES/ED=16 (机制要求), I/MT/DF/P0=0; init_expert_location 按臂核对 (ES->datafore_e3, DF->datafore_e3_remap, MT->placement_e3_fair, 其余 trivial); ED eplb=True iters=256; actual_o32 每 boot 验证 (avg_output_tokens>=31).

## 1. 服务指标 (描述性, 每臂 n=3 轮, 不做显著性声明)

### PROV 场景 (prover_1024_out32_ie, conc 256, timeout 1800)
| arm | tps_mean | tps_min-max | lat_mean_ms | lat_p99_ms | tpot_mean_ms | bench_s_mean |
|---|---|---|---|---|---|---|
| I | 26.54 | 26.22-26.76 | 9592.7 | 11227.4 | 224.23 | 38.6 |
| ES | 28.06 | 27.26-28.63 | 9065.7 | 10315.7 | 218.09 | 36.5 |
| ED | 24.84 | 22.96-26.06 | 10272.2 | 11980.7 | 238.54 | 41.3 |
| MT | 28.01 | 26.66-28.72 | 9014.8 | 10689.0 | 225.05 | 36.6 |
| DF | 28.06 | 27.49-28.43 | 9062.8 | 10369.2 | 216.31 | 36.5 |
| P | 26.67 | 26.5-26.76 | 9539.2 | 11808.4 | 238.15 | 38.4 |

### PBD 场景 (pbd_prover_book_double_out32_ie 5344, conc 256 — conc delta vs pbd mother conc32 O1 已声明, timeout 1800)
| arm | tps_mean | tps_min-max | lat_mean_ms | lat_p99_ms | tpot_mean_ms | bench_s_mean |
|---|---|---|---|---|---|---|
| I | 5.87 | 5.84-5.9 | 43339.9 | 64845.8 | 798.78 | 909.7 |
| ES | 6.06 | 5.96-6.12 | 42004.4 | 63289.9 | 779.64 | 881.6 |
| ED | 6.11 | 6.02-6.28 | 41768.7 | 66388.2 | 803.36 | 874.7 |
| MT | 5.99 | 5.98-6.0 | 42531.3 | 65206.4 | 795.11 | 892.1 |
| DF | 5.94 | 5.88-5.98 | 42773.5 | 64601.0 | 771.76 | 899.8 |
| P | 6.13 | 5.83-6.35 | 41439.9 | 75204.1 | 745.2 | 872.9 |

## 2. 诚实观察 (描述性)
- PROV: ES/DF/MT 同处顶部带 (tps 28.01-28.06), I=26.54, P0=26.67, ED 最低 (24.84, 且跨轮方差最大 22.96-26.06). 静态重布局三臂相对 I 基线约 +5.4~+5.7%, ED 动态重平衡相对 I 约 -6.4% (n=3, 不构成显著性声明).
- PBD: 全臂窄带 5.87-6.13 (极差 ~4.4%), 无清晰分离; P0/ED 名义居前, I 名义居后.
- 两场景臂间差距形态不同 (PROV 分离度大于 PBD), 为如实测量结果; 不作机制归因超出数据支持的声明.
- 本表为 conc256 闭环恒并发口径的服务指标, 非 SLO 口径; T3 (开环到达, SLO 冻结) 未启动 (作者指示后置).

## 3. 产物
- analysis/EXPORT_t2_service_table.tsv (36 行逐 boot), boots/ (36+2 smoke), boots_fail1/ (10 首次失败证据), LEDGER_t2.tsv (含 DEFECT_FIX 修正行), code_lock/ (ORDER_TABLE+shas), PROTOCOL_t2.md (冻结).
