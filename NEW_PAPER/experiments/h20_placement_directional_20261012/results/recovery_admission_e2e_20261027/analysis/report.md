# recovery_admission_e2e_20261027 端到端比较报告

完成度: 28/28 (目标 28/28)。本报告由 report_gen.py 从 summary.json 渲染, 全部为实测事实; 统计口径见 PROTOCOL.md 第7节 (主指标 total_time_s 全服务墙钟, 同报 tps, 按轮配对, 均值差 + 95% 区间 paired t df=3, 双侧配对 t 检验, P1 对六臂 Holm 校正)。禁止与任何历史均值拼接。

## 1. 全部 boot 原始值 (主指标 total_time_s, 秒)

| 臂 | R1 | R2 | R3 | R4 | 均值 |
|---|---|---|---|---|---|
| I Identity | 103.6 | 104.0 | 98.9 | 98.2 | 101.2 |
| ES 静态EPLB | 92.7 | 95.5 | 89.7 | 87.1 | 91.2 |
| ED 动态EPLB | 108.2 | 110.3 | 105.2 | 104.4 | 107.0 |
| MT MoETuner风格 | 85.5 | 79.8 | 80.4 | 80.5 | 81.5 |
| DF DataFore风格 | 102.0 | 102.3 | 99.6 | 99.7 | 100.9 |
| P0 AERO-base | 90.1 | 85.7 | 95.7 | 88.2 | 89.9 |
| P1 AERO-RG | 90.9 | 91.3 | 90.4 | 95.8 | 92.1 |

## 2. 同报: 实际输出吞吐 tps (ok/total_time_s)

| 臂 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| I | 79.1 | 78.8 | 82.8 | 83.4 |
| ES | 88.4 | 85.8 | 91.3 | 94.1 |
| ED | 75.7 | 74.3 | 77.9 | 78.4 |
| MT | 95.8 | 102.7 | 101.9 | 101.8 |
| DF | 80.3 | 80.1 | 82.3 | 82.2 |
| P0 | 90.9 | 95.6 | 85.6 | 92.9 |
| P1 | 90.2 | 89.7 | 90.6 | 85.5 |

## 3. 配对统计 (P1 对各臂, 差值 = P1 - X, 负值表示 P1 更快)

| 对比 | n对 | 均值差(s) | 95%CI | t | p(双侧) | p(Holm) | 含义 |
|---|---|---|---|---|---|---|---|
| P1 vs P0 | 4 | 2.18 | [-7.00, 11.36] | 0.755 | 0.5052 | 1.0000 | 机制效应 (恢复准入) |
| P1 vs I | 4 | -9.07 | [-16.81, -1.33] | -3.728 | 0.0336 | 0.1009 | 全系统 vs Identity |
| P1 vs ES | 4 | 0.87 | [-8.02, 9.76] | 0.312 | 0.7752 | 1.0000 | 全系统 vs 静态EPLB |
| P1 vs ED | 4 | -14.92 | [-22.11, -7.74] | -6.611 | 0.0070 | 0.0423 | 全系统 vs 动态EPLB |
| P1 vs MT | 4 | 10.56 | [3.99, 17.12] | 5.119 | 0.0144 | 0.0703 | 全系统 vs MoETuner风格 |
| P1 vs DF | 4 | -8.78 | [-14.19, -3.37] | -5.168 | 0.0141 | 0.0703 | 全系统 vs DataFore风格 |

检验方法(命名登记): 双侧配对 t 检验 (round-paired, df=3, tcrit=3.182445); P1 对六臂 p 值统一 Holm 步进校正; scipy=True。

## 4. 机制记录 (P0/P1 全部 boot)

### P0 AERO-base
- r1: 驻留={"TRACK": 12, "QUIET": 2, "RECOVER": 4} 采样窗口=18 committed=18 recorded_steps=228 转移数=7 RG注释行=0 RG策略转移=0 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0
- r2: 驻留={"TRACK": 12, "QUIET": 2, "RECOVER": 4} 采样窗口=18 committed=18 recorded_steps=228 转移数=7 RG注释行=0 RG策略转移=0 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0
- r3: 驻留={"TRACK": 12, "QUIET": 2, "RECOVER": 4} 采样窗口=18 committed=18 recorded_steps=228 转移数=7 RG注释行=0 RG策略转移=0 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0
- r4: 驻留={"TRACK": 12, "QUIET": 2, "RECOVER": 4} 采样窗口=18 committed=18 recorded_steps=228 转移数=7 RG注释行=0 RG策略转移=0 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0

### P1 AERO-RG
- r1: 驻留={"TRACK": 6, "QUIET": 10} 采样窗口=16 committed=16 recorded_steps=134 转移数=1 RG注释行=3 RG策略转移=1 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0
- r2: 驻留={"TRACK": 6, "QUIET": 10} 采样窗口=16 committed=16 recorded_steps=135 转移数=1 RG注释行=4 RG策略转移=1 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0
- r3: 驻留={"TRACK": 6, "QUIET": 10} 采样窗口=16 committed=16 recorded_steps=136 转移数=1 RG注释行=4 RG策略转移=1 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0
- r4: 驻留={"TRACK": 6, "QUIET": 10} 采样窗口=16 committed=16 recorded_steps=135 转移数=1 RG注释行=4 RG策略转移=1 首次迁移窗口=2 迁移批(decode_record_batches)=0 swaps_started=2 issued窗口=[1, 2] planhash8秩=True 计划ops=0

## 5. 轮1 机制时间线案例 (预选)

### P0 AERO-base r1 逐窗口 (w=窗口, st=状态, ev=事件, tr=转移)

| w | st_before | st_after | ev | tr | z | h_rate |
|---|---|---|---|---|---|---|
| 1 | TRACK | TRACK | init |  | -- | 0.08989 |
| 2 | TRACK | TRACK | hold |  | 0.157 | 0.02218 |
| 3 | TRACK | TRACK | hold |  | 0.060 | 0.00935 |
| 4 | TRACK | TRACK | confirmed_grow |  | 0.034 | 0.00922 |
| 5 | TRACK | TRACK | hold |  | 0.055 | 0.00940 |
| 6 | TRACK | QUIET | hold | Y | 0.051 | 0.00894 |
| 6 | QUIET | RECOVER | confirmed_grow | Y | 0.036 | 0.01003 |
| 7 | RECOVER | RECOVER | hold |  | 0.042 | 0.00934 |
| 8 | RECOVER | TRACK | hold | Y | 0.049 | 0.00943 |
| 9 | TRACK | TRACK | confirmed_grow |  | 0.048 | 0.00932 |
| 10 | TRACK | TRACK | hold |  | 0.053 | 0.00897 |
| 11 | TRACK | QUIET | hold | Y | 0.033 | 0.00933 |
| 11 | QUIET | RECOVER | confirmed_grow | Y | 0.017 | 0.00938 |
| 12 | RECOVER | RECOVER | hold |  | 0.034 | 0.00943 |
| 13 | RECOVER | TRACK | hold | Y | 0.025 | 0.00931 |
| 14 | TRACK | TRACK | confirmed_grow |  | 0.037 | 0.00898 |
| 15 | TRACK | TRACK | hold |  | 0.038 | 0.00933 |
| 16 | TRACK | QUIET | hold | Y | 0.022 | 0.00924 |

### P1 AERO-RG r1 逐窗口 (w=窗口, st=状态, ev=事件, tr=转移)

| w | st_before | st_after | ev | tr | z | h_rate |
|---|---|---|---|---|---|---|
| 1 | TRACK | TRACK | init |  | -- | 0.08982 |
| 2 | TRACK | TRACK | hold |  | 0.156 | 0.02272 |
| 3 | TRACK | TRACK | hold |  | 0.060 | 0.00976 |
| 4 | TRACK | TRACK | confirmed_grow |  | 0.034 | 0.00961 |
| 5 | TRACK | TRACK | hold |  | 0.055 | 0.00978 |
| 6 | TRACK | QUIET | hold | Y | 0.051 | 0.00929 |
| 6 | QUIET | QUIET | grow_blocked_headroom |  | 0.035 | 0.01047 |
| 6 | QUIET | QUIET | confirmed_grow |  | 0.014 | 0.00981 |
| 6 | QUIET | QUIET | hold |  | 0.027 | 0.01019 |
| 7 | QUIET | QUIET | hold |  | 0.047 | 0.00957 |
| 7 | QUIET | QUIET | confirmed_grow |  | 0.025 | 0.00989 |
| 7 | QUIET | QUIET | hold |  | 0.020 | 0.00982 |
| 7 | QUIET | QUIET | hold |  | 0.018 | 0.00950 |
| 8 | QUIET | QUIET | confirmed_grow |  | 0.010 | 0.00964 |
| 8 | QUIET | QUIET | hold |  | 0.029 | 0.00975 |
| 8 | QUIET | QUIET | hold |  | 0.013 | 0.00964 |

## 6. 基线臂机制事实 (ED 触发次数 / ES 注入 / 完整性)

| boot | completeness | ED rebalance次数 | banner | 注入证据条数 |
|---|---|---|---|---|
| rge2e_ES_r1 | True | 0 | False | 3 |
| rge2e_ED_r1 | True | 8 | True | 3 |
| rge2e_ES_r2 | True | 0 | False | 3 |
| rge2e_ED_r2 | True | 8 | True | 3 |
| rge2e_ES_r3 | True | 0 | False | 3 |
| rge2e_ED_r3 | True | 8 | True | 3 |
| rge2e_ES_r4 | True | 0 | False | 3 |
| rge2e_ED_r4 | True | 8 | True | 3 |

## 7. 数据完整性

- status=ok 且 completeness_pass=True 且 bench_rc=0 的 boot 数: 28/28
- 全部 28 boot 原始值见 analysis/per_boot_raw.tsv 与 analysis/summary.json; 逐 boot 现场在 boots/<label>/。
