# recovery_admission_e2e_20261027 归档 README

恢复准入策略修改(AERO-RG) + 七臂多baseline端到端比较。本目录为 GitHub 精选档; 全档(含逐boot server.log/perreq 与流文件)在 /data/minghua/sjq/PAPER_EXPERIMENTS/recovery_admission_e2e_20261027/(清单 ARCHIVE_SHA256SUMS.txt); 工作现场 /workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/recovery_admission_e2e_20261027/。

## 结论速览 (完成度 28/28)

- 主指标: 全服务墙钟 total_time_s (保留全部在线成本); 同报 tps。4轮x7臂, 同一冻结流 prover_256tok_out1_x4 (8192请求, sha b312ba50...), O=1, conc=256, 统一 --disable-cuda-graph, mem 0.8。
- I Identity: 均值墙钟 101.2s 逐轮 ['103.6', '104.0', '98.9', '98.2']
- ES 静态EPLB: 均值墙钟 91.2s 逐轮 ['92.7', '95.5', '89.7', '87.1']
- ED 动态EPLB(auto/16/256, E*=e5冻结): 均值墙钟 107.0s 逐轮 ['108.2', '110.3', '105.2', '104.4']
- MT MoETuner风格(Gurobi ILP1): 均值墙钟 81.5s 逐轮 ['85.5', '79.8', '80.4', '80.5']
- DF DataFore风格(贪心重映射): 均值墙钟 100.9s 逐轮 ['102.0', '102.3', '99.6', '99.7']
- P0 AERO-base(现有版本): 均值墙钟 89.9s 逐轮 ['90.1', '85.7', '95.7', '88.2']
- P1 AERO-RG(新版恢复准入): 均值墙钟 92.1s 逐轮 ['90.9', '91.3', '90.4', '95.8']

## 配对统计 (差值 = P1 - X, 负值表示 P1 更快; 双侧配对t df=3; Holm 校正)

- P1 vs P0: n=4 均值差=2.18s CI95=[-7.00, 11.36] p=0.5052 Holm=1.0000
- P1 vs I: n=4 均值差=-9.07s CI95=[-16.81, -1.33] p=0.0336 Holm=0.1009
- P1 vs ES: n=4 均值差=0.87s CI95=[-8.02, 9.76] p=0.7752 Holm=1.0000
- P1 vs ED: n=4 均值差=-14.92s CI95=[-22.11, -7.74] p=0.0070 Holm=0.0423
- P1 vs MT: n=4 均值差=10.56s CI95=[3.99, 17.12] p=0.0144 Holm=0.0703
- P1 vs DF: n=4 均值差=-8.78s CI95=[-14.19, -3.37] p=0.0141 Holm=0.0703

## 机制对照 (恢复准入)

- P0 AERO-base: 驻留 TRACK/QUIET/RECOVER=12/2/4 模式跨boot稳定, 7转移(窗口6,6,8,11,11,13,16), 其中 w6/w11 的 QUIET->RECOVER 由孤立 confirmed_grow 触发(P1c)。
- P1 AERO-RG: 唯一改动=孤立 confirmed_grow 不再作为变化证据; 实测驻留 6/10/0 RECOVER, 仅 w6 TRACK->QUIET, [AERO-RG] 注释行3-4/boot, 策略标签转移1, 8秩 planhash 全一致; 离线: 30/30分支测试 + apsd2_P_b1 回放(orig精确/new0逐字节同/rg1反事实与实测吻合)。
- 基线臂事实: ED 每boot实际 rebalance 次数与 ES 注入证据见 analysis/report.md 第6节。

## 目录

- PROTOCOL.md 预注册协议(0-9节) / LEDGER_rge2e.tsv append-only 治理日志
- code_lock/: sp_state.py.orig_ba572f93(原版备份) + AERO_RG.patch.diff(79行权威diff)
- order/: 锁定顺序表(种子20261027, 轮1-3循环移位, 轮4反转)
- scripts/: 七臂启动器+runner+mech_check+三个queue+归档/git脚本(均sha登记于LEDGER)
- tests_rgv/: 离线验证(30分支检查+回放) / placements/: 三份sha锁定静态素材
- analysis/: analyze+report 渲染脚本, summary.json, report.md(28原始值+全部统计), per_boot_raw.tsv
- boots_curated/: 31 boots 小件(bench json, MECH_CHECK, boot_meta, sp_run_end, v2p1, bench.log); server.log/perreq 大件在 /data 全档
