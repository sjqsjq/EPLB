# EPLB 在线专家放置 · 近期实验结果汇总（2026-10-27/28 三战役 + 诊断，供外部审阅）

整理时间 2026-10-30。全部内容取自各战役冻结协议（PROTOCOL_*.md）、预注册描述性分析（descriptive_*.py 在看结果前写成）、append-only LEDGER 与锁定源码核验记录。所有数字为 n=2 轮/臂的描述性统计，不做显著性断言。

## 0. 系统与公共方法

- 模型/硬件：Qwen3-235B-A22B-FP8，8×H20，tp8 dp8 ep8，DeepEP auto + DeepGEMM，FP8，disable-cuda-graph，disable-radix-cache，mem-fraction 0.8。
- 在线机制（sglang 锁定构建，pb_oeplb 控制器四文件 sha256 登记于各战役 code_lock）：窗口化统计（检测窗宽 W∈{8,16,32,64} 步，自适应升降），z 漂移触发信号，one-at-a-time 专家 swap，v2.1 冻结参数（KAPPA/Z_LO/Z_HI 等），反调参条款（机制事实不 gate 运行 rc）。
- 压测：bench_client_v2.py（冻结 sha 06668243…），固定 jsonl 请求流，全行 max_tokens=1、temperature=0，conc=32（吞吐以 prefill 为主）；每请求 perreq 时间戳用于相位切分。
- 六臂（pb/pbd）：
  - I = pinned fair 基线（identity 型固定放置）；
  - ES / ED / MT / DF = 四种静态 paper-lineage 放置（placement 文件字节冻结、sha 锁）；机制日志事实：ED 在运行中有 batched rebalance 事件（日志行 pb 169 / pbd 297），其余静态臂仅 init 1 行；
  - P0 = 在线自适应控制器（AERO-base rev1 冻结参数，RG=0 纯 V1，同种子）。
- 度量定义（源码核验，防止误读）：h_rate = headroom/新分配数，是物理槽空间失衡度，**不是**「还可以加速多少」的 headroom 读数；z = 窗口分布相对 EMA(β=0.5) 参考的漂移触发信号，非失衡度；无 payback solver；离线/纸面分析模型一律不表述为已实现的在线算法。
- spent_frac 口径：server.log 中 "swap(s) done (Xms)" 行按 8 rank 求和后除以服务时长（与标定口径一致）；控制器内部预算检查为单 rank 累计/elapsed。

## 1. pb 域切换战役（prover→book，n=3048：prover 2048 + book 1000；12/12 boots，2026-10-27 12:23–14:12Z，零中止）

per-boot 吞吐（req/s；相位切分 idx2048；book early[2048:2560) / late[2560:3048)）：

| boot | total | prover | book | b_early | b_late |
|---|---|---|---|---|---|
| I_r1 | 7.00 | 9.70 | 4.37 | 4.24 | 4.28 |
| ES_r1 | 7.30 | 10.56 | 4.38 | 4.36 | 4.18 |
| ED_r1 | 7.20 | 9.90 | 4.59 | 4.58 | 4.38 |
| MT_r1 | 7.50 | 11.12 | 4.39 | 4.31 | 4.26 |
| DF_r1 | 7.60 | 11.27 | 4.50 | 4.46 | 4.31 |
| P0_r1 | 7.30 | 10.97 | 4.29 | 4.41 | 3.97 |
| I_r2 | 6.80 | 9.13 | 4.44 | 4.43 | 4.23 |
| ES_r2 | 7.20 | 10.42 | 4.37 | 4.30 | 4.22 |
| ED_r2 | 7.10 | 9.58 | 4.52 | 4.53 | 4.30 |
| MT_r2 | 7.50 | 11.19 | 4.47 | 4.46 | 4.26 |
| DF_r2 | 7.30 | 11.19 | 4.23 | 3.98 | 4.30 |
| P0_r2 | 7.60 | 10.61 | 4.72 | 4.68 | 4.55 |

- prover 段（校准域）：DF 11.23 ≈ MT 11.16 > P0 10.79 > ES 10.49 > ED 9.74 > I 9.42 —— 静态校准收益兑现（DF/MT vs I 约 +19%）。
- book 段（切换域）：ED 4.56 > P0 4.50 > MT 4.43 > I 4.41 > ES 4.38 > DF 4.36 —— prover 段 19% 的臂间差距在 book 段收敛为约 11% 极差；prover 校准不迁移。
- P0 切换动态（v2p1 rank0 日志）：边界检测成功且失衡恢复真实——

| boot | 窗口数 | 边界窗 | h_rate pre→spike→plateau | z spike→plateau | swaps | grow_blocked |
|---|---|---|---|---|---|---|
| pb_P0_r1 | 85 | w27 | 0.0106→0.0490→0.0118 | 45.3→0.76 | 27 | ×40 |
| pb_P0_r2 | 56 | w30 | 0.0103→0.0904→0.0104 | 17.1→0.80 | 6 | ×9 |

  两轮失衡度均完全恢复到切换前基线（≈0.0103），但吞吐跟随 swap 扰动量分裂：r1 27 次 swap ↔ book 4.29 全臂最低、b_late 3.97；r2 仅 6 次 swap ↔ book 4.72 全臂最高、b_late 4.55 无跌幅。
- 对照（更早的 pc prover→code 战役，code 段）：静态 DF/MT/I 约 32–34 req/s，P0 28.76/30.56 —— code 域 P0 落后 8–12%；且 pc 期恢复被 run 尾截断（失衡平台 0.0175，边界后仅 10 个观察窗，约完成 75%），这是 pbd 扩流的直接动机。
- 离线失衡网格（同一份计数 × 各布局重放，纸面分析、非在线算法）：域适配 greedy 布局 book hr≈0.00002、code hr≈0.0001，即布局层面收益上限很大；但 P0 实测切换后 0.0104–0.0118 远未及 ceiling，兑现收益被压缩到 O(1)%。

## 2. pbd 扩流复制战役（prover→book 加倍，n=5344：prover 3344 + book 2000；12/12，17:59–20:58Z，零中止）

研究问题：pb 的臂排序与 P0 的 r1/r2 swap 大 spread（27 vs 6）是否为窗口预算/相位伪影。

per-boot 吞吐（req/s；切分 idx3344；book early[3344:4344)/late[4344:5344)）：

| boot | total | prover | book | b_early | b_late | lat_ms | ttft_ms |
|---|---|---|---|---|---|---|---|
| I_r1 | 6.70 | 9.64 | 4.41 | 4.40 | 4.30 | 4751.1 | 4751.0 |
| ES_r1 | 6.80 | 10.25 | 4.35 | 4.33 | 4.25 | 4663.8 | 4663.8 |
| ED_r1 | 6.80 | 9.64 | 4.56 | 4.55 | 4.43 | 4666.4 | 4666.3 |
| MT_r1 | 7.10 | 11.08 | 4.39 | 4.37 | 4.28 | 4498.8 | 4498.7 |
| DF_r1 | 7.10 | 11.05 | 4.42 | 4.40 | 4.28 | 4481.8 | 4481.7 |
| P0_r1 | 7.10 | 10.96 | 4.43 | 4.75 | 4.01 | 4480.5 | 4480.4 |
| I_r2 | 6.80 | 9.80 | 4.41 | 4.39 | 4.30 | 4717.4 | 4717.3 |
| ES_r2 | 7.00 | 10.54 | 4.40 | 4.40 | 4.28 | 4578.5 | 4578.4 |
| ED_r2 | 7.00 | 10.04 | 4.60 | 4.57 | 4.49 | 4559.7 | 4559.7 |
| MT_r2 | 7.10 | 11.20 | 4.42 | 4.42 | 4.28 | 4462.3 | 4462.2 |
| DF_r2 | 7.20 | 11.25 | 4.49 | 4.48 | 4.36 | 4410.8 | 4410.7 |
| P0_r2 | 7.20 | 11.02 | 4.48 | 4.48 | 4.30 | 4454.5 | 4454.4 |

- prover 段排序 100% 复制 pb：DF 11.15 ≈ MT 11.14 > P0 10.99 > ES 10.39 > ED 9.84 > I 9.72。
- book 段：首尾复制（ED 4.58 最高、ES 4.38 最低），中段 4.38–4.45 在轮间噪声内换序：P0 4.45 ≈ DF 4.45 > I 4.41 > MT 4.40。
- P0 两轮决策轨迹逐窗一致（109 窗、边界 w48、23 swaps、1824 步、z spike 109.0→plateau 1.98、事件计数相同），产物文件独立（sha/size 不同）→ 决策确定性成立；pb 的 27-vs-6 spread 支持读作窗口预算/相位伪影：1.75× 流长下观察窗增多，swap 数收敛到中间值 23。
- 诊断（DIAGNOSIS_P0_BOOK，作者问「book 段 P0 收益为何不显著」）因果链：
  1. P0 响应快：切换窗 w49 z spike 137.7，early 段 4 次 swap，b_early 4.75 = pbd 全部 12 boots 最高（> ED 4.55）；
  2. 但 23 个 swap 窗中 17 个落在 book_late 并持续到最终窗 w108；swap burst 与 tps 凹陷对齐：r1 bucket[4344:4594] 含 w75–w82 共 5 个 swap 窗 → 2.90 req/s（−34%），ED/DF/I/MT 同桶平稳 3.85–4.28；
  3. 代码语义（window_v2.py 锁定源）：grow_blocked_headroom = 分布已平稳（z<z_lo）但残余可恢复负载 h_w ≥ θ_h_low → 保持 W=16 高频监测；one-at-a-time swap 下残差始终不收敛（z 反复再尖峰 22.6/9.8/5.8；plateau 1.98 vs pb-r1 0.76）；config.py 自身记载该失效模式（同源噪声计数构建的 swap plan 可能追逐 phantom skew；实测 swap/headroom 比 1.26 与 1.09，兑现 ceiling≈0）；
  4. 天然对照：pb P0_r2 是唯一 book_late 零 swap 的 P0 轮 → b_late 4.55、book 4.72 为两战役最高（> ED 4.52）；
  5. 净算术：early 增益 − late churn ≈ 4.45 ≈ 静态 DF 4.45。
- swap 成本标定（spent_frac，rank 求和口径）：pb_P0_r1 0.4046（216 事件，b_late 3.97）；pb_P0_r2 0.0884（48 事件，book 4.72，HEALTHY）；pbd_P0_r1 0.2654（192 事件）；pbd_P0_r2 0.2373（192 事件）。单次 swap 均值约 1.0–1.2s，每个 swap 窗约 8 次迁移。

## 3. bsk 单域旋钮试验（纯 book n=2000，字节取自 pbd 流 book 段；6 臂×2 轮=12/12，2026-10-27 21:0x → 10-28 00:13Z；EXPLORATORY）

作者授权问题：treadmill 病理是否单域即现？两个已实现未启用的旋钮（OEPLB_BIAS_GATE、OEPLB_SWAP_BUDGET_FRAC，锁定代码内 env 读取，零代码改动）在单域是否有好效果？

臂：P0（RG=0 基线）/ P0B（+bias_gate=0.5，语义为 margin≥2×噪声）/ P0S（+swap_budget_frac=0.08，标定锚点为健康轮 pb_P0_r2 的 8.84%）/ P0F（两者）/ ED / DF。

per-boot（half1/half2 = 前后各 1000 请求；spent_frac rank 求和口径；bias_skip 为日志下界指标）：

| boot | tps | half1 | half2 | swap_ev | spent_frac | budget_exh | bias_skip_lb | wins/sw_wins | z_tail20 |
|---|---|---|---|---|---|---|---|---|---|
| P0_r1 | 4.6 | 4.982 | 4.212 | 192 | 0.4531 | 0 | 0 | 64/23 | 1.428 |
| P0_r2 | 4.5 | 4.541 | 4.307 | 176 | 0.3923 | 0 | 0 | 64/21 | 1.298 |
| P0B_r1 | 4.5 | 4.831 | 4.118 | 192 | 0.3755 | 0 | 0 | 64/23 | 1.428 |
| P0B_r2 | 4.6 | 4.659 | 4.418 | 176 | 0.4280 | 0 | 0 | 64/21 | 1.298 |
| P0S_r1 | 4.5 | 4.518 | 4.412 | 168 | 0.4062 | 8 | 0 | 59/20 | 1.154 |
| P0S_r2 | 4.6 | 4.712 | 4.361 | 176 | 0.3103 | 0 | 0 | 64/21 | 1.298 |
| P0F_r1 | 4.6 | 4.948 | 4.212 | 192 | 0.4471 | 0 | 0 | 64/23 | 1.428 |
| P0F_r2 | 4.6 | 4.777 | 4.314 | 192 | 0.3452 | 0 | 0 | 64/23 | 1.428 |
| ED_r1 | 4.5 | 4.427 | 4.349 | 0 | 0 | 0 | 0 | — | — |
| ED_r2 | 4.5 | 4.347 | 4.444 | 0 | 0 | 0 | 0 | — | — |
| DF_r1 | 4.4 | 4.413 | 4.334 | 0 | 0 | 0 | 0 | — | — |
| DF_r2 | 4.4 | 4.397 | 4.297 | 0 | 0 | 0 | 0 | — | — |

臂均值：P0F tps 4.60；P0/P0B/P0S 4.55；ED 4.50；DF 4.40。half1：P0F 4.863 > P0 4.761 > P0B 4.745 > P0S 4.615 > DF 4.405 ≈ ED 4.387。half2：ED 4.396 > P0S 4.386 > DF 4.316 > P0B 4.268 ≈ P0F 4.263 ≈ P0 4.260。

按预注册解释规则（R1–R4）：

- **R1 病理单域确认**：基线 P0 无任何域切换仍呈现完整 treadmill——spent_frac 0.453/0.392（高于 switch 场景 0.24–0.27，同口径），half1 4.761→half2 4.260，swap 持续到最终窗 w64，z_tail 1.43/1.30 不收敛，grow_blocked_headroom 31/64 窗。病理是 book 负载（热点专家漂移）内生的。
- **R1 附加（回答作者问题）**：单域全程净效果 P0 族 tps 4.55–4.60 > ED 4.50 > DF 4.40 —— 前半程适配增益（half1 高出参照约 8%）大于后半程损耗（half2 低约 3%），2000 请求窗口内在线适配净正。
- **R2 budget 旋钮**：仅 P0S_r1 启动（8 rank 于 22:40:55 同刻打出 swap budget exhausted，首次 swap 后 41s）。实现语义发现（锁定源核验 + 日志时间戳）：判据是瞬时比率 cum_swap_s/elapsed 而非闩锁——exhausted 之后仍有 144/168 条 swap-done 行，swap 持续到 w58；日志文案 no further swaps will be issued 夸大了实际语义。启动轮效果方向符合设计：churn 被稀释（20 vs 23 swap 窗，id 序列不同），half2 4.412 vs 基线 r1 4.212（P0 族最好，≈ED 4.396），收尾三桶 4.21/4.38/4.30 vs 基线 4.08/3.84，z_tail 1.154 vs 1.428；代价是 half1 4.518 vs 4.982（elapsed 钟含 serving 起点，早期适配阶段 swap 相对昂贵，预算被过早烧掉）。净 tps 持平（4.5 vs 4.6）。P0S_r2 与 P0F 两轮从未启动（0 行、轨迹与对应基线轮恒等）——瞬时比率是否越阈取决于计时噪声，属刀锋触发。
- **R3 bias gate：ENGAGEMENT FAILURE**（按预注册规则如实报告，不作为旋钮结果）：P0B/P0F 四轮 skip 日志 0 行，且 P0B 轨迹与基线逐窗恒等（r1：23 swap 窗 id 序列一致、z_tail 1.428；r2：21 窗、1.298）→ gate 从未改变任何决策。机制解释（推断）：bias=c/sqrt(n_per_layer)，book 长 prompt（约 4k token）×conc32×W16 步窗 → n_per_layer 巨大 → bias 趋 0，0.5×margin 平凡满足。phantom skew 假设在此设定下不可判定。
- **R4 轮间方差警示**：单域轨迹轮间非确定（P0_r1 192ev/23sw vs P0_r2 176ev/21sw；P0F_r2 在零旋钮启动下抽到 r1 型轨迹 192ev/23sw/z1.428）——与 pbd switch 流上 P0 两轮逐决策一致不同；n=2 下旋钮归因力弱。

## 4. 跨战役结论（描述性）

1. 静态校准收益域特异：prover 校准的 DF/MT 在 prover 段 +19% vs I，但在 book 段不兑现（DF book 4.36–4.45）；ED 是 book 段稳定最优（pb 4.56 / pbd 4.58 / bsk half2 4.396）且有周期性 batched rebalance 机制日志佐证。
2. 在线适配 P0 的失衡恢复真实、快速且确定（h_rate 回到基线、z 低平台、pbd 两轮逐窗一致），但吞吐收益被 swap churn 自身吃掉：one-at-a-time swap 下残余失衡不收敛，形成后期 swap treadmill（w 持续到终窗、burst 与 tps 凹陷对齐、每 swap 约 1.0–1.2s、病理轮 spent_frac 24–45% vs 健康轮 8.8%，rank 求和口径）。
3. churn 是主要矛盾的直接证据：pb_P0_r2（6 swaps）book 4.72 两战役最高；pbd P0 b_early 4.75 全场最高但 b_late 4.01–4.30；bsk 中唯一 budget 启动轮 half2 恢复至参照水平。
4. 两个现成旋钮在本设定下未给出稳定净收益：bias_gate=0.5 完全不活跃；swap_budget_frac=0.08 刀锋触发、触发轮早晚损益相抵净持平。
5. 单域回答（作者问题）：单域下 P0 族全程净优（tps > ED > DF），但后半程病理同样存在；switch 场景 book 段打平是「早期增益在 prover 段兑现、book 段只剩 churn」的算术结果。
6. 候选后续（均未动手，需作者授权）：budget 判据闩锁化或 warmup 豁免（代码改动）；bias_gate 取大一个量级重测；或接受现状、论文侧按上述事实定位 P0。

## 5. 冻结产物与守卫链（关键 sha256 前 8 位）

- 锁定代码（/opt/conda/.../sglang/srt/managers/pb_oeplb）：async_swapper 2e4652a0、controller 4964cfc8、sp_state 525e88f0、window_v2 810c54da；冻结 pc launcher cfbdaee7、bsk launcher 2896f550；bench_client_v2 06668243。
- 流：pb n=3048 sha 2da34672…；pbd n=5344 sha f864f236…（BUILD 门 0 mismatch）；bsk n=2000 sha 2e9fe867…（字节保持 sed 抽取）。
- 每战役：PROTOCOL（预注册端点+解释规则）、LEDGER（append-only，pb 完整/pbd 43 行/bsk 26 行）、B_data/registry.json（provenance+标定）、scripts（boot/queue/mech_check/launcher，逐 boot sha 校验，mismatch=rc2 halt）、analysis（descriptive_*.py 先于看结果写成 + json/md + 诊断 out 文件）、每 boot 完整产物（server.log、bench json、perreq jsonl、boot_meta、MECH_CHECK、v2p1_rank0.jsonl、sp_run_end.json）。
- 12/12 boots 三战役全部 bench_rc=0 且 mech_rc=0，零 ABORT，INFRA 重启预算未动用。
