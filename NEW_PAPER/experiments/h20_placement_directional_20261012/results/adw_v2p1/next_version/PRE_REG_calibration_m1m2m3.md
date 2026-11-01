# PRE_REG — M1/M2/M3 校准与机制验证（冻结件）

冻结时间：2026-10-18（首次启动前）。作者指令：15 次启动上限，M1→M2→M3，不新增方案分支；测量前填齐全部参数并冻结；不能支持的统计保证明确留空。
本文件 + DRAFT_window_admission_method_v1_1.md + 冻结 commit（代码/驱动/清单 sha）共同构成锁定协议。**冻结后不改任何值；如必须偏离，先记录偏离再执行，不静默更改。**

## A. 冻结参数表

| 参数 | 值 | 语义/来源 |
|---|---|---|
| OEPLB_PROBE_R | 12 窗 | ramp 排除（封存战役：W1 交换集中 w1–w2，ramp 漂移在前四分之一；12 窗=192 步≈68s） |
| OEPLB_PROBE_NREF / NCAND / NFB | 4 / 3 / 4 窗 | 参考段 64 步、候选段 96 步（W32）、回退段 64 步；探测跨度 12 窗（含触发窗 w12；段窗 11：REF w13–16 / CAND w17–19 / FB w20–23，EVAL 于 w23 末）/ 224 步（W1 run 步数占比 ≈23%，W2 ≈11%） |
| AAA 臂段窗数 | REF 4 / CAND 6 / FB 4（全 W16） | 段 forward 预算与 ABA 完全对应（64/96/64 步）：ABA CAND=3 窗×W32=96 步，AAA CAND=6 窗×W16=96 步；估计器只用段内 Σspan/Σsteps，与窗数无关 |
| 增长门控范围 | 全 run（gate/aaa 模式） | 凡 W_after>W_before 的建议一律 restore+门控（event=grow_gated_probe，含 ARMED 段与 DONE 后、含收缩后回涨）；理由：AAA=全程 W16 基线、两臂均不永久准入、封存 A 臂会在 w1–w2 涨至 W64 而破坏 W16 基线；检测/收缩/一致性不动 |
| K（T-b 最小 c_b 样本） | 8 窗 | **校准期 T-b 只计算记录、不作触发门槛**（确定性触发替代，见 C 节偏离登记）；生产 T-b 区间规格 = 留空 |
| ε_risk | 2,000,000 µs（2s） | 风险**中止阈值**（非硬损失界）：候选段每窗末 excess_cum = Σ(候选步数×max(0, rate_cand_running − rate_ref))，由归约后 MAX 计时序列计算；超阈 → 下一窗回退，事件 probe_abort_risk；**过冲（excess_cum−ε_risk）与单窗粒度滞后记录在案** |
| FORCE_ABORT_AT（仅 M2-abort） | 1 | 机制配置：首个候选窗末强制走中止路径；该 run 标记 MECHANISM，永不入性能统计 |
| M2-abort ε_risk | 1,000 µs | 机制配置（配合上项验证过冲记录） |
| ε_prac（校准临时分类） | 0.01 × rate_ref | 仅用于 AAA 误判测量与三分类报告：Δ_net ≤ −ε_prac → favorable；≥ +ε_prac → unfavorable；其间/中止/污染/布局混杂/anomaly → UNKNOWN。**非生产准入阈值**（生产值 = 留空） |
| ε_log（M1 自成本门槛） | 边界新增中位 ≤ 1 ms/窗；每 forward 新增 ≤ 1 µs | 账目 = _prof_probe_cum_ms（新增仪表累计）；每 forward 项以代码审查（仅计数器自增）+ PROF avg_record_us 对封存参照变化 ≤ +10% 核验（参照 = 运行时从封存 <cond>_A__r1 server.log 末窗 PROF 块导出的 8-rank 中位数：W1=115.54µs、W2=114.67µs。**2026-10-18 冻结后-门前更正**：原括号常数 81.61µs 为转录错误，出自密封 ADDENDUM R2 恒等式核算值（81.61×89770 calls=7326ms），与任何封存末窗 PROF 实测不符；容差 +10% 不变，参照锚点仅由封存（校准前）artifacts 决定，详见 §F #5） |
| 校准触发（M2/M3） | window_id ≥ R 且 T-c 满足的首个窗末 | T-c = 本窗提交事件不在 S3 族且 alert_k=0；延迟触发记录实际触发窗 |
| COMMIT（全程） | 0 | 两臂均不永久准入；EVAL 照常计算记录 |
| 布局混杂旗标 | 段内 p2l_version 变化 → 段旗标 → 结果 UNKNOWN（缺省规则，冻结） | 指纹（8B blake2b, rank0）另作跨臂可比性记录 |
| anomaly 规则 | 段内出现 hold（未提交）窗 → 段旗标 anomaly → 结果 UNKNOWN | 封存历史零 hold，预期不触发 |
| S3 事件族 | urgent_shrink, confirmed_shrink, forced_shrink_uncertain, urgent_shrink_capped, w_desync_freeze, confirm_fail_freeze | S3 触发 → 探测 UNKNOWN-污染，检测器接管 W，段截断，全部保留 |
| 计时跨 rank 聚合 | span/bcost 取 confirm 归约 MAX；steps/true_steps 以 MIN==MAX 验证 lockstep；**禁止求和** | 决策序列 = MAX 序列（最慢 rank 支配） |
| dwell 语义（冻结） | probe_start：ctrl.W=32, dwell=0, dwell_dir=None（检测器可立即收缩——安全优先）；probe_fallback/abort：ctrl.W=16, dwell=dwell_grow(2), dwell_dir="grow"；stable_s/alert_k/episode_shrinks 不被探测转移改动 | v1 §5.1 开放项在此定值 |

## B. 冻结执行顺序（15 次计时启动；序号即执行序）

| seq | label | 条件 | mode | 特殊配置 |
|---|---|---|---|---|
| 1 | cal_m1_w1_a | W1 | off | — |
| 2 | cal_m1_w1_b | W1 | off | — |
| 3 | cal_m1_w2_a | W2 | off | — |
| 4 | cal_m1_w2_b | W2 | off | — |
| 5 | cal_m2_w1_norm | W1 | gate | ε_risk=2s |
| 6 | cal_m2_w1_abort | W1 | gate | **MECHANISM**：FORCE_ABORT_AT=1, ε_risk=1ms |
| 7 | cal_m2_w2_obs | W2 | gate | ε_risk=2s |
| 8 | cal_m3_b1_aaa | W1 | aaa | — |
| 9 | cal_m3_b1_aba | W1 | gate | — |
| 10 | cal_m3_b2_aaa | W1 | aaa | — |
| 11 | cal_m3_b2_aba | W1 | gate | — |
| 12 | cal_m3_b3_aaa | W2 | aaa | — |
| 13 | cal_m3_b3_aba | W2 | gate | — |
| 14 | cal_m3_b4_aaa | W2 | aaa | — |
| 15 | cal_m3_b4_aba | W2 | gate | — |

M3 配对 block = {b1:(8,9), b2:(10,11), b3:(12,13), b4:(14,15)}；block 内 AAA 先、ABA 后（相邻启动，降低时间漂移）；顺序冻结，不因结果调整。
所有启动共同配置：controller=21、A 臂冻结 knobs（与封存 env_A 逐字相同）、EA_W=16、EA_ALPHA=0.0、数据集 sha 锁定（W1 f0391f7c…、W2 9d852a75…）、并发 256、bench 超时 3600s、每启动卫生流程与封存驱动相同（kill→GPU clear→sha 校验→boot 轮询≤2400s→bench→工件→核验→teardown）。

## C. 阶段门槛（只读机制字段，永不读端点值作门槛）

- **M1→M2**：4/4 VALID（ok=N、errors=0、无 fatal/desync）**且** 行为不变性：逐窗 (window_id, steps_actual, W_before, W_after, event, committed) 序列、Σsteps、swap 发起数与封存 A 臂同条件 r1 完全一致（W1 参照 20 窗/992 步/grow@w4,w7，W2 参照 37 窗/1952 步；参照 = 封存 r1 jsonl；封存六轮间决策序列已验证完全一致）。swap 发起数 = committed 且 W_after≠W_before 的窗数（两侧同法从 jsonl 导出）。**z 不作 bit 级门槛**（冻结前修正 2026-10-18：z 为真实流量信号，封存同臂六轮间本身即有抖动——W1 zmax 0.803–1.084、W2 zmax 614–652——而决策序列六轮全同）；M1 逐窗描述性报告 max|Δz| 与 z 到最近决策阈值（z_lo/z_hi/z_urgent）的类间隔余量；任何窗 z 跨越阈值类或决策序列不一致 → HALT 报告，不进 M2。账目 ≤ ε_log（超账 → HALT 报告）。
- **M2→M3**：3/3 VALID **且** 机制审计全过：许可至多一次；触发窗=12（或记录的延迟触发）；段窗数与预算一致；正常路径 EVAL 后 W=16；abort 路径在 FORCE_ABORT_AT 窗后回退且过冲被记录；全程 h_min==h_max、desync=0、steps MIN==MAX；probe_state 序列与状态机定义一致；W2 观察 run 的域事件交互按 S3/DONE 语义处理。任何失败 → HALT 报告，不进 M3。
- **M3**：无门槛（数据收集）；分析一次性在全部 15 次启动结束后进行（阶段门槛只读机制字段）。

## D. 预算纪律

- 15 次**计时启动**上限（bench 已开始才计数）。零数据 boot 失败 = INFRA 事件：同格一次重 boot，INFRA 上限 3；INFRA 不占 15 计数；超限 → HALT 报告缺口。
- RESULT 失败（ok<N 或 errors>0）→ 退役不重试，占 15 计数；预算耗尽 → 报告缺口，不追加。
- 失败启动不移动后续顺序；缺格在报告中以 MISSING 标记。

## E. 统计方法（校准期）与留空表

方法 = **描述性**：逐启动 Δ_net 点值（µs/step 与相对 %）；AAA 四值对 ε_prac 的分类（误判表现）；ABA 四值的极差/分布（估计稳定性）；完整服务差 = 同 block ABA−AAA 的 total_time_s；污染/中止/anomaly 计数与率；账目（每边界/每 forward 新增）。**不做推断检验、不报置信区间**（如需区间一律标"经验区间（无覆盖保证）"）。

| 留空项（尚不能支持，不用假设值补齐） | 值 |
|---|---|
| Δ_net 区间覆盖保证 | （空白） |
| 生产准入阈值与区间方法 | （空白） |
| 生产 T-b 收益区间规格 | （空白） |
| TTFT p99 约束 δ | （空白） |
| ε_prac 正式值 | （空白；0.01×rate_ref 仅校准临时） |
| W64 任何参数 | （空白；v1 不授权） |
| 最小可分辨效应（功效） | （空白；由 M3 噪声底在下一预注册计算） |

## F. 偏离登记（冻结时已知、事先声明）

1. 校准触发为确定性（window_id≥R + T-c），生产触发条件 T-b 只记录不门槛——理由：AAA/ABA 配对对齐 + 机制聚焦；影响：本轮不验证生产触发路径（由离线单元测试覆盖）。
2. M2-abort 使用机制配置（FORCE_ABORT_AT=1、ε_risk=1ms）——标记 MECHANISM，隔离于一切性能统计。
3. Δ_net 为观察性估计：背景趋势经 ABA 差分校正，**不称已隔离因果**；残留假设 A1–A4 随结果报告。
4. M3 数据仅开发校准：不回填封存结论、不预挑正式战役参数、不作部署收益宣称。
5. **冻结后-门前更正（2026-10-18，M1 执行中、gate 评估前）**：ε_log 行的 avg_record_us 封存参照常数 81.61µs 为转录错误。发现经过：为编写分析脚本核对字段时，观察到 calib W1 运行末窗 PROF avg_record_us≈121µs 与冻结上限 89.77µs（81.61×1.10）矛盾；随即核对封存 artifacts，发现封存 A 臂六轮末窗 PROF 8-rank 中位数为 113.8–126.9µs，81.61 与任何封存末窗值均不符——其真实出处是密封 ADDENDUM R2 恒等式核算（81.61×89770 calls=7326ms，legacy 成本重分类分析中的中间量）。修正：gate 参照改为运行时从封存 <cond>_A__r1 server.log 末窗 PROF 块导出（8-rank 中位数，W1=115.54µs→cap 127.09、W2=114.67µs→cap 126.14），容差 +10% 不变，消除硬编码转录风险。**该修正的方向与幅度完全由校准开始前的封存数据决定，非依据校准观测调参**；时序与理由全文记录于此，并同步更新 deployed_shas_calib_m1m2m3.txt 中 m1 check 条目。描述性备注：封存同臂轮间自身波动即达 115→127µs（约 ±5%），calib M1 实测 117–121µs 落于封存轮间范围内。
6. **冻结后更正 #2（2026-10-18，M2 gate 首次评估时）**：m2_mechanism_check.py 的 H9_w32_only_probe_start 实现错误——原实现把"每个 committed 且 W_after==32 且 event≠probe_start 的窗"都判违规，与冻结时间线自相矛盾（CAND 段按设计以 W32 运行 n_cand 窗：保持窗 w17/w18 的 W_before==32、probe_seg==CAND，冻结前 70 项单元测试已验证该时间线）。首次 gate 评估 3/3 运行仅该项 FAIL（norm/obs 标记 w17,w18；abort 标记 w17），机制数据本身正确：转移级不变量（committed W 转移仅 probe_start/probe_fallback/probe_abort_risk，16→32 仅经 probe_start）本就通过，且独立扫描证实全部 W32 窗 = probe_start(w16) + CAND 保持窗、无任何 W64。修正后判定**更严格而非放宽**：W_after==32 合法当且仅当 (a) event==probe_start 或 (b) W_before==32 ∧ probe_seg==CAND（转移条件之外新增段归属条件）。发现于 gate 评估时（M2 数据已产生），修正依据 = 冻结设计文档 + 冻结前单元测试，与 M2 观测值无关；不改动任何机制代码/数据，不构成调参通过。同步更新 sha 锁 m2 check 条目。

## G. 交付清单（对应作者指令 6）

15 启动逐次结果表；AAA 误判表；ABA Δ_net 稳定性表；污染/中止/anomaly 率；完整服务差；新增成本账目；M1 行为不变性核验记录；M2 机制审计记录；全部保留事件清单；若无法可靠辨识 → 限制声明（不反复调到通过）。成功标准 = 测量可信、机制正确、估计器可接受检验；不以正收益判定成功。

## H. 代码冻结

实现文件（sha 见 deployed_shas_calib_m1m2m3.txt，冻结 commit 记录）：OEPLB/src/probe_admission.py（新）、controller.py、config.py、OEPLB/scripts/run_grid_bench.py（追加单调钟字段）；window_v2.py 冻结不动（810c54da…）；window_v2p1.py / v2p1_signals.py 不修改。部署位置 /opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/。驱动 run_calibration_m1m2m3.sh + run_list_calibration.csv（sha 入锁）。
