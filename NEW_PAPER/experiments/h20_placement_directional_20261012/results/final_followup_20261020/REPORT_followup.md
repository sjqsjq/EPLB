# final_followup_20261020 — 总报告(阶段八交付物)

**日期**:2026-10-21。**执行仓库**:/workspace/EPLB(仅本地提交,未 push)。
**授权指令**:作者八阶段指令(修复迁移可靠性;补官方执行路径对照、真正行使的采样消融、SLO 评价;沿用论文三项研究内容;旧 200 次实验/失败/协议/代码锁/原分析全部保留;新旧不混池;≤2 修复轮次;前置门不满足的包不执行、节省不挪用;不显著≠等价;未行使≠无用)。
**关键提交链**:2f775a7(代码锁)→ e19ab5b(阶段一)→ 26d3bba(阶段二 v1 冻结)→ 395a434(阶段三协议冻结)→ 413fed2(第二轮闭环 v2)→ bab38ca(阶段三完成)→ d16e037(阶段四 64 boots)→ 5038cd1f(阶段四分析)。

## 1. 阶段一(CPU 复核)— 完成
`ERRATA_addendum_20261020.md` + `cpu_audit/`:完成度分解(94.43% 主口径 / 94.13% 严格口径,差值 2305 = 两个部分成功格,逐项对账)、A2@E3 p=0.0102 为未校正 raw p(Holm 后 0.0512/0.0615,均不显著)、A3 解释修正(采样操纵未行使 → 对比不携带采样因果信息)、confirmed_grow 语义修正(内部控制事件)、逐臂配置导出(config_table.csv)。追加式勘误,旧报告未改写。

## 2. 阶段二(迁移可靠性修复)— 完成,修复轮次 2/2 用尽
- **第一轮(v1)**:MIGRATION-RACE FIX-1(post 前 empty_cache+sync)+ FIX-2(返回前 sync)+ DIAG-1(PLANHASH)。冻结 async_swapper 85a2dac882177667 / controller 4964cfc8e8af397a。GPU 正确性台账 6/6。
- **B1 事故**:prep B1(P@E3)在 window#17 挂死(8 卡 PLANHASH 一致、SeqNum=5 COALESCED 全发起永不完成、{0,1,5,6}/{2,3,4,7} 分区由 1.5s 完成顺序悖论证明)。失败现场全保留(`phase3_prep/boots/prep3_B1_P_E3/` + watchdog core dumps)。
- **第二轮(v2,根因闭环)**:根因 = NCCL 合并组预算切分死锁(pytorch/pytorch#174288 类,OPEN;NCCL 源码切分点"could vary by rank")。FIX-3 = 按 peer 升序分组提交 + 逐组 host 等待(单 peer 子组两侧镜像对称 ⇒ 切分一致;升序+等待 ⇒ 死锁自由)。DIAG-2 = 镜像对称日志。离线确定性证明:single 模式 iter0 复现 B1 签名死锁,perpeer 40/40 canary 0。v2 冻结 async_swapper **2e4652a0306cb315**(controller 不变)。验证:离线 j4/j5 17/17;实机 R2×4 全 PASS(1527 镜像对);同步成本保留(p2p 22.4–26.4ms/300ops vs v1 20.1ms)。控制律/阈值/S/H/W/α/β/候选规则/动作预算未动;未禁用迁移、未关 adaptive、未削事务。
- **治理红线(现行)**:修复轮次 2/≤2 用尽。任何后续权重/映射错误或同类迁移故障 ⇒ 停全部性能包、报告阻塞,不得自行开第三轮。
- 证据:`correctness/ROOT_CAUSE_ROUND2.md`、`correctness/ACCEPTANCE_EVIDENCE_v2.md`、`freeze/code_fixed_v2/`。

## 3. 阶段三(前置准备,≤12 prep)— 完成,9/12 用毕
- C1/C2(I 臂 SLO 校准)→ **SLO 参考冻结**(2026-10-20T09:24:53Z,先于任何对比臂):E3 ref_p95_TTFT=6.7419s;E4 ref_p95_TTFT=7.3127s、ref_p95_TPOT=0.08313s;Goodput@k,k∈{0.8,1.0,1.2},主 k=1.0;GOOD=error==null∧TTFT≤k·ref∧(E4)TPOT≤k·ref。
- A1/A2(官方 E_native 基线)PASS。
- B1(v1)FAIL 保留;**B1r/B2/B3/B4(v2)4/4 PASS**(0 watchdog,断言全绿)。
- **采样资格裁定(冻结规则,池化 4 个 v2 B boots):NOT ELIGIBLE** — S>1 share=0.0(<0.10);recording_ratio=0.0126(<0.039062,通过)但规则要求两条同时成立 ⇒ **阶段六 STOP**。裁定 boot 与事后数据不替换(PROTOCOL.md ADDENDUM A1)。

## 4. 阶段四(64-boot 核心补充)— 完成,正式预算 64/104
- 锁定矩阵 `phase4_core/supplement_matrix_locked.csv`(sha d3ff2b09…,开跑前验证,一字未改;code_freeze 列为 v1 旧值,实际执行版本 v2 见 `PHASE4_CODE_VERSION_ADDENDUM.md`,不回填)。
- 64/64 PASS(05:13:56–10:23:26Z),327,680 请求 0 错误,0 watchdog,16 个 P boot 镜像对称全过(5706 对),每 boot 流 sha + v2 双组件 sha 守卫。
- 分析(`phase4_core/analysis/`,预注册:8 块配对 bootstrap 10000 抽、seed 20261021、百分位 95% CI、Holm 按流×指标、P−F 次级):

| 流 | 指标 | P−I | P−E(官方) | P−F(次级) |
|---|---|---|---|---|
| E3 | tps | **+1.14**(p_H<1e-4) | **+2.86**(p_H<1e-4) | +0.11(ns, 0.88) |
| E3 | Goodput@1.0 | **−0.060**(p_H<1e-4) | +0.005(ns, 0.88) | +0.021(ns, 0.37) |
| E4 | tps | **−226.3**(p_H<1e-4) | **+1735.7**(p_H<1e-4) | +149.5(raw p=0.0006) |
| E4 | Goodput@1.0 | **−0.148**(p_H<1e-4) | **+0.787**(p_H<1e-4) | +0.086(raw p=0.0106) |

- E 臂 E4 Goodput@k 全 k=0(tpot_p95 0.307s = 3.7×参考);E 臂执行方式(关 CUDA graph + recorder 常开)为预注册偏差,已披露。
- P 臂 S 状态(描述性):E3 2/8 boot 达 S=4(share 0.1875),E4 全 S=1;issued windows E3=112、E4=99;F 臂 E4 windows 12–20。
- 旧四流总 G 未更新、未混池;新补充池标签 s_*,独立分析。

## 5. 阶段五(PH256,条件 ≤16)— **未启动,待作者**
门条件原文在上下文压缩中丢失且本会话全部工具无法读取 runner 主机对话记录(专项核查确认);pod 侧文物仅确立预算包络与命名先例(旧 P1 "PH@C_hi=256")。按指令"前置条件不满足的包不执行、节省不挪用":不自动启动。见 `phase4_core/PHASE5_GATE_GAP.md`。待作者:(a) 重述门条件,或 (b) 确认跳过。

## 6. 阶段六(采样消融 ≤24)— **STOP(资格门 NOT ELIGIBLE)**
未行使≠无用:正式包中 E3 有 2/8 boot 行使了 S=4 升级(描述性);但冻结资格规则按指定 prep boots 池化裁定为 0.0,判决维持,不做事后替换。关于采样机制的因果结论**不可写**。

## 7. 阶段七(预算台账)
| 台账 | 预算 | 已用 | 状态 |
|---|---|---|---|
| 正式(formal) | ≤104 | 64(阶段四) | 阶段五 ≤16 待定;阶段六 ≤24 未执行不挪用 |
| 准备(prep) | ≤12 | 9(C1,C2,A1,A2,B1-FAIL,B1r,B2,B3,B4) | 3 备用保留,不挪用 |
| GPU 正确性(阶段二 v1) | ≤6 | 6 | 台账闭合(commit 26d3bba 期) |
| 第二轮验证(独立台账) | — | 离线套件 + 4 实机 | ACCEPTANCE_EVIDENCE_v2.md 记录 |
| 失败保留 | — | B1(挂死现场+cores)、旧战役 5 挂死格、2 部分成功格 | 全保留,零删除 |

无按均值/p/boot-draw 的样本删除;无预算自扩;无核心控制律改动;无新参数搜索。

## 8. 阶段八(交付物)
- **一页"可以写/不能写/仍未知"**:`ONEPAGE_claims.md`(本目录)。
- 明细:SUPP_RESULTS.json / RESULTS_SUPP.md(阶段四)、SAMPLING_ELIGIBILITY.json、SLO_REFERENCE_FROZEN.json、ROOT_CAUSE_ROUND2.md、ACCEPTANCE_EVIDENCE_v2.md、ERRATA_addendum_20261020.md、LEDGER_phase4.tsv、LEDGER_prep.tsv、LEDGER_r2.tsv、freeze/(v1+v2)、code_lock/。
- 旧战役(final_plan_20261019)一切原样保留。
