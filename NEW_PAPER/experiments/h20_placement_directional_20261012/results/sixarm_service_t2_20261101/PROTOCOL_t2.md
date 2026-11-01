# PROTOCOL_t2 - 六臂完整服务评价 (sixarm_service_t2_20261101)
FROZEN before any GPU run. Governing spec = author messages 2026-10-31 and 2026-11-01 (D1 路径A, D3 O32变体, RG=0, 零冗余核对, prompt内容交集核对, 三轮x两场景独立启动=36正式启动, 顺序提前冻结, 保留全部结果, 7.5h硬上限, 不看成绩改规则).

## 0. 授权与决策记录
- D1 = 路径A: 六臂统一 deepep normal + CUDA graph off.
- D3 = pbd O32 变体物化获批 (原始流只读).
- P 臂冻结 OEPLB_SP_RG=0 单一变体 (XD P0_r1 params=0 先例实证).
- 矩阵: 固定三轮, 两场景独立启动, 36 次正式启动; ORDER_TABLE_t2.json 提前冻结 (场景前缀标签 PROV_*/PBD_*); 全部结果保留 (rc!=0 亦归档不删).
- T3 正式压测后置, 待本战役收口后另行安排.

## 1. 六臂 (launch_arm_xd.sh 字节拼接, 唯一路径 delta: BASE auto->normal)
| 臂 | 配置 | 冗余 (实证) |
|---|---|---|
| I | BASE, trivial, 无 PB | ep_num_redundant_experts=0 |
| ES | BASE(normal) + --ep-num-redundant-experts 16 + --init-expert-location datafore_e3_placement.json | 16 (机制必需) |
| ED | BASE(normal) + --enable-eplb --eplb-algorithm auto + R16 + --eplb-rebalance-num-iterations 256 (XD flags 字节同) | 16 (机制必需) |
| MT | BASE + --init-expert-location placement_e3_fair.json | 0 |
| DF | BASE + --init-expert-location datafore_e3_remap_placement.json | 0 |
| P0 | pb_env_P (V21 冻结常数 + SP adaptive S=1 W=16 SEED=20261018 QUIET_STREAK=3 RECOVER_MIN_CYCLES=2 REV=1, XD 字节同) + OEPLB_SP_RG=0 + PB flags | 0 |
- 零冗余核对结论 (XD 全 boots server.log 实证): I/MT/DF/P=0, ES/ED=16; 本战役沿用, 零新参数.
- ES/ED 在 normal 为 XD 历史原配; I/MT/DF/P 由 auto->normal 为路径A delta (非机制/参数改动), smoke 门覆盖.

## 2. 流与画像分离
- 场景 PROV: prover_1024_out32_ie.jsonl (1024 行, sha256 0f8b5a50..., O32 ie, conc 256, bench_timeout 1800, G3 先例同口径).
- 场景 PBD: pbd_prover_book_double_out32_ie.jsonl (5344 行, sha256 cdd1d864..., 本轮由原始 pbd 流按 build_streams_g3.py 变换规则物化: max_tokens 1->32 共5344处, ignore_eos false->true 共3344处 (book 块原已 true), temperature=0 不变, 其余字段字节保留; 原始流 f864f236... 只读). conc 256, bench_timeout 1800.
- 声明 delta: pbd 母战役 (O1 prefill 重制度) 用 conc 32; 本战役两场景统一 O32 服务口径 conc 256 (与 PROV/G3/XD-E3 先例一致), 事前冻结, 非中途调参.
- 画像 x 评测交集 (prompt 内容逐条 sha256 精确比对, 本轮实证): e3(8192) 交 pbd(5344) = 0; e3 交 prover(1024) = 0; pbd 交 prover = 0. 域相交 (E3 含 prover/book 各2048条同域), 内容零交集.
- 画像独立性: ES/DF <- E3 capture 画像 (字节相同 ddc3943d...); MT <- P_e3.npz -> ILP1 (ddb38bfd...); ED <- 运行时统计; P0 <- V21 离线校准常数; I <- 无. 全部历史域数据, 与评测流按上表零内容交集.

## 3. 矩阵与顺序 (冻结)
- SMOKE 2 boots (零正式数据, prover_512_smoke.jsonl): PROV_I_s0 (plain x normal 类), PROV_P0_s0 (PB adaptive x normal 类). 任一失败 => 报具体问题, STOP.
- 正式 36 boots 按 ORDER_TABLE_t2.json: 场景为主块 (PROV 18 -> PBD 18); 场景内轮转 r1: ES ED MT DF P0 I; r2: ED MT DF P0 I ES; r3: MT DF P0 I ES ED (XD/pbd 轮转方案延续).
- 每 boot 独立启动独立拆卸 (两场景独立启动, 无双段续跑).

## 4. 每 boot 会计 (全部保留)
- bench json: tps, ttft/tpot/lat mean/p50/p95/p99, ok==nreq, total_time_s, avg_output_tokens (实际 O32 验证).
- MECH_CHECK 事实: ServerArgs 中 deepep_mode=normal, disable_cuda_graph=True, ep_num_redundant_experts 按臂表, init_expert_location 按臂 (ES/DF/MT), ED eplb flags, P0 SP env 摘要; ready_s 工程开销单列.
- v2p1/controller 日志仅 P0 臂产生, 原样归档.

## 5. 守卫与 STOP
- boot 级守卫: 流 sha (按场景); pb 代码 4 文件 sha (async_swapper 2e4652a0.../controller 4964cfc8.../sp_state 525e88f0.../window_v2 810c54da..., 部署态本轮复核一致); launcher launch_arm_t2.sh sha; bench_client_v2 sha 06668243...; placement lock (PL_ES/PL_DF/PL_MT 与 XD 冻结值相同, 缺失 => BOOT_ABORT).
- rc=2 => GUARD_ABORT 停队; rc=1 => 记录继续 (INFRA 零数据重启 <= 2); 无自动重试; 不静默缩臂.

## 6. 预算
- GPU: smoke 0.3h + PROV 18 x ~4.3min ~ 1.3h + PBD 18 x ~8.7min ~ 2.6h => T2 <= 4.3h; 本轮 T1+T2 <= 5.6h < 7.5h 硬上限. 实测 boot 依据: XD ready+bench+teardown wall 4.9-5.4min (E3 190s 段), G3 PROV 段 ~35-45s, pbd O1 段 740-797s@conc32 (本战役 conc256 重估 ~5min, 若实测超预算 20% 则如实报告, 不砍轮次).

## 7. 产物与纪律
- boots/LABEL/...; analysis/; code_lock/{placement_shas.env, ORDER_TABLE_t2.json}; LEDGER_t2.tsv append-only; 命名 t2_*; 关键路径程序化拼接.
