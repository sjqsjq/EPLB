# INCIDENT_E4F_hang — m_E4_F_b4 服务器中途崩溃（NCCL 悬挂 → watchdog abort）

日期：2026-10-19 00:25–00:47Z（发现于 00:40Z 监测心跳）
相位：MAIN（168 格主矩阵），格 seq=9，label=m_E4_F_b4（F 臂 × E4 fullduplex，block 4）
分类（按 plan §9.9 / FREEZE §7）：**方法崩溃 = RESULT_RETIRE（完成率结果，不重试、不删除）**
状态：驱动存活，战役继续；本格由驱动在 bench 自然结束或 BENCH_TIMEOUT（01:24:30Z）后按修正后检查器判定。

## 1. 时间线（全部取自本格 server.log / bench.log / 台账，UTC）

| 时刻 | 事件 |
|---|---|
| 00:23:19 | F 臂 server 启动（enable_pb_oeplb=True，deepep auto，mem_fraction 0.8） |
| 00:24:01 | ARMFLAGS 横幅 8 rank 齐：sp_mode=fixed sp_s=1 sp_h=16 sp_w=16 sp_rev=0 ✓（机制标记在位且一致） |
| 00:24:30 | bench 开始（e4_fullduplex.jsonl，2048 reqs，conc=256）；window#1 issued 268 swaps |
| 00:25:31 | 最后一条正常 Decode batch（gen tput ~803 tok/s/rank，queue 0） |
| 00:25:32 | window#55 issued **300 swaps**（= max_total_ops 上限；385024 tok global）；TIMING begin(): batch_isend_irecv=6.1ms n_ops=300；**NCCL COALESCED SeqNum=14 于此刻入队后永久悬挂**（由 watchdog 600s 反推） |
| 00:26–00:35 | 前端仍接受请求并发出 200 响应头（流 body 永不完成）；HTTP 日志 200 OK 累计至 2048 条（含悬挂流的头部） |
| 00:35:32 | NCCL watchdog 超时（Timeout=600000ms）：rank0/3/5 等 `Watchdog caught collective operation timeout: WorkNCCL(SeqNum=14, OpType=COALESCED, NumelIn=0, NumelOut=0)`；`terminate called ... c10::DistBackendError` → **Fatal Python error: Aborted ×4** |
| 00:36:32 | DP6 scheduler `RuntimeError: gloo ... Connection closed by peer`（级联）；调度器全部死亡，GPU 释放至 4MiB |
| 00:40:08 | 监测心跳发现：16 分钟无新 verdict + GPU 空闲 → 启动调查 |
| 00:46–00:47 | 取证：/health 超时（调度器死）、/get_model_info 200（前端半死）；主进程 563650 仍存活；bench（pid 569472）仍在按每请求 600s 超时逐波排空悬挂请求 |

## 2. 根因链（证据 → 假设分级）

**确证（日志直接证据）**：F 臂固定窗机制在高负载下正常推进（55 窗、13 次异步迁移启动、
批次达 300-op 上限），第 14 次迁移的 `batch_isend_irecv`（NCCL COALESCED P2P 批）在所有
rank 上永久未完成 → 600s watchdog → 4 rank SIGABRT → gloo 级联 → 服务器死亡。

**假设（留待 M2 诊断验证，≤4 boot 预算，战役后执行）**：固定 W=16 强制节奏 × 持续 decode
负载 ⇒ 异步权重迁移与 forward 集合通信在高节奏下发生竞态（某 rank 的 isend/irecv 配对
序列与其 EP 对等体发散 → COALESCED 批永久等待）。非确定性、暴露量依赖。

**关键对照（排除“大批次必然死锁”）**：
- calib f_1_16_16 共 10 次 VALID 运行同样打出 300-op 上限批次（stage1 r1：104 window 行），无悬挂；
- 同一 E4 流上 P 臂（自适应节奏，m_E4_P_b4，VALID t=90.42s）发出 299-op 批次、240 window 行，无悬挂；
- E4 上 A1（P+全记录）两格 VALID。
⇒ 差异变量是 **F 的强制迁移节奏在持续 decode 下的暴露量**：本格 62 秒内 55 窗/13 次迁移启动，
远超任何已封存 F 运行（calib 每次约 13 window 事件/rank，且 prefill 主导）。

## 3. 处置（全部先于本格判定完成，逐项引用冻结规则）

1. **不人工干预在飞格**：bench 由 BENCH_TIMEOUT=3600s（冻结值）或自然结束收敛；驱动自行判 verdict。
2. **检查器窄修正（工具对齐 plan §9.9，非规则变更）**：原 integrity_check_fp.py 将
   “bench.json 缺失”一律判 INTEGRITY_FAIL → 会误触发 HALT。plan §9.9 明文：方法崩溃/OOM/超时 =
   完成率结果（RESULT_RETIRE）。修正：仅当 server.log 含明确致命崩溃标记
   （watchdog collective timeout / Fatal Python error: Aborted / CUDA OOM / c10::DistBackendError /
   Scheduler hit an exception）且 bench.json 缺失或不可解析时，判 RESULT_RETIRE（方法崩溃类）；
   无崩溃证据的缺件仍判 INTEGRITY_FAIL。判定永不涉及终点数值。
   - 旧版封存：`C_configs/integrity_check_fp_pre_amendment_20261019.py`（sha f127d6e6bd8e…）
   - 新版 sha：a44e1e34725876ae…（驱动 meta.txt 每格记录 integ_sha，前后格可区分）
   - 离线验证：崩溃格 → RESULT_RETIRE ✓；健康格 m_E4_P_b4 → VERDICT:VALID t=90.42 不变 ✓
   - 已判 8 格 verdict **不重判**。
3. **矩阵不裁剪**：剩余 7 个 E4_F 格照常尝试（冻结矩阵，无中期跳格授权）。若竞态假设成立，
   各格将以 boot(~80s)+≤3600s 超时后 RESULT_RETIRE，E4×F 完成率本身即为可报告发现
   （F 固定窗机制在 decode-heavy 工作负载下的存活率）；P-vs-I / P-vs-E\* 确证比较不受影响；
   次级 P-vs-F@E4 按完成率如实申报。
4. 台账（campaign_marks.tsv）：INCIDENT + AMENDMENT 条目；PROGRESS.md #10。

## 4. 证据文件清单

- `main/m_E4_F_b4/server.log`（L139-150 ARMFLAGS；L3972-3980 watchdog；L4040/4115/4193/4271 Aborted；L4315 gloo；末段 PB-OEPLB window#55 + TIMING begin）
- `main/m_E4_F_b4/sp_run_end.json`（mode=fixed S/H/W=1/16/16、sp_fwd_total=890、window_count=55、swaps_started_total=13）
- `main/m_E4_F_b4/v2p1_rank0.jsonl`（55 行窗口记录）
- `main/m_E4_F_b4/bench.log`（bench 启动行；无结果段 = 未自然完成）
- 对照：`calib/stage1/c1_f_1_16_16_r1/server.log`、`main/m_E4_P_b4/server.log`
- 驱动日志：`logs/driver_MAIN.out`（BOOT_OK 00:24:20Z 后无 verdict = 在飞）

---

## ADDENDUM（事故 #2，2026-10-19）

同签名悬挂在 **P 臂** 再次发生：m_E3_P_b6（window#17，299 swaps，batch_isend_irecv COALESCED SeqNum=5，01:47:20Z 悬挂 → 01:57:20Z watchdog ranks 5/6/0/1 → abort 级联 → RESULT_RETIRE）。本文档中"F 臂强制节奏暴露"的框架**被取代/泛化**：该竞态发生在任何经 async swapper `batch_isend_irecv` 提交 swap 的臂上（F 与 P 均中招；I/A4/E 迄今免疫），是概率性而非确定性（同 run 内悬挂前 4 批大迁移成功；P 臂此前 4 格 VALID）。详见 `INCIDENT_E3P_hang.md`。
