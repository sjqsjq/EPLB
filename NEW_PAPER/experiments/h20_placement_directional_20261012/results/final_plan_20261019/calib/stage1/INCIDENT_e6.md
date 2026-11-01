# INCIDENT: e6 候选在校准流上机制不可操作（2026-10-18T22:20Z MECH_RETIRE → HALT → 自主修正）
## 现象
c1_e6_r1（E auto R16 period1000，"默认周期参照"）：bench 干净（ok=1280 errors=0 t=77.14s），横幅 `will rebalance per 1000 iterations` ✓，零 Traceback，但 `rebalance start` 0 次 → integrity_check_fp.py 判 MECH_RETIRE（"E arm: no rebalance actually triggered"）→ 驱动按治理 HALT。
## 根因（机制-工作负载必然，非构建错误）
- 触发链：model_runner.forward 每 forward pass 调 eplb_manager.on_forward_pass_end() → generator `for _ in range(period): yield` 计满才 rebalance（eplb_manager.py L41-53, model_runner.py L2681）。
- 每个 DP rank 各有 EPLBManager，各记一行日志。已测 E 细胞：e1_r2 32 行=4 事件×8 rank（period64 ⇒ passes≥256）；e4_r2 128 行=16 事件×8（period16 ⇒ passes≥256 且 <272）；e2_r1/e3_r1 同 e1。**冻结校准流（calib_screen 1280 reqs conc256）总 forward pass ≈ 256-271**。
- period=1000 > 271 ⇒ e6 在该流上 rebalance 次数恒为 0。e6_r2 同参数同流，结果确定相同，不再消耗 boot。
- 佐证自洽：1280 reqs 中 1024 条 O=1（~130 prefill pass）+ 256 条 O=128 并发（~128 decode pass）≈ 260 pass。
## 决定（预注册修正，作者授权自主执行；在任何 stage-1 排序前记录）
1. e6_r1 维持 MECH_RETIRE 记录：retire 不重试，消耗 1 次计时 boot（marks 在案）。
2. e6_r2 从 run-list 剔除（确定性不可操作，不浪费 boot）；CALIB1 修正为 20 可操作细胞，TIMED_CAP 22→20；其余行冻结顺序不变。CSV 重生成、新 sha 锁嵌入驱动。
3. e6 不进入 stage-1 排序。**报告事实**：该 sglang 版本 EPLB 默认 period(1000) 在本项目校准工作负载尺度（~270 pass）下从不触发 rebalance——"默认周期参照"在此类基准上等价于 identity+EPLB 开销。此事实进冻结文件与最终报告。
4. e5（period256）保留：passes≥256 已在 4 个 E 细胞一致观测，预期恰好触发 1 次；若其 r 细胞出现 0 触发 → 同 MECH_RETIRE 流程处理并同法记录。
5. 预算核对：校准 boot 消耗 = 20（修正后上限）+ 1（e6_r1 已退役）= 21 ≤ 方案 22 screen 预算；确认阶段 ≤8 不变。
## 证据文件
c1_e6_r1/{server.log,bench.json,verdict.txt,meta.txt,meta_env.txt}；对照 c1_e1_r2/c1_e4_r2/c1_e2_r1/c1_e3_r1 server.log 计数；marks TSV 22:20:24Z HALT 行。
