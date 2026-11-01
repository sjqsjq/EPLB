# 迁移可靠性根因报告（followup 阶段二取证）

日期: 2026-10-20 · 对象: final_plan_20261019 主矩阵 7 个退役 boot（5 hang + 2 crash-partial）
代码锁: `$G/code_lock/`（修复前 src 与 deployed 逐文件 sha256 相同，见 SHA256SUMS_code_lock.txt；git 2f775a7）

## 1. 结论

1. **“299-swap batch-mismatch（计划失配）”诊断不成立**。全部 7 个 boot 在挂起窗口，8 个 rank 的
   `[PB-OEPLB-DIAG] layers_touched / total_ops / avg_ratio_before / avg_ratio_after` 逐位一致
   （含 avg_ratio_after 三位小数，如 E3: 94/299/1.349/1.082）；v2.1 决策哈希校验 h_min==h_max、
   desync=false；rank0 的 p2l_version == swaps_started（E3=4，E4_b4=13，E4_b7=6）；
   7 个 boot 中 `[PB-OEPLB] error / finish error / finalize error / P2P swap failed / RANK DESYNC`
   出现次数全部为 0。计划输入（all_reduce 后的 global_load + p2l）无 rank 分歧。
2. **最小反例（同 boot 内自证）**：
   - m_E3_P_b3 / m_E3_F_b4：window#9 issued **299** ops，8 rank 全部 `swap(s) done` 并提交
     （swaps done 计数 32 = 4 次提交 × 8 rank）；window#17 同样 **299** ops 挂死。
   - m_E4_F_b4：window#19 与 window#37 两次 **300** ops（=max_total_ops 上限）成功提交；
     window#55 **300** ops 挂死。
   - m_E4_F_b7：window#19 **300** ops 成功；window#37 **299** ops 挂死。
   同等规模合法事务既有成功又有失败 ⇒ 失败不是事务大小或计划内容的确定函数，是**时序竞态**。
3. **根因：迁移 P2P 合并组与下一 forward 的 DeepEP 自旋 kernel 之间的跨 rank 循环等待**。
   机制链（与全部日志签名一致）：
   a. `async_swapper.begin()` 中 `req.wait()` 对 NCCL 工作是**流序等待而非主机阻塞**
      （torch 2.9 WorkNCCL 语义），begin() 在 P2P 组 kernel 仍在 GPU 上会合时即返回主机；
   b. 快 rank 返回后立即在事件循环提交 forward N+1，DeepEP dispatch/combine 为**持续自旋
      kernel**，占住 SM 并等待全体 EP rank 参与 forward N+1；
   c. 慢 rank（plan[0] 参与者因 VERIFY 校验的 `.item()` 主机阻塞，async_swapper.py:175）
      尚未返回，永远不会提交 forward N+1；
   d. 慢 rank 的 P2P 组 kernel 需要快 rank 同组 kernel 的会合才能推进，而快 rank 上
      自旋 kernel 抢占 SM/会合资源且其等待对象（慢 rank 的 N+1）被 c 卡死 ⇒
      **{P2P组会合} ↔ {forward N+1 自旋kernel} 循环等待**，600s 后 NCCL watchdog 中止。
   e. 大事务（≈300 ops × ~27.5MB fp8 权重）把 P2P 组执行时长拉长到与 (b) 的提交窗口
      必然重叠；小事务（如 49 ops）在快 rank 提交 N+1 之前已完成会合，故历史上未暴露。
4. **各 boot 的“等待方/完成方”rank 划分不同**（竞速结果不同），与机制一致：
   | boot | 挂起窗口 | plan ops | COALESCED超时ranks | begin内主机阻塞ranks | 组已完成ranks |
   |---|---|---|---|---|---|
   | m_E3_P_b3 | #17 | 299 | 0,1,5,6 (Seq5) | 0,5 | 2,3,4,7 |
   | m_E3_P_b6 | #17 | 299 | 0,1,5,6 (Seq5) | 同签名 | 同签名 |
   | m_E3_F_b3 | #17 | 299 | 0,1,5,6 (Seq5) | 0,5 | 2,3,4,7 |
   | m_E3_F_b4 | #17 | 299 | 0,1,5,6 (Seq5) | 0,5 | 2,3,4,7 |
   | m_E3_F_b8 | #17 | 299 | 0,1,5,6 (Seq5) | 同签名 | 同签名 |
   | m_E4_F_b4 | #55 | 300 | 0,2,3,5 (Seq14) | 0,3 | 1,4,6,7 |
   | m_E4_F_b7 | #37 | 299 | 2,3,4,5,6 (Seq7) | 4 | 0,1,7 |
   注：`issued` 日志行在快 rank 上出现不代表组完成（主机侧 wait 非阻塞）——E4_F_b4 的
   rank2,5 与 E4_F_b7 的 rank2,3,5,6 均打印 issued 后仍 watchdog 超时；仅无任何超时的
   rank（如 E3 的 2,3,4,7）其组真正完成。SeqNum=5/7/14 == 该 boot 第 5/7/14 个 P2P 合并组
   （PB 的 all_reduce 走 gloo/CPU 张量路径，不消耗 default_pg 的 NCCL SeqNum；
   sp_swaps=4/6/13 与 SeqNum-1 严格对应）。
5. 崩溃尾部签名（gloo `Connection closed by peer`、scheduler RuntimeError）为 watchdog
   abort 后控制面塌缩的**次生**现象，非首发故障。

## 2. 最小修复（round 1）

文件: `OEPLB/src/async_swapper.py`（同步部署到 `/opt/conda/.../sglang/srt/managers/pb_oeplb/`）
原则: 不改自适应阈值、S/H/W 状态表、α/β、候选规则、动作预算；不禁用迁移；不削小事务；
新增同步的成本保留并在性能包中如实计入。

- **FIX-1（提交前排空）**: `batch_isend_irecv` 之前 `torch.cuda.synchronize()`。
  begin() 入口时在飞 GPU 工作只有 forward N 的尾部 kernel（forward N+1 在任何 rank 上都
  尚未提交，因为 begin 在 forward N 的 on_forward_pass_end 内被调用）；forward N 的集合
  已全 rank 排队（8/8 rank 都进入了 begin，见证据），排空必然在有限时间完成。
  ⇒ P2P 组 kernel 总在**静默 GPU**上启动，不会被自旋 kernel 饿死。
- **FIX-2（返回前真完成）**: wait 循环之后 `torch.cuda.synchronize()`（try 与 retry 两条路径）。
  ⇒ 任何 rank 在自身 P2P 组真正完成前不得返回事件循环，故不存在“快 rank 提交 N+1 自旋
  kernel 而慢 rank 组未完成”的状态；循环等待的两个前提（并发窗口）被同时移除。
- **DIAG-1（计划哈希可见性）**: `controller.py` 计划构建后逐 rank 记录
  `[PB-OEPLB-PLANHASH] window#k rank=r n_ops=n plan_hash=sha256(...)[:16]`，
  把“计划一致”从事后推断升级为逐窗口可测证据。纯诊断，不改行为。

为何修复能关闭该路径：竞态需要 (i) P2P 组 kernel 与服务 kernel 在同一 GPU 上并发、
(ii) 某 rank 在自身组未完成时继续提交新工作。FIX-1 消除 (i)（组总在静默 GPU 上会合），
FIX-2 消除 (ii)（返回即完成）。两者同时失效才可能重现挂死，而各自失效都被对方兜底：
即使某 rank 排空前仍有残余 kernel，FIX-2 也保证它不会带着未完成组回到事件循环；
即使某 rank 的 synchronize 提前返回，FIX-1 保证对端 kernel 已在静默 GPU 上排队，
其主机会在自身 FIX-2 处等待会合完成——不存在第三方资源持有者（自旋 kernel）可插入。
失败模式残余：某 rank 在 begin 之前死亡/永不进入（不在本缺陷类内）→ 由既有 600s
watchdog 与 RESULT_RETIRE 台账兜底，行为与旧版一致。

## 3. 验证计划（本阶段内）

- CPU（先行）：torchrun×8 gloo 进程组驱动**真实** `AsyncSwapExecutor.begin()`：
  空计划 / 1 op / 8 ops / 299 / 300 / 300+同rank混合 / 连续多窗口 / 权重与 scale canary
  置换正确性 / busy 语义。断言无挂死（超时看门狗）、逐 rank 权重置换一致。
- GPU（≤6 个独立 8-rank job，固定事务数与超时）：J1 旧代码复现路径（大计划+并发负载）；
  J2 修复代码同场景；J3 空/小计划回归；J4 连续迁移窗口；J5 边界 299/300/相邻；
  J6 端到端正确性（weights/scale/p2l/stats 逐rank一致、commit version、canary 输出）。

## 4. 补充证据 6：与历史 SeqNum-分歧缺陷的签名区分

`async_swapper.py` 类 docstring 记载了早期 chunked 版本的缺陷：“uneven participation
makes the sequence numbers diverge … watchdog reported SeqNum=4 on some ranks and
SeqNum=5 on others”。该缺陷类的签名是**不同 rank 报告不同 SeqNum**。本次 7 个退役 boot
中，同一 boot 内所有报告 rank 的 SeqNum 完全一致（E3 五个 boot 全为 5；E4_F_b4 全为 14；
E4_F_b7 全为 7），且单批 `batch_isend_irecv` 全 rank 各参与一次 ⇒ 序列号对齐、参与均匀，
排除“参与不均/序列分歧”缺陷类，支持“一致计划的会合停滞（竞态）”结论。
