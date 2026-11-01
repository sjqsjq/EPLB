# REPORT — sp_coupling_v1 DEV seq 2 MECH_RETIRE HALT（已解除：checker 判据缺陷，非运行时机制错误）

日期: 2026-10-18 11:50Z (HALT) → 12:0xZ (诊断+修复+重判)  执行者: 执行AI（作者全权授权、在睡觉）

## 1. 事件时间线
- 11:36:25Z DEV 驱动启动 (12 计时启动, 冻结臂序)
- 11:43:32Z seq 1 (W1 b1 F_sparse) VERDICT=VALID t=353.73s（校准 W1 包络 352–357s 内）
- 11:50:44Z seq 2 (W1 b1 A) VERDICT=MECH_RETIRE t=352.17s (checker rc=2)
- 11:50:50Z 驱动按治理 HALT (exit 8, 状态保全, kill_server 执行, GPU 清空 4MiB, 两驱动 PID 转 defunct)

## 2. 触发判据与根因
- 唯一 mech 项: "S1 gate: opportunities 938 != sp_fwd 1010"
- 根因: integrity_check_sp.py 把 S1 门的精确恒等式 (opps==sp_fwd) 套用到 A 臂。该恒等式仅当全程 S=1 成立；A 臂按设计在 QUIET  residency 期间 S=4（本次 6 个 QUIET 周期）。冒烟 A 臂恰好从未离开 TRACK（2 次 swap 打断 quiet_streak），故该缺陷未在 P0e 暴露。
- 分类: 检查器判据缺陷（工具类），非运行时机制错误。

## 3. 运行时机制完整性证据（seq 2 全部通过）
- jsonl 72 行: fatal=0, desync=0, lockstep_fail=0, off-table S/H/W=0, HOLD=0
- residency 和 (47+6+19=72) == h_cycles (72)；steps 核算 47×16+6×16+19×8=1000 == true_steps_total
- PD-9 门账本推导界（数据无关）: opps=938 ∈ [916,950]（steps_s1=904, steps_q=96, q_eps=6, 每进出各 1 block-edge 松弛, idle 全按 S1 计费的上界）
- bench: ok=4000 errors=0 concurrency=256 label/dataset 匹配
- 结论: 该启动的服务与机制记录完整有效。

## 4. 修复（integrity_check_sp.py, 两处, 均在计时数据之外可推导）
1. A 臂改用账本推导界 lo=steps_s1+steps_q//4−2·q_eps, hi=(fwd−steps_q)+ceil(steps_q/4)+2·q_eps；固定臂保留精确恒等式（S1: opps==sp_fwd; S4: |4·opps−sp_fwd|≤4）。界公式只用 sp_run_end 自身提交的 residency/transitions 账本，不含任何端点测量值。
2. 修复自身引入的变量遮蔽 (res 字典覆盖了 result 失败列表 → 短暂把 A 臂误判 RESULT_RETIRE；重判定稿前发现并改名 res_h)。
- 修复后离线复核 5 个既存工件: seq1 VALID, seq2 VALID (t=352.17 原值), smoke×3 VALID —— 与驱动原判一致（除 seq2 被纠正）。

## 5. 治理核对
- W/H/S 值、全部阈值、臂序、数据集: 未动（计时开始后不得改 — 遵守）
- seq 2 未重跑、未替换工件: 同一工件用修正判据重判（不挑选有利重复）
- 失败启动 retire 不重试: 不适用（启动本身成功, 是判定工具误分类）
- state.tsv 追加 CHECKER_FIX 与 VERDICT_REJUDGED 审计行（append-only, 原 MECH_RETIRE 行保留）
- 先例: 校准战役 analyzer STAGE_END 工具缺陷修复后继续 (commit 466d387)
- HALT 报告义务: 本文件 + PROGRESS.md 更新 + git commit（LOCAL ONLY）

## 6. 附带科学发现（P2 素材, 不构成任何调参依据）
A 臂 W1 出现 QUIET 边界抖动: 6 次 QUIET 进入 (h_rate 0.0080–0.0086 < θ=0.0101) 后 1–2 周期内即因 h_rate 0.0109–0.0126 ≥ θ 退出到 RECOVER；z 全程远低于 z_lo (0.35–0.86 vs 22.6)，退出全部由 h_rate 触发。进入/退出共用同一 θ 且无滞回带（PD-2/PD-3 冻结设计如此），h_rate 在 θ 附近漂移时抖动是设计内行为。后果: 11 次封存, QUIET 常驻仅 6/72 周期, 机会减少仅 7.1% (938/1010)。S4 采样是否抬高/放大 h_rate 噪声将由 F_sparse vs F_dense 的 h_rate 分布对比回答（PD-1 §4.3 核查, P2 报告）。按治理不做任何阈值/滞回修改。

## 7. 决定
HALT 解除, DEV 从 seq 3 续跑（resume 跳过已有 VERDICT 的 seq1/seq2）。预算核算: timed starts 2/12 已用, INFRA 0/3。
