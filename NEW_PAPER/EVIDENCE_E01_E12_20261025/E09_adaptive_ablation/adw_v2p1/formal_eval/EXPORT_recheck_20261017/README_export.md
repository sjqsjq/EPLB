# EXPORT_recheck_20261017 — 只读导出（复核用，无 GPU、无新运行）

来源工件：48 个启动目录（bench.json/meta.txt/verdict.json/server.log/v2p1_rank0.jsonl）
+ driver_state.tsv；git 提交 7569fd2（FINAL 报告同库）。导出时间 2026-10-17。
方向与统计口径不变；本导出仅供复核，不构成新分析规则。

## 文件
1. launches_48.csv — 48 次启动：seq/label/条件/臂/轮/位置/控制器/boot/bench起止(UTC)/
   verdict/total_time_s/tps/ok/errors/实际补全token数(total_comp_tokens)/延迟分位。
2. windows_A_M.csv — W1/W2 × A/M × 6 轮逐窗全字段（jsonl 原样 29 字段）+
   server_ts_dp0（server.log ADW21 w# 行时间戳，秒粒度）+ interval_s_from_prev +
   s_per_step_derived + remap_version_bump（p2l/det_map 版本递增=布局重映射提交）。
3. windows_F_legacy_anchors.csv — F 臂 legacy 控制器仅有：每 4 窗一条 PROF 锚点
   （DP0，含 calls/record/allreduce/planbuild/finalize 累计 ms）+ STABILITY 行。
   Δcalls/94 导出锚点间步数 → s_per_step_derived。
4. swap_events_all_arms.csv — A/M/F 全部规划/迁移事件行（swap_issued/swap_done/
   timing_begin/diag，全 8 rank，含时间戳、swap 数、token 数、耗时、失衡比前后）。
5. STALENESS_CHECK_descriptive.md — 描述性复核表（s/step 分桶、h_rate 走势、A−M 时间差定位）。

## MISSING 登记（工件中不存在，未做任何填补）
- prompt/输入 token 数：bench.json 未记录（MISSING）。数据集 sha 锁定、四臂同一文件，
  故输入 token 总量按设计跨臂恒定；实际补全 token = total_comp_tokens（O1: max_tokens=1）。
- tpot_*：None（max_tokens=1，无 inter-token 间隔，设计性缺失）。
- F 臂逐窗 W_before/W_after：legacy 不逐窗打印；由 server_args pb_oeplb_sync_window=16
  导出为 config 值（列内标注 config），逐窗 logged 值 = MISSING。
- F 臂逐窗 steps_actual 与逐窗时间戳：MISSING（仅每 4 窗 PROF 锚点 + swap 事件时刻）。
- A/M 逐窗墙钟：server.log 时间戳为秒粒度（±1s）；控制器内部窗时长未记录，
  interval_s_from_prev 为相邻窗提交时刻差（含该窗全部 forward + 边界开销）。
- L 臂逐窗：按本次要求未导出（launches_48.csv 含 L 的启动级数据）。
- M 臂 z 值：惰性 κ=1.0 → z 数值无物理意义（机制对照用），原样导出。

## 一致性自检
- A/M：ADW21 w# 行数 == jsonl 行数（A/M 24 个启动全部一致，ts_join_issues 列为空）。
- steps_actual = 该窗实际运行步数（= W_before；grow/shrink 在窗末边界生效）。
  Σsteps 为**已记录步数下界**（bench 结束于窗中时尾段 0..W−1 步不提交不记录）：
  W1 六轮 A=M=992；W2 六轮 A=1952、M=2000（差异判读见 STALENESS_CHECK 注意事项 2/3）。
- verdict 48/48 VALID 与 driver_state.tsv 一致；launches_48.csv 的 total_time_s 与
  REPORT T1 表逐一相同（同源 bench.json）。
- launches_48.csv 中 A/M 行的 ea_w_csv 字段为带引号的 "16(init,live)"（内含逗号，
  标准 CSV 引号规则）；用逗号裸切会错位，请用 CSV 解析器读取。

## ADDENDUM（作者三项追问，2026-10-17 追加）
见 ADDENDUM_recheck_clarifications.md：①A−M 曲线对齐规则及原 (c)(d) 节偏差更正
（matched-boundary 重算）；②legacy 147ms/窗 = record 122.1（per-forward 持续累计，
与 W 无关）+ 边界项 24.5ms；③最终布局同一性 = MISSING（无 p2l 内容/swap ID，
间接指纹提示三臂布局大概率不同）；④"全程 s/step"分母为已记录步数**下界**，
非独立记录的真实总 forward 数（任何工件均未记录真实总步数）。
