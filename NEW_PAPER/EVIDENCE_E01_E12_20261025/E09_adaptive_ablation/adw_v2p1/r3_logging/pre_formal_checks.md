# 正式实验前只读核查（用户三问）— pre_formal_checks

日期：2026-10-17（冻结 07:25:26+00:00 之后、行为验证 07:34:19 结束之后编写）。
方法声明：**只读**——仅查阅现有代码/日志/工件 + 对既有 JSONL 的离线重算；
未启动任何 GPU 进程，未改动任何冻结参数、脚本或结果文件。

## ① 固定窗口臂是否关闭 _det_load、第二次 scatter、ADW 确认通信

**代码层结论：三者均为 v2.1 专属机制，由 `_wctrl_version == 21` 硬门控；
版本 ≠21 的固定窗口臂在构造上就不执行它们（_det_load 甚至不分配）。**

证据（deployed controller.py，sha256 24ccf304…，行号为部署文件）：
- 版本读取与分派：L196 `self._wctrl_version = int(getattr(cfg,'window_controller_version',1))`；
  L203-211：`==2 → _init_window_control_v2`；`==21 → _init_window_control_v2p1`；
  其余（除 1）raise。合法版本 {1(legacy), 2(frozen v2), 21(v2.1)}，默认 1
  （config.py L109 / L192 `OEPLB_WINDOW_CONTROLLER`）。
- _det_load 分配：仅 L1310，位于 `_init_window_control_v2p1` 内 → 只有版本 21 分配；
  L276-277 用 `getattr(self,'_det_load',None)` 防御性守卫覆盖 legacy。
- 第二次 scatter_add（记录路径，每层每 forward）：L359 `if self._wctrl_version == 21:` → L364
  `self._det_load[layer_id].scatter_add_(…)`。planner 自身的 `self.load` scatter 恒开
  （属 planner 累加器，所有 OEPLB 模式都有，非 ADW 机制）。
- 窗边界 _det_load 清零/结转检查：L496→L504；warmstart/refine 重排清空：L638/L713；
  swap 提交镜像 remap：L1731→L1740-1742。全部 `== 21` 门控。
- ADW v2.1 确认通信（det SUM all_reduce + confirm MIN/MAX all_reduce，phase 1-4）：
  全部位于 `_window_control_v2p1` 内（det clone L1422 起），唯一调用点 L946-952
  `if self._wctrl_version == 21: self._window_control_v2p1(…)`。版本 2 走
  `_window_control_v2`（v2 自己的另一套 det/collective）；版本 1 两者都不调用。

**先例（v2 时代正式比较，step6_formal_protocol_E.md L30-31）**：
V2 臂 = `OEPLB_WINDOW_CONTROLLER=2` + 冻结 knobs；**固定臂 F-global =
`OEPLB_WINDOW_CONTROLLER=1`（v1 byte-identical legacy path）+ EA_W=16** →
v2 时代正式比较的固定臂确实三者全关（自适应臂独自承担其开销）。

**本 r3 战役的实际情况（须区分）**：4 次固定 W logging run 用的是
`OEPLB_WINDOW_CONTROLLER=21` + 惰性 knobs（run_v2p1_logging.sh L25-26，
KAPPA=1.0、z 带 1e12/2e12/3e12、θ 1e-12/2.0、N_MIN=0）→ **三项机制全开**。
这是设计使然（校准需要 z/h_rate 样本），且 logging run 不是性能臂
（calibration_plan_v2p1.md §1.3：“不含任何跨臂比较，不是性能战役”）。
实测仪表化开销（PROF，每 rank 全程，rank 并行不相加）：
- W8（132 窗）：record 13.9s、allreduce 2.65s（window#132 末次报告）；
- W16（66 窗）：record 12.9s、allreduce 2.68s；
- 行为验证（32 窗）：record 11.1–11.5s、allreduce 0.79–0.82s、
  c_window_ms median 1.544ms×32。
（对照 bench 墙钟 361–399s，record 路径 ≈3%；窗数越多 allreduce 越大。）

**对 v2.1 正式实验的含义**：v2.1 正式比较协议**尚未撰写**
（calibration_plan_v2p1.md §5.3：“正式比较（任何跨臂性能结论）需要独立协议文档
+ 单独批准 + 单独预算”）。固定臂有两种合法构造，公平性后果不同，必须在协议中显式预注册：
- (A) controller 1（v2 时代先例）：三项机制全关 → 自适应臂独自承担
  record/collective 开销（量级如上）；差值 = W 自适应 + 仪表化成本之和。
- (B) controller 21 + 惰性 knobs（本战役 logging 模式）：仪表化对称开启 →
  差值隔离 W 自适应本身；但“固定臂”也承担 det/确认通信成本。
两种都不是错误，但不可混用、不可事后选择。

## ② 冻结参数是否尚未在 W1/W2 或其他评估集测过服务时间

**是——评估独立性完好。冻结后的 v2.1 参数只在 W0（=校准集）上跑过。**

- 时间线：FREEZE 07:25:26 → 行为验证 boot 07:26:59、bench 07:28:09-07:34:13、
  DONE 07:34:19（driver.log）。07:34:19 之后新建文件仅 4 个报告工件
  （behavior_analysis_v2p1.json / FINAL_REPORT_v2p1_r3.md /
  behavior_driver_console.log / driver.log），无任何 GPU 启动。
- 全部 5 个 v2.1 bench JSON（v21calib_W8/16/32/64 + v21behavior）
  dataset 均为 calibration.jsonl（uniq -c = 5）；benchmarks/results 中
  v21behavior.json（07:34）为最新。
- 冻结 κ 字符串 3875.1768798828125 全仓仅出现 2 处：
  run_v2p1_behavior.sh:23 + LOCK_record_r3_logging.md。
- W1 = w1_stable_code_4000.jsonl、W2 = w2_switch_code_chinese_8000.jsonl、
  evaluation.jsonl 存在于 workload 目录，但从未被 v2.1 knobs 触碰。
  现存 W1/W2 服务时间记录（如 s6c_W2_M1p.json，Oct 16 18:14，716.39s；
  addendum2 中 V2 s1/s2 步数）全部早于冻结、属 v2 时代战役（v2/消融 knobs）。
- 条款依据：calibration_plan_v2p1.md §6“校准数据 = W0-only；
  evaluation/W1/W2 不进入任何拟合、验收或诊断”。

推论（与既有声明一致）：冻结参数的服务时间目前**只有 W0 口径**，
W0 是校准集、不是留出验证集；任何 W1/W2/evaluation 上的性能结论都还是空白，
须待独立正式协议 + 单独批准后测量。

## ③ 报告中 2009/4018/6027 与真实边界 2075/4052/6068

**是——报告边界 = 预注册的 W0 四等分近似；真实构造边界 = 2075/4052/6068。**

- 真实构造（handoff_addendum2_workloads.md §1 W0 表，2026-10-16 程序化统计）：
  sharegpt 0–2074（2075 条）/ prover 2075–4051（1977）/ book 4052–6067（2016）/
  code 6068–8036（1969）；预定义切换点 B1=2075、B2=4052、B3=6068。
- r3 预注册（calibration_plan_v2p1.md §2 L115；fit_v2p1_calibration.py L39
  `BOUNDS=[2009,4018,6027]  # 0-based`，GUARD=±250）：8037×k/4 →
  2009.25/4018.5/6027.75 取整。与真实边界偏差 +66/+34/+41，全部在 ±250 护栏内。
- （split_manifest.json 是另一根轴：16000 行源数据 → calibration 8037 /
  evaluation 7963 的 sha 规则拆分；与 W0 内部 4 域块边界无关。）

**检测延迟口径（离线重算，同一映射约定“累计 steps_actual 比例 × 8037”）：**

| 边界 | 近似口径 containing idx / 首个 shrink / 延迟 | 真实口径 | 是否一致 |
|---|---|---|---|
| 2009 vs 2075 | idx15 (wid16, W=32, hold) / wid24 / 9 | idx15 / wid24 / 9 | 一致 |
| 4018 vs 4052 | idx21 (wid22, W=64, hold) / wid24 / 3 | idx21 / wid24 / 3 | 一致 |
| 6027 vs 6068 | idx27 (wid28, confirmed_grow) / 无 / — | idx27 / 无 / — | 一致 |

→ **行为验证报告的检测延迟（9/3/未检出）对边界口径不敏感。**

**拟合样本敏感性（report-only，未重拟合）**：按真实边界重新分类 4 个 logging run：
- 池变化：Z_s 182→184、Z_c 61→59。仅 W8 计数变化（101→103 / 30→28；
  翻转窗 wid29 z=232、wid62 z=1527、wid95 z=1204 change→stationary，
  wid39 z=2.0e5 stationary→change，全部合格）；W16 两窗互换（wid20 z=6.17e5
  s→c、wid31 z=1898 c→s，池计数不变 49/16）；W32/W64 无翻转。
- 复现校验：按预注册近似规则离线重算六参数与冻结值**逐位一致（Δ=0）**，
  证明下述敏感性数字可信。
- 若当初用真实边界（假设性重算，非重拟合）：κ −3.0%（3875.18→3758.66）、
  z_lo −32.4%（22.64→15.30；p90 尾部分位数，对移出池的 2 个极大 z 敏感）、
  z_hi +0.76%、z_urgent +0.76%、θ_r_low −0.27%、θ_r_act −0.43%。
- 行为显著性检查：行为验证 31 个 z（0.518–771.5）与 4 个 logging run 中
  **没有任何窗的 z 落在两个 z_lo 候选之间 (15.30, 22.64]** →
  带分类（quiet/midband）在全部既有 v2.1 数据上不受口径影响。
- 处置：按反调参条款（不静默重拟合；改 knob = 新预注册 + 重冻结），
  **冻结值保持预注册口径不变**；本敏感性仅存档为诊断。正式协议应写明
  检测延迟采用哪套边界口径（建议：真实构造边界 2075/4052/6068，
  并注明已验证与近似口径在行为验证数据上给出相同延迟）。

## 结论一览

| 问题 | 答案 | 对正式实验的要求 |
|---|---|---|
| ① 固定臂三机制 | 代码硬门控 ==21：版本 1/2 固定臂构造性全关；v2 时代先例固定臂=版本 1；r3 logging 臂=版本 21 惰性（全开，非性能臂） | 正式协议必须预注册固定臂构造（A: controller 1，开销不对称；B: controller 21 inert，仪表化对称），二选一、不得事后挑 |
| ② 冻结后 W1/W2 | 从未测过；冻结后唯一启动=行为验证（W0）；W1/W2/evaluation 干净 | 评估独立性成立；正式实验将是冻结参数的首次独立测量 |
| ③ 边界口径 | 报告值=等分近似（预注册）；真实边界 2075/4052/6068；行为延迟两口径相同；拟合样本 ≤4 窗翻转、κ −3%/z_lo −32%（尾部）仅作敏感性存档 | 冻结不回改；正式协议写明延迟口径（建议真实边界） |
