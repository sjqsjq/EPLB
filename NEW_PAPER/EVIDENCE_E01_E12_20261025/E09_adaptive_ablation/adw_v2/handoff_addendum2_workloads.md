# 增补文件 2：W0/W1/W2 工作负载完整说明（回应外部 AI 五问）

日期：2026-10-16。本文件是 handoff_step6_full_data.md 与 handoff_addendum_perstart.md 的第二份增补，
所有数字均为当日对冻结文件重新程序化统计所得（非手抄），sha256 与 workload_inventory_plan.md 冻结值核对一致。

## 0. 一句话回答

- W0 = calibration.jsonl（8037 条，4 个连续域块、3 个预定义切换点）；W1 = spliced_code_chinese_O1.jsonl 的
  前 4000 行（纯 code、0 切换，逐行字节一致已验证）；W2 = 该拼接文件全量 8000 行（code×4000 → chinese×4000，
  1 个预定义切换点在行 4000）。
- 三者均为 O1 协议：每行 max_tokens=1、temperature=0、ignore_eos=False；36 次启动全部验证
  total_comp_tokens == num_requests（每请求恰好 1 个输出 token），服务时间为 prefill 主导，tps ≈ req/s。
- 请求顺序 = 文件行序（asyncio.gather + Semaphore(256) 按行序发放）；切换方式为文件级连续域块拼接，
  无洗牌；控制器不接收任何域标签或切换时刻。
- calibration 与 evaluation：文件级/ID 级/决策级分离成立（ID 交集为 0；阈值只在 W0 上拟合；
  evaluation.jsonl 未用于任何 step-6 启动）；但内容级分离不完全——见 §3 与新增限制 L9。
- Phase A 网格与 Phase B 正式比较使用了完全相同的请求：同一条件同一文件（bench.json 内 dataset_path 相同、
  sha256 冻结且当日复验一致、发放语义相同），只有控制器配置不同。三重证据见 §4。

## 1. 逐条件画像

### W0 — calibration.jsonl（校准条件，4 域 3 切换）

| 项目 | 值 |
|---|---|
| 来源 | /data/minghua/sjq/OEPLBdata/datasets/multi_domain/ 真实请求池（prefill_heavy_universal 谱系） |
| 行数 | 8037 |
| 域块（行区间，条数） | sharegpt 0–2074（2075）、prover 2075–4051（1977）、book 4052–6067（2016）、code 6068–8036（1969） |
| 预定义切换点 | 3 个：行 2075（B1）、4052（B2）、6068（B3） |
| prompt 字符数（全文） | min 787 / median 4118 / max 15694 / mean 4289 |
| 各段 mean 字符数 | sharegpt 2937、prover 1067、book 4394、code 8842 |
| 唯一 prompt 数 | 5185 |
| 输出 | max_tokens=1、temperature=0、ignore_eos=False（全部行） |
| 引擎步数（实测） | V2 s1=1056、s2=1032；消融臂 1048/1056/1048 |
| 推算 prefill 总量 | ≈8.45–8.65M tokens（步数×8192 tok/步，饱和 chunked prefill），≈1052–1077 tok/请求 |

### W1 — w1_stable_code_4000.jsonl（稳定单域条件，0 切换）

| 项目 | 值 |
|---|---|
| 来源与选取规则 | spliced_code_chinese_O1.jsonl 的前 4000 行；预注册规则"第一个带标签的连续单域块"，跑前冻结 |
| 与源文件关系 | 逐行字节一致（已验证） |
| 行数 | 4000，全部 domain=code（行含 domain 字段），ids seg1_code_0..3999 |
| 切换点 | 0 |
| prompt 字符数 | min 4475 / median 8940 / max 11551 / mean 8837 |
| 唯一 prompt 数 | 125，每条恰好重复 32 次（重复直方图 {32:125}） |
| 输出 | 同 O1（max_tokens=1、temp=0、ignore_eos=False） |
| 引擎步数（实测） | V2 s1=992、s2=976 |
| 推算 prefill 总量 | ≈8.00–8.13M tokens，≈2000–2032 tok/请求 |

### W2 — w2_switch_code_chinese_8000.jsonl（预定义真实切换条件，1 切换）

| 项目 | 值 |
|---|---|
| 来源 | /data/minghua/sjq/OEPLBdata/datasets/multi_domain/spliced_code_chinese_O1.jsonl 全量（sha256 相同） |
| 行数 | 8000 |
| 域块 | code 行 0–3999；chinese 行 4000–7999 |
| 预定义切换点 | 1 个：行 4000 |
| prompt 字符数（code 段） | min 4475 / median 8940 / max 11551 / mean 8837（与 W1 完全同分布——W1 即此段） |
| prompt 字符数（chinese 段） | min 2055 / median 3094 / max 6878 / mean 3164 |
| prompt 字符数（全文） | min 2055 / median 5567 / max 11551 / mean 6000 |
| 唯一 prompt 数 | 250（125 code + 125 chinese），每条恰好重复 32 次 |
| 输出 | 同 O1 |
| 引擎步数（实测） | V2 s1=1976、s2=1976；消融臂 1984/1968/1976 |
| 推算 prefill 总量 | ≈16.12–16.25M tokens，≈2015–2032 tok/请求 |

注：输入长度的文件级度量是字符数；token 级数字是由引擎步数×8192 tok/步反推的执行级估计
（饱和 chunked prefill 下每步 8192 token，n_fresh/步=6,160,384=94 层×8 topk×8192 已验证），
末尾不足一步的尾差会造成 ±1 步以内的偏差。

### 参照：evaluation.jsonl（已知回归基准，step-6 未使用）

7963 行；段边界 sharegpt(0)/prover(1925)/book(3948)/code(5932)；字符 min 781/median 4147/max 16203/mean 4348；
5047 唯一 prompt；与 calibration 的 ID 交集为 0，但共享 1383 个唯一 prompt（同一有限真实请求池抽取）。
按主导指令它只是"已知回归"，不是未见验证集，且未进入任何 step-6 启动。

## 2. 请求顺序、发放与切换方式（三条件相同）

- 客户端为 /workspace/EPLB/OEPLB/scripts/run_grid_bench.py：按文件行序读入全部请求，
  asyncio.gather + Semaphore(256) 发放——发放顺序 = 文件行序，同时在飞 ≤256 条。
- warmup 仅为 conc 次 GET /health，不触达模型与 OEPLB/控制器状态（脚本内有显式设计注释：
  历史上真实请求 warmup 会预先推进控制器状态/触发交换，故禁用）。
- 计时从 warmup 之后的 t_start 开始，到全部请求完成为止（wall = total_time_s），
  含全部在线开销（检查、重规划、迁移），与证据链口径一致。
- "切换"只存在于请求序列中（连续域块拼接）；控制器不接收域标签、切换时刻或任何监督信息，
  只能从路由统计（z、h_w）自行发现。
- 并发 256、O1 载荷、发放语义在全部 36 次启动中一致；臂间唯一差异是控制器配置。

## 3. calibration 与 evaluation 是否分离（分层回答）

**文件/ID 级：分离。** calibration.jsonl 与 evaluation.jsonl 为两个文件，ID 交集 = 0。

**决策级：分离成立。** 控制器全部阈值/旋钮（kappa、z_lo/z_hi/z_urgent、theta_h_low/theta_h_act、n_min 等）
只在 W0=calibration 上拟合，从未在 W1/W2 上重调；W1/W2 由预注册规则在跑前选定并冻结；
evaluation.jsonl 未参与 step-6 任何启动，也未参与拟合。

**内容级：不完全分离（新发现，登记为 L9）。** 对唯一 prompt 哈希做逐段交集，当日验证：

| 交集 | 唯一 prompt 数 | 位置 |
|---|---|---|
| cal ∩ w1 | 125（W1 全部） | 全部落在 calibration 的 code 段 |
| cal ∩ w2 | 125 | 仅 code 半；chinese 半与 calibration 各段交集均为 0 |
| eva ∩ w1 | 125（W1 全部） | 全部落在 evaluation 的 code 段 |
| eva ∩ w2 | 125 | 仅 code 半；chinese 半与 evaluation 交集为 0 |
| w1 ∩ w2 | 125 | W1 即 W2 的 code 半（字节级前缀） |
| cal ∩ eva | 1383 | 同一有限池 |

含义：
1. W1 的 125 个唯一 prompt（各重复 32 次）全部出现在 calibration 与 evaluation 的 code 段中——
   W1 在内容上不是新颖的，新颖性只在"文件/序列/位置"层面（稳定单域长驻留序列本身未参与开发）。
2. W2 的切换目标域（chinese，125 唯一 prompt）不在 calibration 也不在 evaluation 中——
   W2 的检测考验（切换后 z 尖峰、h_w 机会）在内容级对切换后一侧仍然成立。
3. 重复结构（×32）是既有拼接文件的固有属性，非本战役构造；三臂收到逐字节相同的文件，
   重复不给任何臂带来优势，臂间比较结论不受影响。受影响的是"未见真实请求"这一表述的强度：
   它只在文件/序列层面与 W2-chinese 内容层面成立，对 code prompt 的内容层面不成立。
4. 这是有限真实 prompt 清单（multi_domain 池）的固有结果；已如实登记为限制 L9
   （deliverable_G_method_paper_ready.md §3）。

## 4. Phase A 网格与 Phase B 正式比较是否用了完全相同的请求：是

三重证据：

1. **同一文件路径**：36 次启动的 bench.json 均记录 dataset_path——calibration.jsonl ×13
   （A 网格 4 + B 正式 6 + C 消融 3）、w1_stable_code_4000.jsonl ×10（A4 + B6）、
   w2_switch_code_chinese_8000.jsonl ×13（A4 + B6 + C3）。同一条件的所有启动读同一文件。
2. **字节级冻结**：三文件 sha256 在 workload_inventory_plan.md 中跑前冻结；每个 phase 边界及当日
   （2026-10-16）复验，current == frozen；W1 另逐行验证与源文件前 4000 行字节一致，
   W2 与源拼接文件 sha256 一致。
3. **相同发放语义**：同一 run_grid_bench.py，行序发放、Semaphore(256)、/health-only warmup、
   warmup 后计时、逐行 O1 载荷（max_tokens=1/temp=0/ignore_eos=False）。

因此 Phase A（定 FL 网格）与 Phase B（V2 vs FG vs FL 正式比较）之间、以及 Phase C 消融，
请求内容、顺序、并发、计时口径完全相同；臂间唯一差异是窗口控制器配置
（固定 W / v2 全机制 / 单机制消融）。

sha256（当日复验值，与冻结值一致）：
- calibration.jsonl: 7d53aa0b07dbc9f11f47af44ee4f4106861f5ace6d4eeca1ba41cdf5b01b17f9
- w1_stable_code_4000.jsonl: f0391f7cdf8edd734a09c65b331d751052b8b1937a0416b3782a409039fe8578
- w2_switch_code_chinese_8000.jsonl: 9d852a756957d3db805fd876ebbad51a0224b9b6c23d23edb5646cdf260362f8
- evaluation.jsonl（step-6 未使用）: 53e5b0e1f74af708037d69f95856aced71374ec55826c96ea1f4134a78fedb1c

## 5. 锚点

- 工作负载目录：/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/inputs/fullservice_v2/workload/
- 来源池：/data/minghua/sjq/OEPLBdata/datasets/multi_domain/（prefill_heavy_universal.jsonl、spliced_code_chinese_O1.jsonl）
- 客户端：/workspace/EPLB/OEPLB/scripts/run_grid_bench.py
- 预注册清单：results/adw_v2/workload_inventory_plan.md
- 逐启动原始记录：results/step6/phaseA|phaseB|phaseC/**/bench.json（dataset_path、total_time_s、
  total_comp_tokens、num_requests）与 v2_rank0.jsonl（逐窗口 z/h_w/W/steps/n_fresh）
