# Benchmark数据分析工具调研报告

## 1. 数据结构分析

### 1.1 目录结构

```
result/benchmarks/
├── Qwen3-Coder-480B-FP8-TP{X}-EP{Y}-DP{Z}/  # 并行度配置
│   ├── bs{N}_in{INPUT}_out{OUTPUT}/          # 负载配置
│   │   └── inference_bs{N}_input{INPUT}_output{OUTPUT}.json  # 结果文件
```

**层级说明**：
- **Level 1**: 并行度配置 - `TP{X}-EP{Y}-DP{Z}`（Tensor Parallelism, Expert Parallelism, Data Parallelism）
- **Level 2**: 负载配置 - `bs{N}_in{INPUT}_out{OUTPUT}`（Batch Size, Input Length, Output Length）
- **Level 3**: JSON结果文件

### 1.2 数据规模

- **并行度配置**: 5种（TP8-EP2, TP8-EP4, TP8-EP8, TP16-EP8, TP16-EP16）
- **Batch Sizes**: 7种（1, 4, 16, 32, 64, 128, 256）
- **Input Lengths**: 8种（1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072）
- **Output Length**: 固定512
- **总文件数**: 278个JSON文件（理论上应该280个，缺失2个）
- **文件大小**: 平均1-2MB每个文件

### 1.3 JSON文件结构

每个JSON文件是**单行**大型对象，包含以下部分：

#### 配置信息（metadata）
```json
{
  "backend": "sglang",
  "dataset_name": "random-ids",
  "random_input_len": 1024,
  "random_output_len": 512,
  "server_info": {
    "tp_size": 16,
    "ep_size": 16,
    "dp_size": 1,
    "nnodes": 4,
    "model_path": "...",
    "mem_fraction_static": 0.8,
    "attention_backend": "trtllm_mha",
    ... (100+ 配置参数)
  }
}
```

#### 性能指标（汇总统计）
```json
{
  "duration": 12.864,
  "completed": 131,
  "total_input_tokens": 134144,
  "total_output_tokens": 67072,

  "request_throughput": 10.18,
  "input_throughput": 10427.61,
  "output_throughput": 5213.80,
  "total_throughput": 15641.41,

  "mean_e2e_latency_ms": 12786.74,
  "median_e2e_latency_ms": 12802.67,
  "p99_e2e_latency_ms": 12851.67,

  "mean_ttft_ms": 1034.12,      // Time to First Token
  "median_ttft_ms": 893.67,
  "p99_ttft_ms": 2150.17,

  "mean_tpot_ms": 23.00,        // Time Per Output Token
  "median_tpot_ms": 23.31,
  "p99_tpot_ms": 24.11,

  "mean_itl_ms": 23.00,         // Inter-Token Latency
  "median_itl_ms": 20.57,
  "p99_itl_ms": 22.95,

  "concurrency": 130.21,
  "max_output_tokens_per_s": 6512.0
}
```

#### 详细数据（数组）
```json
{
  "input_lens": [1024, 1024, ...],      // N个样本的输入长度
  "output_lens": [512, 512, ...],       // N个样本的输出长度
  "ttfts": [0.534, 0.533, ...],         // N个样本的TTFT（秒）
  "itls": [                              // N×M二维数组
    [1.782, 0.017, 0.020, ...],         // 第1个样本的M个token间延迟
    [1.756, 0.019, 0.021, ...],         // 第2个样本的M个token间延迟
    ...
  ]
}
```

**关键观察**：
- `completed`字段表示实际完成的请求数（如131），与batch_size不一定相等
- `itls`是二维数组，每个请求包含约511个token的延迟（output_len - 1）
- TTFT单位是秒，ITL单位是秒，但汇总统计是毫秒

---

## 2. 工具和方法调研

### 2.1 Python数据处理库

#### 已测试的库可用性（当前环境）

| 库 | 状态 | 用途 |
|---|---|---|
| Python 3.6+ | ✅ 可用 | 基础环境 |
| json | ✅ 可用 | JSON解析（标准库） |
| csv | ✅ 可用 | CSV读写（标准库） |
| statistics | ✅ 可用 | 基础统计（标准库） |
| collections | ✅ 可用 | 数据结构（标准库） |
| re | ✅ 可用 | 正则表达式（标准库） |
| pandas | ❌ 未安装 | 数据分析（**推荐安装**） |
| numpy | ❌ 未安装 | 数值计算（**推荐安装**） |
| matplotlib | ❌ 未安装 | 可视化（**推荐安装**） |
| seaborn | ❌ 未安装 | 高级可视化 |
| plotly | ❌ 未安装 | 交互式可视化 |
| gnuplot | ❌ 未安装 | 命令行绘图 |

#### 推荐安装的核心库

```bash
# 方案1：使用pip安装（推荐）
pip3 install --user pandas numpy matplotlib seaborn

# 方案2：使用conda（如果可用）
conda install pandas numpy matplotlib seaborn

# 可选：交互式可视化
pip3 install --user plotly jupyter
```

**选择理由**：
- **pandas**: 数据处理的事实标准，DataFrame API非常适合表格数据
- **numpy**: 数值计算基础，pandas依赖
- **matplotlib**: Python可视化基础库，生成论文级图表
- **seaborn**: 基于matplotlib的高级接口，统计可视化更简单

### 2.2 备选方案（标准库实现）

如果无法安装外部库，可以使用Python标准库实现基础功能：

**优点**：
- 无需安装依赖
- 轻量级，快速运行

**缺点**：
- 代码更复杂
- 缺少高级分析功能
- 无法生成图表
- 性能较差（处理大量数据时）

**适用场景**：
- 快速原型验证
- 服务器环境限制
- 简单的统计分析

### 2.3 可视化方案

#### 方案A：Matplotlib/Seaborn（推荐）
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 生成PNG/PDF图表
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='parallelism', y='output_throughput')
plt.savefig('result/figures/throughput_by_parallelism.png', dpi=300)
```

**优点**：
- 论文级图表质量
- 完全控制样式
- 支持多种导出格式（PNG, PDF, SVG）

**缺点**：
- 静态图表
- 学习曲线略陡

#### 方案B：Plotly（交互式）
```python
import plotly.express as px

# 生成交互式HTML图表
fig = px.line(df, x='input_len', y='output_throughput', color='parallelism')
fig.write_html('result/figures/throughput_interactive.html')
```

**优点**：
- 交互式探索（缩放、悬停、过滤）
- 生成独立HTML文件，易分享
- 现代化界面

**缺点**：
- HTML文件较大
- 需要浏览器查看

#### 方案C：Jupyter Notebook（开发环境）
```bash
pip3 install --user jupyter
jupyter notebook
```

**优点**：
- 代码和可视化结合
- 交互式开发
- 易于分享（.ipynb文件）

**缺点**：
- 需要额外学习
- 不适合自动化流程

#### 方案D：导出到Excel/Google Sheets（低技术门槛）
```python
df.to_excel('result/benchmark_analysis.xlsx', index=False)
```

**优点**：
- 用户友好
- 支持手动分析和图表
- 无需编程知识

**缺点**：
- 手动操作，不可重现
- 不适合大规模数据

---

## 3. 推荐工作流设计

### 3.1 工作流概览

```
┌─────────────────┐
│  JSON Files     │  (278 files, ~1-2MB each)
│  result/        │
│  benchmarks/    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Step 1: Parse  │  parse_benchmarks.py
│ JSON → CSV     │  → tmp/raw_data.csv
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Step 2: Clean  │  clean_data.py
│ & Filter       │  - 删除损坏文件
└────────┬────────┘  - 过滤异常值
         │           - 添加计算字段
         ↓           → tmp/cleaned_data.csv
┌─────────────────┐
│ Step 3:        │  analyze_data.py
│ Analysis       │  - 分组统计
└────────┬────────┘  - 性能对比
         │           - 趋势分析
         ↓           → result/analysis/
┌─────────────────┐      - summary_stats.csv
│ Step 4:        │      - comparison_tables.csv
│ Visualization  │  plot_results.py
└────────┬────────┘  → result/figures/
         │              - throughput_by_*.png
         ↓              - latency_heatmap.png
┌─────────────────┐      - scaling_analysis.png
│ Final Report   │
│ (Markdown/PDF) │
└─────────────────┘
```

### 3.2 详细步骤

#### Step 1: 解析JSON → CSV

**输入**: `result/benchmarks/**/*.json`
**输出**: `tmp/raw_data.csv`

**功能**：
- 扫描所有JSON文件
- 从路径提取配置信息（TP, EP, DP, BS, Input, Output）
- 从JSON提取性能指标
- 合并为单个CSV文件

**关键字段**：
```
tp_size, ep_size, dp_size, nnodes,
batch_size, input_len, output_len,
duration, completed,
total_input_tokens, total_output_tokens,
request_throughput, input_throughput, output_throughput, total_throughput,
mean_e2e_latency_ms, median_e2e_latency_ms, p99_e2e_latency_ms,
mean_ttft_ms, median_ttft_ms, p99_ttft_ms,
mean_tpot_ms, median_tpot_ms, p99_tpot_ms,
mean_itl_ms, median_itl_ms, p99_itl_ms,
concurrency, max_output_tokens_per_s
```

#### Step 2: 数据清洗

**输入**: `tmp/raw_data.csv`
**输出**: `tmp/cleaned_data.csv`

**清洗规则**：
1. **删除损坏数据**：JSON解析失败的文件
2. **过滤无效数据**：
   - `duration <= 0`
   - `completed == 0`
   - 关键指标为空（throughput, latency）
3. **异常值检测**：
   - 使用IQR方法检测output_throughput异常值
   - 标记但不删除（添加`is_outlier`字段）
   - 记录原因到日志
4. **添加计算字段**：
   - `parallelism`: "TP{X}-EP{Y}-DP{Z}"
   - `workload`: "BS{N}_IN{INPUT}"
   - `total_gpus`: `tp_size * ep_size * dp_size * nnodes`
   - `tokens_per_gpu`: `total_output_tokens / total_gpus / duration`

**异常值判断标准**：
- IQR方法：Q1 - 1.5×IQR, Q3 + 1.5×IQR
- 基于上下文：同配置下偏差超过2个标准差
- 手动规则：throughput过低（< 50 tokens/s）可能是系统故障

#### Step 3: 数据分析

**输入**: `tmp/cleaned_data.csv`
**输出**: `result/analysis/*.csv` + 日志

**分析维度**：

1. **并行度对比**（固定workload）：
   - 对比TP8-EP2, TP8-EP4, TP8-EP8, TP16-EP8, TP16-EP16
   - 指标：throughput, latency, efficiency
   - 结果：`parallelism_comparison.csv`

2. **负载扩展性**（固定parallelism）：
   - 对比不同batch size和input length
   - 指标：throughput scaling, latency scaling
   - 结果：`workload_scaling.csv`

3. **效率分析**：
   - GPU利用率：`tokens_per_gpu`
   - 并行效率：`throughput / total_gpus`
   - 结果：`efficiency_analysis.csv`

4. **Roofline分析**（如果适用）：
   - 理论峰值 vs 实际性能
   - Bottleneck识别（compute vs memory）

#### Step 4: 可视化

**输入**: `tmp/cleaned_data.csv` + `result/analysis/*.csv`
**输出**: `result/figures/*.png`

**关键图表**：

1. **Throughput对比图**：
   - Bar chart: 不同并行度的throughput
   - 分组：按batch size
   - `throughput_by_parallelism.png`

2. **Latency对比图**：
   - Box plot: TTFT, TPOT, E2E latency分布
   - 分组：按并行度
   - `latency_comparison.png`

3. **扩展性曲线**：
   - Line chart: batch size vs throughput
   - 多条线：不同并行度配置
   - `scaling_curves.png`

4. **热力图**：
   - Heatmap: batch size × input length → throughput
   - 分面：每个并行度一个子图
   - `throughput_heatmap.png`

5. **效率分析**：
   - Scatter plot: total_gpus vs tokens_per_gpu
   - 颜色：并行度配置
   - `gpu_efficiency.png`

---

## 4. 数据兼容性设计

随着实验进行，数据会持续增加，需要考虑以下兼容性问题：

### 4.1 新增并行度配置

**场景**：添加新的TP/EP/DP组合

**解决方案**：
- 解析脚本自动识别新配置（从路径提取）
- 分析脚本动态检测所有配置
- 可视化脚本使用colormap自动分配颜色

**代码示例**：
```python
# 动态获取所有并行度配置
parallelism_configs = df['parallelism'].unique()
# 自动分配颜色
colors = plt.cm.tab10(range(len(parallelism_configs)))
```

### 4.2 新增负载配置

**场景**：添加新的batch size、input/output length

**解决方案**：
- 解析脚本从文件名提取，无需硬编码
- 分析脚本按值排序，自动适应范围

### 4.3 JSON格式变化

**场景**：SGLang更新，添加新字段或修改字段名

**解决方案**：
1. **向后兼容**：使用`.get()`而非直接索引
```python
throughput = data.get('output_throughput', None)  # 安全
# 而不是 data['output_throughput']  # 可能KeyError
```

2. **字段映射**：维护字段别名映射表
```python
FIELD_ALIASES = {
    'output_throughput': ['output_throughput', 'decode_throughput'],
    'mean_ttft_ms': ['mean_ttft_ms', 'ttft_mean'],
}
```

3. **版本检测**：从JSON中提取版本号
```python
version = data.get('version', 'unknown')
if version >= '0.6.0':
    # 使用新字段
else:
    # 使用旧字段
```

### 4.4 大规模数据处理

**场景**：文件数超过1000个，单机内存不足

**解决方案**：
1. **流式处理**：逐文件解析，增量写入CSV
2. **分块读取**：pandas的`chunksize`参数
```python
for chunk in pd.read_csv('raw_data.csv', chunksize=1000):
    process(chunk)
```
3. **数据库存储**：使用SQLite或DuckDB
4. **分布式处理**：Dask或Spark（如需要）

---

## 5. 测试结果总结

### 5.1 成功验证的功能

✅ **JSON解析**：
- 成功解析1.5MB的单行JSON文件
- 提取所有关键配置和性能指标
- 处理速度：约278个文件 < 10秒

✅ **CSV转换**：
- 276/278文件成功转换（2个损坏）
- 生成36列的汇总CSV
- 文件大小：~50KB（紧凑格式）

✅ **标准库分析**：
- 使用statistics模块计算mean/median/stdev
- 按并行度、batch size、input length分组
- IQR异常值检测

### 5.2 发现的数据质量问题

⚠️ **损坏的JSON文件**（2个）：
```
result/benchmarks/Qwen3-Coder-480B-FP8-TP16-EP16-DP1/bs1_in4096_out512/...
result/benchmarks/Qwen3-Coder-480B-FP8-TP16-EP16-DP1/bs256_in2048_out512/...
```
→ 建议重新运行这些配置

⚠️ **缺失的配置**（2个）：
```
TP16-EP8, BS=128, INPUT=131072
TP16-EP8, BS=256, INPUT=131072
```
→ 目录存在但JSON文件缺失

⚠️ **异常值**（25个，9%）：
- 主要是高throughput配置（BS=128/256, INPUT=1024/2048）
- 可能是真实的高性能点，不是错误
- 建议保留但标记

### 5.3 初步性能洞察

📊 **并行度对比**（平均throughput）：
1. TP8-EP2: **1549 tokens/s** ⭐ 最佳
2. TP16-EP8: 1526 tokens/s
3. TP8-EP4: 1465 tokens/s
4. TP8-EP8: 1338 tokens/s
5. TP16-EP16: 1156 tokens/s

**观察**：EP规模增大，性能反而下降，可能是通信开销

📊 **负载扩展性**：
- Batch size扩展性良好：BS=256是BS=1的13倍throughput
- Input length敏感：131K输入比1K输入慢30倍

---

## 6. 最佳实践建议

### 6.1 开发流程

1. **初始阶段**：使用标准库快速验证
2. **迭代阶段**：安装pandas/matplotlib，完善分析
3. **发布阶段**：生成报告和图表

### 6.2 代码组织

```
benchmark/
├── scripts/               # 数据处理脚本
│   ├── parse_json.py     # 解析JSON
│   ├── clean_data.py     # 数据清洗
│   ├── analyze_data.py   # 数据分析
│   └── plot_results.py   # 可视化
├── lib/                   # 共享库
│   ├── data_loader.py    # 数据加载器
│   ├── metrics.py        # 指标计算
│   └── plotting.py       # 绘图工具
├── config/                # 配置文件
│   └── analysis_config.yaml
├── tmp/                   # 临时文件
│   ├── raw_data.csv
│   └── cleaned_data.csv
├── result/                # 结果输出
│   ├── analysis/         # 分析结果CSV
│   └── figures/          # 图表PNG/PDF
├── test/                  # 测试脚本
└── doc/                   # 文档
```

### 6.3 脚本规范

- **参数化**：使用argparse，避免硬编码路径
- **日志记录**：使用logging模块，输出详细日志
- **错误处理**：捕获异常，继续处理其他文件
- **进度显示**：使用tqdm显示进度条（可选）
- **可重现性**：记录时间戳、版本号、配置参数

### 6.4 数据管理

- **版本控制**：使用Git跟踪脚本，不跟踪数据文件
- **备份**：定期备份`result/benchmarks/`原始数据
- **清理**：定期清理`tmp/`临时文件
- **归档**：完成实验后，压缩存档完整结果

---

## 7. 下一步行动计划

### 7.1 环境准备

```bash
# 1. 安装必要的Python库
pip3 install --user pandas numpy matplotlib seaborn

# 2. 验证安装
python3 -c "import pandas, numpy, matplotlib; print('OK')"
```

### 7.2 开发优先级

**高优先级**（核心功能）：
1. ✅ JSON解析器（已完成测试）
2. ✅ CSV转换器（已完成测试）
3. ⏳ 数据清洗脚本（待开发）
4. ⏳ 基础分析脚本（待开发）

**中优先级**（增强功能）：
5. ⏳ 可视化脚本（待开发）
6. ⏳ 报告生成器（待开发）

**低优先级**（高级功能）：
7. ⏳ 交互式Dashboard
8. ⏳ 自动化CI集成

### 7.3 测试计划

- [ ] 测试所有并行度配置的分析
- [ ] 测试损坏文件的错误处理
- [ ] 测试大规模数据（1000+ 文件）
- [ ] 测试生成的图表质量
- [ ] 用户验收测试

---

## 8. 参考资源

### 8.1 文档

- Pandas官方文档: https://pandas.pydata.org/docs/
- Matplotlib官方文档: https://matplotlib.org/stable/contents.html
- Seaborn官方文档: https://seaborn.pydata.org/
- Python标准库statistics: https://docs.python.org/3/library/statistics.html

### 8.2 教程

- Pandas 10 minutes: https://pandas.pydata.org/docs/user_guide/10min.html
- Matplotlib tutorials: https://matplotlib.org/stable/tutorials/index.html
- 数据可视化最佳实践: https://clauswilke.com/dataviz/

### 8.3 示例代码

所有测试脚本位于：
- `test/parse_json_test.py` - JSON解析示例
- `test/scan_all_benchmarks.py` - 批量扫描示例
- `test/json_to_csv_test.py` - CSV转换示例
- `test/stdlib_analysis_test.py` - 标准库分析示例

---

**文档版本**: v1.0
**创建日期**: 2025-12-22
**作者**: Claude Code
**状态**: 调研完成，待开发实现
