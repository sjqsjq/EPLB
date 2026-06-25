# Benchmark数据分析 - 快速开始指南

## 概览

本项目提供了一套完整的工具来处理和分析SGLang benchmark结果数据。

**数据规模**：
- 278个JSON文件（~1-2MB每个）
- 5种并行度配置
- 7种batch sizes × 8种input lengths = 56种负载配置

---

## 环境准备

### 必需的Python库

```bash
# 推荐：安装pandas/numpy/matplotlib（功能最完整）
pip3 install --user pandas numpy matplotlib seaborn

# 验证安装
python3 -c "import pandas, numpy, matplotlib; print('OK')"
```

### 可选工具

```bash
# 交互式可视化
pip3 install --user plotly

# Jupyter Notebook（用于探索式分析）
pip3 install --user jupyter
```

---

## 快速测试

所有测试脚本都已准备好，位于`test/`目录。

### 1. 测试JSON解析

```bash
python3 test/parse_json_test.py
```

**输出**：
- 解析单个JSON文件
- 显示配置信息、并行度、性能指标

### 2. 扫描所有benchmark文件

```bash
python3 test/scan_all_benchmarks.py
```

**输出**：
- 统计所有并行度配置和负载配置
- 检测缺失的数据文件

### 3. 转换JSON到CSV

```bash
python3 test/json_to_csv_test.py
```

**输出**：
- 生成`tmp/benchmark_summary.csv`
- 显示前5行预览

### 4. 数据分析（使用标准库）

```bash
python3 test/stdlib_analysis_test.py
```

**输出**：
- 按并行度分组统计
- 按batch size分组统计
- 按input length分组统计
- 固定workload对比不同并行度
- 异常值检测

---

## 查看测试结果

### 生成的CSV文件

```bash
# 查看CSV文件前10行
head -10 tmp/benchmark_summary.csv

# 统计行数和列数
wc -l tmp/benchmark_summary.csv
head -1 tmp/benchmark_summary.csv | tr ',' '\n' | wc -l
```

### 查看统计摘要

```bash
python3 test/stdlib_analysis_test.py > tmp/analysis_summary.txt
cat tmp/analysis_summary.txt
```

---

## 数据结构说明

### CSV字段（36列）

**配置信息**：
- `tp_size`, `ep_size`, `dp_size`, `nnodes`
- `batch_size`, `input_len`, `output_len`

**吞吐量指标**：
- `request_throughput`, `input_throughput`, `output_throughput`, `total_throughput`

**延迟指标**：
- `mean_e2e_latency_ms`, `median_e2e_latency_ms`, `p99_e2e_latency_ms`
- `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms`
- `mean_tpot_ms`, `median_tpot_ms`, `p99_tpot_ms`
- `mean_itl_ms`, `median_itl_ms`, `p99_itl_ms`

**其他**：
- `duration`, `completed`
- `total_input_tokens`, `total_output_tokens`
- `concurrency`, `max_output_tokens_per_s`

---

## 常见问题

### Q1: 为什么有些JSON文件解析失败？

**A**: 有2个文件可能损坏：
```
bs1_in4096_out512/inference_bs1_input4096_output512.json
bs256_in2048_out512/inference_bs256_input2048_output512.json
```
建议重新运行这些配置的benchmark。

### Q2: 为什么CSV只有276行，不是280行？

**A**:
- 2个JSON文件解析失败（损坏）
- 2个配置的JSON文件缺失
- 实际可用数据：276个配置

### Q3: 异常值是否应该删除？

**A**: 不建议删除。测试发现25个异常值（9%），主要是：
- 高吞吐量配置（BS=128/256, INPUT=1024/2048）
- 可能是真实的高性能点
- 建议保留但标记为异常值

### Q4: 如何添加新的分析？

**A**: 参考`test/stdlib_analysis_test.py`，核心步骤：
1. 加载CSV：`data = load_csv()`
2. 转换数值：`data = convert_numeric(data)`
3. 分组/过滤：`filtered = [row for row in data if condition]`
4. 计算统计：`statistics.mean(values)`

---

## 下一步计划

### 已完成（调研阶段）

✅ 数据结构分析
✅ JSON解析器测试
✅ CSV转换器测试
✅ 基础统计分析测试
✅ 异常值检测测试

### 待开发（实现阶段）

⏳ 数据清洗脚本
⏳ 高级分析脚本
⏳ 可视化脚本
⏳ 自动化报告生成

---

## 文件组织

```
benchmark/
├── test/                              # 测试脚本（已完成）
│   ├── parse_json_test.py            # ✅ JSON解析测试
│   ├── scan_all_benchmarks.py        # ✅ 文件扫描测试
│   ├── json_to_csv_test.py           # ✅ CSV转换测试
│   └── stdlib_analysis_test.py       # ✅ 数据分析测试
├── tmp/                               # 临时文件
│   ├── benchmark_summary.csv         # 生成的汇总CSV
│   └── analysis_summary.txt          # 分析结果文本
├── doc/                               # 文档
│   ├── DATA_ANALYSIS_RESEARCH.md     # 详细调研报告
│   └── QUICK_START.md                # 本文件
└── result/                            # 原始数据
    └── benchmarks/                    # 278个JSON文件
```

---

## 联系和反馈

如有问题或建议，请查看：
- 详细调研报告：`doc/DATA_ANALYSIS_RESEARCH.md`
- 项目说明：`CLAUDE.md`

**文档版本**: v1.0
**创建日期**: 2025-12-22
