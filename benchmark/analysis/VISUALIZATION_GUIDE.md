# Benchmark 可视化指南

## 快速开始

### 1. 启动 Jupyter Notebook

```bash
cd analysis
jupyter notebook benchmark_visualization.ipynb
```

或者使用 JupyterLab：

```bash
cd analysis
jupyter lab benchmark_visualization.ipynb
```

### 2. 运行所有 Cells

在 Jupyter 界面中：
- 点击 **Cell** → **Run All** 运行所有分析
- 或者按 **Shift + Enter** 逐个运行 cell

### 3. 查看生成的图表

所有图表会保存到 `analysis/figures/` 目录：

```bash
ls -lh figures/*.png
```

## Notebook 内容结构

### 第 1 节：导入库和加载数据
- 加载 `benchmark_summary.csv` 和 `benchmark_requests.csv`
- 创建配置标签（TP-EP 组合）
- 显示数据概览

### 第 2 节：数据概览
- 性能指标基本统计
- 数据样例展示

### 第 3 节：吞吐量分析
**3.1 不同并行配置的吞吐量对比**
- 柱状图：各配置的平均吞吐量
- 箱线图：吞吐量分布

**3.2 批量大小对吞吐量的影响**
- 线图：不同配置的批量大小扩展性
- 聚合趋势图：整体批量大小影响

**3.3 输入长度对吞吐量的影响**
- 按配置的输入长度趋势
- 整体输入长度影响

**3.4 热力图：批量大小 × 输入长度**
- 每个配置的二维性能热力图

### 第 4 节：TTFT（首 Token 延迟）分析
- 不同配置的 TTFT 对比
- 批量大小对 TTFT 的影响
- 输入长度对 TTFT 的影响
- TTFT vs 吞吐量散点图

### 第 5 节：TPOT（每 Token 时间）分析
- 不同配置的 TPOT 对比
- 批量大小和输入长度的影响
- TPOT vs 吞吐量关系（反比）

### 第 6 节：端到端延迟分析
- E2E 延迟分布箱线图
- 批量大小和输入长度的影响
- P99 延迟对比

### 第 7 节：综合性能对比
**7.1 雷达图：多维性能对比**
- 每个配置的归一化性能雷达图
- 指标：吞吐量、TTFT、TPOT、E2E 延迟

**7.2 综合评分排名**
- 基于归一化指标的综合评分
- 性能排名柱状图

### 第 8 节：最佳配置推荐
根据不同场景推荐最佳配置：
1. 最高吞吐量
2. 最低延迟
3. 最低 TTFT（最快响应）
4. 平衡配置（中等批量）
5. 长文本处理

### 第 9 节：请求级数据分析
基于 `benchmark_requests.csv` 的详细分析：
- TTFT 分布直方图
- TTFT 累积分布函数（CDF）
- 请求序列 TTFT 趋势
- 详细统计信息

### 第 10 节：导出所有图表
列出所有生成的图表文件

## 生成的图表文件

运行完整个 notebook 后，会生成以下图表（保存在 `figures/` 目录）：

1. **throughput_by_config.png** - 各配置吞吐量对比
2. **throughput_by_batch_size.png** - 批量大小影响
3. **throughput_by_input_len.png** - 输入长度影响
4. **throughput_heatmap.png** - 二维热力图（每个配置）
5. **ttft_analysis.png** - TTFT 综合分析
6. **tpot_analysis.png** - TPOT 综合分析
7. **e2e_latency_analysis.png** - E2E 延迟分析
8. **performance_radar.png** - 性能雷达图（所有配置）
9. **composite_score.png** - 综合评分排名
10. **request_level_analysis.png** - 请求级详细分析

## 自定义分析

### 修改样本配置

在第 9 节中，可以修改以下变量来分析不同配置：

```python
sample_config = 'TP16-EP16'  # 修改为你想分析的配置
sample_bs = 128              # 修改批量大小
sample_input = 8192          # 修改输入长度
```

### 添加新的图表

参考现有代码模式，可以轻松添加新的分析图表：

```python
# 示例：自定义分析
fig, ax = plt.subplots(figsize=(12, 6))

# 你的分析代码
# ...

plt.savefig('figures/my_custom_chart.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 性能优化建议

### 如果数据量很大

1. **只运行部分 cells**: 不需要运行所有分析，可以选择性运行
2. **降低图表分辨率**: 修改 `dpi=300` 为 `dpi=150`
3. **减少热力图数量**: 在第 3.4 节中限制配置数量

### 如果内存不足

```python
# 在加载数据后，只保留需要的列
df_summary = df_summary[['config_label', 'batch_size', 'input_len',
                         'output_throughput', 'mean_ttft_ms', 'mean_tpot_ms']]
```

## 导出报告

### 导出为 HTML

```bash
jupyter nbconvert --to html benchmark_visualization.ipynb
```

生成的 `benchmark_visualization.html` 包含所有分析结果和图表。

### 导出为 PDF（需要 LaTeX）

```bash
jupyter nbconvert --to pdf benchmark_visualization.ipynb
```

### 导出为 Python 脚本

```bash
jupyter nbconvert --to python benchmark_visualization.ipynb
```

生成 `benchmark_visualization.py`，可以直接运行：

```bash
python benchmark_visualization.py
```

## 常见问题

### Q1: 导入错误 "No module named 'xxx'"

**解决方案**：安装缺失的包

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Q2: 图表不显示

**解决方案**：确保使用了正确的后端

```python
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg', 'nbAgg'
```

在 Jupyter Notebook 中，添加 magic command：

```python
%matplotlib inline
```

### Q3: 字体显示问题（中文乱码）

**解决方案**：配置中文字体

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
```

### Q4: 内存错误

**解决方案**：分批加载数据或增加系统内存

```python
# 分批读取
df_summary = pd.read_csv('data/benchmark_summary.csv', nrows=100)
```

## 进阶使用

### 交互式图表（Plotly）

可以将 matplotlib 图表替换为 Plotly 交互式图表：

```python
import plotly.express as px

fig = px.bar(df_summary.groupby('config_label')['output_throughput'].mean().reset_index(),
             x='config_label', y='output_throughput',
             title='Throughput by Configuration')
fig.show()
```

### 动画图表

展示性能随时间/配置的变化：

```python
import matplotlib.animation as animation

# 创建动画
# ...
```

### 自动化报告生成

结合 `papermill` 自动执行 notebook 并生成报告：

```bash
pip install papermill
papermill benchmark_visualization.ipynb output_report.ipynb
```

## 最佳实践

1. **重启 Kernel 后重新运行**: 确保结果可重现
2. **保存中间结果**: 处理大数据时保存处理后的数据
3. **添加注释**: 在自定义分析中添加 markdown 说明
4. **版本控制**: 使用 git 跟踪 notebook 变更
5. **清理输出**: 提交前清除所有输出（Cell → All Output → Clear）

## 相关文件

- `export_to_csv.py` - CSV 数据导出脚本
- `README.md` - 完整使用文档
- `data/benchmark_summary.csv` - 汇总数据
- `data/benchmark_requests.csv` - 请求级数据

## 技术支持

如有问题，请检查：
1. Python 版本：≥ 3.7
2. Jupyter 版本：`jupyter --version`
3. 依赖包版本：`pip list | grep -E "pandas|matplotlib|seaborn"`

祝分析愉快！
