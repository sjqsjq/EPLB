# Benchmark 数据分析工具

此目录包含用于导出和分析 SGLang benchmark 结果的脚本和数据。

## 并行配置说明

### TP、EP、DP 参数

- **TP (Tensor Parallelism)**: 张量并行度
- **EP (Expert Parallelism)**: 专家并行度（MoE 模型）
- **DP (Data Parallelism)**: 数据并行度

### GPU 总数计算

```
GPU 总数 = max(TP, EP, DP)
```

### Attention TP 和 MoE TP 计算

不同的 DP 值会影响实际的并行策略：

- **Attention TP** = TP / DP
- **MoE TP** = TP / EP

**重要区别示例：**

虽然以下两个配置的 GPU 总数都是 16，但它们的并行策略完全不同：

1. **TP16-EP16-DP1**:
   - GPU 总数: max(16, 16, 1) = 16
   - Attention TP: 16/1 = 16
   - MoE TP: 16/16 = 1

2. **TP16-EP16-DP16**:
   - GPU 总数: max(16, 16, 16) = 16
   - Attention TP: 16/16 = 1
   - MoE TP: 16/16 = 1

因此，**TP16-EP16-DP1** 和 **TP16-EP16-DP16** 是两个不同的配置，在性能表现上会有显著差异。

## 目录结构

```
analysis/
├── export_to_csv.py           # 主导出脚本
├── data/                      # 输出数据目录
│   ├── benchmark_summary.csv  # 汇总统计数据
│   ├── benchmark_requests.csv # 请求级详细数据
│   ├── .processed_files.txt   # 去重追踪文件
│   └── export.log             # 导出日志
└── README.md                  # 本文档
```

## 快速开始

### 首次导出

```bash
# 导出所有 benchmark 数据
python analysis/export_to_csv.py
```

这将扫描 `result/benchmarks` 目录下所有 JSON 文件，并生成两个 CSV 文件。

### 增量更新

当你运行新的 benchmark 测试后，再次运行脚本即可：

```bash
# 只处理新文件，不重复处理
python analysis/export_to_csv.py
```

脚本会自动识别并只处理新增的 JSON 文件。

### 强制重建

如果需要重新生成所有数据：

```bash
# 忽略追踪文件，重新处理所有 JSON
python analysis/export_to_csv.py --force
```

## 命令行选项

```bash
python analysis/export_to_csv.py [选项]

选项:
  -h, --help              显示帮助信息
  -i, --input DIR         指定 JSON 输入目录 (默认: result/benchmarks)
  -o, --output DIR        指定 CSV 输出目录 (默认: analysis/data)
  -f, --force             强制完全重建，忽略追踪文件
  --summary-only          只导出汇总 CSV
  --requests-only         只导出请求级 CSV
  -b, --batch-size N      批量处理大小 (默认: 50)
  -v, --verbose           显示详细日志
```

### 使用示例

```bash
# 只导出汇总数据
python analysis/export_to_csv.py --summary-only

# 只导出请求级数据
python analysis/export_to_csv.py --requests-only

# 自定义输入输出路径
python analysis/export_to_csv.py \
  --input /path/to/benchmarks \
  --output /path/to/output

# 详细日志模式
python analysis/export_to_csv.py --verbose
```

## CSV 文件说明

### benchmark_summary.csv

每行代表一个 benchmark 配置的汇总统计数据。

**列数**: 37 列
**行数**: ~390 行（取决于实际 benchmark 数量）
**文件大小**: ~130 KB

**主要列说明**:

| 列名 | 说明 |
|-----|------|
| `tp_size` | Tensor Parallel 大小 |
| `ep_size` | Expert Parallel 大小 |
| `dp_size` | Data Parallel 大小 |
| `nnodes` | 节点数量 |
| `batch_size` | 批量大小 |
| `input_len` | 输入长度（tokens）|
| `output_len` | 输出长度（tokens）|
| `duration` | 测试持续时间（秒）|
| `completed` | 完成的请求数 |
| `request_throughput` | 请求吞吐量（req/s）|
| `input_throughput` | 输入吞吐量（tokens/s）|
| `output_throughput` | 输出吞吐量（tokens/s）|
| `mean_ttft_ms` | 平均首 token 时间（毫秒）|
| `median_ttft_ms` | 中位数首 token 时间 |
| `p99_ttft_ms` | P99 首 token 时间 |
| `mean_tpot_ms` | 平均每 token 时间（毫秒）|
| `mean_e2e_latency_ms` | 平均端到端延迟（毫秒）|
| `p99_e2e_latency_ms` | P99 端到端延迟 |
| `concurrency` | 平均并发度 |

**完整列列表**（37 列）:
- 配置: `tp_size`, `ep_size`, `dp_size`, `nnodes`, `batch_size`, `input_len`, `output_len`
- 时长: `duration`, `completed`, `concurrency`
- Token 统计: `total_input_tokens`, `total_output_tokens`, `total_input_text_tokens`, `total_output_tokens_retokenized`
- 吞吐量: `request_throughput`, `input_throughput`, `output_throughput`, `total_throughput`
- E2E 延迟: `mean_e2e_latency_ms`, `median_e2e_latency_ms`, `std_e2e_latency_ms`, `p99_e2e_latency_ms`
- TTFT: `mean_ttft_ms`, `median_ttft_ms`, `std_ttft_ms`, `p99_ttft_ms`
- TPOT: `mean_tpot_ms`, `median_tpot_ms`, `std_tpot_ms`, `p99_tpot_ms`
- ITL: `mean_itl_ms`, `median_itl_ms`, `std_itl_ms`, `p95_itl_ms`, `p99_itl_ms`
- 其他: `max_output_tokens_per_s`, `max_concurrent_requests`

### benchmark_requests.csv

每行代表一个请求的详细数据。

**列数**: 11 列
**行数**: ~51,000 行（390 个配置 × 平均 131 个请求）
**文件大小**: ~1-3 MB

**列说明**:

| 列名 | 说明 |
|-----|------|
| `tp_size` | Tensor Parallel 大小 |
| `ep_size` | Expert Parallel 大小 |
| `dp_size` | Data Parallel 大小 |
| `nnodes` | 节点数量 |
| `batch_size` | 批量大小 |
| `input_len` | 配置的输入长度 |
| `output_len` | 配置的输出长度 |
| `request_id` | 请求 ID（从 0 开始）|
| `actual_input_len` | 实际输入长度 |
| `actual_output_len` | 实际输出长度 |
| `ttft_ms` | 首 token 时间（毫秒）|

**注意**:
- 不包含 `generated_texts`（生成文本）以控制文件大小
- 不包含 `itls`（token 间延迟数组）详细数据，汇总统计已在 summary CSV 中

## 数据分析示例

### 使用 Pandas

```python
import pandas as pd

# 加载数据
df_summary = pd.read_csv('analysis/data/benchmark_summary.csv')
df_requests = pd.read_csv('analysis/data/benchmark_requests.csv')

# 示例 1: 查看 TP16-EP16 配置的性能
df_tp16_ep16 = df_summary[
    (df_summary['tp_size'] == 16) &
    (df_summary['ep_size'] == 16)
]
print(df_tp16_ep16[['batch_size', 'input_len', 'output_throughput']])

# 示例 2: 批量大小对吞吐量的影响
batch_analysis = df_summary.groupby('batch_size')['output_throughput'].agg([
    'mean', 'std', 'min', 'max', 'count'
])
print(batch_analysis)

# 示例 3: 输入长度对延迟的影响
input_len_analysis = df_summary.groupby('input_len')['mean_ttft_ms'].mean()
print(input_len_analysis)

# 示例 4: 请求级 TTFT 分布
import matplotlib.pyplot as plt
df_subset = df_requests[
    (df_requests['tp_size'] == 16) &
    (df_requests['batch_size'] == 128)
]
plt.hist(df_subset['ttft_ms'], bins=50)
plt.xlabel('TTFT (ms)')
plt.ylabel('Frequency')
plt.title('TTFT Distribution (TP16, BS128)')
plt.show()

# 示例 5: 找出最佳配置
best_config = df_summary.nlargest(10, 'output_throughput')[
    ['tp_size', 'ep_size', 'batch_size', 'input_len', 'output_throughput']
]
print("Top 10 configurations by throughput:")
print(best_config)
```

### 使用标准库（无需 Pandas）

```python
import csv

# 读取汇总数据
with open('analysis/data/benchmark_summary.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# 过滤特定配置
tp16_data = [row for row in data if row['tp_size'] == '16']

# 计算平均值
throughputs = [float(row['output_throughput']) for row in tp16_data if row['output_throughput']]
avg_throughput = sum(throughputs) / len(throughputs)
print(f"Average throughput for TP16: {avg_throughput:.2f} tokens/s")

# 按批量大小分组
from collections import defaultdict
batch_groups = defaultdict(list)
for row in data:
    batch_size = row['batch_size']
    if row['output_throughput']:
        batch_groups[batch_size].append(float(row['output_throughput']))

for bs in sorted(batch_groups.keys(), key=int):
    values = batch_groups[bs]
    print(f"Batch size {bs}: avg={sum(values)/len(values):.2f}, count={len(values)}")
```

## 去重机制

脚本使用 `.processed_files.txt` 追踪已处理的文件，避免重复处理。

### 追踪文件格式

```
/cpfs01/user/nebula_model/sjq-workspace/benchmark/result/benchmarks/Qwen3-Coder-480B-FP8-TP16-EP16-DP1/bs128_in1024_out512/inference_bs128_input1024_output512.json
/cpfs01/user/nebula_model/sjq-workspace/benchmark/result/benchmarks/Qwen3-Coder-480B-FP8-TP16-EP16-DP1/bs128_in8192_out512/inference_bs128_input8192_output512.json
...
```

### 手动管理追踪文件

```bash
# 查看已处理文件数量
wc -l analysis/data/.processed_files.txt

# 清空追踪文件（下次运行将重新处理所有文件）
rm analysis/data/.processed_files.txt

# 查看最近处理的文件
tail -20 analysis/data/.processed_files.txt
```

## 错误处理

脚本会自动处理以下错误情况：

1. **JSON 解析错误**: 跳过文件并记录到日志
2. **缺失字段**: 使用 `None` 填充
3. **数组长度不匹配**: 使用最短数组长度
4. **文件 I/O 错误**: 记录错误并继续处理其他文件

所有错误和警告都记录在 `analysis/data/export.log` 文件中。

### 查看日志

```bash
# 查看完整日志
cat analysis/data/export.log

# 查看错误信息
grep ERROR analysis/data/export.log

# 查看警告信息
grep WARNING analysis/data/export.log

# 实时监控导出进度
tail -f analysis/data/export.log
```

## 性能说明

- **首次完整导出**: ~2-3 秒（390 个 JSON 文件）
- **增量更新**: < 1 秒（典型场景）
- **内存使用**: 批量处理机制确保内存占用低（< 500 MB）

## 常见问题

### Q: 如何处理已修改的 JSON 文件？

A: 使用 `--force` 选项重新生成全部数据：
```bash
python analysis/export_to_csv.py --force
```

### Q: 可以只更新某个配置的数据吗？

A: 可以删除追踪文件中对应配置的行，然后重新运行脚本：
```bash
# 删除 TP16-EP16 配置的追踪记录
grep -v "TP16-EP16" analysis/data/.processed_files.txt > tmp.txt
mv tmp.txt analysis/data/.processed_files.txt

# 重新运行脚本
python analysis/export_to_csv.py
```

### Q: CSV 文件太大怎么办？

A:
1. 使用 `--summary-only` 只导出汇总数据（~130KB）
2. 使用 Pandas 的 `chunksize` 参数分块读取
3. 考虑使用 Parquet 格式（需修改脚本）

### Q: 如何查看特定配置的数据？

A: 使用命令行工具快速过滤：
```bash
# 查看 TP16-EP16 的所有配置
grep "^16,16," analysis/data/benchmark_summary.csv

# 查看批量大小为 128 的配置
awk -F',' '$5==128' analysis/data/benchmark_summary.csv
```

## 与现有脚本的关系

- `test/json_to_csv_test.py`: 原型脚本，保留作为参考
- `analysis/export_to_csv.py`: 生产级脚本，功能更完整

建议使用 `analysis/export_to_csv.py` 进行日常数据导出。

## 未来扩展

如需包含更详细的数据（如 ITL 数组、生成文本），可以：

1. 修改 `REQUESTS_COLUMNS` 添加额外字段
2. 在 `parse_json_to_request_rows()` 中提取相应数据
3. 注意文件大小可能显著增加

## 许可与贡献

此脚本为内部工具，如有改进建议请联系维护者。
