# Benchmark 自动化工具实现指导文档

> **目标读者**: Claude Code Agent
> **文档目的**: 提供清晰的实现指导，用于构建自动化 LLM benchmark 工具

---

## 📋 项目概述

### 功能目标
构建一个自动化 benchmark 工具，系统地测试不同 batch size 和 input length 组合下的 LLM 推理性能。

### 核心组件
```
benchmark/
├── config.yaml              # 配置文件（用户编辑）
├── run_benchmark.sh         # SSH 入口脚本（已有 SSH 框架）
├── benchmark.py             # 核心测试逻辑（需要实现）
└── results/                 # 结果目录
    ├── model-TP-EP-DP/
    │   ├── inference_bs1_input1k_output1k.json
    │   ├── inference_bs4_input1k_output1k.json
    │   └── ...
    └── traces/
        └── model-TP-EP-DP/
            ├── trace_bs1_input1k.json.gz
            └── ...
```

---

## 🎯 实现计划

### Phase 1: 配置系统 (config.yaml)
设计配置文件结构，包含所有可配置参数。

### Phase 2: 核心测试逻辑 (benchmark.py)
实现多轮测试循环、队列监控、结果保存等核心功能。

### Phase 3: 入口脚本 (run_benchmark.sh)
将参数传递给 benchmark.py（SSH 框架已存在）。

### Phase 4: 测试和验证
验证所有功能正常工作。

---

## 📝 详细实现指导

## 一、配置文件设计 (config.yaml)

### 1.1 配置文件结构

```yaml
# ============================================
# Benchmark 配置文件
# ============================================

# 服务器配置
server:
  base_url: "http://127.0.0.1:34567"
  backend: "sglang"                    # sglang, vllm, lmdeploy 等
  model_path: "/cpfs01/user/nebula_model/llm_weight/Qwen3-Coder-480B-A35B-Instruct-FP8"

# 测试参数
benchmark:
  dataset: "random-ids"                # 数据集类型
  output_len: 1024                     # 固定输出长度
  warmup_requests: 3                   # 预热请求数

  # 测试矩阵
  batch_sizes: [1, 4, 16, 32, 64, 128, 256]
  input_lengths: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]  # 1k, 2k, 4k, ..., 128k

  # 队列限制
  max_queue_requests: 15               # 队列请求数阈值

# 结果路径
output:
  result_base_path: "./results"        # 结果根目录
  save_traces: true                    # 是否保存 trace 文件

# Profiler 配置
profiler:
  enable: false                        # 是否启用 profiler
  activities: ["CPU", "GPU"]           # profiler 活动类型

# PD 分离模式配置（可选）
pd_separated:
  enable: false                        # 是否启用 PD 分离模式
  prefill_url: "http://127.0.0.1:30010"
  decode_url: "http://127.0.0.1:30020"

# 其他选项
options:
  print_requests: true                 # 默认启用请求打印
  output_details: true                 # 保存详细结果
```

### 1.2 配置读取示例代码

```python
import yaml
from pathlib import Path
from typing import Dict, List, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 验证必需字段
    required_fields = [
        "server.base_url",
        "server.backend",
        "benchmark.batch_sizes",
        "benchmark.input_lengths",
    ]

    for field in required_fields:
        keys = field.split('.')
        value = config
        for key in keys:
            if key not in value:
                raise ValueError(f"Missing required config field: {field}")
            value = value[key]

    return config
```

---

## 二、核心测试逻辑 (benchmark.py)

### 2.1 整体架构

```python
#!/usr/bin/env python3
"""
Benchmark 自动化测试工具

功能：
1. 遍历 batch_sizes 和 input_lengths 组合
2. 动态监控服务器队列状态
3. 智能跳过会导致队列堆积的配置
4. 保存结果和 trace 文件
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import requests

class BenchmarkRunner:
    """Benchmark 运行器"""

    def __init__(self, config_path: str):
        """初始化

        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.max_bs_limits = {}  # 记录每个 input 的 bs 上限

    def load_config(self, config_path: str) -> Dict:
        """加载配置"""
        # 实现见 1.2 节
        pass

    def get_server_info(self) -> Dict:
        """获取服务器信息，包括 TP/EP/DP 和队列状态"""
        pass

    def run_single_test(self, batch_size: int, input_len: int) -> Dict:
        """运行单次测试"""
        pass

    def should_skip_test(self, input_len: int, batch_size: int) -> bool:
        """判断是否应该跳过此测试"""
        pass

    def save_results(self, results: Dict, batch_size: int, input_len: int):
        """保存测试结果"""
        pass

    def run_benchmark(self):
        """运行完整的 benchmark 测试"""
        pass

def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark 自动化工具")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    # 其他参数可以覆盖 config 中的值
    parser.add_argument("--backend", help="后端类型")
    parser.add_argument("--base-url", help="服务器 URL")
    parser.add_argument("--enable-profiler", type=bool, help="启用 profiler")
    # ... 其他可选参数

    args = parser.parse_args()

    runner = BenchmarkRunner(args.config)
    # 用命令行参数覆盖配置
    if args.backend:
        runner.config['server']['backend'] = args.backend
    # ... 其他覆盖

    runner.run_benchmark()

if __name__ == "__main__":
    main()
```

### 2.2 关键函数实现

#### 2.2.1 获取服务器信息

```python
def get_server_info(self) -> Dict:
    """获取服务器信息

    返回格式：
    {
        "model_name": "Qwen3-480B-...",
        "tp": 8,
        "ep": 4,
        "dp": 1,
        "waiting_reqs": 5,  # 当前队列中的请求数
        "running_reqs": 2,  # 正在处理的请求数
        ...
    }
    """
    base_url = self.config['server']['base_url']

    try:
        response = requests.get(f"{base_url}/get_server_info", timeout=10)
        response.raise_for_status()
        server_info = response.json()

        # 提取 TP/EP/DP 信息
        # 注意：具体字段名可能需要根据实际 API 响应调整
        result = {
            "model_name": server_info.get("model_name", "unknown"),
            "tp": server_info.get("tp_size", 1),
            "ep": server_info.get("ep_size", 1),
            "dp": server_info.get("dp_size", 1),
        }

        # 提取队列信息
        # 对于 PD 分离模式，可能有 decode/prefill 两个部分
        if "decode" in server_info:
            # PD 分离模式
            decode_info = server_info["decode"][0] if isinstance(server_info["decode"], list) else server_info["decode"]
            result["waiting_reqs"] = decode_info.get("#waiting_reqs", 0)
            result["running_reqs"] = decode_info.get("#running_reqs", 0)
        else:
            # 普通模式
            result["waiting_reqs"] = server_info.get("#waiting_reqs", 0)
            result["running_reqs"] = server_info.get("#running_reqs", 0)

        return result

    except Exception as e:
        print(f"警告: 无法获取服务器信息: {e}")
        return {
            "model_name": "unknown",
            "tp": 1, "ep": 1, "dp": 1,
            "waiting_reqs": 0,
            "running_reqs": 0,
        }
```

**💡 提示词 for Agent:**
> - `/get_server_info` API 的具体响应格式可能因 SGLang 版本而异
> - 需要处理 PD 分离模式下的特殊响应结构（`server_info["decode"]`）
> - `#waiting_reqs` 是队列中等待处理的请求数，这是判断是否跳过测试的关键指标
> - 错误处理很重要，避免因服务器暂时不可用而导致整个测试失败

#### 2.2.2 运行单次测试

```python
def run_single_test(self, batch_size: int, input_len: int) -> Dict:
    """运行单次测试

    Args:
        batch_size: 批次大小
        input_len: 输入长度

    Returns:
        测试结果字典
    """
    config = self.config
    warmup = config['benchmark']['warmup_requests']
    output_len = config['benchmark']['output_len']

    # 实际请求数 = batch_size + warmup
    num_prompts = batch_size + warmup

    # 构建 bench_serving.py 命令
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", config['server']['backend'],
        "--base-url", config['server']['base_url'],
        "--dataset-name", config['benchmark']['dataset'],
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--warmup-requests", str(warmup),
    ]

    # 添加可选参数
    if config.get('options', {}).get('print_requests', False):
        cmd.append("--print-requests")

    if config.get('options', {}).get('output_details', False):
        cmd.append("--output-details")

    # Profiler 配置
    if config.get('profiler', {}).get('enable', False):
        cmd.append("--profile")
        activities = config['profiler'].get('activities', ["CPU", "GPU"])
        cmd.extend(["--profile-activities"] + activities)

    # PD 分离模式
    if config.get('pd_separated', {}).get('enable', False):
        cmd.append("--pd-separated")
        cmd.extend(["--profile-prefill-url", config['pd_separated']['prefill_url']])
        cmd.extend(["--profile-decode-url", config['pd_separated']['decode_url']])

    # 输出文件
    output_file = self._get_output_filename(batch_size, input_len)
    cmd.extend(["--output-file", output_file])

    print(f"\n{'='*60}")
    print(f"运行测试: BS={batch_size}, Input={input_len//1024}k, Output={output_len}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # 执行命令
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )

        print(result.stdout)

        # 读取结果文件（最后一行 JSON）
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    result_data = json.loads(lines[-1])
                    return result_data

        return {}

    except subprocess.CalledProcessError as e:
        print(f"错误: 测试失败")
        print(f"stderr: {e.stderr}")
        return {"error": str(e)}
    except subprocess.TimeoutExpired:
        print(f"错误: 测试超时")
        return {"error": "timeout"}
    except Exception as e:
        print(f"错误: {e}")
        return {"error": str(e)}
```

**💡 提示词 for Agent:**
> - `num_prompts = batch_size + warmup`: warmup 请求会被跳过，所以总请求数要加上 warmup 数
> - 使用 `subprocess.run()` 调用 `bench_serving.py`，设置合理的超时时间
> - 结果文件是 JSONL 格式（每行一个 JSON），需要读取最后一行
> - 错误处理：捕获 subprocess 异常、超时、文件读取错误等
> - 打印详细日志，方便调试和监控进度

#### 2.2.3 判断是否跳过测试

```python
def should_skip_test(self, input_len: int, batch_size: int) -> Tuple[bool, str]:
    """判断是否应该跳过此测试

    逻辑：
    1. 如果之前的 input 已经记录了 bs 上限，且当前 bs 超过该上限，则跳过
    2. input 越大，bs 上限只会相等或更小

    Args:
        input_len: 输入长度
        batch_size: 批次大小

    Returns:
        (是否跳过, 跳过原因)
    """
    # 检查是否已有 bs 上限记录
    if self.max_bs_limits:
        # 获取所有小于等于当前 input_len 的 bs 上限中的最小值
        applicable_limits = [
            bs_limit for inp, bs_limit in self.max_bs_limits.items()
            if inp <= input_len
        ]

        if applicable_limits:
            min_limit = min(applicable_limits)
            if batch_size > min_limit:
                reason = f"BS {batch_size} 超过了基于之前测试的上限 {min_limit}"
                return True, reason

    return False, ""

def check_and_record_queue_status(
    self,
    input_len: int,
    batch_size: int,
    test_result: Dict
) -> bool:
    """检查队列状态并记录 bs 上限

    Args:
        input_len: 输入长度
        batch_size: 批次大小
        test_result: 测试结果

    Returns:
        是否应该停止测试更大的 bs（当前 input）
    """
    max_queue = self.config['benchmark']['max_queue_requests']

    # 获取服务器队列信息
    server_info = self.get_server_info()
    waiting_reqs = server_info.get('waiting_reqs', 0)

    print(f"  队列状态: waiting_reqs={waiting_reqs}, 阈值={max_queue}")

    if waiting_reqs > max_queue:
        # 记录此 input 的 bs 上限
        self.max_bs_limits[input_len] = batch_size

        print(f"  ⚠️  队列超限! 记录 Input={input_len//1024}k 的 BS 上限为 {batch_size}")
        print(f"  跳过更大的 BS，继续测试下一个 Input")

        return True  # 停止测试更大的 bs

    return False  # 继续测试
```

**💡 提示词 for Agent:**
> - **核心逻辑**: Input 越大，GPU 显存压力越大，能支持的 BS 越小
> - `max_bs_limits` 字典记录每个 input_len 的 BS 上限
> - 当队列超过阈值时，记录当前 BS 为上限，跳过后续更大的 BS
> - 对于更大的 input，自动应用之前记录的 BS 上限
> - 这样可以避免不必要的测试，节省时间

#### 2.2.4 保存结果

```python
def _get_output_filename(self, batch_size: int, input_len: int) -> str:
    """生成输出文件名

    格式: {result_path}/model-TP-EP-DP/{mode}_bs{bs}_input{input}_output{output}.json

    Args:
        batch_size: 批次大小
        input_len: 输入长度（token数）

    Returns:
        文件路径
    """
    # 获取服务器信息
    server_info = self.get_server_info()

    # 提取模型名称（去掉路径，只保留最后一部分）
    model_path = self.config['server'].get('model_path', 'unknown')
    model_name = Path(model_path).name

    # 构建模型标识: model-TP{tp}-EP{ep}-DP{dp}
    tp = server_info.get('tp', 1)
    ep = server_info.get('ep', 1)
    dp = server_info.get('dp', 1)
    model_id = f"{model_name}-TP{tp}-EP{ep}-DP{dp}"

    # 确定模式
    if self.config.get('pd_separated', {}).get('enable', False):
        # PD 分离模式 - 需要分别测试 prefill 和 decode
        # 这里暂时使用 "inference" 作为默认
        # 实际应用中，可能需要两次调用，分别指定 mode
        mode = "inference"  # 或 "prefill" / "decode"
    else:
        mode = "inference"

    # 构建文件名
    output_len = self.config['benchmark']['output_len']
    input_str = f"{input_len//1024}k" if input_len >= 1024 else str(input_len)
    output_str = f"{output_len//1024}k" if output_len >= 1024 else str(output_len)

    filename = f"{mode}_bs{batch_size}_input{input_str}_output{output_str}.json"

    # 完整路径
    result_base = self.config['output']['result_base_path']
    result_dir = Path(result_base) / model_id
    result_dir.mkdir(parents=True, exist_ok=True)

    return str(result_dir / filename)

def save_trace_file(self, batch_size: int, input_len: int):
    """保存 profiler trace 文件

    Trace 文件由 bench_serving.py 的 --profile 参数生成，
    存储在服务器的 SGLANG_TORCH_PROFILER_DIR 目录中。

    需要从服务器复制到结果目录。
    """
    if not self.config.get('profiler', {}).get('enable', False):
        return

    if not self.config['output'].get('save_traces', False):
        return

    # 获取模型标识
    server_info = self.get_server_info()
    model_path = self.config['server'].get('model_path', 'unknown')
    model_name = Path(model_path).name
    tp = server_info.get('tp', 1)
    ep = server_info.get('ep', 1)
    dp = server_info.get('dp', 1)
    model_id = f"{model_name}-TP{tp}-EP{ep}-DP{dp}"

    # Trace 目标路径
    result_base = self.config['output']['result_base_path']
    trace_dir = Path(result_base) / "traces" / model_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    input_str = f"{input_len//1024}k"
    trace_filename = f"trace_bs{batch_size}_input{input_str}.json.gz"
    trace_path = trace_dir / trace_filename

    # 从服务器复制 trace 文件
    # 注意：这里需要知道服务器 trace 文件的位置
    # 通常由环境变量 SGLANG_TORCH_PROFILER_DIR 指定
    # 可能需要通过 API 获取，或者使用固定路径

    print(f"  保存 trace 文件到: {trace_path}")

    # TODO: 实现从服务器复制 trace 文件的逻辑
    # 这可能需要：
    # 1. 调用 /stop_profile API
    # 2. 从响应中获取 trace 文件路径
    # 3. 使用 scp 或 API 下载文件
    # 4. 压缩为 .gz 格式
```

**💡 提示词 for Agent:**
> - 文件名格式严格按照: `{mode}_bs{bs}_input{input}_output{output}.json`
> - 从 `/get_server_info` API 获取 TP/EP/DP 参数
> - 目录结构: `result_path/model-TP-EP-DP/` 和 `result_path/traces/model-TP-EP-DP/`
> - Trace 文件需要从服务器获取，可能需要额外的 API 调用或文件传输
> - **暂时不考虑 PD 分离模式的 prefill/decode 区分**，统一使用 "inference"

#### 2.2.5 主测试循环

```python
def run_benchmark(self):
    """运行完整的 benchmark 测试

    测试策略：
    1. 外层循环：遍历 input_lengths（从小到大）
    2. 内层循环：遍历 batch_sizes（从小到大）
    3. 每次测试前检查是否应该跳过
    4. 测试后检查队列状态，决定是否继续测试更大的 bs
    """
    input_lengths = self.config['benchmark']['input_lengths']
    batch_sizes = self.config['benchmark']['batch_sizes']

    print("\n" + "="*70)
    print("开始 Benchmark 测试")
    print("="*70)
    print(f"Input Lengths: {input_lengths}")
    print(f"Batch Sizes: {batch_sizes}")
    print(f"Output Length: {self.config['benchmark']['output_len']}")
    print(f"Max Queue Requests: {self.config['benchmark']['max_queue_requests']}")
    print("="*70 + "\n")

    total_tests = 0
    skipped_tests = 0
    failed_tests = 0

    # 外层循环: Input Length（从小到大）
    for input_len in input_lengths:
        print(f"\n{'#'*70}")
        print(f"# 测试 Input Length: {input_len//1024}k ({input_len} tokens)")
        print(f"{'#'*70}\n")

        # 内层循环: Batch Size（从小到大）
        for batch_size in batch_sizes:
            # 检查是否应该跳过
            should_skip, skip_reason = self.should_skip_test(input_len, batch_size)
            if should_skip:
                print(f"⏭️  跳过: BS={batch_size}, Input={input_len//1024}k")
                print(f"   原因: {skip_reason}\n")
                skipped_tests += 1
                continue

            # 运行测试
            total_tests += 1
            test_result = self.run_single_test(batch_size, input_len)

            # 检查是否失败
            if "error" in test_result:
                print(f"❌ 测试失败: {test_result['error']}\n")
                failed_tests += 1
                # 失败也视为达到上限，记录并跳过后续更大的 bs
                self.max_bs_limits[input_len] = batch_size
                break

            # 保存 trace 文件
            self.save_trace_file(batch_size, input_len)

            # 检查队列状态
            should_stop = self.check_and_record_queue_status(
                input_len, batch_size, test_result
            )

            if should_stop:
                # 队列超限，停止测试更大的 bs
                break

            # 短暂等待，让服务器稳定
            time.sleep(2)

        print(f"\n完成 Input={input_len//1024}k 的所有测试")
        if input_len in self.max_bs_limits:
            print(f"  记录的 BS 上限: {self.max_bs_limits[input_len]}")

    # 打印总结
    print("\n" + "="*70)
    print("Benchmark 测试完成!")
    print("="*70)
    print(f"总测试数: {total_tests}")
    print(f"跳过测试: {skipped_tests}")
    print(f"失败测试: {failed_tests}")
    print(f"成功测试: {total_tests - failed_tests}")
    print("\nBS 上限记录:")
    for input_len, bs_limit in sorted(self.max_bs_limits.items()):
        print(f"  Input {input_len//1024}k: BS <= {bs_limit}")
    print("="*70 + "\n")
```

**💡 提示词 for Agent:**
> - **双层循环**: 外层 input_len，内层 batch_size
> - 每次测试前调用 `should_skip_test()` 检查是否跳过
> - 测试后调用 `check_and_record_queue_status()` 检查队列
> - 如果队列超限或测试失败，**立即跳出内层循环**，继续下一个 input
> - 测试之间加短暂延迟（如 2 秒），让服务器稳定
> - 详细的日志输出，方便监控进度和调试

---

## 三、入口脚本 (run_benchmark.sh)

### 3.1 脚本设计

```bash
#!/bin/bash
#
# Benchmark 入口脚本
#
# 功能：
# 1. 解析命令行参数
# 2. SSH 到容器中执行 benchmark.py
# 3. 将结果从容器复制回本地（可选）
#

set -e  # 遇到错误立即退出

# 默认值
CONFIG_FILE="config.yaml"
ENABLE_PROFILER="false"
BACKEND="sglang"
WARMUP_REQUESTS=3
BASE_URL="http://127.0.0.1:34567"
MODEL_PATH="/cpfs01/user/nebula_model/llm_weight/Qwen3-Coder-480B-A35B-Instruct-FP8"
DATASET="random-ids"
PRINT_REQUESTS="true"
PD_SEPARATED="false"
PROFILE_PREFILL_URL=""
PROFILE_DECODE_URL=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --enable-profiler)
            ENABLE_PROFILER="$2"
            shift 2
            ;;
        --warmup-requests)
            WARMUP_REQUESTS="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --print-requests)
            PRINT_REQUESTS="true"
            shift
            ;;
        --pd-separated)
            PD_SEPARATED="true"
            shift
            ;;
        --profile-prefill-url)
            PROFILE_PREFILL_URL="$2"
            shift 2
            ;;
        --profile-decode-url)
            PROFILE_DECODE_URL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 构建 benchmark.py 命令
BENCHMARK_CMD="python3 benchmark.py --config $CONFIG_FILE"

# 添加可选参数（覆盖 config.yaml）
if [ "$BACKEND" != "" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --backend $BACKEND"
fi

if [ "$ENABLE_PROFILER" == "true" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --enable-profiler"
fi

if [ "$WARMUP_REQUESTS" != "" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --warmup-requests $WARMUP_REQUESTS"
fi

if [ "$BASE_URL" != "" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --base-url $BASE_URL"
fi

if [ "$MODEL_PATH" != "" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --model $MODEL_PATH"
fi

if [ "$DATASET" != "" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --dataset $DATASET"
fi

if [ "$PRINT_REQUESTS" == "true" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --print-requests"
fi

if [ "$PD_SEPARATED" == "true" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --pd-separated"

    if [ "$PROFILE_PREFILL_URL" != "" ]; then
        BENCHMARK_CMD="$BENCHMARK_CMD --profile-prefill-url $PROFILE_PREFILL_URL"
    fi

    if [ "$PROFILE_DECODE_URL" != "" ]; then
        BENCHMARK_CMD="$BENCHMARK_CMD --profile-decode-url $PROFILE_DECODE_URL"
    fi
fi

echo "================================================"
echo "运行 Benchmark 测试"
echo "================================================"
echo "配置文件: $CONFIG_FILE"
echo "命令: $BENCHMARK_CMD"
echo "================================================"
echo ""

# SSH 到容器执行（假设 SSH 框架已存在）
# 这里使用占位符，实际实现依赖于现有的 SSH 框架
# ssh user@container "cd /workspace && $BENCHMARK_CMD"

# 或者直接在本地执行（如果已经在容器中）
eval $BENCHMARK_CMD

echo ""
echo "================================================"
echo "Benchmark 测试完成"
echo "================================================"
```

**💡 提示词 for Agent:**
> - 脚本使用 `set -e` 确保遇到错误立即退出
> - 所有参数都是可选的，有合理的默认值
> - 命令行参数会覆盖 `config.yaml` 中的值
> - SSH 框架已存在，只需要调用即可（具体实现可能是项目特定的）
> - 如果在容器内运行，可以直接执行 `benchmark.py`

---

## 四、完整工作流程

### 4.1 执行流程图

```
┌─────────────────────────────────────────┐
│  用户调用 run_benchmark.sh              │
│  (可选参数覆盖 config.yaml)             │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  SSH 到容器 / 本地执行                   │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  benchmark.py 主程序                     │
│  - 加载 config.yaml                      │
│  - 用命令行参数覆盖配置                  │
│  - 初始化 BenchmarkRunner                │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  获取服务器信息                          │
│  - 调用 /get_server_info API            │
│  - 提取 TP/EP/DP 参数                    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  开始双层循环测试                        │
│  外层: input_lengths (1k → 128k)        │
│  内层: batch_sizes (1 → 256)            │
└─────────────┬───────────────────────────┘
              │
              ▼
       ┌──────────────┐
       │ 每个 input_len │
       └──────┬─────────┘
              │
              ▼
       ┌──────────────┐
       │ 每个 batch_size│
       └──────┬─────────┘
              │
              ▼
    ┌─────────────────────┐
    │ 1. should_skip_test? │
    │    (检查 bs 上限)     │
    └─────┬──────────┬─────┘
          │          │
        是 │          │ 否
          │          │
          ▼          ▼
    ┌──────┐   ┌─────────────────────────┐
    │ 跳过  │   │ 2. run_single_test       │
    └──────┘   │    - 调用 bench_serving  │
               │    - 保存结果            │
               └─────┬───────────────────┘
                     │
                     ▼
               ┌─────────────────────────┐
               │ 3. save_trace_file       │
               │    (如果启用 profiler)   │
               └─────┬───────────────────┘
                     │
                     ▼
               ┌─────────────────────────┐
               │ 4. check_queue_status    │
               │    - 获取 waiting_reqs   │
               │    - 判断是否超过阈值    │
               └─────┬──────────┬────────┘
                     │          │
              超限/失败│          │ 正常
                     │          │
                     ▼          ▼
               ┌──────────┐  ┌─────────┐
               │ 记录上限  │  │ 继续测试 │
               │ break    │  │ 下一个bs │
               └──────────┘  └─────────┘
                     │          │
                     └────┬─────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ 继续下一个    │
                   │ input_len     │
                   └──────┬────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ 所有测试完成  │
                   │ 打印总结      │
                   └──────────────┘
```

### 4.2 测试示例

假设配置：
- `input_lengths = [1024, 2048, 4096, 8192]`  (1k, 2k, 4k, 8k)
- `batch_sizes = [1, 4, 16, 32, 64, 128]`
- `max_queue_requests = 15`

**执行过程：**

```
# Input = 1k
  BS = 1   → 成功, queue = 0   ✓
  BS = 4   → 成功, queue = 3   ✓
  BS = 16  → 成功, queue = 8   ✓
  BS = 32  → 成功, queue = 18  ✗ 超限! 记录 max_bs_limits[1024] = 32
  → 跳过 BS = 64, 128

# Input = 2k
  BS = 1   → 成功, queue = 0   ✓
  BS = 4   → 成功, queue = 4   ✓
  BS = 16  → 成功, queue = 12  ✓
  BS = 32  → 跳过（基于 1k 的上限）
  BS = 64  → 跳过
  BS = 128 → 跳过

# Input = 4k
  BS = 1   → 成功, queue = 1   ✓
  BS = 4   → 成功, queue = 5   ✓
  BS = 16  → 成功, queue = 17  ✗ 超限! 记录 max_bs_limits[4096] = 16
  → 跳过 BS = 32, 64, 128

# Input = 8k
  BS = 1   → 成功, queue = 2   ✓
  BS = 4   → 成功, queue = 8   ✓
  BS = 16  → 跳过（基于 4k 的上限 16）
  BS = 32  → 跳过
  BS = 64  → 跳过
  BS = 128 → 跳过

总结:
  总测试: 10 次
  跳过: 11 次
  BS 上限记录:
    - Input 1k: BS <= 32
    - Input 4k: BS <= 16
```

---

## 五、关键技术要点

### 5.1 bench_serving.py 参数对应关系

| 需求 | bench_serving.py 参数 | 说明 |
|------|----------------------|------|
| Batch Size | `--num-prompts` | 实际值 = BS + warmup |
| Input Length | `--random-input-len` | Token 数量 |
| Output Length | `--random-output-len` | Token 数量 |
| Dataset | `--dataset-name random-ids` | 使用真实文本分布 |
| Warmup | `--warmup-requests` | 预热请求数 |
| Backend | `--backend sglang` | 推理引擎 |
| Server URL | `--base-url` | 服务器地址 |
| Profiler | `--profile` | 启用性能分析 |
| Print Requests | `--print-requests` | 打印请求日志 |
| Output File | `--output-file` | 结果 JSONL 文件 |

### 5.2 关键 API

#### `/get_server_info`

**请求:**
```bash
curl http://127.0.0.1:34567/get_server_info
```

**响应示例（普通模式）:**
```json
{
  "model_name": "Qwen3-Coder-480B-A35B-Instruct-FP8",
  "tp_size": 8,
  "ep_size": 4,
  "dp_size": 1,
  "#running_reqs": 3,
  "#waiting_reqs": 12,
  "internal_states": [...]
}
```

**响应示例（PD 分离模式）:**
```json
{
  "prefill": [{
    "model_name": "...",
    "#running_reqs": 1,
    "#waiting_reqs": 3,
    ...
  }],
  "decode": [{
    "model_name": "...",
    "#running_reqs": 5,
    "#waiting_reqs": 18,
    ...
  }]
}
```

**关键字段:**
- `tp_size` / `ep_size` / `dp_size`: 并行配置
- `#waiting_reqs`: 队列中等待的请求数 ⚠️ **这是判断阈值的关键指标**
- `#running_reqs`: 正在处理的请求数

### 5.3 结果文件示例

**输出文件:** `results/Qwen3-480B-TP8-EP4-DP1/inference_bs16_input4k_output1k.json`

```json
{
  "tag": null,
  "backend": "sglang",
  "dataset_name": "random-ids",
  "request_rate": "inf",
  "max_concurrency": null,
  "random_input_len": 4096,
  "random_output_len": 1024,
  "duration": 45.23,
  "completed": 16,
  "total_input_tokens": 65536,
  "total_output_tokens": 16384,
  "request_throughput": 0.35,
  "input_throughput": 1448.5,
  "output_throughput": 362.1,
  "total_throughput": 1810.6,
  "mean_ttft_ms": 125.3,
  "median_ttft_ms": 118.2,
  "p99_ttft_ms": 245.6,
  "mean_tpot_ms": 12.5,
  "median_tpot_ms": 11.8,
  "p99_tpot_ms": 18.3,
  "mean_itl_ms": 12.5,
  "median_itl_ms": 11.8,
  "p99_itl_ms": 18.3,
  "concurrency": 15.8,
  "server_info": {
    "model_name": "Qwen3-Coder-480B-A35B-Instruct-FP8",
    "tp_size": 8,
    "ep_size": 4,
    "dp_size": 1
  }
}
```

---

## 六、实现检查清单

### ✅ Phase 1: 配置系统
- [ ] 创建 `config.yaml` 模板
- [ ] 实现 `load_config()` 函数
- [ ] 添加配置验证逻辑
- [ ] 支持命令行参数覆盖配置

### ✅ Phase 2: 核心逻辑
- [ ] 实现 `BenchmarkRunner` 类
- [ ] 实现 `get_server_info()` - 获取 TP/EP/DP 和队列状态
- [ ] 实现 `should_skip_test()` - 基于 bs 上限判断
- [ ] 实现 `run_single_test()` - 调用 bench_serving.py
- [ ] 实现 `check_and_record_queue_status()` - 队列监控
- [ ] 实现 `save_results()` - 保存 JSON 结果
- [ ] 实现 `save_trace_file()` - 保存 profiler trace
- [ ] 实现 `run_benchmark()` - 主测试循环

### ✅ Phase 3: 入口脚本
- [ ] 创建 `run_benchmark.sh`
- [ ] 实现参数解析
- [ ] 集成现有 SSH 框架
- [ ] 添加错误处理

### ✅ Phase 4: 测试验证
- [ ] 单元测试：配置加载
- [ ] 单元测试：服务器信息获取
- [ ] 集成测试：完整测试流程
- [ ] 边界测试：队列阈值触发
- [ ] 错误测试：服务器不可用、测试超时等

---

## 七、常见问题和调试建议

### Q1: 如何调试 `get_server_info()` 返回的数据格式？

**建议:**
```python
# 添加详细日志
import json

server_info = self.get_server_info()
print("=== Server Info Debug ===")
print(json.dumps(server_info, indent=2))
print("=========================")
```

### Q2: 测试中途失败如何恢复？

**建议:**
- 添加断点续传功能：
  ```python
  # 在测试前检查结果文件是否已存在
  output_file = self._get_output_filename(batch_size, input_len)
  if Path(output_file).exists():
      print(f"  结果文件已存在，跳过: {output_file}")
      return json.loads(Path(output_file).read_text().split('\n')[-1])
  ```

### Q3: 如何处理 PD 分离模式？

**建议（暂不实现）:**
- 目前所有测试使用 `mode="inference"`
- 未来如需支持，可以：
  1. 在配置中添加 `pd_separated.mode: "prefill" | "decode"`
  2. 修改 `_get_output_filename()` 使用对应的 mode
  3. 分别调用两次 `bench_serving.py`，指定不同的 URL

### Q4: Trace 文件如何获取？

**建议:**
- Profiler 输出路径由服务器环境变量 `SGLANG_TORCH_PROFILER_DIR` 指定
- 可能需要：
  1. 调用 `/stop_profile` API 获取 trace 文件路径
  2. 使用文件系统路径或 scp 复制文件
  3. 压缩为 `.gz` 格式

---

## 八、代码模板总结

### 完整的 benchmark.py 骨架

```python
#!/usr/bin/env python3
"""
LLM Benchmark 自动化测试工具
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import requests


class BenchmarkRunner:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.max_bs_limits = {}

    def load_config(self, config_path: str) -> Dict:
        """加载和验证配置"""
        # 见 1.2 节
        pass

    def get_server_info(self) -> Dict:
        """获取服务器信息（TP/EP/DP, 队列状态）"""
        # 见 2.2.1 节
        pass

    def should_skip_test(self, input_len: int, batch_size: int) -> Tuple[bool, str]:
        """判断是否跳过测试"""
        # 见 2.2.3 节
        pass

    def run_single_test(self, batch_size: int, input_len: int) -> Dict:
        """运行单次测试"""
        # 见 2.2.2 节
        pass

    def check_and_record_queue_status(
        self, input_len: int, batch_size: int, test_result: Dict
    ) -> bool:
        """检查队列并记录 bs 上限"""
        # 见 2.2.3 节
        pass

    def _get_output_filename(self, batch_size: int, input_len: int) -> str:
        """生成输出文件名"""
        # 见 2.2.4 节
        pass

    def save_trace_file(self, batch_size: int, input_len: int):
        """保存 profiler trace"""
        # 见 2.2.4 节
        pass

    def run_benchmark(self):
        """主测试循环"""
        # 见 2.2.5 节
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backend", help="覆盖 backend")
    parser.add_argument("--base-url", help="覆盖 base_url")
    parser.add_argument("--enable-profiler", action="store_true")
    # ... 其他参数

    args = parser.parse_args()

    runner = BenchmarkRunner(args.config)

    # 命令行参数覆盖配置
    if args.backend:
        runner.config['server']['backend'] = args.backend
    if args.base_url:
        runner.config['server']['base_url'] = args.base_url
    if args.enable_profiler:
        runner.config['profiler']['enable'] = True

    runner.run_benchmark()


if __name__ == "__main__":
    main()
```

---

## 九、给 AI Agent 的关键提示

### 🎯 核心设计原则

1. **从小到大遍历**: Input 从小到大，BS 从小到大
2. **智能跳过**: 基于已知 BS 上限，避免不必要的测试
3. **队列监控**: 每次测试后检查 `#waiting_reqs`，超过阈值则记录上限
4. **渐进式限制**: Input 越大，BS 上限只会相等或更小
5. **详细日志**: 打印每一步操作，方便调试和监控

### 🔧 实现技巧

1. **错误处理**: 所有外部调用（API、subprocess）都要有 try-except
2. **超时设置**: subprocess 调用设置合理超时（如 1 小时）
3. **文件路径**: 使用 `pathlib.Path` 处理路径，自动创建目录
4. **配置覆盖**: 命令行参数优先级高于配置文件
5. **结果解析**: JSONL 文件每行一个 JSON，读取最后一行

### ⚠️ 常见陷阱

1. **num_prompts 计算**: `num_prompts = batch_size + warmup_requests`
2. **队列字段名**: 注意 `#waiting_reqs` 的 `#` 符号
3. **PD 分离响应**: 需要处理 `server_info["decode"]` 的特殊结构
4. **文件名格式**: 严格按照 `{mode}_bs{bs}_input{input}_output{output}.json`
5. **Input 单位**: 配置中是 token 数（如 1024），文件名中是 "1k"

### 📚 参考资料

- `bench_serving.py` 完整参数: 见前面的探索报告
- API 文档: `/get_server_info` 返回格式
- 输出文件: JSONL 格式，每行一个完整 JSON 对象

---

## 十、快速启动示例

### 示例 1: 最小化测试（调试用）

```yaml
# config_minimal.yaml
server:
  base_url: "http://127.0.0.1:34567"
  backend: "sglang"

benchmark:
  dataset: "random-ids"
  output_len: 128
  warmup_requests: 1
  batch_sizes: [1, 4]
  input_lengths: [1024, 2048]
  max_queue_requests: 10

output:
  result_base_path: "./test_results"
  save_traces: false

profiler:
  enable: false
```

```bash
python3 benchmark.py --config config_minimal.yaml
```

### 示例 2: 完整测试

```bash
./run_benchmark.sh \
  --backend sglang \
  --base-url http://127.0.0.1:34567 \
  --enable-profiler true \
  --warmup-requests 3
```

---

## 附录：完整的 config.yaml 模板

```yaml
# ============================================
# LLM Benchmark 配置文件
# ============================================

# 服务器配置
server:
  base_url: "http://127.0.0.1:34567"
  backend: "sglang"
  model_path: "/cpfs01/user/nebula_model/llm_weight/Qwen3-Coder-480B-A35B-Instruct-FP8"

# 测试参数
benchmark:
  dataset: "random-ids"
  output_len: 1024
  warmup_requests: 3

  # 测试矩阵
  batch_sizes: [1, 4, 16, 32, 64, 128, 256]
  input_lengths: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

  # 队列限制
  max_queue_requests: 15

# 结果路径
output:
  result_base_path: "./results"
  save_traces: true

# Profiler 配置
profiler:
  enable: false
  activities: ["CPU", "GPU"]

# PD 分离模式（暂不实现）
pd_separated:
  enable: false
  prefill_url: "http://127.0.0.1:30010"
  decode_url: "http://127.0.0.1:30020"

# 其他选项
options:
  print_requests: true
  output_details: true
```

---

**文档版本**: v1.0
**最后更新**: 2025-01-22
**作者**: Claude Code Team
**目标读者**: Claude Code Agent

---

## 总结

本文档提供了完整的实现指导，包括：
- ✅ 配置系统设计
- ✅ 核心逻辑实现
- ✅ 关键函数详解
- ✅ 工作流程图
- ✅ 代码模板
- ✅ 调试建议
- ✅ AI Agent 提示词

请按照 Phase 1 → Phase 2 → Phase 3 → Phase 4 的顺序实施。每个 Phase 完成后进行测试验证，确保功能正常再进入下一阶段。

**祝实现顺利！** 🚀
