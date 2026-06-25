#!/usr/bin/env python3
"""
Export benchmark data from JSON files to CSV format

将 benchmark JSON 数据导出到两个 CSV 文件：
1. benchmark_summary.csv: 汇总统计数据（每个配置一行）
2. benchmark_requests.csv: 请求级详细数据（每个请求一行）

使用增量更新机制，支持重复运行不重复处理。
"""

import json
import csv
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set


# ============================================================================
# CSV Schema 定义
# ============================================================================

SUMMARY_COLUMNS = [
    # 配置信息 (8)
    'tp_size', 'ep_size', 'dp_size', 'nnodes',
    'batch_size', 'input_len', 'output_len',
    'moe_a2a_backend',

    # 时长与完成 (3)
    'duration', 'completed', 'concurrency',

    # Token 统计 (4)
    'total_input_tokens', 'total_output_tokens',
    'total_input_text_tokens', 'total_output_tokens_retokenized',

    # 吞吐量 (4)
    'request_throughput', 'input_throughput',
    'output_throughput', 'total_throughput',

    # E2E 延迟 (4)
    'mean_e2e_latency_ms', 'median_e2e_latency_ms',
    'std_e2e_latency_ms', 'p99_e2e_latency_ms',

    # TTFT (4)
    'mean_ttft_ms', 'median_ttft_ms',
    'std_ttft_ms', 'p99_ttft_ms',

    # TPOT (4)
    'mean_tpot_ms', 'median_tpot_ms',
    'std_tpot_ms', 'p99_tpot_ms',

    # ITL (5)
    'mean_itl_ms', 'median_itl_ms', 'std_itl_ms',
    'p95_itl_ms', 'p99_itl_ms',

    # 其他 (2)
    'max_output_tokens_per_s', 'max_concurrent_requests',
]

# PD 分离模式的 CSV Schema
PD_SUMMARY_COLUMNS = [
    # Prefill 配置 (3)
    'prefill_tp_size', 'prefill_ep_size', 'prefill_dp_size',

    # Decode 配置 (3)
    'decode_tp_size', 'decode_ep_size', 'decode_dp_size',

    # 测试配置 (3)
    'batch_size', 'input_len', 'output_len',

    # 服务器信息 (3)
    'router_manager', 'routers_count', 'workers_count',

    # 时长与完成 (3)
    'duration', 'completed', 'concurrency',

    # Token 统计 (4)
    'total_input_tokens', 'total_output_tokens',
    'total_input_text_tokens', 'total_output_tokens_retokenized',

    # 吞吐量 (4)
    'request_throughput', 'input_throughput',
    'output_throughput', 'total_throughput',

    # E2E 延迟 (4)
    'mean_e2e_latency_ms', 'median_e2e_latency_ms',
    'std_e2e_latency_ms', 'p99_e2e_latency_ms',

    # TTFT (4)
    'mean_ttft_ms', 'median_ttft_ms',
    'std_ttft_ms', 'p99_ttft_ms',

    # TPOT (4)
    'mean_tpot_ms', 'median_tpot_ms',
    'std_tpot_ms', 'p99_tpot_ms',

    # ITL (5)
    'mean_itl_ms', 'median_itl_ms', 'std_itl_ms',
    'p95_itl_ms', 'p99_itl_ms',

    # 其他 (2)
    'max_output_tokens_per_s', 'max_concurrent_requests',
]

REQUESTS_COLUMNS = [
    # 配置信息 (8)
    'tp_size', 'ep_size', 'dp_size', 'nnodes',
    'batch_size', 'input_len', 'output_len',
    'moe_a2a_backend',

    # 请求数据 (5)
    'request_id', 'actual_input_len', 'actual_output_len', 'ttft_ms', 'decode_time',
]


# ============================================================================
# 文件发现与去重追踪
# ============================================================================

def find_json_files(root_dir: str = 'result/benchmarks') -> List[Path]:
    """扫描所有 benchmark JSON 文件"""
    root = Path(root_dir)
    if not root.exists():
        logging.warning(f"目录不存在: {root_dir}")
        return []

    files = sorted(root.glob('**/*.json'))
    logging.info(f"发现 {len(files)} 个 JSON 文件")
    return files


def load_processed_files(tracking_file: Path) -> Set[str]:
    """加载已处理文件集合"""
    if not tracking_file.exists():
        return set()

    with open(tracking_file, 'r') as f:
        processed = {line.strip() for line in f if line.strip()}

    logging.info(f"已处理 {len(processed)} 个文件")
    return processed


def save_processed_files(tracking_file: Path, processed_files: Set[str]):
    """保存已处理文件集合"""
    tracking_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracking_file, 'w') as f:
        for path in sorted(processed_files):
            f.write(f"{path}\n")

    logging.info(f"更新追踪文件: {tracking_file}")


def get_unprocessed_files(all_files: List[Path], processed_files: Set[str]) -> List[Path]:
    """过滤出未处理的文件"""
    all_paths = {str(f.resolve()) for f in all_files}
    new_paths = all_paths - processed_files
    new_files = [Path(p) for p in sorted(new_paths)]

    logging.info(f"待处理 {len(new_files)} 个新文件")
    return new_files


# ============================================================================
# 配置提取（复用 test/json_to_csv_test.py 逻辑）
# ============================================================================

def extract_model_name_from_path(json_path: Path) -> str:
    """从文件路径提取模型名称"""
    path_str = str(json_path)

    # 提取 result/benchmarks/ 后的目录名
    match = re.search(r'result/benchmarks/([^/]+)/', path_str)
    if not match:
        return 'Unknown'

    dir_name = match.group(1)

    # 移除 TP/EP/DP 配置部分和 -nodeepep 后缀
    # 例如: Qwen3-Coder-480B-FP8-TP8-EP8-DP8-nodeepep -> Qwen3-Coder-480B-FP8
    model_name = re.sub(r'-TP\d+.*$', '', dir_name)

    # 处理 PD 分离模式
    # 例如: Qwen3-Coder-480B-FP8-Prefill-TP8-EP8-DP8-Decode-TP8-DP8-EP8 -> Qwen3-Coder-480B-FP8
    model_name = re.sub(r'-Prefill-.*$', '', model_name)

    return model_name


def extract_config_from_path(json_path: Path) -> Dict[str, Optional[int]]:
    """从文件路径提取配置信息"""
    path_str = str(json_path)

    # 提取模型名称
    model_name = extract_model_name_from_path(json_path)

    # 检查是否为 PD 分离模式
    if 'Prefill' in path_str and 'Decode' in path_str:
        # PD 分离模式: Prefill-TP16-EP16-DP16-Decode-TP48-EP48-DP48
        prefill_match = re.search(r'Prefill-TP(\d+)-EP(\d+)-DP(\d+)', path_str)
        decode_match = re.search(r'Decode-TP(\d+)-EP(\d+)-DP(\d+)', path_str)
        bs_match = re.search(r'bs(\d+)_in(\d+)_out(\d+)', path_str)

        return {
            'is_pd_separation': True,
            'model_name': model_name,
            'prefill_tp': int(prefill_match.group(1)) if prefill_match else None,
            'prefill_ep': int(prefill_match.group(2)) if prefill_match else None,
            'prefill_dp': int(prefill_match.group(3)) if prefill_match else None,
            'decode_tp': int(decode_match.group(1)) if decode_match else None,
            'decode_ep': int(decode_match.group(2)) if decode_match else None,
            'decode_dp': int(decode_match.group(3)) if decode_match else None,
            'batch_size': int(bs_match.group(1)) if bs_match else None,
            'input_len': int(bs_match.group(2)) if bs_match else None,
            'output_len': int(bs_match.group(3)) if bs_match else None,
        }
    else:
        # 常规模式
        tp_match = re.search(r'TP(\d+)', path_str)
        ep_match = re.search(r'EP(\d+)', path_str)
        dp_match = re.search(r'DP(\d+)', path_str)
        bs_match = re.search(r'bs(\d+)_in(\d+)_out(\d+)', path_str)

        # 检测是否有 -nodeepep 后缀
        has_nodeepep_suffix = '-nodeepep' in path_str

        return {
            'is_pd_separation': False,
            'model_name': model_name,
            'tp': int(tp_match.group(1)) if tp_match else None,
            'ep': int(ep_match.group(1)) if ep_match else None,
            'dp': int(dp_match.group(1)) if dp_match else None,
            'batch_size': int(bs_match.group(1)) if bs_match else None,
            'input_len': int(bs_match.group(2)) if bs_match else None,
            'output_len': int(bs_match.group(3)) if bs_match else None,
            'has_nodeepep_suffix': has_nodeepep_suffix,
        }


# ============================================================================
# 数据解析 - Summary
# ============================================================================

def parse_json_to_summary_row(json_path: Path) -> Optional[Dict]:
    """解析 JSON 文件并返回汇总数据行"""
    config = extract_config_from_path(json_path)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"无效 JSON: {json_path}: {e}")
        return None
    except IOError as e:
        logging.error(f"无法读取: {json_path}: {e}")
        return None

    server_info = data.get('server_info', {})

    # 检查是否为 PD 分离模式
    if config.get('is_pd_separation'):
        # PD 分离模式
        row = {
            # Prefill 配置
            'prefill_tp_size': config['prefill_tp'],
            'prefill_ep_size': config['prefill_ep'],
            'prefill_dp_size': config['prefill_dp'],

            # Decode 配置
            'decode_tp_size': config['decode_tp'],
            'decode_ep_size': config['decode_ep'],
            'decode_dp_size': config['decode_dp'],

            # 测试配置
            'batch_size': config['batch_size'],
            'input_len': config['input_len'],
            'output_len': config['output_len'],

            # 服务器信息
            'router_manager': server_info.get('router_manager'),
            'routers_count': server_info.get('routers_count'),
            'workers_count': server_info.get('workers_count'),

            # 时长与完成
            'duration': data.get('duration'),
            'completed': data.get('completed'),
            'concurrency': data.get('concurrency'),

            # Token 统计
            'total_input_tokens': data.get('total_input_tokens'),
            'total_output_tokens': data.get('total_output_tokens'),
            'total_input_text_tokens': data.get('total_input_text_tokens'),
            'total_output_tokens_retokenized': data.get('total_output_tokens_retokenized'),

            # 吞吐量
            'request_throughput': data.get('request_throughput'),
            'input_throughput': data.get('input_throughput'),
            'output_throughput': data.get('output_throughput'),
            'total_throughput': data.get('total_throughput'),

            # E2E 延迟
            'mean_e2e_latency_ms': data.get('mean_e2e_latency_ms'),
            'median_e2e_latency_ms': data.get('median_e2e_latency_ms'),
            'std_e2e_latency_ms': data.get('std_e2e_latency_ms'),
            'p99_e2e_latency_ms': data.get('p99_e2e_latency_ms'),

            # TTFT
            'mean_ttft_ms': data.get('mean_ttft_ms'),
            'median_ttft_ms': data.get('median_ttft_ms'),
            'std_ttft_ms': data.get('std_ttft_ms'),
            'p99_ttft_ms': data.get('p99_ttft_ms'),

            # TPOT
            'mean_tpot_ms': data.get('mean_tpot_ms'),
            'median_tpot_ms': data.get('median_tpot_ms'),
            'std_tpot_ms': data.get('std_tpot_ms'),
            'p99_tpot_ms': data.get('p99_tpot_ms'),

            # ITL
            'mean_itl_ms': data.get('mean_itl_ms'),
            'median_itl_ms': data.get('median_itl_ms'),
            'std_itl_ms': data.get('std_itl_ms'),
            'p95_itl_ms': data.get('p95_itl_ms'),
            'p99_itl_ms': data.get('p99_itl_ms'),

            # 其他
            'max_output_tokens_per_s': data.get('max_output_tokens_per_s'),
            'max_concurrent_requests': data.get('max_concurrent_requests'),
        }
    else:
        # 常规模式
        tp = config['tp']
        ep = config['ep']
        dp = config['dp']
        has_nodeepep_suffix = config.get('has_nodeepep_suffix', False)

        # 确定 moe_a2a_backend 值
        # 如果 TP=DP=EP 且不带 -nodeepep 后缀，则为 deepep，否则为 none
        if tp == dp == ep and not has_nodeepep_suffix:
            moe_a2a_backend = 'deepep'
        else:
            moe_a2a_backend = 'none'

        row = {
            # 配置信息
            'tp_size': tp,
            'ep_size': ep,
            'dp_size': dp,
            'nnodes': server_info.get('nnodes'),
            'batch_size': config['batch_size'],
            'input_len': config['input_len'],
            'output_len': config['output_len'],
            'moe_a2a_backend': moe_a2a_backend,

            # 时长与完成
            'duration': data.get('duration'),
            'completed': data.get('completed'),
            'concurrency': data.get('concurrency'),

            # Token 统计
            'total_input_tokens': data.get('total_input_tokens'),
            'total_output_tokens': data.get('total_output_tokens'),
            'total_input_text_tokens': data.get('total_input_text_tokens'),
            'total_output_tokens_retokenized': data.get('total_output_tokens_retokenized'),

            # 吞吐量
            'request_throughput': data.get('request_throughput'),
            'input_throughput': data.get('input_throughput'),
            'output_throughput': data.get('output_throughput'),
            'total_throughput': data.get('total_throughput'),

            # E2E 延迟
            'mean_e2e_latency_ms': data.get('mean_e2e_latency_ms'),
            'median_e2e_latency_ms': data.get('median_e2e_latency_ms'),
            'std_e2e_latency_ms': data.get('std_e2e_latency_ms'),
            'p99_e2e_latency_ms': data.get('p99_e2e_latency_ms'),

            # TTFT
            'mean_ttft_ms': data.get('mean_ttft_ms'),
            'median_ttft_ms': data.get('median_ttft_ms'),
            'std_ttft_ms': data.get('std_ttft_ms'),
            'p99_ttft_ms': data.get('p99_ttft_ms'),

            # TPOT
            'mean_tpot_ms': data.get('mean_tpot_ms'),
            'median_tpot_ms': data.get('median_tpot_ms'),
            'std_tpot_ms': data.get('std_tpot_ms'),
            'p99_tpot_ms': data.get('p99_tpot_ms'),

            # ITL
            'mean_itl_ms': data.get('mean_itl_ms'),
            'median_itl_ms': data.get('median_itl_ms'),
            'std_itl_ms': data.get('std_itl_ms'),
            'p95_itl_ms': data.get('p95_itl_ms'),
            'p99_itl_ms': data.get('p99_itl_ms'),

            # 其他
            'max_output_tokens_per_s': data.get('max_output_tokens_per_s'),
            'max_concurrent_requests': data.get('max_concurrent_requests'),
        }

    return row


# ============================================================================
# 数据解析 - Requests
# ============================================================================

def parse_json_to_request_rows(json_path: Path, config: Dict) -> List[Dict]:
    """解析 JSON 文件并返回请求级数据行"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"解析失败: {json_path}: {e}")
        return []

    server_info = data.get('server_info', {})
    input_lens = data.get('input_lens', [])
    output_lens = data.get('output_lens', [])
    ttfts = data.get('ttfts', [])
    itls = data.get('itls', [])  # Inter-token latencies for each request

    # 验证数组长度
    if not input_lens:
        logging.warning(f"{json_path}: 无 input_lens 数据")
        return []

    lengths = [len(input_lens), len(output_lens), len(ttfts)]
    if itls:  # Only include itls in length check if it exists
        lengths.append(len(itls))
    if len(set(lengths)) > 1:
        logging.warning(f"{json_path}: 数组长度不匹配 {lengths}")
        num_requests = min(lengths)
    else:
        num_requests = lengths[0]

    # 计算 moe_a2a_backend
    tp = config['tp']
    ep = config['ep']
    dp = config['dp']
    has_nodeepep_suffix = config.get('has_nodeepep_suffix', False)

    if tp == dp == ep and not has_nodeepep_suffix:
        moe_a2a_backend = 'deepep'
    else:
        moe_a2a_backend = 'none'

    rows = []
    for i in range(num_requests):
        try:
            # Calculate decode_time as sum of all inter-token latencies for this request
            decode_time = None
            if itls and i < len(itls):
                itl_list = itls[i]
                if isinstance(itl_list, list) and itl_list:
                    decode_time = sum(itl_list)
                elif isinstance(itl_list, (int, float)):
                    # If itls is a single value, use it directly
                    decode_time = itl_list

            row = {
                # 配置信息
                'tp_size': tp,
                'ep_size': ep,
                'dp_size': dp,
                'nnodes': server_info.get('nnodes'),
                'batch_size': config['batch_size'],
                'input_len': config['input_len'],
                'output_len': config['output_len'],
                'moe_a2a_backend': moe_a2a_backend,

                # 请求数据
                'request_id': i,
                'actual_input_len': input_lens[i],
                'actual_output_len': output_lens[i],
                'ttft_ms': ttfts[i],
                'decode_time': decode_time,
            }
            rows.append(row)
        except IndexError as e:
            logging.error(f"{json_path} 请求 {i}: {e}")
            continue

    return rows


# ============================================================================
# CSV 导出
# ============================================================================

def export_summary_csv(rows: List[Dict], output_path: Path, mode: str = 'w'):
    """导出汇总数据到 CSV"""
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 判断是否需要写入表头
    write_header = (mode == 'w') or not output_path.exists()

    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logging.info(f"导出 {len(rows)} 行到 {output_path}")


def export_pd_summary_csv(rows: List[Dict], output_path: Path, mode: str = 'w'):
    """导出 PD 分离模式汇总数据到 CSV"""
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 判断是否需要写入表头
    write_header = (mode == 'w') or not output_path.exists()

    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PD_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logging.info(f"导出 {len(rows)} 行 PD 数据到 {output_path}")


def export_requests_csv(rows: List[Dict], output_path: Path, mode: str = 'w'):
    """导出请求级数据到 CSV"""
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = (mode == 'w') or not output_path.exists()

    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REQUESTS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    logging.info(f"导出 {len(rows)} 行到 {output_path}")


# ============================================================================
# 主流程
# ============================================================================

def export_benchmarks(
    input_dir: str = 'result/benchmarks',
    output_dir: str = 'analysis/data',
    incremental: bool = True,
    export_summary: bool = True,
    export_requests: bool = True,
    batch_size: int = 50,
):
    """
    导出 benchmark 数据到 CSV

    Args:
        input_dir: JSON 文件所在目录
        output_dir: CSV 输出目录
        incremental: 是否增量更新
        export_summary: 是否导出汇总 CSV
        export_requests: 是否导出请求级 CSV
        batch_size: 批量处理大小
    """
    output_path = Path(output_dir)
    summary_csv = output_path / 'benchmark_summary.csv'
    pd_summary_csv = output_path / 'PD_benchmark.csv'
    requests_csv = output_path / 'benchmark_requests.csv'
    tracking_file = output_path / '.processed_files.txt'

    # 发现所有文件
    all_files = find_json_files(input_dir)
    if not all_files:
        logging.warning("未发现任何 JSON 文件")
        return

    # 增量模式：过滤已处理文件
    if incremental:
        processed = load_processed_files(tracking_file)
        files_to_process = get_unprocessed_files(all_files, processed)

        if not files_to_process:
            logging.info("所有文件已处理，无需更新")
            return
    else:
        files_to_process = all_files
        processed = set()
        logging.info("强制重建模式：处理所有文件")

    # 批量处理文件 - 按模型分组
    summary_rows_by_model = {}  # {model_name: [rows]}
    pd_summary_rows = []
    request_rows_by_model = {}  # {model_name: [rows]}
    processed_count = 0
    error_count = 0

    # 记录每个模型的CSV写入模式
    csv_modes = {}  # {model_name: mode}
    pd_csv_mode = 'w' if not incremental or not pd_summary_csv.exists() else 'a'

    for i, json_file in enumerate(files_to_process):
        logging.debug(f"处理 {i+1}/{len(files_to_process)}: {json_file.name}")

        # 提取配置信息
        config = extract_config_from_path(json_file)
        is_pd_separation = config.get('is_pd_separation', False)
        model_name = config.get('model_name', 'Unknown')

        # 解析汇总数据
        if export_summary:
            summary_row = parse_json_to_summary_row(json_file)
            if summary_row:
                if is_pd_separation:
                    pd_summary_rows.append(summary_row)
                else:
                    # 按模型分组
                    if model_name not in summary_rows_by_model:
                        summary_rows_by_model[model_name] = []
                    summary_rows_by_model[model_name].append(summary_row)
                processed_count += 1
            else:
                error_count += 1
                continue  # 汇总数据解析失败，跳过请求级数据

        # 解析请求级数据（仅常规模式）
        if export_requests and not is_pd_separation:
            req_rows = parse_json_to_request_rows(json_file, config)
            if req_rows:
                # 按模型分组
                if model_name not in request_rows_by_model:
                    request_rows_by_model[model_name] = []
                request_rows_by_model[model_name].extend(req_rows)

        # 批量导出
        if (i + 1) % batch_size == 0 or (i + 1) == len(files_to_process):
            # 导出常规汇总数据 - 按模型分别导出
            if export_summary:
                for model_name, rows in summary_rows_by_model.items():
                    if rows:
                        # 生成模型专属的CSV文件名
                        model_summary_csv = output_path / f'{model_name}_benchmark_summary.csv'

                        # 确定写入模式
                        if model_name not in csv_modes:
                            csv_modes[model_name] = 'w' if not incremental or not model_summary_csv.exists() else 'a'

                        export_summary_csv(rows, model_summary_csv, mode=csv_modes[model_name])
                        csv_modes[model_name] = 'a'  # 后续批次使用追加模式

                summary_rows_by_model.clear()

            # 导出 PD 分离汇总数据
            if export_summary and pd_summary_rows:
                export_pd_summary_csv(pd_summary_rows, pd_summary_csv, mode=pd_csv_mode)
                pd_summary_rows.clear()
                pd_csv_mode = 'a'  # 后续批次使用追加模式

            # 导出请求级数据 - 按模型分别导出
            if export_requests:
                for model_name, rows in request_rows_by_model.items():
                    if rows:
                        # 生成模型专属的CSV文件名
                        model_requests_csv = output_path / f'{model_name}_benchmark_requests.csv'

                        # 确定写入模式
                        req_mode_key = f'{model_name}_req'
                        if req_mode_key not in csv_modes:
                            csv_modes[req_mode_key] = 'w' if not incremental or not model_requests_csv.exists() else 'a'

                        export_requests_csv(rows, model_requests_csv, mode=csv_modes[req_mode_key])
                        csv_modes[req_mode_key] = 'a'  # 后续批次使用追加模式

                request_rows_by_model.clear()

            logging.info(f"进度: {i+1}/{len(files_to_process)}")

    # 更新追踪文件
    if incremental:
        processed.update(str(f.resolve()) for f in files_to_process)
        save_processed_files(tracking_file, processed)

    # 输出统计
    logging.info("=" * 60)
    logging.info(f"导出完成:")
    logging.info(f"  成功处理: {processed_count} 个文件")
    logging.info(f"  错误: {error_count} 个文件")
    if export_summary:
        # 列出所有生成的模型CSV文件
        model_csvs = sorted(output_path.glob('*_benchmark_summary.csv'))
        logging.info(f"  生成 {len(model_csvs)} 个模型汇总 CSV:")
        for csv_file in model_csvs:
            logging.info(f"    - {csv_file.name}")
        logging.info(f"  PD 分离汇总 CSV: {pd_summary_csv}")
    if export_requests:
        # 列出所有生成的请求级CSV文件
        request_csvs = sorted(output_path.glob('*_benchmark_requests.csv'))
        logging.info(f"  生成 {len(request_csvs)} 个模型请求级 CSV:")
        for csv_file in request_csvs:
            logging.info(f"    - {csv_file.name}")
    logging.info("=" * 60)


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='导出 benchmark JSON 数据到 CSV 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 增量更新（默认）
  python analysis/export_to_csv.py

  # 强制完全重建
  python analysis/export_to_csv.py --force

  # 只导出汇总数据
  python analysis/export_to_csv.py --summary-only

  # 只导出请求级数据
  python analysis/export_to_csv.py --requests-only

  # 自定义路径
  python analysis/export_to_csv.py --input result/benchmarks --output analysis/data

  # 详细日志
  python analysis/export_to_csv.py --verbose
        '''
    )

    parser.add_argument(
        '--input', '-i',
        default='result/benchmarks',
        help='JSON 文件输入目录 (默认: result/benchmarks)'
    )

    parser.add_argument(
        '--output', '-o',
        default='analysis/data',
        help='CSV 文件输出目录 (默认: analysis/data)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制完全重建（忽略追踪文件）'
    )

    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='仅导出汇总 CSV'
    )

    parser.add_argument(
        '--requests-only',
        action='store_true',
        help='仅导出请求级 CSV'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=50,
        help='批量处理大小 (默认: 50)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细日志输出'
    )

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = Path(args.output) / 'export.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info("开始导出 benchmark 数据")
    logging.info("=" * 60)

    # 执行导出
    export_benchmarks(
        input_dir=args.input,
        output_dir=args.output,
        incremental=not args.force,
        export_summary=not args.requests_only,
        export_requests=not args.summary_only,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
