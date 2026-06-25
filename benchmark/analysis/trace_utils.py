#!/usr/bin/env python3
"""
Trace Analysis Utilities

Utilities for trace file discovery, configuration parsing, and incremental processing.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set


def extract_config_from_trace_path(trace_path: Path) -> Dict[str, Optional[any]]:
    """
    Extract configuration from trace file path using regex patterns.

    Expected path pattern:
        result/traces/Qwen3-480B-TP{X}-EP{Y}-DP{Z}/
        bs{N}_in{INPUT}_out{OUTPUT}/{timestamp}-TP-{rank}-EP-{rank}.trace.json.gz

    Args:
        trace_path: Path to trace file

    Returns:
        Dictionary with configuration fields:
        - model_name: str (e.g., 'Qwen3-480B')
        - tp_size, ep_size, dp_size: int
        - batch_size, input_len, output_len: int
        - timestamp: float
        - tp_rank, ep_rank: int

    Example:
        >>> path = Path('result/traces/Qwen3-480B-TP16-EP8-DP1/bs32_in2048_output512/1766401812.8294094-TP-0-EP-0.trace.json.gz')
        >>> config = extract_config_from_trace_path(path)
        >>> config['tp_size']
        16
    """
    path_str = str(trace_path)

    # Pattern 1: Model and parallelism configuration
    # Matches: Qwen3-480B-TP16-EP8-DP1 or Qwen3-Coder-480B-FP8-TP8-EP8-DP8
    config_match = re.search(
        r'(Qwen3[^/]*?)-TP(\d+)-EP(\d+)-DP(\d+)',
        path_str
    )

    # Pattern 2: Batch configuration
    # Matches: bs32_in2048_out512 or bs32_in2048_output512
    batch_match = re.search(
        r'bs(\d+)_in(\d+)_out(?:put)?(\d+)',
        path_str
    )

    # Pattern 3: File name with rank information
    # Matches: 1766401812.8294094-TP-0-EP-0.trace.json.gz
    # or: 1766592555.4233096-TP-0-DP-0-EP-0.trace.json.gz (with DP)
    file_match = re.search(
        r'([\d.]+)-TP-(\d+)(?:-DP-(\d+))?-EP-(\d+)\.trace\.json\.gz',
        path_str
    )

    # Extract model name (remove parallelism suffix for cleaner name)
    model_name = None
    if config_match:
        model_name = config_match.group(1)

    return {
        # Model configuration
        'model_name': model_name,
        'tp_size': int(config_match.group(2)) if config_match else None,
        'ep_size': int(config_match.group(3)) if config_match else None,
        'dp_size': int(config_match.group(4)) if config_match else None,

        # Batch configuration
        'batch_size': int(batch_match.group(1)) if batch_match else None,
        'input_len': int(batch_match.group(2)) if batch_match else None,
        'output_len': int(batch_match.group(3)) if batch_match else None,

        # Trace metadata
        'timestamp': float(file_match.group(1)) if file_match else None,
        'tp_rank': int(file_match.group(2)) if file_match else None,
        'dp_rank': int(file_match.group(3)) if file_match and file_match.group(3) else None,
        'ep_rank': int(file_match.group(4)) if file_match else None,
    }


def find_trace_files(root_dir: str = 'result/traces') -> List[Path]:
    """
    Recursively discover all trace files in directory.

    Args:
        root_dir: Root directory to search (default: 'result/traces')

    Returns:
        Sorted list of Path objects for all *.trace.json.gz files

    Example:
        >>> files = find_trace_files('result/traces')
        >>> len(files)
        160
    """
    root = Path(root_dir)
    if not root.exists():
        logging.warning(f"Directory does not exist: {root_dir}")
        return []

    files = sorted(root.glob('**/*.trace.json.gz'))
    logging.info(f"Found {len(files)} trace files in {root_dir}")
    return files


def load_processed_trace_files(tracking_file: Path) -> Set[str]:
    """
    Load set of already-processed trace files from tracking file.

    The tracking file contains one absolute file path per line.
    This enables incremental updates by skipping already-processed files.

    Args:
        tracking_file: Path to .processed_trace_files.txt

    Returns:
        Set of absolute file paths (as strings) that have been processed

    Example:
        >>> tracking_file = Path('analysis/data/.processed_trace_files.txt')
        >>> processed = load_processed_trace_files(tracking_file)
        >>> len(processed)
        160
    """
    if not tracking_file.exists():
        logging.info(f"Tracking file does not exist: {tracking_file}")
        return set()

    with open(tracking_file, 'r') as f:
        processed = {line.strip() for line in f if line.strip()}

    logging.info(f"Loaded {len(processed)} processed files from {tracking_file}")
    return processed


def save_processed_trace_files(tracking_file: Path, processed_files: Set[str]):
    """
    Save set of processed trace files to tracking file.

    Args:
        tracking_file: Path to .processed_trace_files.txt
        processed_files: Set of absolute file paths (as strings)

    Example:
        >>> tracking_file = Path('analysis/data/.processed_trace_files.txt')
        >>> processed = {'/path/to/file1.trace.json.gz', '/path/to/file2.trace.json.gz'}
        >>> save_processed_trace_files(tracking_file, processed)
    """
    tracking_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracking_file, 'w') as f:
        for path in sorted(processed_files):
            f.write(f"{path}\n")

    logging.info(f"Saved {len(processed_files)} processed files to {tracking_file}")


def get_unprocessed_trace_files(
    all_files: List[Path],
    processed_files: Set[str]
) -> List[Path]:
    """
    Filter out already-processed files to get new files for processing.

    Args:
        all_files: List of all discovered trace files
        processed_files: Set of absolute paths that have been processed

    Returns:
        List of Path objects for files that have not been processed

    Example:
        >>> all_files = find_trace_files('result/traces')
        >>> processed = load_processed_trace_files(Path('analysis/data/.processed_trace_files.txt'))
        >>> new_files = get_unprocessed_trace_files(all_files, processed)
        >>> len(new_files)
        10
    """
    all_paths = {str(f.resolve()) for f in all_files}
    new_paths = all_paths - processed_files
    new_files = [Path(p) for p in sorted(new_paths)]

    logging.info(f"Found {len(new_files)} new files to process (out of {len(all_files)} total)")
    return new_files


def validate_config(config: Dict[str, Optional[any]]) -> bool:
    """
    Validate that extracted configuration has required fields.

    Args:
        config: Configuration dictionary from extract_config_from_trace_path

    Returns:
        True if config is valid, False otherwise
    """
    required_fields = [
        'tp_size', 'ep_size', 'dp_size',
        'batch_size', 'input_len', 'output_len',
        'timestamp', 'tp_rank', 'ep_rank'
    ]

    for field in required_fields:
        if config.get(field) is None:
            logging.warning(f"Missing required field: {field}")
            return False

    return True


if __name__ == '__main__':
    # Simple test
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test configuration extraction
    test_paths = [
        'result/traces/Qwen3-480B-TP16-EP8-DP1/bs32_in2048_output512/1766401812.8294094-TP-0-EP-0.trace.json.gz',
        'result/traces/Qwen3-480B-TP8-EP8-DP8/bs32_in2048_output512/1766592555.4233096-TP-0-DP-0-EP-0.trace.json.gz',
        'result/traces/Qwen3-Coder-480B-FP8-TP16-EP16-DP1/bs256_in800_out100/1766401812-TP-7-EP-3.trace.json.gz',
    ]

    print("Testing configuration extraction:")
    print("=" * 80)
    for path_str in test_paths:
        path = Path(path_str)
        config = extract_config_from_trace_path(path)
        print(f"\nPath: {path.name}")
        print(f"  Model: {config['model_name']}")
        print(f"  Parallelism: TP={config['tp_size']} EP={config['ep_size']} DP={config['dp_size']}")
        print(f"  Batch config: bs={config['batch_size']} in={config['input_len']} out={config['output_len']}")
        print(f"  Rank: TP={config['tp_rank']} EP={config['ep_rank']} DP={config['dp_rank']}")
        print(f"  Valid: {validate_config(config)}")

    print("\n" + "=" * 80)
    print("Testing file discovery:")
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = 'result/traces'

    files = find_trace_files(root_dir)
    if files:
        print(f"Found {len(files)} files")
        print(f"First file: {files[0]}")
        print(f"Last file: {files[-1]}")
