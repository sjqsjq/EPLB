#!/usr/bin/env python3
"""
Trace Aggregator

Compute aggregate metrics from kernel events.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from trace_parser import KernelEvent


class TraceAggregator:
    """
    Aggregate kernel events into summary statistics.

    Computes per-rank metrics (category breakdown, phase breakdown, GPU idle time)
    and cross-rank statistics (mean, std, load imbalance).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def aggregate_per_rank(
        self,
        kernels: List[KernelEvent],
        config: Dict
    ) -> Dict:
        """
        Compute per-rank summary metrics.

        Groups kernels by category (communication/memory/compute) and phase
        (prefill/decode), then computes counts, times, percentages, and
        statistics for each group.

        Args:
            kernels: List of KernelEvent objects
            config: Configuration dictionary (tp_size, batch_size, etc.)

        Returns:
            Dictionary with all metrics for trace_summary.csv

        Example:
            >>> aggregator = TraceAggregator()
            >>> summary = aggregator.aggregate_per_rank(kernels, config)
            >>> summary['compute_time_pct']
            85.3
        """
        if not kernels:
            self.logger.warning("No kernels to aggregate")
            return self._empty_summary(config)

        # Initialize summary with configuration
        summary = config.copy()

        # Group kernels by category
        by_category = {'communication': [], 'memory': [], 'compute': []}
        for k in kernels:
            if k.category in by_category:
                by_category[k.category].append(k)

        # Group kernels by phase
        by_phase = {'prefill': [], 'decode': []}
        for k in kernels:
            if k.phase in by_phase:
                by_phase[k.phase].append(k)

        # Overall metrics
        total_time_us = sum(k.duration_us for k in kernels)
        summary['total_kernels'] = len(kernels)
        summary['total_time_us'] = total_time_us

        # Trace duration (wall-clock time from first to last kernel)
        if kernels:
            timestamps = [k.timestamp_us for k in kernels]
            durations = [k.duration_us for k in kernels]
            trace_start = min(timestamps)
            trace_end = max(ts + dur for ts, dur in zip(timestamps, durations))
            summary['trace_duration_us'] = trace_end - trace_start
        else:
            summary['trace_duration_us'] = 0.0

        # GPU idle time
        idle_time_us, idle_pct = self.compute_gpu_idle_time(kernels)
        summary['gpu_idle_time_us'] = idle_time_us
        summary['gpu_idle_pct'] = idle_pct

        # Category metrics
        for category, cat_kernels in by_category.items():
            prefix = category
            summary[f'{prefix}_count'] = len(cat_kernels)

            if cat_kernels:
                cat_time_us = sum(k.duration_us for k in cat_kernels)
                summary[f'{prefix}_time_us'] = cat_time_us
                summary[f'{prefix}_time_pct'] = (cat_time_us / total_time_us * 100) if total_time_us > 0 else 0.0

                # Statistics
                durations = [k.duration_us for k in cat_kernels]
                summary[f'{prefix}_mean_duration_us'] = np.mean(durations)
                summary[f'{prefix}_p99_duration_us'] = np.percentile(durations, 99)
            else:
                summary[f'{prefix}_time_us'] = 0.0
                summary[f'{prefix}_time_pct'] = 0.0
                summary[f'{prefix}_mean_duration_us'] = 0.0
                summary[f'{prefix}_p99_duration_us'] = 0.0

        # Phase metrics
        for phase, phase_kernels in by_phase.items():
            prefix = phase
            summary[f'{prefix}_count'] = len(phase_kernels)

            if phase_kernels:
                phase_time_us = sum(k.duration_us for k in phase_kernels)
                summary[f'{prefix}_time_us'] = phase_time_us
                summary[f'{prefix}_time_pct'] = (phase_time_us / total_time_us * 100) if total_time_us > 0 else 0.0

                # Mean duration
                durations = [k.duration_us for k in phase_kernels]
                summary[f'{prefix}_mean_duration_us'] = np.mean(durations)
            else:
                summary[f'{prefix}_time_us'] = 0.0
                summary[f'{prefix}_time_pct'] = 0.0
                summary[f'{prefix}_mean_duration_us'] = 0.0

        return summary

    def compute_gpu_idle_time(
        self,
        kernels: List[KernelEvent]
    ) -> Tuple[float, float]:
        """
        Compute GPU idle time using interval merging algorithm.

        This algorithm handles multi-stream parallelism by merging overlapping
        kernel execution intervals. The idle time is calculated as the gaps
        between merged intervals.

        Algorithm:
        1. Extract all kernel intervals: [(start_ts, end_ts), ...]
        2. Sort intervals by start time
        3. Merge overlapping intervals (handles parallel streams)
        4. Calculate: idle_time = total_time - sum(merged_intervals)

        Time complexity: O(n log n) for sorting
        Space complexity: O(n) for intervals

        Args:
            kernels: List of KernelEvent objects

        Returns:
            Tuple of (idle_time_us, idle_pct)

        Example:
            >>> aggregator = TraceAggregator()
            >>> kernels = [...]  # Some kernel events
            >>> idle_time, idle_pct = aggregator.compute_gpu_idle_time(kernels)
            >>> print(f"GPU idle: {idle_pct:.2f}%")
        """
        if not kernels:
            return 0.0, 0.0

        # Step 1: Extract intervals (start_time, end_time)
        intervals = [
            (k.timestamp_us, k.timestamp_us + k.duration_us)
            for k in kernels
        ]

        # Step 2: Sort by start time
        intervals.sort()

        # Step 3: Merge overlapping intervals
        merged = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                # No overlap: add new interval
                merged.append([start, end])
            else:
                # Overlap: extend current interval
                merged[-1][1] = max(merged[-1][1], end)

        # Step 4: Calculate idle time
        # Total time span from first kernel start to last kernel end
        total_time = intervals[-1][1] - intervals[0][0]

        # Busy time is sum of all merged intervals
        busy_time = sum(end - start for start, end in merged)

        # Idle time is the gaps
        idle_time = total_time - busy_time

        # Percentage
        idle_pct = (idle_time / total_time * 100) if total_time > 0 else 0.0

        return idle_time, idle_pct

    def aggregate_cross_rank(
        self,
        rank_summaries: List[Dict]
    ) -> Dict:
        """
        Aggregate metrics across multiple ranks.

        Computes mean, std, min, max for each metric across ranks.
        Also calculates load imbalance metrics.

        Args:
            rank_summaries: List of per-rank summary dictionaries

        Returns:
            Dictionary with cross-rank aggregate metrics

        Example:
            >>> aggregator = TraceAggregator()
            >>> rank_summaries = [...]  # List of per-rank summaries
            >>> cross_rank = aggregator.aggregate_cross_rank(rank_summaries)
            >>> cross_rank['load_imbalance_pct']
            5.2
        """
        if not rank_summaries:
            self.logger.warning("No rank summaries to aggregate")
            return {}

        # Extract common configuration (should be same across ranks)
        first = rank_summaries[0]
        result = {
            'model_name': first.get('model_name'),
            'tp_size': first.get('tp_size'),
            'ep_size': first.get('ep_size'),
            'dp_size': first.get('dp_size'),
            'batch_size': first.get('batch_size'),
            'input_len': first.get('input_len'),
            'output_len': first.get('output_len'),
            'num_ranks': len(rank_summaries),
        }

        # Metrics to aggregate
        metrics = [
            'total_kernels',
            'total_time_us',
            'gpu_idle_pct',
            'communication_time_pct',
            'communication_mean_duration_us',
            'memory_time_pct',
            'memory_mean_duration_us',
            'compute_time_pct',
            'compute_mean_duration_us',
            'prefill_time_pct',
            'decode_time_pct',
        ]

        # Compute mean, std, min, max for each metric
        for metric in metrics:
            values = [s.get(metric, 0) for s in rank_summaries]

            # Filter out None values
            values = [v for v in values if v is not None]

            if values:
                result[f'{metric}_mean'] = np.mean(values)
                result[f'{metric}_std'] = np.std(values)
                result[f'{metric}_min'] = np.min(values)
                result[f'{metric}_max'] = np.max(values)
            else:
                result[f'{metric}_mean'] = 0.0
                result[f'{metric}_std'] = 0.0
                result[f'{metric}_min'] = 0.0
                result[f'{metric}_max'] = 0.0

        # Compute load imbalance
        # Load imbalance = (max_time - min_time) / mean_time * 100
        total_times = [s.get('total_time_us', 0) for s in rank_summaries]
        total_times = [t for t in total_times if t > 0]

        if total_times:
            mean_time = np.mean(total_times)
            max_time = np.max(total_times)
            min_time = np.min(total_times)

            if mean_time > 0:
                result['load_imbalance_pct'] = (max_time - min_time) / mean_time * 100
            else:
                result['load_imbalance_pct'] = 0.0

            # Identify bottleneck rank
            result['max_rank_id'] = int(np.argmax(total_times))
            result['min_rank_id'] = int(np.argmin(total_times))
        else:
            result['load_imbalance_pct'] = 0.0
            result['max_rank_id'] = None
            result['min_rank_id'] = None

        # Category-specific load imbalance
        for category in ['communication', 'memory', 'compute']:
            times = [s.get(f'{category}_time_us', 0) for s in rank_summaries]
            times = [t for t in times if t > 0]

            if times:
                mean_time = np.mean(times)
                max_time = np.max(times)
                min_time = np.min(times)

                if mean_time > 0:
                    result[f'{category}_imbalance_pct'] = (max_time - min_time) / mean_time * 100
                else:
                    result[f'{category}_imbalance_pct'] = 0.0
            else:
                result[f'{category}_imbalance_pct'] = 0.0

        # Prefill/decode ratio
        prefill_pcts = [s.get('prefill_time_pct', 0) for s in rank_summaries]
        decode_pcts = [s.get('decode_time_pct', 0) for s in rank_summaries]

        if decode_pcts and np.mean(decode_pcts) > 0:
            ratios = [p / d if d > 0 else 0 for p, d in zip(prefill_pcts, decode_pcts)]
            result['prefill_decode_ratio_mean'] = np.mean(ratios)
            result['prefill_decode_ratio_std'] = np.std(ratios)
        else:
            result['prefill_decode_ratio_mean'] = 0.0
            result['prefill_decode_ratio_std'] = 0.0

        return result

    def _empty_summary(self, config: Dict) -> Dict:
        """Create empty summary with all metrics set to zero."""
        summary = config.copy()

        # Overall
        summary['total_kernels'] = 0
        summary['total_time_us'] = 0.0
        summary['trace_duration_us'] = 0.0
        summary['gpu_idle_time_us'] = 0.0
        summary['gpu_idle_pct'] = 0.0

        # Categories
        for category in ['communication', 'memory', 'compute']:
            summary[f'{category}_count'] = 0
            summary[f'{category}_time_us'] = 0.0
            summary[f'{category}_time_pct'] = 0.0
            summary[f'{category}_mean_duration_us'] = 0.0
            summary[f'{category}_p99_duration_us'] = 0.0

        # Phases
        for phase in ['prefill', 'decode']:
            summary[f'{phase}_count'] = 0
            summary[f'{phase}_time_us'] = 0.0
            summary[f'{phase}_time_pct'] = 0.0
            summary[f'{phase}_mean_duration_us'] = 0.0

        return summary


if __name__ == '__main__':
    # Simple test
    import sys
    from trace_parser import TraceParser
    from trace_utils import extract_config_from_trace_path
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python3 trace_aggregator.py <trace_file.trace.json.gz>")
        sys.exit(1)

    trace_file = sys.argv[1]

    # Parse trace
    parser = TraceParser()
    kernels, metadata = parser.parse(trace_file)

    # Extract config
    config = extract_config_from_trace_path(Path(trace_file))

    # Aggregate
    aggregator = TraceAggregator()
    summary = aggregator.aggregate_per_rank(kernels, config)

    print("\n" + "=" * 80)
    print("AGGREGATION RESULTS")
    print("=" * 80)

    # Configuration
    print("\nConfiguration:")
    print(f"  Model: {summary['model_name']}")
    print(f"  Parallelism: TP={summary['tp_size']} EP={summary['ep_size']} DP={summary['dp_size']}")
    print(f"  Batch: bs={summary['batch_size']} in={summary['input_len']} out={summary['output_len']}")
    print(f"  Rank: TP={summary['tp_rank']} EP={summary['ep_rank']}")

    # Overall
    print("\nOverall:")
    print(f"  Total kernels: {summary['total_kernels']}")
    print(f"  Total time: {summary['total_time_us']/1e6:.2f} s")
    print(f"  Trace duration: {summary['trace_duration_us']/1e6:.2f} s")
    print(f"  GPU idle: {summary['gpu_idle_pct']:.2f}%")

    # Category breakdown
    print("\nCategory breakdown:")
    for category in ['communication', 'memory', 'compute']:
        count = summary[f'{category}_count']
        time_pct = summary[f'{category}_time_pct']
        mean_dur = summary[f'{category}_mean_duration_us']
        p99_dur = summary[f'{category}_p99_duration_us']
        print(f"  {category:15s}: {count:5d} kernels, {time_pct:5.2f}% time, "
              f"mean={mean_dur:8.2f}us, p99={p99_dur:10.2f}us")

    # Phase breakdown
    print("\nPhase breakdown:")
    for phase in ['prefill', 'decode']:
        count = summary[f'{phase}_count']
        time_pct = summary[f'{phase}_time_pct']
        mean_dur = summary[f'{phase}_mean_duration_us']
        print(f"  {phase:15s}: {count:5d} kernels, {time_pct:5.2f}% time, mean={mean_dur:8.2f}us")

    # Validation: category percentages should sum to ~100%
    total_pct = sum(summary[f'{cat}_time_pct'] for cat in ['communication', 'memory', 'compute'])
    print(f"\nValidation: Category percentages sum to {total_pct:.2f}% (should be ~100%)")
