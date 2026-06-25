#!/usr/bin/env python3
"""
Trace Parser

Parse PyTorch profiler trace files (Chrome Trace Format) and extract kernel events.
"""

import gzip
import json
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class KernelEvent:
    """Represents a single GPU kernel execution."""
    # Kernel identification
    name: str
    category: str  # 'communication', 'memory', 'compute'
    phase: str     # 'prefill', 'decode'

    # Timing
    timestamp_us: float
    duration_us: float

    # Execution context
    device: int
    stream: int
    correlation: int

    # Kernel configuration
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]

    # Performance metrics
    occupancy_pct: float
    registers_per_thread: int
    shared_memory_bytes: int


class TraceParser:
    """
    Parse Chrome Trace Format files from PyTorch profiler.

    Memory optimization: Only retains ~33K events (kernel + cuda_runtime)
    out of 12.9M total events, reducing memory usage by 99.75%.
    """

    # Kernel classification patterns (case-insensitive)
    # Support both underscore and non-underscore variants (e.g., allreduce and all_reduce)
    COMM_PATTERNS = r'nccl|all_?reduce|all_?gather|all_?to_?all|broadcast|reduce_?scatter|deep_?ep|p2p|send|recv'
    MEM_PATTERNS = r'mem'

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse(self, trace_file_path: str) -> Tuple[List[KernelEvent], Dict]:
        """
        Parse trace file and extract kernel events.

        Args:
            trace_file_path: Path to .trace.json.gz file

        Returns:
            Tuple of (kernel_events, metadata):
            - kernel_events: List of KernelEvent objects
            - metadata: Dictionary with trace metadata

        Raises:
            json.JSONDecodeError: If trace file is malformed
            gzip.BadGzipFile: If trace file is corrupt

        Example:
            >>> parser = TraceParser()
            >>> kernels, metadata = parser.parse('trace.json.gz')
            >>> len(kernels)
            12975
        """
        self.logger.info(f"Parsing trace file: {trace_file_path}")

        # Load gzipped JSON
        try:
            with gzip.open(trace_file_path, 'rt') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {trace_file_path}: {e}")
            raise
        except gzip.BadGzipFile as e:
            self.logger.error(f"Corrupt gzip file {trace_file_path}: {e}")
            raise

        # Extract metadata
        metadata = {
            'schema_version': data.get('schemaVersion'),
            'display_time_unit': data.get('displayTimeUnit', 'ms'),
            'device_properties': data.get('deviceProperties', []),
            'distributed_info': data.get('distributedInfo', {}),
        }

        # Extract events array
        events = data.get('traceEvents', [])
        total_events = len(events)
        self.logger.info(f"Total events in trace: {total_events}")

        # Filter events (memory optimization: discard 12M+ events)
        kernel_events_raw, cuda_rt_events = self._filter_events(events)
        self.logger.info(
            f"Filtered events: {len(kernel_events_raw)} kernel, "
            f"{len(cuda_rt_events)} cuda_runtime "
            f"(kept {len(kernel_events_raw) + len(cuda_rt_events)}/{total_events}, "
            f"{(len(kernel_events_raw) + len(cuda_rt_events))/total_events*100:.2f}%)"
        )

        # Build correlation map: correlation_id -> launch_method
        corr_map = self._build_correlation_map(cuda_rt_events)
        self.logger.debug(f"Built correlation map with {len(corr_map)} entries")

        # Process kernel events
        kernel_events = []
        for event in kernel_events_raw:
            try:
                kernel = self._process_kernel_event(event, corr_map)
                kernel_events.append(kernel)
            except Exception as e:
                self.logger.warning(f"Failed to process kernel event: {e}")
                continue

        self.logger.info(f"Successfully parsed {len(kernel_events)} kernel events")
        return kernel_events, metadata

    def _filter_events(
        self,
        events: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Single-pass filter to extract kernel and cuda_runtime events.

        This is the key memory optimization: we discard 93.7% of events
        (python_function, cpu_op) that are not needed for kernel analysis.

        Args:
            events: List of all events from trace file

        Returns:
            Tuple of (kernel_events, cuda_runtime_events)
        """
        kernel_events = []
        cuda_rt_events = []

        for event in events:
            cat = event.get('cat')
            if cat == 'kernel':
                kernel_events.append(event)
            elif cat == 'cuda_runtime':
                cuda_rt_events.append(event)
            # Discard all other events (python_function, cpu_op, etc.)

        return kernel_events, cuda_rt_events

    def _build_correlation_map(self, cuda_rt_events: List[Dict]) -> Dict[int, str]:
        """
        Build mapping from correlation ID to launch method name.

        The correlation field links kernel events to their corresponding
        cuda_runtime launch events. This allows us to determine whether
        a kernel was launched via cudaLaunchKernel (prefill) or
        cudaGraphLaunch (decode).

        Args:
            cuda_rt_events: List of cuda_runtime events

        Returns:
            Dictionary mapping correlation ID to launch method name
        """
        corr_map = {}

        for event in cuda_rt_events:
            # Extract correlation ID
            args = event.get('args', {})
            correlation = args.get('correlation')
            if correlation is None:
                continue

            # Extract launch method from event name
            name = event.get('name', '')
            corr_map[correlation] = name

        return corr_map

    def _process_kernel_event(
        self,
        event: Dict,
        corr_map: Dict[int, str]
    ) -> KernelEvent:
        """
        Process a single kernel event into KernelEvent object.

        Args:
            event: Raw kernel event dictionary
            corr_map: Correlation map from _build_correlation_map

        Returns:
            KernelEvent object
        """
        # Basic fields
        name = event.get('name', '')
        timestamp_us = event.get('ts', 0.0)
        duration_us = event.get('dur', 0.0)

        # Args
        args = event.get('args', {})
        device = args.get('device', 0)
        stream = args.get('stream', 0)
        correlation = args.get('correlation', 0)

        # Grid and block dimensions
        grid = args.get('grid', [1, 1, 1])
        block = args.get('block', [1, 1, 1])
        if not isinstance(grid, list) or len(grid) != 3:
            grid = [1, 1, 1]
        if not isinstance(block, list) or len(block) != 3:
            block = [1, 1, 1]

        # Performance metrics
        occupancy_pct = args.get('est. achieved occupancy %', 0.0)
        registers_per_thread = args.get('registers per thread', 0)
        shared_memory_bytes = args.get('shared memory', 0)

        # Classify kernel
        category = self._classify_kernel(name)
        phase = self._detect_phase(correlation, corr_map)

        return KernelEvent(
            name=name,
            category=category,
            phase=phase,
            timestamp_us=timestamp_us,
            duration_us=duration_us,
            device=device,
            stream=stream,
            correlation=correlation,
            grid=tuple(grid),
            block=tuple(block),
            occupancy_pct=occupancy_pct,
            registers_per_thread=registers_per_thread,
            shared_memory_bytes=shared_memory_bytes,
        )

    def _classify_kernel(self, kernel_name: str) -> str:
        """
        Classify kernel into communication, memory, or compute category.

        Classification rules (case-insensitive):
        - Communication: Contains nccl, allreduce, allgather, broadcast, etc.
        - Memory: Contains 'mem'
        - Compute: Everything else

        Args:
            kernel_name: Full kernel name

        Returns:
            Category string: 'communication', 'memory', or 'compute'

        Example:
            >>> parser = TraceParser()
            >>> parser._classify_kernel('ncclDevKernel_AllReduce')
            'communication'
            >>> parser._classify_kernel('cudaMemcpyAsync')
            'memory'
            >>> parser._classify_kernel('at::native::elementwise_kernel')
            'compute'
        """
        name_lower = kernel_name.lower()

        # Check communication patterns
        if re.search(self.COMM_PATTERNS, name_lower):
            return 'communication'

        # Check memory patterns
        if re.search(self.MEM_PATTERNS, name_lower):
            return 'memory'

        # Default to compute
        return 'compute'

    def _detect_phase(self, correlation: int, corr_map: Dict[int, str]) -> str:
        """
        Detect whether kernel is in prefill or decode phase.

        Phase detection:
        - Decode: Launch method contains 'graph' (cudaGraphLaunch)
        - Prefill: All other launch methods (cudaLaunchKernel, etc.)

        Args:
            correlation: Correlation ID from kernel event
            corr_map: Correlation map from _build_correlation_map

        Returns:
            Phase string: 'prefill' or 'decode'

        Example:
            >>> parser = TraceParser()
            >>> corr_map = {123: 'cudaLaunchKernel', 456: 'cudaGraphLaunch'}
            >>> parser._detect_phase(123, corr_map)
            'prefill'
            >>> parser._detect_phase(456, corr_map)
            'decode'
        """
        if correlation in corr_map:
            launch_method = corr_map[correlation]
            if 'graph' in launch_method.lower():
                return 'decode'

        # Default to prefill if correlation not found or not a graph launch
        return 'prefill'


if __name__ == '__main__':
    # Simple test
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python3 trace_parser.py <trace_file.trace.json.gz>")
        print("\nExample:")
        print("  python3 trace_parser.py result/traces/Qwen3-480B-TP16-EP8-DP1/bs32_in2048_output512/1766401812.8294094-TP-0-EP-0.trace.json.gz")
        sys.exit(1)

    trace_file = sys.argv[1]

    # Parse trace file
    parser = TraceParser()
    try:
        kernels, metadata = parser.parse(trace_file)

        print("\n" + "=" * 80)
        print("TRACE PARSING RESULTS")
        print("=" * 80)

        # Summary statistics
        print(f"\nTotal kernels: {len(kernels)}")

        # Category breakdown
        categories = {}
        for k in kernels:
            categories[k.category] = categories.get(k.category, 0) + 1

        print("\nCategory breakdown:")
        for cat, count in sorted(categories.items()):
            pct = count / len(kernels) * 100
            print(f"  {cat:15s}: {count:6d} ({pct:5.2f}%)")

        # Phase breakdown
        phases = {}
        for k in kernels:
            phases[k.phase] = phases.get(k.phase, 0) + 1

        print("\nPhase breakdown:")
        for phase, count in sorted(phases.items()):
            pct = count / len(kernels) * 100
            print(f"  {phase:15s}: {count:6d} ({pct:5.2f}%)")

        # Top 10 longest kernels
        top_kernels = sorted(kernels, key=lambda k: k.duration_us, reverse=True)[:10]
        print("\nTop 10 longest kernels:")
        for i, k in enumerate(top_kernels, 1):
            print(f"  {i:2d}. {k.duration_us:10.2f} us | {k.category:13s} | {k.phase:7s} | {k.name[:60]}")

        # Metadata
        print("\nMetadata:")
        print(f"  Display time unit: {metadata['display_time_unit']}")
        print(f"  Devices: {len(metadata['device_properties'])}")
        dist_info = metadata['distributed_info']
        if dist_info:
            print(f"  Distributed: rank={dist_info.get('rank')}, world_size={dist_info.get('world_size')}")

    except Exception as e:
        logging.error(f"Failed to parse trace: {e}", exc_info=True)
        sys.exit(1)
