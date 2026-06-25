# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a shell-based distributed deployment system for SGLang inference services across multiple nodes using Pouch containers. It's designed for multi-node GPU clusters running large language models with tensor parallelism (TP), expert parallelism (EP), and data parallelism (DP).

## Script Execution Flow

### Quick Start Workflow

```bash
# 1. Deploy containers on 4 nodes
./run_container.sh --cur-node 4 --master-ip 11.139.21.81

# 2. Start SGLang server
./run_server.sh --master-ip 11.139.21.81 --command "python -m sglang.launch_server --port 34567 --tp-size 16 --ep-size 8 --nnodes 4"

# 3. Run benchmarks
./run_client.sh --nnodes 4 --input-len 2048 --output-len 512
# OR automated matrix testing:
./run_benchmark.sh

# 4. Analyze results
python3 analysis/export_to_csv.py
cd analysis && jupyter nbconvert --to notebook --execute benchmark_visualization.ipynb

# 5. Cleanup
./cleanup_containers.sh
```

### Detailed Script Flow

#### 1. `run_container.sh` - Container Deployment
```bash
./run_container.sh --cur-node 4 --master-ip 11.139.21.81
```

**What it does:**
- Checks GPU availability on nodes from `ip_list` (priority order)
- Selects N nodes with free GPUs, avoiding conflicts with existing deployments
- Launches Pouch containers: `lsw_sglang_benchmark_rank{0,1,2,3}`
- Mounts workspace/model paths (identical inside/outside container)
- Configures SSH keys between containers
- Saves node list to `tmp/nodelist_<master_ip>` for parallel deployments

**Key features:**
- **Parallel deployments**: Different master IPs = independent deployments on different node sets
- **Conflict detection**: Automatically skips nodes used by other deployments
- **Path consistency**: `/cpfs01/user/...` same inside and outside containers

#### 2. `run_server.sh` - Server Launch
```bash
./run_server.sh --master-ip 11.139.21.81 --command "python -m sglang.launch_server --port 34567 --tp-size 16 --ep-size 8 --dp-size 16 --nnodes 4"
```

**What it does:**
- Reads node list from `tmp/nodelist_<master_ip>`
- Cleans previous processes (`pkill sglang`, `pkill python`)
- Parses command to extract TP/EP/DP/port
- **Auto-augments** missing parameters:
  - `--model-path` from config.yaml
  - `--host 0.0.0.0`
  - `--node-rank {0,1,2,3}` per node
  - `--dist-init-addr <master_ip>:31000` if not specified
  - `--chunked-prefill-size` (auto-calculated: base_size × DP)
- Launches server on all nodes via SSH + pouch exec
- Polls `/health` endpoint until ready (600s timeout)

**Logs:**
- Master (rank 0): `log/worker/MMDD_hhmm_TP{X}EP{Y}DP{Z}_node{IP}_rank0.log`
- Workers (rank 1+): `log/worker/MMDD_hhmm_TP{X}EP{Y}DP{Z}_node{IP}_rank{N}.log`

#### 3. `run_client.sh` - Single Benchmark
```bash
./run_client.sh --nnodes 4 --bs-per-rank 256 --input-len 2048 --output-len 512 --save-result
```

**What it does:**
- Calculates batch size: `bs_per_rank × nnodes × gpus_per_node`
- Runs `python3 -m sglang.bench_one_batch_server` in master container
- Saves results to `result/benchmarks/<model>-TP{X}-EP{Y}-DP{Z}/bs{N}_in{INPUT}_out{OUTPUT}/`

#### 4. `run_benchmark.sh` / `benchmark.py` - Automated Testing
```bash
./run_benchmark.sh --base-url http://11.139.21.81:34567 --skip-existing
```

**What it does:**
- Loads test matrix from config.yaml (batch_sizes × input_lengths)
- For each combination:
  - Checks if result exists (skip if `--skip-existing`)
  - Monitors queue depth via `/get_model_info`
  - Skips larger batch sizes if queue exceeds threshold
  - Calls `run_client.sh` with parameters
- **Smart queue management**: Prevents server overload
- **Resume capability**: Incremental testing

#### 5. `cleanup_containers.sh` - Cleanup
```bash
./cleanup_containers.sh
```

**What it does:**
- Iterates through ALL nodes in `ip_list`
- Stops/removes containers: `lsw_sglang_benchmark*`
- Kills processes: `sglang.launch_server`, `python.*launch_server`
- Removes temp files: `tmp/nodelist_*`

### Key Design Principles

- **Pure shell scripts**: Simple, debuggable (vs complex Python orchestration)
- **Parallel deployments**: Use different `--master-ip` for independent server groups
- **Auto-augmentation**: `run_server.sh` adds missing SGLang parameters automatically
- **Remote execution**: All operations via `ssh_util.py` (SSH + pouch exec)
- **Conflict avoidance**: Deployment tracking prevents node conflicts

### Node Selection Strategy

**How `run_container.sh` selects nodes:**

1. **Priority-based selection**: Nodes are selected from `ip_list` in order (first nodes in list have priority)
2. **Master node first**: The specified `--master-ip` is always selected as rank 0
3. **Fill remaining ranks**: Selects next available nodes from `ip_list` to fill ranks 1, 2, 3...
4. **Conflict detection**: Checks `tmp/nodelist_*` files to detect existing deployments and skips occupied nodes

**Important for parallel deployments:**

When launching multiple deployments in parallel, they may select overlapping nodes because conflict detection only sees deployments that completed BEFORE the current one starts. To avoid conflicts:

**Option 1: Sequential deployment** (safest)
```bash
# Deploy and wait for each to complete before starting next
./run_container.sh --cur-node 4 --master-ip 11.139.21.81  # Waits to complete
./run_container.sh --cur-node 4 --master-ip 11.139.21.82  # Detects server 1, avoids its nodes
./run_container.sh --cur-node 4 --master-ip 11.139.21.88  # Detects servers 1 & 2
```

**Option 2: Manual node allocation** (for parallel deployment)
```bash
# Manually ensure master IPs are far apart in ip_list to minimize overlap
# Example: Use nodes 0, 5, 10, 15 as masters
./run_container.sh --cur-node 4 --master-ip 11.139.21.81  # Selects: 81, 86, 87, 77
./run_container.sh --cur-node 4 --master-ip 11.139.21.97  # Selects: 97, 81, 86, 87 (may overlap!)
./run_container.sh --cur-node 4 --master-ip 11.139.21.96  # Selects: 96, 81, 86, 87 (may overlap!)
```

**Best practice**: Deploy sequentially or ensure you have enough nodes (N deployments × 4 nodes/deployment = total nodes needed)

## Configuration

### config.yaml Structure

```yaml
# Node topology
max_nodes: 15           # Total available nodes
current_nodes: 4        # Nodes to use for this run
gpus_per_node: 4        # GPUs per node
ip_list: [...]          # Node IPs in priority order

# Paths (absolute, must match inside/outside containers)
workspace_path: "/cpfs01/user/nebula_model/sjq-workspace/benchmark"
model_path: "/cpfs01/user/nebula_model/llm_weight"
model_name: "Qwen3-Coder-480B-A35B-Instruct-FP8"

# Container settings
image: "mirrors-ssl.aliyuncs.com/lmsysorg/sglang:v0.5.6-cu129-arm64"

# NCCL environment (passed to containers)
nccl_env: {...}

# SGLang defaults (augmented into server command)
sglang_defaults:
  max_dispatch_tokens_per_rank: 256
  mem_fraction_static: 0.8
  chunked_prefill_base_size: 16384  # Base size for chunked prefill (auto-calculated: chunked-prefill-size = base * DP)
  ...

# Profiling
profiler_config:
  enable_torch_profiler: true  # PyTorch profiler via HTTP API
  torch_profiler_output_path: "./result/traces"
  # Note: nsys commands are provided directly in run_server.sh --command parameter
```

### Critical Configuration Notes

- **ip_list**: Order matters - nodes listed first are selected first when launching containers
- **workspace_path**: Must be absolute and accessible on all nodes
- **mem_fraction_static**: Typically set to 0.8 for memory management
- **chunked_prefill_base_size**: Base size for chunked prefill (default: 16384). The actual `--chunked-prefill-size` is auto-calculated by run_server.sh:
  - If `--deepep-mode low_latency`: `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK * DP_SIZE`
  - Otherwise: `chunked_prefill_base_size * DP_SIZE`

### Automated Benchmark Configuration

The `benchmark` section in config.yaml controls automated matrix testing:

```yaml
benchmark:
  # Test matrix (all combinations will be tested)
  test_matrix:
    batch_sizes: [1, 4, 16, 32, 64, 128, 256]
    input_lengths: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

  # Test parameters
  test_params:
    dataset: "random-ids"
    output_len: 512
    warmup_requests: 3

  # Queue monitoring
  queue_limits:
    max_queue_requests: 40  # Skip larger batch sizes if exceeded

  # Behavior
  behavior:
    skip_existing: true  # Resume capability
    sleep_between_tests: 2
    test_timeout: 3600

  # Profiler
  profiler:
    enable: false
    use_http_api: true
```

### Understanding TP, EP, DP Configurations

**CRITICAL:** Configurations with the same GPU count can have vastly different performance characteristics:

- **TP16-EP16-DP1**:
  - Total GPUs: max(16, 16, 1) = 16
  - Attention TP: 16/1 = 16 (high attention parallelism)
  - MoE TP: 16/16 = 1 (no MoE parallelism)

- **TP16-EP16-DP16**:
  - Total GPUs: max(16, 16, 16) = 16
  - Attention TP: 16/16 = 1 (no attention parallelism)
  - MoE TP: 16/16 = 1 (no MoE parallelism)

These represent fundamentally different parallelism strategies and will show significantly different performance profiles despite using the same number of GPUs.

## Common Commands

### Launch Containers
```bash
# Single deployment on 4 nodes
./run_container.sh --cur-node 4 --master-ip 11.139.21.81

# Parallel deployments (different master IPs = independent deployments)
./run_container.sh --cur-node 4 --master-ip 11.139.21.81  # Deployment 1
./run_container.sh --cur-node 4 --master-ip 11.139.21.86  # Deployment 2
```

### Start Server
```bash
# Basic server (script auto-adds --model-path, --host, --node-rank, --dist-init-addr)
./run_server.sh --master-ip 11.139.21.81 --command "python -m sglang.launch_server \
  --port 34567 --tp-size 16 --ep-size 8 --dp-size 16 --nnodes 4 \
  --attention-backend trtllm_mha --enable-dp-attention"

# With nsys profiling (provide complete nsys command)
./run_server.sh --master-ip 11.139.21.81 --command "nsys profile --trace=cuda,nvtx ... python -m sglang.launch_server ..."

# Parallel servers (different master IPs)
./run_server.sh --master-ip 11.139.21.81 --command "..."  # Server 1
./run_server.sh --master-ip 11.139.21.86 --command "..."  # Server 2
```

### Run Benchmark
```bash
# Default benchmark
./run_client.sh --nnodes 4 --input-len 2000 --output-len 100

# Custom batch size and save results
./run_client.sh --nnodes 4 --bs-per-rank 512 --input-len 4096 --output-len 512 --save-result

# Batch size calculation: batch_size = tokens_per_rank × num_nodes × gpus_per_node
```

### Automated Benchmark Matrix Testing
```bash
# Run automated tests across all batch sizes and input lengths from config.yaml
./run_benchmark.sh

# With custom server URL
./run_benchmark.sh --base-url http://11.139.21.81:34567

# Enable profiler and skip existing results (resume capability)
./run_benchmark.sh --enable-profiler --skip-existing

# Python script for more control
python3 benchmark.py --config config.yaml --verbose
```

**Key Features:**
- Dual-loop testing: Tests all combinations of batch_sizes × input_lengths from config.yaml
- Intelligent queue monitoring: Skips larger batch sizes if queue limit exceeded
- Resume capability: Skips tests with existing results when using `--skip-existing`
- Optional profiler integration via HTTP API
- Organized output: `result/benchmarks/<model>-TP{X}-EP{Y}-DP{Z}/bs{N}_in{INPUT}_out{OUTPUT}/`

### Profiling
```bash
# One-stop profiling script (recommended)
./run_profile_benchmark.sh --server http://11.139.21.81:34567 \
  --nnodes 4 --input-len 800 --output-len 100

# Manual profiling
./profiler_util.sh start --server http://11.139.21.81:34567
./run_client.sh --nnodes 4 --input-len 800 --output-len 100
./profiler_util.sh stop --server http://11.139.21.81:34567 \
  --bs 256 --input-len 800 --output-len 100
```

### Cleanup
```bash
# Clean all containers and processes on ALL nodes in ip_list
./cleanup_containers.sh
```

### Check Node Status
```bash
# Check GPU availability
python3 ssh_util.py check_gpu 11.139.21.81 --gpus 4

# Execute command on node
python3 ssh_util.py exec_on_node 11.139.21.81 "nvidia-smi"

# Execute command in container
python3 ssh_util.py exec_in_container 11.139.21.81 lsw_sglang_benchmark_rank0 "ps aux"

# Check running containers
for node in 11.139.21.81 11.139.21.86 11.139.21.87 11.139.21.77; do
  echo "=== Node $node ==="
  python3 ssh_util.py exec_on_node $node "echo 'Alibaba@12#\$' | sudo -S pouch ps"
done
```

## Data Analysis and Visualization

### Export Benchmark Results to CSV

After running benchmarks, export results to CSV for analysis:

```bash
# First-time export (processes all JSON files in result/benchmarks/)
cd /cpfs01/user/nebula_model/sjq-workspace/benchmark
python3 analysis/export_to_csv.py

# Incremental update (only processes new files)
python3 analysis/export_to_csv.py

# Force rebuild (reprocess all files)
python3 analysis/export_to_csv.py --force

# Export only summary data (faster, smaller files)
python3 analysis/export_to_csv.py --summary-only

# Verbose mode with detailed logging
python3 analysis/export_to_csv.py --verbose
```

**Outputs:**
- `analysis/data/benchmark_summary.csv` - Aggregate metrics (37 columns):
  - Configuration: tp_size, ep_size, dp_size, nnodes, batch_size, input_len, output_len
  - Throughput: request_throughput, input_throughput, output_throughput, total_throughput
  - Latency: mean/median/p99 for E2E latency, TTFT, TPOT, ITL
  - Statistics: duration, completed requests, concurrency
- `analysis/data/benchmark_requests.csv` - Per-request details (11 columns):
  - Configuration identifiers
  - Request-level metrics: request_id, actual_input_len, actual_output_len, ttft_ms
- `analysis/data/.processed_files.txt` - Automatic deduplication tracker
- `analysis/data/export.log` - Export process logs

**Features:**
- Automatic deduplication: Only processes new JSON files
- Batch processing: Handles large datasets efficiently
- Error handling: Skips corrupt files, logs issues
- Resume capability: Incremental updates without reprocessing

### Visualize Benchmark Results

Generate performance charts from exported CSV data:

```bash
cd analysis

# Generate all visualizations (creates executed notebook + figures)
jupyter nbconvert --to notebook --execute benchmark_visualization.ipynb

# Or run interactively
jupyter notebook benchmark_visualization.ipynb
```

**Generated Figures** (saved to `analysis/figures/`):
- `throughput_by_batch_size.png` - Throughput scaling across batch sizes
- `throughput_by_config.png` - Configuration comparison
- `throughput_by_input_len.png` - Impact of input length on throughput
- `throughput_heatmap.png` - Batch size × Input length heatmap
- `ttft_analysis.png` - Time to First Token (TTFT) analysis
- `tpot_analysis.png` - Time Per Output Token (TPOT) analysis
- `e2e_latency_analysis.png` - End-to-end latency distribution
- `gpu_scaling_analysis.png` - GPU scaling efficiency
- `performance_radar.png` - Multi-metric radar chart comparison
- `composite_score.png` - Overall performance scores

**Requirements:**
```bash
pip install -r analysis/requirements.txt  # pandas, matplotlib, seaborn, jupyter
```

See `analysis/README.md` and `analysis/VISUALIZATION_GUIDE.md` for detailed documentation.

### Analyze Results with Pandas

```python
import pandas as pd

# Load exported data
df_summary = pd.read_csv('analysis/data/benchmark_summary.csv')
df_requests = pd.read_csv('analysis/data/benchmark_requests.csv')

# Example: Find best configurations by throughput
best = df_summary.nlargest(10, 'output_throughput')[
    ['tp_size', 'ep_size', 'dp_size', 'batch_size', 'input_len', 'output_throughput']
]
print(best)

# Example: Analyze batch size impact
batch_analysis = df_summary.groupby('batch_size')['output_throughput'].agg([
    'mean', 'std', 'min', 'max', 'count'
])
print(batch_analysis)
```

## File Locations

### Logs
- **Master**: `log/master/YYMMDDHHmm.log` (rank 0 server output)
- **Workers**: `log/worker/rank{N}_YY_MM_DD_HH_mm.log` (rank 1+ server output)

### Results
- **Benchmarks**: `result/benchmarks/Qwen3-Coder-480B-FP8-TP{X}-EP{Y}-DP{Z}/bs{N}_in{INPUT}_out{OUTPUT}/benchmark_YYYYMMDD_HHMMSS.txt`
- **Traces**: `result/traces/Qwen3-Coder-480B-FP8-TP{X}-EP{Y}-DP{Z}/bs{N}_in{INPUT}_out{OUTPUT}/<timestamp>-TP-{rank}-EP-{rank}.trace.json.gz`
- **Nsys Traces**: `result/nsys_traces/rank{N}_<timestamp>.nsys-rep`

### Analysis Outputs
- **CSV Data**: `analysis/data/benchmark_summary.csv`, `analysis/data/benchmark_requests.csv`
- **Figures**: `analysis/figures/*.png` (throughput, latency, scaling visualizations)
- **Logs**: `analysis/data/export.log` (CSV export process logs)
- **Tracking**: `analysis/data/.processed_files.txt` (deduplication tracker)

### Temporary Files
- **Node List**: `tmp/sglang_nodes.txt` (created by run_container.sh, read by run_server.sh)


## Important Implementation Constraints

### Node-Rank Parameter
The `--node-rank` parameter is CRITICAL for multi-node distributed deployments:
- Must be unique per node (0, 1, 2, ...)
- Automatically added by `run_server.sh` based on node index
- User should NEVER manually add `--node-rank` to the command

### Path Consistency
Containers mount directories with identical paths inside/outside:
```bash
-v /cpfs01/user/nebula_model/sjq-workspace/benchmark:/cpfs01/user/nebula_model/sjq-workspace/benchmark
-v /cpfs01/user/nebula_model/llm_weight:/cpfs01/user/nebula_model/llm_weight
```
This avoids path resolution errors in distributed deployments.

### Permission Management
Log/result directories are created by scripts with the actual user's permissions (not root), even though containers run as root. This avoids permission issues when accessing files from host.

## Troubleshooting Common Issues

### "Not enough available nodes"
1. Run `./cleanup_containers.sh` to clean stale containers
2. Check GPU availability: `python3 ssh_util.py check_gpu <node_ip>`
3. Verify nodes are reachable: `ssh henry.sjq@<node_ip> nvidia-smi`

### Server timeout during launch
1. Check master log: `tail -f log/master/$(ls -t log/master/ | head -1)`
2. Check worker logs: `tail -f log/worker/rank*.log`
3. Common issues:
   - NCCL initialization hanging (check `dist-init-addr`)
   - Model loading errors (check `model_path`)
   - GPU memory issues (reduce `mem_fraction_static`)

### Connection refused during benchmark
1. Verify server health: `curl http://11.139.21.81:34567/health`
2. Check port matches server command
3. Verify containers are running: `python3 ssh_util.py exec_on_node <node_ip> "echo 'Alibaba@12#\$' | sudo -S pouch ps"`

### Permission denied on log files
```bash
sudo chown -R $(whoami):$(id -gn) log/ result/
```

## Development Guidelines

### Modifying guideline.md
The `guideline.md` file contains the original requirements (in Chinese). Any modifications to this file must be approved by the user first.

### Adding New Features
When adding features to the scripts:
1. Maintain the simple shell-based architecture
2. Use `ssh_util.py` for all remote operations
3. Add new parameters to `config.yaml` with sensible defaults
4. Update `doc/USAGE.md` with usage examples
5. Ensure proper error handling and user feedback

### Testing Changes
Always test with a minimal deployment first:
```bash
# Single-node test
./cleanup_containers.sh
./run_container.sh --cur-node 1 --master-ip 11.139.21.81
./run_server.sh --command "python -m sglang.launch_server --port 34567 --tp-size 4 --nnodes 1"
./run_client.sh --nnodes 1 --input-len 800 --output-len 100
```

## Differences from NVL72-GB200

This is a simplified version of the NVL72-GB200 orchestrator:

| Aspect | NVL72-GB200 | This Implementation |
|--------|-------------|---------------------|
| Language | Python orchestrator | Pure shell scripts |
| Coordination | Signal files, state machines | Direct execution |
| Profiling | Fully integrated | Manual/separate scripts |
| TP/EP Iteration | Automatic sweeps | Single config per run |
| Complexity | High (multi-layer orchestration) | Low (direct commands) |

The shell-based approach trades automation for simplicity and debuggability.
