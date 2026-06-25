# SGLang Distributed Deployment - Usage Guide

This guide explains how to use the simplified SGLang distributed deployment system for running inference services across multiple nodes with Pouch containers.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

The deployment system consists of 4 main scripts:

1. **run_container.sh** - Launch Pouch containers on multiple nodes
2. **run_server.sh** - Deploy distributed SGLang server across containers
3. **run_client.sh** - Run benchmark tests
4. **cleanup_containers.sh** - Clean up all containers and processes
5. **profiler_util.sh** (optional) - Control profiler for performance analysis

## Prerequisites

1. SSH access to all nodes in the cluster
2. Sudo password for pouch commands (default: `617178Sjq`)
3. Python 3 with PyYAML installed
4. Pouch container engine on all nodes
5. NVIDIA GPUs and drivers on all nodes
6. Model files available at the specified model_path
7. SSH keys configured (for inter-container communication)

## Configuration

### config.yaml

Edit `config.yaml` to match your environment:

```yaml
# Node Configuration
max_nodes: 4
current_nodes: 4
gpus_per_node: 4

# IP List (priority order)
ip_list:
  - "11.139.21.86"
  - "11.139.21.81"  # Typically the master
  - "11.139.21.87"
  - "11.139.21.77"

master_ip: "11.139.21.81"

# Container Configuration
image: "hub.docker.alibaba-inc.com/tre-ai-infra/sglang:v0.5.2-cu129-gb200"

# Path Configuration
workspace_path: "/cpfs01/user/nebula_model/sjq-workspace/benchmark"
model_path: "/cpfs01/user/nebula_model/llm_weight"
model_name: "Qwen3-Coder-480B-A35B-Instruct-FP8"

# Output Paths (relative to workspace_path)
log_path: "./log"
result_path: "./result"
profiler_path: "./result/traces"

# NCCL Environment Variables
nccl_env:
  SGLANG_LOCAL_IP_NIC: "eth0"
  GLOO_SOCKET_IFNAME: "eth0"
  # ... (add all NCCL settings)
```

### Key Configuration Parameters

- **ip_list**: List of node IPs in priority order (nodes listed first will be selected first)
- **master_ip**: IP of the master node (should be in ip_list)
- **workspace_path**: Absolute path to workspace (must be same inside/outside containers)
- **model_path**: Absolute path to model directory
- **nccl_env**: NCCL environment variables for distributed GPU communication

## Quick Start

### Basic 4-Node Deployment

```bash
# 1. Configure (edit config.yaml if needed)
vim config.yaml

# 2. Launch containers on 4 nodes
./run_container.sh --cur-node 4 --master-ip 11.139.21.81

# 3. Start distributed server
./run_server.sh --command "python -m sglang.launch_server \
  --port 30000 \
  --tp-size 16 \
  --ep-size 16 \
  --nnodes 4 \
  --dist-init-addr 11.139.21.81:31000 \
  --attention-backend trtllm_mha \
  --enable-dp-attention"

# 4. Run benchmark
./run_client.sh --tokens-per-rank 256 --num-nodes 4 \
  --input-len 2000 --output-len 100

# 5. Clean up when done
./cleanup_containers.sh
```

## Detailed Usage

### 1. Launching Containers

**Command:**
```bash
./run_container.sh --cur-node N --master-ip IP [--config FILE]
```

**Options:**
- `--cur-node N`: Number of nodes to use
- `--master-ip IP`: IP address of the master node
- `--config FILE`: Configuration file (default: config.yaml)

**What it does:**
1. Checks GPU availability on all nodes from ip_list
2. Selects N nodes with free GPUs (prioritizes by ip_list order)
3. Launches Pouch containers with:
   - Workspace and model directories mounted
   - GPU access enabled
   - NCCL environment variables set
4. Configures SSH keys in containers
5. Creates log directories
6. Saves node list to `/tmp/sglang_nodes.txt`

**Output:**
```
Selected nodes:
  Rank 0: 11.139.21.81 (container: sglang_benchmark_node0)
  Rank 1: 11.139.21.86 (container: sglang_benchmark_node1)
  Rank 2: 11.139.21.87 (container: sglang_benchmark_node2)
  Rank 3: 11.139.21.77 (container: sglang_benchmark_node3)
```

### 2. Starting the Server

**Command:**
```bash
./run_server.sh --command "SGLANG_COMMAND" [--config FILE]
```

**Options:**
- `--command "..."`: Base SGLang server launch command
- `--config FILE`: Configuration file (default: config.yaml)

**The command will be automatically augmented with:**
- `--model-path` (if not present)
- `--host 0.0.0.0` (if not present)
- `--node-rank N` (for each node)

**Example commands:**

Basic distributed setup:
```bash
./run_server.sh --command "python -m sglang.launch_server \
  --port 30000 \
  --tp-size 16 \
  --ep-size 16 \
  --nnodes 4 \
  --dist-init-addr 11.139.21.81:31000"
```

With advanced features:
```bash
./run_server.sh --command "python -m sglang.launch_server \
  --port 30000 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --nnodes 4 \
  --dist-init-addr 11.139.21.81:31000 \
  --attention-backend trtllm_mha \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --ep-dispatch-algorithm fake \
  --stream-out \
  --moe-a2a-backend deepep \
  --disable-cuda-graph"
```

**Logs:**
- Master: `log/master/YYMMDDHHmm.log`
- Workers: `log/worker/rank{N}_YY_MM_DD_HH_mm.log`

### 3. Running Benchmarks

**Command:**
```bash
./run_client.sh [OPTIONS]
```

**Options:**
- `--tokens-per-rank N`: Tokens per rank (default: 256)
- `--num-nodes N`: Number of nodes (default: auto-detected from node list)
- `--input-len N`: Input sequence length (default: 2000)
- `--output-len N`: Output sequence length (default: 100)
- `--base-url URL`: Server base URL (default: auto-detected)
- `--model-path PATH`: Model path for tokenizer (default: from config)
- `--config FILE`: Configuration file (default: config.yaml)

**Batch size calculation:**
```
batch_size = tokens_per_rank × num_nodes × gpus_per_node
```

**Examples:**

Basic benchmark:
```bash
./run_client.sh --tokens-per-rank 256 --num-nodes 4
```

Custom workload:
```bash
./run_client.sh \
  --tokens-per-rank 512 \
  --num-nodes 4 \
  --input-len 4096 \
  --output-len 512 \
  --base-url http://11.139.21.81:30000
```

### 4. Profiling (Optional)

**Start profiler:**
```bash
./profiler_util.sh start --server http://11.139.21.81:30000
```

**Run your workload** (e.g., run_client.sh)

**Stop profiler:**
```bash
./profiler_util.sh stop --server http://11.139.21.81:30000
```

**Check status:**
```bash
./profiler_util.sh status --server http://11.139.21.81:30000
```

**Trace files location:**
```
result/traces/model-TP-EP-DP/rank{TP-EP}/*.gz
```

Example: `result/traces/Qwen3-480B-TP16-EP16-DP1/rank0-0/trace.gz`

### 5. Cleanup

**Command:**
```bash
./cleanup_containers.sh [CONFIG_FILE]
```

**What it does:**
1. Stops and removes all sglang containers on ALL nodes (from ip_list)
2. Kills stray sglang and python processes
3. Cleans up local files (/tmp/sglang_nodes.txt)
4. Verifies cleanup

**Important:** This cleans ALL nodes in ip_list, not just active nodes.

## Troubleshooting

### Container Launch Issues

**Problem:** Not enough available nodes
```
Error: Not enough available nodes
  Requested: 4
  Available: 2
```

**Solutions:**
1. Check GPU availability: `python3 ssh_util.py check_gpu 11.139.21.81`
2. Clean up existing containers: `./cleanup_containers.sh`
3. Check if nodes are reachable: `ssh henry.sjq@11.139.21.81 "nvidia-smi"`

### Server Launch Issues

**Problem:** Server timeout
```
Timeout waiting for server (600s elapsed)
```

**Solutions:**
1. Check master log: `tail -f log/master/$(ls -t log/master/ | head -1)`
2. Check worker logs: `tail -f log/worker/rank*`
3. Verify containers are running: `python3 ssh_util.py exec_on_node 11.139.21.81 "echo 'Alibaba@12#\$' | sudo -S pouch ps"`
4. Increase timeout or check for errors in logs

### Client Benchmark Issues

**Problem:** Connection refused
```
Error: Connection refused to http://11.139.21.81:30000
```

**Solutions:**
1. Verify server is running: `curl http://11.139.21.81:30000/health`
2. Check server logs for errors
3. Verify port number matches server configuration
4. Check firewall settings

### Permission Issues

**Problem:** Log files owned by root
```
Permission denied: log/master/202512151730.log
```

**Solutions:**
1. Log directories are created with your user permissions
2. If containers create root-owned files, fix with:
   ```bash
   sudo chown -R $(whoami):$(id -gn) log/ result/
   ```

### SSH Issues

**Problem:** SSH key not configured
```
Warning: No SSH key found at ~/.ssh/id_rsa
```

**Solutions:**
1. Generate SSH key: `ssh-keygen -t rsa -b 4096`
2. Copy to nodes: `ssh-copy-id henry.sjq@11.139.21.81`
3. Rerun run_container.sh

## Advanced Usage

### Custom NCCL Configuration

Edit `config.yaml` nccl_env section:
```yaml
nccl_env:
  NCCL_DEBUG: "INFO"  # Set to "TRACE" for detailed debugging
  NCCL_IB_DISABLE: "0"  # Enable InfiniBand if available
  # Add custom NCCL settings
```

### Using Different Models

Update config.yaml:
```yaml
model_path: "/path/to/models"
model_name: "YourModel-Name"
```

Or override in run_client.sh:
```bash
./run_client.sh --model-path /path/to/models/YourModel-Name
```

### Multi-Workload Testing

Run multiple benchmarks in sequence:
```bash
for input_len in 1024 2048 4096 8192; do
  for output_len in 100 500 1000; do
    echo "Testing input=$input_len output=$output_len"
    ./run_client.sh --input-len $input_len --output-len $output_len
    sleep 5
  done
done
```

### Monitoring

**Check container status:**
```bash
for node in 11.139.21.81 11.139.21.86 11.139.21.87 11.139.21.77; do
  echo "=== Node $node ==="
  python3 ssh_util.py exec_on_node $node \
    "echo 'Alibaba@12#\$' | sudo -S pouch ps --filter name=sglang_benchmark"
done
```

**Monitor GPU usage:**
```bash
for node in 11.139.21.81 11.139.21.86 11.139.21.87 11.139.21.77; do
  echo "=== Node $node ==="
  ssh henry.sjq@$node nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
done
```

**Tail all logs:**
```bash
# Master log
tail -f log/master/$(ls -t log/master/ | head -1)

# All worker logs
tail -f log/worker/rank*.log
```

## File Structure

```
benchmark/
├── config.yaml                 # Configuration
├── run_container.sh           # Container launcher
├── run_server.sh              # Server launcher
├── run_client.sh              # Client benchmark
├── cleanup_containers.sh      # Cleanup script
├── profiler_util.sh           # Profiler control
├── ssh_util.py                # SSH utility
├── remote_utils.py            # Remote utilities (from NVL72-GB200)
├── log/                       # Logs
│   ├── master/                # Master node logs
│   └── worker/                # Worker node logs
├── result/                    # Results
│   ├── traces/                # Profiler traces
│   └── benchmarks/            # Benchmark results
└── doc/                       # Documentation
    └── USAGE.md               # This file
```

## Summary

The simplified SGLang deployment system provides:
- Automatic GPU availability checking
- Multi-node container orchestration
- Distributed server deployment
- Benchmark client with configurable workloads
- Optional profiling support
- Easy cleanup

For issues or questions, check the logs in `log/master/` and `log/worker/`.
