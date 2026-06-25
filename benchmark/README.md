# SGLang Distributed Deployment System

A simplified shell-based system for deploying SGLang inference services across multiple nodes using Pouch containers.

## Quick Start

```bash
# 1. Configure your environment
vim config.yaml  # Edit IP list, model path, etc.

# 2. Launch containers on 4 nodes
./run_container.sh --cur-node 4 --master-ip 11.139.21.81

# 3. Start distributed server (TP=16, EP=16, 4 nodes)
./run_server.sh --command "python -m sglang.launch_server \
  --port 34567 --tp-size 16 --ep-size 16 --nnodes 4 \
  --dist-init-addr 11.139.21.81:31000 \
  --attention-backend trtllm_mha --enable-dp-attention"

# 4. Run benchmark
./run_client.sh --tokens-per-rank 256 --num-nodes 4 \
  --input-len 2000 --output-len 100

# 5. Clean up
./cleanup_containers.sh
```

## Features

- **Automatic GPU Detection**: Checks GPU availability across all nodes before deployment
- **Priority-Based Node Selection**: Selects nodes based on ip_list order in config
- **Multi-Node Container Orchestration**: Launches and configures Pouch containers on multiple nodes
- **Command Augmentation**: Automatically adds missing parameters to SGLang server commands
- **Distributed Logging**: Separate logs for master and worker nodes
- **Health Checking**: Polls server health endpoint to verify readiness
- **Batch Size Calculation**: Automatically calculates optimal batch size based on topology
- **Profiling Support**: Optional profiler control via HTTP API
- **Aggressive Cleanup**: Ensures all containers and processes are cleaned across all nodes

## Architecture

```
Master Node                          Worker Nodes
┌─────────────────┐                 ┌─────────────────┐
│ run_container.sh│────SSH───────▶  │ Pouch Container │
│ run_server.sh   │                 │ - GPU Access    │
│ run_client.sh   │                 │ - Volume Mounts │
└─────────────────┘                 │ - NCCL Config   │
        │                            └─────────────────┘
        │                                    │
        ▼                                    ▼
┌─────────────────┐                 ┌─────────────────┐
│ Pouch Container │◀───SSH (keys)──▶│ SGLang Server   │
│ - SGLang Server │                 │ - Rank 1-N      │
│ - Rank 0        │                 │ - Distributed   │
│ - Benchmark     │                 └─────────────────┘
└─────────────────┘
```

## Scripts

### Core Scripts

- **config.yaml** - Configuration file (nodes, paths, NCCL settings)
- **run_container.sh** - Launch Pouch containers on multiple nodes
- **run_server.sh** - Deploy distributed SGLang server
- **run_client.sh** - Run benchmark tests
- **cleanup_containers.sh** - Clean all containers and processes
- **profiler_util.sh** - Control profiler (optional)

### Utilities

- **ssh_util.py** - SSH utility with CLI interface for shell scripts
- **remote_utils.py** - Remote operations library (from NVL72-GB200)

## Configuration

Edit `config.yaml` to set:

```yaml
# Node settings
max_nodes: 4
current_nodes: 4
gpus_per_node: 4
ip_list: [list of node IPs in priority order]
master_ip: "11.139.21.81"

# Container settings
image: "hub.docker.alibaba-inc.com/tre-ai-infra/sglang:v0.5.2-cu129-gb200"

# Paths (absolute)
workspace_path: "/cpfs01/user/nebula_model/sjq-workspace/benchmark"
model_path: "/cpfs01/user/nebula_model/llm_weight"
model_name: "Qwen3-Coder-480B-A35B-Instruct-FP8"

# SGLang server defaults
sglang_defaults:
  chunked_prefill_base_size: 16384  # Auto-calculated: chunked-prefill-size = base * DP

# NCCL environment variables
nccl_env: {...}
```

## Usage Examples

### Basic Deployment

```bash
# Single-node deployment (TP=4)
./run_container.sh --cur-node 1 --master-ip 11.139.21.81
./run_server.sh --command "python -m sglang.launch_server --port 34567 --tp-size 4 --nnodes 1"
./run_client.sh --tokens-per-rank 256 --num-nodes 1
```

### Multi-Node with TP and EP

```bash
# 4-node deployment (TP=16, EP=16)
./run_container.sh --cur-node 4 --master-ip 11.139.21.81
./run_server.sh --command "python -m sglang.launch_server \
  --port 34567 --tp-size 16 --ep-size 16 --nnodes 4 \
  --dist-init-addr 11.139.21.81:31000 \
  --attention-backend trtllm_mha --enable-dp-attention \
  --moe-dense-tp-size 1 --enable-dp-lm-head"

# Run benchmark (output to console)
./run_client.sh --nnodes 4 --input-len 4096 --output-len 512 \
  --base-url http://11.139.21.81:34567

# Run benchmark and save results to file
./run_client.sh --nnodes 4 --input-len 4096 --output-len 512 \
  --base-url http://11.139.21.81:34567 --save-result
```

**Result organized at**: `result/benchmarks/Qwen3-Coder-480B-FP8-TP8-EP8-DP1/bs256_in4096_out512/benchmark_YYYYMMDD_HHMMSS.txt`

### With Profiling

**Option 1: One-stop script (Recommended)**
```bash
# Automatic profiling with organized traces
./run_profile_benchmark.sh --server http://11.139.21.81:34567 \
  --nnodes 2 --input-len 800 --output-len 100

# With custom batch size per rank
./run_profile_benchmark.sh --server http://11.139.21.81:34567 \
  --nnodes 4 --input-len 2000 --output-len 200 --bs-per-rank 64
```

**Option 2: Manual steps**
```bash
# 1. Start profiler
./profiler_util.sh start --server http://11.139.21.81:34567

# 2. Run workload
./run_client.sh --nnodes 4 --input-len 800 --output-len 100 \
  --base-url http://11.139.21.81:34567

# 3. Stop profiler (automatically organizes traces)
./profiler_util.sh stop --server http://11.139.21.81:34567 \
  --bs 256 --input-len 800 --output-len 100
```

**Traces organized at**: `result/traces/Qwen3-Coder-480B-FP8-TP8-EP8-DP1/bs256_in800_out100/*.gz`

## Directory Structure

```
benchmark/
├── config.yaml                # Configuration
├── run_container.sh          # Container launcher
├── run_server.sh             # Server launcher
├── run_client.sh             # Client benchmark
├── cleanup_containers.sh     # Cleanup script
├── profiler_util.sh          # Profiler control
├── ssh_util.py               # SSH utility
├── remote_utils.py           # Remote utilities
├── guideline.md              # Implementation guidelines
├── README.md                 # This file
├── log/                      # Logs
│   ├── master/              # Master node logs (YYMMDDHHmm.log)
│   └── worker/              # Worker logs (rank{N}_YY_MM_DD_HH_mm.log)
├── result/                   # Results
│   ├── traces/              # Profiler traces (model-TP-EP-DP/rank{TP-EP}/*.gz)
│   └── benchmarks/          # Benchmark results
├── doc/                      # Documentation
│   └── USAGE.md             # Detailed usage guide
└── test/                     # Test files (if any)
```

## Logging

### Master Node
- Location: `log/master/YYMMDDHHmm.log`
- Contains: Rank 0 server output, initialization logs

### Worker Nodes
- Location: `log/worker/rank{N}_YY_MM_DD_HH_mm.log`
- Contains: Rank N server output, distributed operations

### Monitoring Logs

```bash
# Master log
tail -f log/master/$(ls -t log/master/ | head -1)

# All worker logs
tail -f log/worker/rank*.log
```

## Troubleshooting

### Common Issues

1. **Not enough available nodes**
   - Check GPU availability: `python3 ssh_util.py check_gpu 11.139.21.81`
   - Clean existing containers: `./cleanup_containers.sh`

2. **Server timeout**
   - Check logs: `tail -f log/master/*.log`
   - Verify containers: `python3 ssh_util.py exec_on_node 11.139.21.81 "echo 'Alibaba@12#\$' | sudo -S pouch ps"`

3. **Connection refused**
   - Check health: `curl http://11.139.21.81:34567/health`
   - Verify port in server command

4. **Permission errors**
   - Fix ownership: `sudo chown -R $(whoami):$(id -gn) log/ result/`

See `doc/USAGE.md` for detailed troubleshooting.

## Key Differences from NVL72-GB200

This is a **simplified architecture** compared to NVL72-GB200:

| Feature | NVL72-GB200 | This Implementation |
|---------|-------------|---------------------|
| Language | Python orchestrator | Pure shell scripts |
| Coordination | Signal files | Direct execution |
| Profiling | Integrated | Manual/separate |
| TP/EP iteration | Automatic | Single config per run |
| Complexity | High (orchestrator layers) | Low (direct commands) |

## Requirements

- SSH access to all nodes
- Sudo password for Pouch commands
- Python 3 with PyYAML
- Pouch container engine
- NVIDIA GPUs and drivers
- Model files at specified path
- SSH keys for inter-container communication

## Documentation

- **README.md** (this file) - Quick overview and examples
- **doc/USAGE.md** - Detailed usage guide with troubleshooting
- **guideline.md** - Original implementation requirements

## Support

For issues or questions:
1. Check logs in `log/master/` and `log/worker/`
2. Review `doc/USAGE.md` for detailed troubleshooting
3. Verify configuration in `config.yaml`
4. Test SSH connectivity: `ssh henry.sjq@<node_ip>`
5. Check GPU availability: `python3 ssh_util.py check_gpu <node_ip>`

## License

Internal use for SGLang benchmarking and deployment.
