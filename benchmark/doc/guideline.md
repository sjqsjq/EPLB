根据下面的指引实现功能，实现过程中可按实际情况修改此文件，但是每次修改此文件必须询问。不要添加类似✅❌之类的符号。
所有测试文件放在./test
所有文档放在./doc

ip list：
11.139.21.86
11.139.21.81
11.139.21.87
11.139.21.77
...(将来会增加)

创建一系列脚本实现sglang自动化部署推理服务, 执行流程：
在master节点(手动)
运行run_container.sh，在多节点启动pouch容器
运行run_server.sh --command "xxx xxx",  拉起server服务
运行run_client.sh --bs xxx --seq_len xxx, 指定负载测试


config.yaml : 默认配置，脚本未指定的参数从这里读取
max_nodes：4 # 最大节点数量，每节点4gpus
current_nodes：4 # 本次运行使用的节点数
image: "hub.docker.alibaba-inc.com/tre-ai-infra/sglang:v0.5.2-cu129-gb200" #镜像
model_path：/cpfs01/user/nebula_model/llm_weight # 模型所在目录
model_name: Qwen3-Coder-480B-A35B-Instruct-FP8
workspace_path：/cpfs01/user/nebula_model/sjq-workspace/benchmark # 运行脚本所在绝对路径
master_ip: 11.139.21.81 #master节点ip
log_path: ./log # 日志文件路径，基于workspace_path的相对路径 
result_path: ./result # 结果文件路径，基于workspace_path的相对路径
profiler_path: ./result/traces  # trace文件路径，基于workspace_path的相对路径
#NCCL config，-e 传入容器
  SGLANG_LOCAL_IP_NIC: "eth0"
  GLOO_SOCKET_IFNAME: "eth0"
  NCCL_SOCKET_IFNAME: "eth0"
  NCCL_DEBUG: "INFO"
  NCCL_DEBUG_SUBSYS: "ENV"
  NCCL_IB_DISABLE: "1"
  NCCL_IBEXT_DISABLE: "1"
  NCCL_IB_TIMEOUT: "22"
  NCCL_P2P_LEVEL: "5"
  NCCL_NVLS_ENABLE: "1"
  NCCL_MNNVL_ENABLE: "1"
  NCCL_CUMEM_ENABLE: "1"

run_container.sh/container.py:
根据配置节点数量，从ip list中寻找gpu未被占用的节点（寻找顺序按ip list排列的顺序，ip填写靠前的优先）。找到足够数量节点后，在每个节点启动并进入容器，启动时使用绝对路径挂载工作目录和模型目录，-v workspace_path:workspace_path 容器内外使用完全相同的路径来避免路径出错的问题,容器启动后保持运行。在容器中，配置ssh，将master节点的密钥复制过来并赋予相应权限。
运行命令： run_container.sh --cur-node 4 --master-ip 11.139.21.81

run_server.sh/server.py:
功能说明：
1. 启动新配置前先清理先前的进程（在各节点容器中运行 pkill -9 sglang, pkill -9 python）
2. 接收 --command 参数传入的启动命令
3. 自动为每个节点添加 --node-rank 参数（关键！分布式训练/推理必需）
4. 自动补充 config 中有但命令中未指定的参数（如 --model-path, --host）
5. 通过 ssh_util 在各节点容器中执行拼接后的完整命令

node-rank 参数说明：
--node-rank 是多节点分布式部署的必需参数，用于标识当前进程所在的节点编号
- node-rank 从 0 开始编号
- 每个节点必须有唯一的 node-rank
- 脚本会自动为每个节点添加对应的 --node-rank 参数

使用示例：
输入命令：
  ./run_server.sh --command "python -m sglang.launch_server --port 34567 --host 0.0.0.0 --tp-size 16 --dp-size 16 --ep-size 16 --nnodes 4 --dist-init-addr 11.139.21.81:31000 --attention-backend trtllm_mha --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --ep-dispatch-algorithm fake --stream-out --moe-a2a-backend deepep --disable-cuda-graph"

实际执行命令（脚本自动拼接）：
  Node 0 容器中:
    python -m sglang.launch_server --model-path /cpfs01/user/nebula_model/llm_weight/Qwen3-Coder-480B-A35B-Instruct-FP8 --port 34567 --host 0.0.0.0 --tp-size 16 --dp-size 16 --ep-size 16 --nnodes 4 --dist-init-addr 11.139.21.81:31000 --attention-backend trtllm_mha --enable-dp-attention --moe-dense-tp-size 1 --enable-dp-lm-head --ep-dispatch-algorithm fake --stream-out --moe-a2a-backend deepep --disable-cuda-graph --node-rank 0

  Node 1 容器中:
    python -m sglang.launch_server ... --node-rank 1

  Node 2 容器中:
    python -m sglang.launch_server ... --node-rank 2

  Node 3 容器中:
    python -m sglang.launch_server ... --node-rank 3

注意事项：
1. 确保 --nnodes 参数与实际启动的节点数一致
2. 对于 TP=16, DP=16, EP=16 的配置，总 GPU 数应为 4 nodes x 4 GPUs = 16 GPUs
3. 如果遇到 DeepEP "Unsupported ranks" 错误，尝试使用节点内 TP 配置（如 TP=4, DP=4, EP=4）

run_client.sh/client.py
只需在master节点的容器中运行，参考下列代码，input/output也应该是可传递的参数，并设置默认值
#!/bin/bash
set -x

# 默认值
TOKENS_PER_RANK_DEFAULT=256
NUM_NODES_DEFAULT=1
BASE_URL="http://11.139.21.81:31000" 

TOKENS_PER_RANK=$TOKENS_PER_RANK_DEFAULT
NUM_NODES=$NUM_NODES_DEFAULT

# 解析命令行参数
# 这里假设 $1 是 tokens_per_rank, $2 是 num_nodes
# 如果你想通过命名参数传递，则需要更复杂的解析，如之前的例子
if [ -n "$1" ]; then
    TOKENS_PER_RANK="$1"
fi
if [ -n "$2" ]; then
    NUM_NODES="$2"
fi

# 在 Bash 中进行数学运算
# 正确的写法是使用 $(( ))
BENCH_BATCH_SIZE=$((TOKENS_PER_RANK * NUM_NODES *4 ))

# 可以通过第三个参数传递 base-url，或者保持硬编码
if [ -n "$3" ]; then
    BASE_URL="$3"
fi


echo "Running sglang.bench_one_batch_server with:"
echo "  Calculated BATCH_SIZE: $BENCH_BATCH_SIZE"
echo "  --model-path /disk1/ldp/models/DeepSeek-R1-0528"
echo "  --base-url $BASE_URL"
echo "  --input-len 2000"
echo "  --output-len 100"
echo "  --skip-warmup"

python3 -m sglang.bench_one_batch_server \
  --model-path /disk1/ldp/models/DeepSeek-R1-0528 \
  --base-url "$BASE_URL" \
  --batch-size "$BENCH_BATCH_SIZE" \
  --input-len 2000 \
  --output-len 100 \
  --skip-warmup

echo "bench.sh finished."

## Profiling功能

支持两种profiling方式：
1. **nsys (Nsight Systems)**: 系统级性能分析，包含CUDA kernel时间线
2. **PyTorch Profiler**: Python/PyTorch层性能分析，包含算子级别信息

### 配置

在config.yaml中配置：
```yaml
profiler_config:
  enable_nsys: false          # 启用nsys (包装python进程)
  enable_torch_profiler: true # 启用PyTorch profiler HTTP API
  nsys_output_path: "./result/nsys_traces"
  torch_profiler_output_path: "./result/traces"
```

### 使用流程

**1. 启动支持profiler的服务器**
```bash
# 确保config.yaml中 enable_torch_profiler: true
./run_server.sh --command "python -m sglang.launch_server --port 34567 \
  --tp-size 16 --ep-size 16 --nnodes 4 \
  --dist-init-addr 11.139.21.81:31000 \
  --attention-backend trtllm_mha"
```

**2. 启动profiling**
```bash
./profiler_util.sh start --server http://11.139.21.81:34567
```
输出示例：
```
Request:
  curl -X POST http://11.139.21.81:34567/start_profile \
    -H 'Content-Type: application/json' \
    -d '{"num_steps": 10, "activities": ["CPU", "CUDA"], "record_shapes": true, "profile_memory": true}'

✓ Profiler started successfully (HTTP 200)
```

**3. 运行workload**
```bash
./run_client.sh --nnodes 4 --input-len 800 --output-len 100
```

**4. 停止profiling**
```bash
./profiler_util.sh stop --server http://11.139.21.81:34567
```
输出示例：
```
Request:
  curl -X POST http://11.139.21.81:34567/stop_profile -m 300

✓ Profiler stopped successfully (HTTP 200)
Traces have been saved.
```

### Trace文件位置

**PyTorch profiler traces:**
```
result/traces/Qwen3-Coder-480B-FP8-TP16-EP16-DP1/
├── bs256_in800_out100/
│   ├── <timestamp>-TP-0-EP-0.trace.json.gz
│   ├── <timestamp>-TP-1-EP-1.trace.json.gz
│   └── ...
├── bs512_in2000_out200/
│   └── ...
...
```

**nsys traces** (如果 enable_nsys: true):
```
result/nsys_traces/
├── rank0_<timestamp>.nsys-rep
├── rank1_<timestamp>.nsys-rep
...
```

### 查看Trace

**1. 解压缩**
```bash
gunzip result/traces/*/*.gz
```

**2. 使用Chrome Tracing**
- 打开 chrome://tracing
- Load JSON文件

**3. 使用Perfetto**
- 访问 https://ui.perfetto.dev
- Open trace file

**4. 分析nsys结果**
```bash
# 查看统计信息
nsys stats result/nsys_traces/rank0_*.nsys-rep

# 打开GUI
nsys-ui result/nsys_traces/rank0_*.nsys-rep
```

### 性能影响

| 模式 | 开销 | 说明 |
|-----|------|------|
| nsys未激活 | < 1% | `--capture-range=cudaProfilerApi`只监听API调用 |
| PyTorch profiler激活 | 5-10% | 记录详细的算子级信息 |
| nsys + PyTorch同时激活 | 10-20% | 完整的系统级+算子级profiling |

### 完整命令示例

```bash
# Step 1: 清理环境
./cleanup_containers.sh

# Step 2: 启动容器
./run_container.sh --cur-node 4 --skip-ssh

# Step 3: 启动服务器（带nsys）
# 在config.yaml中设置: enable_nsys: true, enable_torch_profiler: true
./run_server.sh --command "python -m sglang.launch_server \
  --port 34567 --host 0.0.0.0 \
  --tp-size 16 --ep-size 16 --nnodes 4 \
  --dist-init-addr 11.139.21.81:31000 \
  --attention-backend trtllm_mha"

# Step 4: 等待服务器就绪
curl http://11.139.21.81:34567/health

# Step 5: 启动profiling
./profiler_util.sh start --server http://11.139.21.81:34567

# Step 6: 运行workload
./run_client.sh --nnodes 4 --input-len 800 --output-len 100

# Step 7: 停止profiling
./profiler_util.sh stop --server http://11.139.21.81:34567

# Step 8: 查看trace文件
find result/traces -name "*.json.gz"
find result/nsys_traces -name "*.nsys-rep"
```

### Troubleshooting

**问题1: API返回404**
```bash
✗ API endpoint not found (HTTP 404)
```
**解决方案:**
1. 检查config.yaml: `enable_torch_profiler: true`
2. 检查服务器日志：
   ```bash
   grep SGLANG_TORCH_PROFILER_DIR log/worker/rank0_*.log
   ```
3. 重启服务器

**问题2: Trace文件未生成**
**解决方案:**
```bash
# 检查目录权限
ls -ld result/traces/

# 检查stop_profile响应
./profiler_util.sh stop --server http://11.139.21.81:34567
```

**问题3: nsys进程未启动**
**解决方案:**
```bash
# 检查nsys是否可用
python3 ssh_util.py exec_in_container 11.139.21.81 lsw_sglang_benchmark_rank0 "which nsys"

# 检查nsys进程
python3 ssh_util.py exec_in_container 11.139.21.81 lsw_sglang_benchmark_rank0 "ps aux | grep nsys"
```

ssh_util.py:
工具类，负责将命令从master节点自动在其他节点或其他节点的容器中执行，运行pouch容器需要sudo密码, 使用-S参数来通过代码输入参数
基于pssh实现在多节点容器运行命令

日志/结果/trace 文件存放路径：
log_path/result_path 从congig.yaml中读取, 注意容器中用户是root，要以master节点实际用户省份创建目录和文件以避免权限问题
master节点主程序日志：log_path/master/YYMMDDHHmm.log
worker container日志：log_path/worker/rank{$rank}_YY_MM_DD_HH_mm.log
结果文件：result_path/model-TP-EP/result.json
trace_file: result_path/traces/model-quant-TP-EP-DP/bs{N}_in{input}_out{output}/{timestamp}-TP-{rank}-EP-{rank}.trace.json.gz
  示例: result/traces/Qwen3-Coder-480B-FP8-TP8-EP8-DP1/bs256_in800_out100/1765916951.7779593-TP-0-EP-0.trace.json.gz

