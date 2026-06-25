#!/bin/bash
#
# bench_eplb_comparison.sh - End-to-end throughput comparison: EPLB ON vs OFF
#
# This script deploys DeepSeek-R1 twice (with and without EPLB),
# measures throughput using synthetic prompts, and reports comparison.
#
# Usage: ./bench_eplb_comparison.sh [--master-ip IP] [--concurrency N] [--num-requests N] [--max-tokens N]
#
# Requirements:
# - run_container.sh, run_server.sh, cleanup_containers.sh available
# - patch_sampler.py available (handles nan from dummy weights)
# - 4 nodes with free GPUs
#

# set -e  # Disabled: individual steps handle errors

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
MASTER_IP="79"
CONCURRENCY=16
NUM_REQUESTS=64
MAX_TOKENS=128
WARMUP_REQUESTS=8
PORT=34567

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --master-ip) MASTER_IP="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --warmup) WARMUP_REQUESTS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo -e "${RED}Unknown: $1${NC}"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-complete IP
IP_PREFIX="11.139.21"
if [[ "$MASTER_IP" =~ ^[0-9]+$ ]]; then
    FULL_IP="${IP_PREFIX}.${MASTER_IP}"
else
    FULL_IP="$MASTER_IP"
    MASTER_IP=$(echo "$MASTER_IP" | awk -F. '{print $NF}')
fi

SERVER_URL="http://${FULL_IP}:${PORT}"
RESULT_DIR="$SCRIPT_DIR/result"
mkdir -p "$RESULT_DIR"

RESULT_FILE="$RESULT_DIR/bench_eplb_comparison.json"

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  EPLB Throughput Comparison Benchmark${NC}"
echo -e "${CYAN}============================================================${NC}"
echo "  Master IP:      $FULL_IP"
echo "  Server URL:     $SERVER_URL"
echo "  Concurrency:    $CONCURRENCY"
echo "  Num requests:   $NUM_REQUESTS"
echo "  Max tokens:     $MAX_TOKENS"
echo "  Warmup:         $WARMUP_REQUESTS"
echo ""

# Function: wait for server health
wait_for_health() {
    local max_wait=${1:-600}
    local start=$(date +%s)
    echo -n "  Waiting for server health"
    while true; do
        if curl -s --connect-timeout 3 "$SERVER_URL/health" 2>/dev/null | grep -q "ok\|ready\|health"; then
            echo -e " ${GREEN}READY${NC}"
            return 0
        fi
        local elapsed=$(( $(date +%s) - start ))
        if [ $elapsed -gt $max_wait ]; then
            echo -e " ${RED}TIMEOUT (${max_wait}s)${NC}"
            return 1
        fi
        echo -n "."
        sleep 5
    done
}

# Function: run throughput benchmark
run_benchmark() {
    local label="$1"
    local output_file="$2"
    
    echo -e "  ${YELLOW}Running benchmark: $label${NC}"
    echo "    Warmup: $WARMUP_REQUESTS requests..."
    
    # Warmup with short requests
    python3 -c "
import requests, json, time
from concurrent.futures import ThreadPoolExecutor

server = '$SERVER_URL'
prompts = [
    'Hello, how are you?',
    'What is 2+2?',
    'Tell me a joke.',
    'What is the capital of France?',
]

def send(i):
    try:
        r = requests.post(f'{server}/v1/completions', json={
            'model': 'DeepSeek-R1',
            'prompt': prompts[i % len(prompts)],
            'max_tokens': 32,
            'temperature': 0.7
        }, timeout=120)
        return r.status_code == 200
    except:
        return False

with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(send, range($WARMUP_REQUESTS)))
print(f'    Warmup done: {sum(results)}/{len(results)} succeeded')
"
    
    echo "    Benchmark: $NUM_REQUESTS requests, concurrency=$CONCURRENCY, max_tokens=$MAX_TOKENS..."
    
    # Actual benchmark
    python3 << PYEOF
import requests, json, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

server = '$SERVER_URL'
concurrency = $CONCURRENCY
num_requests = $NUM_REQUESTS
max_tokens = $MAX_TOKENS
output_file = '$output_file'

prompts = [
    'Explain the theory of general relativity in simple terms.',
    'Write a short essay about machine learning applications.',
    'Describe the architecture of modern distributed systems.',
    'What are the main challenges in quantum computing?',
    'Explain how neural networks learn from data.',
    'Describe the history and evolution of programming languages.',
    'What is the significance of the Higgs boson discovery?',
    'Explain consensus algorithms in distributed computing.',
    'How does garbage collection work in modern runtimes?',
    'Describe the TCP/IP protocol stack and its layers.',
    'What are the principles of functional programming?',
    'Explain the CAP theorem and its implications.',
    'How do modern GPUs achieve parallel computation?',
    'Describe the evolution of computer memory hierarchies.',
    'What is the role of attention in transformer models?',
    'Explain microservice architecture patterns and tradeoffs.',
]

results = []
errors = 0

def send_request(idx):
    prompt = prompts[idx % len(prompts)]
    start = time.time()
    try:
        r = requests.post(f'{server}/v1/completions', json={
            'model': 'DeepSeek-R1',
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': 0.7
        }, timeout=300)
        elapsed = time.time() - start
        if r.status_code == 200:
            data = r.json()
            usage = data.get('usage', {})
            return {
                'success': True,
                'latency': elapsed,
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
            }
        else:
            return {'success': False, 'latency': elapsed, 'error': r.status_code}
    except Exception as e:
        return {'success': False, 'latency': time.time() - start, 'error': str(e)}

overall_start = time.time()
with ThreadPoolExecutor(max_workers=concurrency) as ex:
    futures = [ex.submit(send_request, i) for i in range(num_requests)]
    for f in as_completed(futures):
        r = f.result()
        results.append(r)
        if not r['success']:
            errors += 1

overall_elapsed = time.time() - overall_start

successful = [r for r in results if r['success']]
total_prompt_tokens = sum(r['prompt_tokens'] for r in successful)
total_completion_tokens = sum(r['completion_tokens'] for r in successful)
total_tokens = total_prompt_tokens + total_completion_tokens
avg_latency = sum(r['latency'] for r in successful) / len(successful) if successful else 0

summary = {
    'label': '$label',
    'num_requests': num_requests,
    'concurrency': concurrency,
    'max_tokens': max_tokens,
    'successful': len(successful),
    'failed': errors,
    'total_time_s': overall_elapsed,
    'avg_latency_s': avg_latency,
    'total_prompt_tokens': total_prompt_tokens,
    'total_completion_tokens': total_completion_tokens,
    'throughput_req_per_s': len(successful) / overall_elapsed if overall_elapsed > 0 else 0,
    'throughput_output_tok_per_s': total_completion_tokens / overall_elapsed if overall_elapsed > 0 else 0,
    'throughput_total_tok_per_s': total_tokens / overall_elapsed if overall_elapsed > 0 else 0,
}

with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'    Successful: {len(successful)}/{num_requests}')
print(f'    Total time: {overall_elapsed:.2f}s')
print(f'    Avg latency: {avg_latency:.2f}s')
print(f'    Throughput: {summary["throughput_req_per_s"]:.3f} req/s')
print(f'    Output tok/s: {summary["throughput_output_tok_per_s"]:.1f}')
print(f'    Total tok/s: {summary["throughput_total_tok_per_s"]:.1f}')

if errors > 0:
    print(f'    WARNING: {errors} requests failed', file=sys.stderr)
PYEOF
}

# Function: deploy and start server
deploy_server() {
    local eplb_flag="$1"  # "0" or "32"
    local label="$2"
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  Deploying: $label${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Step 0: Ensure clean state - force remove any existing containers
    echo -e "  ${YELLOW}[0/4] Ensuring clean state...${NC}"
    # Create temporary nodelist if needed for cleanup
    local tmp_nodelist="$SCRIPT_DIR/tmp/nodelist_${FULL_IP}"
    if [ ! -f "$tmp_nodelist" ]; then
        mkdir -p "$SCRIPT_DIR/tmp"
        # Read first 4 IPs from config that have free GPUs
        python3 -c "
import yaml, json, subprocess, sys
sys.path.insert(0, '$SCRIPT_DIR')
with open('$SCRIPT_DIR/config.yaml') as f:
    config = yaml.safe_load(f)
ips = config.get('ip_list', [])[:8]
# Just output the first few IPs - containers will be on these
for ip in ips[:4]:
    print(ip)
" > "$tmp_nodelist" 2>/dev/null || true
    fi
    bash cleanup_containers.sh --master-ip "$MASTER_IP" 2>&1 | tail -3
    # Also directly remove containers on all possible nodes
    for node_ip in $(python3 -c "
import yaml
with open('$SCRIPT_DIR/config.yaml') as f:
    config = yaml.safe_load(f)
for ip in config.get('ip_list', [])[:6]:
    print(ip)
"); do
        for r in 0 1 2 3; do
            python3 ssh_util.py exec_on_node "$node_ip"                 "pouch rm -f sjq_sglang_benchmark_rank${r} 2>/dev/null || true" 2>/dev/null &
        done
    done
    wait
    rm -f "$tmp_nodelist" 2>/dev/null
    sleep 3
    
    # Step 1: Deploy containers
    echo -e "  ${YELLOW}[1/4] Deploying containers...${NC}"
    if ! bash run_container.sh --cur-node 4 --master-ip "$MASTER_IP" 2>&1 | tail -10; then
        echo "    run_container.sh returned error, checking if containers exist anyway..."
    fi
    
    # Step 2: Apply sampler patch (handles nan from dummy weights)
    echo -e "  ${YELLOW}[2/4] Applying sampler patch...${NC}"
    NODE_LIST_FILE="$SCRIPT_DIR/tmp/nodelist_${FULL_IP}"
    if [ ! -f "$NODE_LIST_FILE" ]; then
        echo -e "${RED}Error: nodelist not found${NC}"
        return 1
    fi
    NODES=($(cat "$NODE_LIST_FILE"))
    
    RANK=0
    for NODE in "${NODES[@]}"; do
        CONTAINER="sjq_sglang_benchmark_rank${RANK}"
        python3 ssh_util.py exec_in_container "$NODE" "$CONTAINER" \
            "python3 $SCRIPT_DIR/patch_sampler.py 2>/dev/null || echo SKIP" 2>&1 | grep -v "^$" | tail -1
        RANK=$((RANK + 1))
    done
    echo "    Sampler patched on ${#NODES[@]} containers"
    
    # Step 3: Start server
    echo -e "  ${YELLOW}[3/4] Starting server (ep-num-redundant-experts=$eplb_flag)...${NC}"
    
    local SERVER_CMD="python -m sglang.launch_server --tp-size 16 --dp-size 4 --port $PORT --deepep-mode low_latency --load-format dummy --ep-num-redundant-experts $eplb_flag"
    
    # Run server in background - don't let it block
    bash run_server.sh --master-ip "$MASTER_IP" --command "$SERVER_CMD" --health-timeout 300 2>&1 | \
        grep -E "ready|READY|health|timeout|Error|error|Added|Launch" | tail -10
    
    # Step 4: Wait for health
    echo -e "  ${YELLOW}[4/4] Verifying server health...${NC}"
    if ! wait_for_health 300; then
        echo -e "${RED}Server failed to start for: $label${NC}"
        return 1
    fi
    
    echo -e "  ${GREEN}Server ready: $label${NC}"
    return 0
}

# Function: cleanup
cleanup() {
    echo -e "  ${YELLOW}Cleaning up containers...${NC}"
    bash cleanup_containers.sh --master-ip "$MASTER_IP" 2>&1 | tail -3
    sleep 5
}

# ═══════════════════════════════════════════════════════════
# PHASE 1: Baseline (No EPLB)
# ═══════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  PHASE 1: Baseline (No EPLB)                           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

BASELINE_FILE="$RESULT_DIR/bench_baseline_no_eplb.json"
PHASE1_START=$(date +%s)

if deploy_server "0" "Baseline (no EPLB)"; then
    run_benchmark "baseline_no_eplb" "$BASELINE_FILE"
    PHASE1_OK=true
else
    echo -e "${RED}Phase 1 deployment failed${NC}"
    PHASE1_OK=false
fi

cleanup
PHASE1_ELAPSED=$(( $(date +%s) - PHASE1_START ))
echo "  Phase 1 elapsed: ${PHASE1_ELAPSED}s"

# ═══════════════════════════════════════════════════════════
# PHASE 2: With EPLB (32 redundant experts)
# ═══════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  PHASE 2: With EPLB (32 redundant experts)             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

EPLB_FILE="$RESULT_DIR/bench_with_eplb.json"
PHASE2_START=$(date +%s)

if deploy_server "32" "With EPLB (32 redundant)"; then
    run_benchmark "with_eplb_32" "$EPLB_FILE"
    PHASE2_OK=true
else
    echo -e "${RED}Phase 2 deployment failed${NC}"
    PHASE2_OK=false
fi

cleanup
PHASE2_ELAPSED=$(( $(date +%s) - PHASE2_START ))
echo "  Phase 2 elapsed: ${PHASE2_ELAPSED}s"

# ═══════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  COMPARISON RESULTS                                     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$PHASE1_OK" = true ] && [ "$PHASE2_OK" = true ]; then
    python3 << COMPARE_EOF
import json

with open('${BASELINE_FILE}') as f:
    baseline = json.load(f)
with open('${EPLB_FILE}') as f:
    eplb = json.load(f)

print('  Configuration          | Req/s    | Output tok/s | Total tok/s | Avg Latency')
print('  -----------------------+----------+--------------+-------------+------------')
print(f'  Baseline (no EPLB)     | {baseline["throughput_req_per_s"]:>7.3f}  | {baseline["throughput_output_tok_per_s"]:>11.1f}  | {baseline["throughput_total_tok_per_s"]:>10.1f}  | {baseline["avg_latency_s"]:>8.2f}s')
print(f'  With EPLB (32 redund.) | {eplb["throughput_req_per_s"]:>7.3f}  | {eplb["throughput_output_tok_per_s"]:>11.1f}  | {eplb["throughput_total_tok_per_s"]:>10.1f}  | {eplb["avg_latency_s"]:>8.2f}s')
print()

# Compute deltas
if baseline['throughput_output_tok_per_s'] > 0:
    tok_delta = (eplb['throughput_output_tok_per_s'] - baseline['throughput_output_tok_per_s']) / baseline['throughput_output_tok_per_s'] * 100
    req_delta = (eplb['throughput_req_per_s'] - baseline['throughput_req_per_s']) / baseline['throughput_req_per_s'] * 100
    lat_delta = (eplb['avg_latency_s'] - baseline['avg_latency_s']) / baseline['avg_latency_s'] * 100
    
    print(f'  EPLB Impact:')
    print(f'    Throughput (output tok/s): {tok_delta:+.2f}%')
    print(f'    Throughput (req/s):        {req_delta:+.2f}%')
    print(f'    Latency:                   {lat_delta:+.2f}%')
    print()
    
    if tok_delta < 0:
        print(f'  => EPLB reduces output throughput by {abs(tok_delta):.2f}% (overhead from rebalancing)')
    else:
        print(f'  => EPLB improves output throughput by {tok_delta:.2f}% (better load balance)')

    # Save combined result
    combined = {
        'benchmark_config': {
            'concurrency': baseline.get('concurrency'),
            'num_requests': baseline.get('num_requests'),
            'max_tokens': baseline.get('max_tokens'),
            'load_format': 'dummy',
            'tp_size': 16,
            'dp_size': 4,
            'deepep_mode': 'low_latency',
        },
        'baseline': baseline,
        'with_eplb': eplb,
        'comparison': {
            'throughput_output_tok_delta_pct': tok_delta,
            'throughput_req_delta_pct': req_delta,
            'latency_delta_pct': lat_delta,
        }
    }
    with open('${RESULT_FILE}', 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'')
    print(f'  Results saved to: ${RESULT_FILE}')
COMPARE_EOF
else
    echo -e "${RED}Cannot compare - one or both phases failed${NC}"
    [ "$PHASE1_OK" != true ] && echo "  Phase 1 (baseline): FAILED"
    [ "$PHASE2_OK" != true ] && echo "  Phase 2 (EPLB): FAILED"
fi

echo ""
echo -e "${GREEN}Benchmark complete.${NC}"
