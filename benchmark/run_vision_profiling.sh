#!/bin/bash
# 简单的视觉模块profiling测试脚本

BASE_URL="${1:-http://127.0.0.1:34567}"
BATCH_SIZE="${2:-32}"
IMAGE_COUNT="${3:-4}"

# 输出目录
OUT_DIR="/cpfs01/user/nebula_model/sjq-workspace/benchmark/result/traces/Qwen3-VL-235B-TP8-EP8-DP1/bs${BATCH_SIZE}_img${IMAGE_COUNT}_1080p_output512"
mkdir -p "$OUT_DIR"

echo "开始profiling测试..."
echo "  Server: $BASE_URL"
echo "  Batch size: $BATCH_SIZE"
echo "  Image count: $IMAGE_COUNT"
echo "  Output dir: $OUT_DIR"

# 启动profiler
curl -X POST "${BASE_URL}/start_profile" \
  -H "Content-Type: application/json" \
  -d "{\"activities\": [\"CPU\", \"GPU\"], \"num_steps\": 30, \"output_dir\": \"${OUT_DIR}\"}"

echo -e "\n等待profiler初始化..."
sleep 2

# 发送带图片的请求
echo "发送图片请求..."
python3 /cpfs01/user/nebula_model/sjq-workspace/benchmark/test_image_requests.py \
  --base-url "$BASE_URL" \
  --num-requests "$BATCH_SIZE" \
  --output-length 512 \
  --image-count "$IMAGE_COUNT" \
  --image-resolution 1080p

# 停止profiler
echo -e "\n停止profiler..."
curl -X POST "${BASE_URL}/stop_profile" -H "Content-Type: application/json"

echo -e "\n等待trace文件写入 (120秒)..."
sleep 120

# 检查trace文件中的视觉处理
echo -e "\n检查vision events..."
latest_trace=$(ls -t "$OUT_DIR"/*.trace.json.gz 2>/dev/null | head -1)
if [ -n "$latest_trace" ]; then
  echo "分析: $latest_trace"
  zcat "$latest_trace" | python3 -c "
import json, sys
data = json.load(sys.stdin)
events = data.get('traceEvents', [])
vision = [e for e in events if 'name' in e and any(k in str(e['name']).lower() for k in ['image', 'vision', 'mm_embed', 'mrope'])]
print(f'Total events: {len(events)}')
print(f'Vision events: {len(vision)}')
if vision[:5]: print('Sample:', [e['name'] for e in vision[:5]])
" || echo "trace分析失败"
else
  echo "未找到trace文件"
fi

echo -e "\n完成!"
