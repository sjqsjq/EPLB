"""Benchmark throughput with/without EPLB using the routing dataset."""
import csv
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_prompts(csv_path, max_prompts=None):
    prompts = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                messages = json.loads(row['prompt'])
                prompts.append(messages)
            except:
                continue
            if max_prompts and len(prompts) >= max_prompts:
                break
    return prompts

def send_request(server_url, messages, max_tokens, request_id):
    start = time.time()
    try:
        resp = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "model": "DeepSeek-R1",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=300,
        )
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            return {
                "id": request_id,
                "success": True,
                "latency": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        else:
            return {"id": request_id, "success": False, "latency": elapsed, "error": resp.status_code}
    except Exception as e:
        return {"id": request_id, "success": False, "latency": time.time() - start, "error": str(e)}

def run_benchmark(server_url, prompts, max_tokens, concurrency, num_requests):
    print(f"\n{'='*60}")
    print(f"Benchmark: {num_requests} requests, concurrency={concurrency}, max_tokens={max_tokens}")
    print(f"Server: {server_url}")
    print(f"{'='*60}")

    results = []
    overall_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(num_requests):
            messages = prompts[i % len(prompts)]
            futures.append(executor.submit(send_request, server_url, messages, max_tokens, i))

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"  req={result['id']} latency={result['latency']:.2f}s "
                      f"prompt={result['prompt_tokens']} completion={result['completion_tokens']}")
            else:
                print(f"  req={result['id']} FAILED: {result.get('error')}")

    overall_elapsed = time.time() - overall_start

    successful = [r for r in results if r["success"]]
    failed = len(results) - len(successful)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)
    total_completion_tokens = sum(r["completion_tokens"] for r in successful)
    total_tokens = total_prompt_tokens + total_completion_tokens

    avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0

    summary = {
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": failed,
        "total_time_s": overall_elapsed,
        "avg_latency_s": avg_latency,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "throughput_req_per_s": len(successful) / overall_elapsed,
        "throughput_output_tok_per_s": total_completion_tokens / overall_elapsed,
        "throughput_total_tok_per_s": total_tokens / overall_elapsed,
    }

    print(f"\n{'─'*60}")
    print(f"Results:")
    print(f"  Successful:          {summary['successful']}/{num_requests}")
    print(f"  Total time:          {summary['total_time_s']:.2f}s")
    print(f"  Avg latency:         {summary['avg_latency_s']:.2f}s")
    print(f"  Throughput (req/s):  {summary['throughput_req_per_s']:.3f}")
    print(f"  Output tokens/s:     {summary['throughput_output_tok_per_s']:.1f}")
    print(f"  Total tokens/s:      {summary['throughput_total_tok_per_s']:.1f}")
    print(f"  Prompt tokens:       {summary['total_prompt_tokens']}")
    print(f"  Completion tokens:   {summary['total_completion_tokens']}")
    print(f"{'─'*60}\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://11.139.21.79:34567")
    parser.add_argument("--data", default="/cpfs01/user/nebula_model/sjq-workspace/benchmark/data/routing_dataset.csv")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prompts = load_prompts(args.data)
    print(f"Loaded {len(prompts)} prompts from {args.data}")

    summary = run_benchmark(args.server, prompts, args.max_tokens, args.concurrency, args.num_requests)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output}")
