#!/usr/bin/env python3
"""
分析真实 Prefill 路由数据下的本地命中率和负载不均衡度
数据来源：Qwen3-30B-A3B-FP8 的 routing_record
用途：验证数学建模文档中的理论假设
"""
import json
import csv
import math
import os

RESULT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(RESULT_DIR, "result/expert/routing_record_Qwen3-30B-A3B-FP8.csv")
JSON_PATH = os.path.join(RESULT_DIR, "result/expert/routing_record_Qwen3-30B-A3B-FP8.json")

def load_data():
    with open(JSON_PATH) as f:
        meta = json.load(f)
    
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        layers = {}
        for row in reader:
            layer_id = int(row[0])
            layers[layer_id] = [int(x) for x in row[1:]]
    
    return meta, layers

def analyze_imbalance(layers, N, sim_ep_list=[8, 16, 32, 64]):
    """模拟不同 EP 下的 GPU 负载不均衡"""
    results = {}
    
    for sim_ep in sim_ep_list:
        n_local = N // sim_ep
        contiguous = []
        roundrobin = []
        lpt = []
        
        for layer_id, counts in layers.items():
            total = sum(counts)
            if total == 0:
                continue
            mu_gpu = total / sim_ep
            
            # 连续分配
            gpu_c = [sum(counts[g*n_local:(g+1)*n_local]) for g in range(sim_ep)]
            contiguous.append(max(gpu_c) / mu_gpu)
            
            # Round-Robin
            gpu_rr = [0] * sim_ep
            for e, c in enumerate(counts):
                gpu_rr[e % sim_ep] += c
            roundrobin.append(max(gpu_rr) / mu_gpu)
            
            # LPT
            gpu_lpt = [0] * sim_ep
            gpu_cnt = [0] * sim_ep
            for e_id, c in sorted(enumerate(counts), key=lambda x: x[1], reverse=True):
                cands = [(gpu_lpt[g], g) for g in range(sim_ep) if gpu_cnt[g] < n_local]
                if cands:
                    _, g = min(cands)
                    gpu_lpt[g] += c
                    gpu_cnt[g] += 1
            lpt.append(max(gpu_lpt) / mu_gpu)
        
        results[sim_ep] = {
            'n_local': n_local,
            'contiguous': contiguous,
            'roundrobin': roundrobin,
            'lpt': lpt,
        }
    
    return results

def print_results(meta, layers, results):
    N = meta['num_experts']
    K = meta['top_k']
    
    print("=" * 80)
    print(f"模型: {meta['model']}, N={N}, K={K}, 总tokens={meta['total_tokens']}")
    print("=" * 80)
    
    # Expert 分布统计
    print("\n[1] Expert 需求分布")
    for lid in sorted(layers.keys()):
        counts = layers[lid]
        total = sum(counts)
        if total == 0:
            continue
        mu = total / N
        sc = sorted(counts, reverse=True)
        cv = math.sqrt(sum((c-mu)**2 for c in counts)/N) / mu
        active = sum(1 for c in counts if c > 0)
        print(f"  L{lid:2d}: max/μ={sc[0]/mu:5.1f}x  CV={cv:.2f}  "
              f"Top1={sc[0]/total*100:4.1f}%  Top4={sum(sc[:4])/total*100:4.1f}%  "
              f"Top10={sum(sc[:10])/total*100:4.1f}%  active={active}/{N}")
    
    # 不均衡对比
    print(f"\n[2] GPU 负载不均衡比 (max_GPU / mean_GPU)")
    print(f"{'EP':>4} {'N_loc':>5} | {'连续(均)':>8} {'连续(P95)':>9} | "
          f"{'RR(均)':>7} {'RR(P95)':>8} | {'LPT(均)':>8} {'LPT(P95)':>9} | {'LPT改善':>7}")
    print("-" * 90)
    for ep in sorted(results.keys()):
        r = results[ep]
        n = len(r['contiguous'])
        
        def stats(arr):
            arr_s = sorted(arr)
            return sum(arr)/n, arr_s[int(n*0.95)]
        
        mc, p95c = stats(r['contiguous'])
        mr, p95r = stats(r['roundrobin'])
        ml, p95l = stats(r['lpt'])
        improve = (1 - ml/mc) * 100
        
        print(f"{ep:4d} {r['n_local']:5d} | {mc:8.2f}x {p95c:8.2f}x | "
              f"{mr:7.2f}x {p95r:7.2f}x | {ml:8.2f}x {p95l:8.2f}x | {improve:6.1f}%")
    
    # 本地命中率
    print(f"\n[3] 本地命中率 & MoE_local 时间估算")
    print(f"{'EP':>4} {'N_loc':>5} {'命中率':>8} | ", end="")
    for B in [4096, 8192, 16384, 32768]:
        print(f"B={B:5d} ", end="")
    print()
    print("-" * 80)
    for ep in sorted(results.keys()):
        n_local = results[ep]['n_local']
        hr = n_local / N
        print(f"{ep:4d} {n_local:5d} {hr*100:7.2f}% | ", end="")
        for B in [4096, 8192, 16384, 32768]:
            local_req = B / ep * K * hr
            moe_local_ms = local_req * 0.05  # ~0.05ms per token-expert pair
            print(f"{moe_local_ms:6.1f}ms", end="")
        print()

if __name__ == "__main__":
    meta, layers = load_data()
    results = analyze_imbalance(layers, meta['num_experts'])
    print_results(meta, layers, results)
