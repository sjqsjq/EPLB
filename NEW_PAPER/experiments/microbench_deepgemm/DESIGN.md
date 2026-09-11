# 算子级 Microbenchmark: DeepGEMM 专家成本模型

## 目标

在 DeepGEMm grouped GEMM kernel 级别(非服务级)测量:
1. **per-expert time vs tokens/expert (N)**: 验证 flat→linear 转折点 n*
2. **per-GPU time vs activated expert count (G)**: 验证 activation floor b
3. **swap cost vs GEMM time**: 验证 swap 何时划算(我们的死区)

## 模型参数
- Qwen3-235B-A22B-FP8: hidden=4096, moe_intermediate=1536, 128 experts, top-8
- EP=8: 16 experts/GPU
- Expert weight(FP8): w13=12MB, w2=6MB, total=18MB/expert/layer
- H20 HBM bandwidth: ~3.35 TB/s -> b = 18MB/3.35TB/s = 5.6us (理论 activation floor)

## 实验 1: per-expert time vs N (TEMPO Fig 2a)
固定 G=1, 扫 N={1,2,4,8,16,32,64,128,256,512,1024,2048,4096}
预期: flat 到 N~128-156, 然后线性 -> 找 n*

## 实验 2: per-GPU time vs G (TEMPO Fig 3b)  
固定 N=1, 扫 G={1,2,4,8,16,32}
预期: time ~ G (b per expert), N=1 和 N=64 重合(flat 区 N 无关)

## 实验 3: swap cost vs GEMM benefit
测 A: P2P swap 18MB expert weight 时间
测 B: 该 expert 在 N tokens 下 GEMM 时间
crossover N*: swap_cost / beta -> 死区 per-expert 版本

## 实验 4: layer-level (连接服务级)
固定 G=16, 扫 total_tokens/GPU={16,64,128,256,512,1024,2048,4096,8192}
-> Fig I (T(r) hinge) 的算子级版本

## 实现
直接调用 deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked
- dummy FP8 expert weights (正确 shape + scale)
- dummy FP8 input tokens (N per expert)  
- masked_m = [N]*G + [0]*(16-G)
- CUDA event timing (100 次取中位数)
- 无需 SGLang server -> 零服务噪声

## 输出图
1. per-expert time vs N (flat->linear, n* 标注)
2. per-GPU time vs G (linear in G, b 标注)  
3. swap_cost 线 vs GEMM time 曲 -> crossover N*
4. layer-level T(r) vs tokens/GPU -> 连接服务级死区
5. 拟合参数 b, beta, n* for Qwen3-235B
