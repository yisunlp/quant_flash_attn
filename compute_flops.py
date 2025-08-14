# -*- coding: utf-8 -*-
"""
benchmark_final.py

本脚本旨在全面对比以下两种高性能Attention实现的 **前向与反向传播** 的吞吐量 (TFLOP/s) 和内存效率：
1. 您定制的、基于Triton的INT8量化Attention实现。
2. TriDao官方的、业界标准的FlashAttention (FP16)实现。

反向传播的性能通过“差值法”进行测量，以避免`retain_graph=True`可能导致的OOM问题。
"""

import torch
import triton.testing
import pandas as pd

# ------------------------------------------------------------------
# 1. 导入必要的函数
# ------------------------------------------------------------------
try:
    from asymmetric_quant import asymmetric_quant_per_token
    from quant_flash_attn import quant_flash_attn
    print("成功导入您的自定义量化Attention函数。")
except ImportError as e:
    print(f"--- 导入错误 ---\n无法导入自定义函数。细节: {e}\n请确保 'quant_flash_attn.py' 和 'asymmetric_quant.py' 文件与本脚本在同一目录下。")
    exit()

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = True
    print("成功导入官方FlashAttention库。")
except ImportError:
    HAS_FLASH_ATTN = False
    print("警告: 未找到官方FlashAttention库，将跳过对其的测试。")

# ------------------------------------------------------------------
# 2. 辅助计算函数
# ------------------------------------------------------------------

def calculate_theoretical_io(Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM, provider, mode):
    """计算Attention的理论总访存量（读+写）。"""
    FP16_SIZE, FP32_SIZE, INT8_SIZE = 2, 4, 1
    
    # --- 前向传播访存 ---
    read_q = Z * H * N_CTX_Q * HEAD_DIM * FP16_SIZE
    write_o = Z * H * N_CTX_Q * HEAD_DIM * FP16_SIZE
    write_lse = Z * H * N_CTX_Q * FP32_SIZE
    common_io_fwd = read_q + write_o + write_lse

    if provider == 'fp16':
        read_k = Z * H * N_CTX_KV * HEAD_DIM * FP16_SIZE
        read_v = Z * H * N_CTX_KV * HEAD_DIM * FP16_SIZE
        io_fwd = common_io_fwd + read_k + read_v
    elif provider == 'quantized':
        read_kv_quant = Z * H * N_CTX_KV * HEAD_DIM * INT8_SIZE
        read_kv_scale = Z * H * N_CTX_KV * FP16_SIZE
        read_kv_zp = Z * H * N_CTX_KV * INT8_SIZE
        io_fwd = common_io_fwd + read_kv_quant + read_kv_scale + read_kv_zp
    
    if mode == 'fwd':
        return io_fwd

    # --- 反向传播额外访存 ---
    read_o_bwd = Z * H * N_CTX_Q * HEAD_DIM * FP16_SIZE
    read_do_bwd = Z * H * N_CTX_Q * HEAD_DIM * FP16_SIZE
    write_dq_bwd = Z * H * N_CTX_Q * HEAD_DIM * FP16_SIZE
    read_lse_bwd = Z * H * N_CTX_Q * FP32_SIZE
    write_delta_bwd = Z * H * N_CTX_Q * FP32_SIZE
    common_io_bwd = read_o_bwd + read_do_bwd + write_dq_bwd + read_lse_bwd + write_delta_bwd
    
    if provider == 'fp16':
        read_q_bwd = Z * H * N_CTX_Q * HEAD_DIM * FP16_SIZE
        read_k_bwd = Z * H * N_CTX_KV * HEAD_DIM * FP16_SIZE
        read_v_bwd = Z * H * N_CTX_KV * HEAD_DIM * FP16_SIZE
        write_dk_bwd = Z * H * N_CTX_KV * HEAD_DIM * FP16_SIZE
        write_dv_bwd = Z * H * N_CTX_KV * HEAD_DIM * FP16_SIZE
        return common_io_bwd + read_q_bwd + read_k_bwd + read_v_bwd + write_dk_bwd + write_dv_bwd
    elif provider == 'quantized':
        read_q_bwd = Z * H * N_CTX_Q * HEAD_DIM * FP16_SIZE
        read_kv_quant_bwd = Z * H * N_CTX_KV * HEAD_DIM * INT8_SIZE
        read_kv_scale_bwd = Z * H * N_CTX_KV * FP16_SIZE
        read_kv_zp_bwd = Z * H * N_CTX_KV * INT8_SIZE
        write_dkv_bwd = Z * H * N_CTX_KV * HEAD_DIM * FP16_SIZE
        return common_io_bwd + read_q_bwd + read_kv_quant_bwd + read_kv_scale_bwd + read_kv_zp_bwd + write_dkv_bwd

# ------------------------------------------------------------------
# 3. 性能对比测试
# ------------------------------------------------------------------

def run_benchmark():
    print("\n" + "="*80)
    print("开始性能对比测试 (使用差值法测量反向传播，避免OOM)")
    print("="*80)
    
    # [核心修正] 将结果收集移到循环外部，以便所有HEAD_DIM的结果都在一张表里
    all_results = {}

    for HEAD_DIM in [64, 128, 256]:
        # --- a. 定义测试参数 ---
        Z, H, N_CTX_Q, N_CTX_KV = 128, 16, 16, 3000
        dtype = torch.float16
        device = "cuda"

        print(f"\n--- 正在测试 HEAD_DIM = {HEAD_DIM} ---")
        
        # --- b. 计算理论FLOPs ---
        flops_fwd = 4 * Z * H * N_CTX_Q * N_CTX_KV * HEAD_DIM
        flops_bwd = flops_fwd * 2.5

        results_per_dim = {}
        providers = ['quantized', 'fp16'] if HAS_FLASH_ATTN else ['quantized']
        names = {'quantized': '我的量化版Triton (INT8)', 'fp16': '官方FlashAttention (FP16)'}

        for provider in providers:
            # print(f"  -> 正在测试 {names[provider]}...")
            
            q = torch.randn((Z, H, N_CTX_Q, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
            kv_fp16 = torch.randn((Z, H, N_CTX_KV, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
            sm_scale = 1.0 / (HEAD_DIM ** 0.5)
            
            if provider == 'quantized':
                kv_quant_data = asymmetric_quant_per_token(kv_fp16)
                fwd_fn = lambda: quant_flash_attn(q, kv_fp16, kv_quant_data, sm_scale)
            else: # fp16
                q_tridao = q.transpose(1, 2).contiguous()
                k_tridao = kv_fp16.transpose(1, 2).contiguous()
                v_tridao = kv_fp16.transpose(1, 2).contiguous()
                fwd_fn = lambda: flash_attn_func(q_tridao, k_tridao, v_tridao, causal=False)

            ms_fwd = triton.testing.do_bench(fwd_fn, warmup=50, rep=1000)
            
            output_for_grad = fwd_fn()
            grad_output = torch.randn_like(output_for_grad)
            
            def fwd_bwd_pass():
                output = fwd_fn()
                output.backward(grad_output)

            tensors_to_clear_grad = [q, kv_fp16] if provider == 'quantized' else [q_tridao, k_tridao, v_tridao]
            ms_fwd_bwd = triton.testing.do_bench(fwd_bwd_pass, grad_to_none=tensors_to_clear_grad, warmup=50, rep=1000)
            
            ms_bwd = ms_fwd_bwd - ms_fwd

            provider_type = provider
            io_fwd = calculate_theoretical_io(Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM, provider_type, 'fwd')
            io_bwd = calculate_theoretical_io(Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM, provider_type, 'bwd')
            
            tflops_fwd = flops_fwd / (ms_fwd * 1e-3) / 1e12
            bw_fwd = io_fwd / (ms_fwd * 1e-3) / 1e9
            
            tflops_bwd = flops_bwd / (ms_bwd * 1e-3) / 1e12 if ms_bwd > 0.0001 else 0.0
            bw_bwd = io_bwd / (ms_bwd * 1e-3) / 1e9 if ms_bwd > 0.0001 else 0.0

            results_per_dim[names[provider]] = {
                'Time Fwd (ms)': ms_fwd, 'TFLOP/s Fwd': tflops_fwd, 'BW Fwd (GB/s)': bw_fwd,
                'Time Bwd (ms)': ms_bwd, 'TFLOP/s Bwd': tflops_bwd, 'BW Bwd (GB/s)': bw_bwd,
            }
        
        all_results[HEAD_DIM] = results_per_dim

    # --- d. 打印所有总结报告 ---
    for head_dim, results in all_results.items():
        print("\n" + "="*80)
        print(f"性能对比总结报告 (HEAD_DIM = {head_dim})")
        print("="*80)
        
        df = pd.DataFrame.from_dict(results, orient='index')
        
        # [核心修正] 调整列名以减少宽度，并设置pandas以获得更好的排版
        df.columns = ['Time Fwd(ms)', 'TFLOP/s Fwd', 'BW Fwd(GB/s)', 
                      'Time Bwd(ms)', 'TFLOP/s Bwd', 'BW Bwd(GB/s)']
        
        # 格式化输出
        for col in df.columns:
            df[col] = df[col].map('{:.2f}'.format)
        
        # 设置pandas显示选项以防止换行
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200) # 设置一个足够宽的宽度
        
        print(df)
        
        if HAS_FLASH_ATTN:
            fwd_speedup = results['官方FlashAttention (FP16)']['Time Fwd (ms)'] / results['我的量化版Triton (INT8)']['Time Fwd (ms)']
            bwd_speedup = results['官方FlashAttention (FP16)']['Time Bwd (ms)'] / results['我的量化版Triton (INT8)']['Time Bwd (ms)']
            print("\n--- 结论 ---")
            print(f"前向传播加速比 (INT8 vs FP16): {fwd_speedup:.2f}x")
            print(f"反向传播加速比 (INT8 vs FP16): {bwd_speedup:.2f}x")

# ------------------------------------------------------------------
# 4. 主程序入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_benchmark()