# -*- coding: utf-8 -*-
"""
benchmark_with_backward.py

本脚本旨在全面对比以下三种Attention实现的前向与反向传播速度：
1. PyTorch原生 scaled_dot_product_attention (FP16 基准).
2. 标准Triton版 Flash Attention V2 (源自 flash_attn.py, FP16).
3. 您的最终版量化Attention (源自 quant_flash_attn.py, INT8输入).
"""

import torch
import triton
import triton.testing
import pandas as pd

# ------------------------------------------------------------------
# 1. 导入所有需要对比的实现
# ------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func as flash_attention_v2
    
    # 导入您的最终版量化Attention
    from quant_flash_attn import quant_flash_attn
    
    # 导入量化函数，用于在测试前准备数据
    from asymmetric_quant import asymmetric_quant_per_token

    print("成功导入所有必要的Attention和量化函数。")
except ImportError as e:
    print(f"--- 导入错误 ---")
    print(f"无法导入自定义函数。细节: {e}")
    print("请确保 'flash_attn.py', 'quant_flash_attn.py', 和 'asymmetric_quant.py' 文件与本脚本在同一目录下。")
    exit()

# ------------------------------------------------------------------
# 2. 速度对比测试
# ------------------------------------------------------------------

# 为前向和反向传播分别创建Benchmark配置
benchmark_configs = []
for mode in ['fwd', 'bwd']:
    benchmark_configs.append(
        triton.testing.Benchmark(
            x_names=['N_CTX_KV'],
            x_vals=[512, 1024, 2048, 4096],
            line_arg='provider',
            line_vals=['pytorch_fp16', 'triton_flash_v2', 'triton_quantized'],
            line_names=[
                'PyTorch FP16 (基准)', 
                'Triton Flash V2 (FP16)', 
                '我的量化版 Triton (INT8)'
            ],
            styles=[('green', '-'), ('blue', '-'), ('purple', '-.')],
            ylabel="TFLOP/s",
            plot_name=f"attention_benchmark_mode_{mode}",
            args={
                'Z': 256, 'H': 16, 'N_CTX_Q': 16, 'HEAD_DIM': 128, 
                'dtype': torch.float16, 'mode': mode
            }
        )
    )

@triton.testing.perf_report(benchmark_configs)
def benchmark_attention(Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM, dtype, mode, provider):
    """
    性能测试函数，覆盖前向和反向传播。
    """
    device = "cuda"
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    # --- a. 准备需要梯度的数据 ---
    q = torch.randn((Z, H, N_CTX_Q, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    # 为了公平对比，所有实现都使用相同的长短序列
    # 对于通用版Flash Attention V2，我们将使用一个不等长的修改版或wrapper
    kv_fp16 = torch.randn((Z, H, N_CTX_KV, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    
    # --- b. 为量化版本提前准备好输入数据 ---
    kv_quant_data = asymmetric_quant_per_token(kv_fp16)

    # --- c. 根据provider选择要计时的函数 ---
    if provider == 'pytorch_fp16':
        fn_fwd = lambda: torch.nn.functional.scaled_dot_product_attention(q, kv_fp16, kv_fp16, is_causal=False, scale=sm_scale)
    
    elif provider == 'triton_flash_v2':
        kv_fp16_dao = kv_fp16.transpose(1,2).contiguous()  # 确保数据是连续的
        kv_fp16_dao = kv_fp16.transpose(1,2).contiguous()
        fn_fwd = lambda: flash_attention_v2(q.transpose(1,2).contiguous(), kv_fp16_dao, kv_fp16_dao, False, sm_scale)

    elif provider == 'triton_quantized':
        fn_fwd = lambda: quant_flash_attn(q, kv_fp16, kv_quant_data, sm_scale)
        
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # --- d. 定义前向和反向的计时函数 ---
    if mode == 'fwd':
        fn = fn_fwd
    elif mode == 'bwd':
        # 先执行一次前向传播（不计时）
        output = fn_fwd()
        # 创建上游梯度
        grad_output = torch.randn_like(output)
        # 计时的函数是反向传播
        fn = lambda: output.backward(grad_output, retain_graph=True)

    # --- e. 运行性能测试 ---
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)

    # --- f. 计算 TFLOP/s ---
    # 使用官方FlashAttention论文中的FLOPs计算惯例
    flops_per_matmul = 2 * Z * H * N_CTX_Q * N_CTX_KV * HEAD_DIM
    total_flops = 2 * flops_per_matmul # Forward pass has 2 matmuls
    
    if mode == 'bwd':
        # Backward pass is approx 2.5x the complexity of forward (2.0 for bwd, 0.5 for recomputation)
        total_flops *= 2.5
        
    tflops = total_flops / (ms * 1e-3) / 1e12
    
    return tflops

# ------------------------------------------------------------------
# 3. 主程序入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("开始对三种Attention实现进行前向与反向传播速度对比测试...")
    print("将分别为 forward 和 backward 生成独立的性能图表。")
    print("="*60)
    
    benchmark_attention.run(print_data=True, show_plots=True, save_path='.')