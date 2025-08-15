# -*- coding: utf-8 -*-
"""
per_token_quant.py

This script provides a high-performance INT8 asymmetric quantization kernel
implemented in Triton, specifically for per-token quantization.

It includes:
1. An autotuned Triton kernel for per-token (or per-row) asymmetric quantization.
2. A precision verification routine to measure quantization error.
3. A performance measurement routine for the Triton kernel.
"""

import torch
import triton
import triton.language as tl

# ==============================================================================
# 1. Triton Kernel and Public API for Per-Token Quantization
# ==============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 256}, num_warps=8),
    ],
    key=['HEAD_DIM'], # The performance is mainly dependent on the head dimension
)
@triton.jit
def _asymmetric_quant_kernel(
    # --- Pointers to Tensors ---
    input_ptr,
    quantized_output_ptr,
    scale_ptr,
    zero_point_ptr,
    # --- Stride Info ---
    stride_z, stride_h, stride_m, stride_k,
    # --- Dimensions ---
    Z, H, M, HEAD_DIM: tl.constexpr,
    # --- Block Size for Tiling within a token vector ---
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Triton kernel for per-token INT8 asymmetric quantization.
    Each program in the grid processes one token vector of size HEAD_DIM.
    """
    # --- 1. Initialization ---
    # The grid is 1D, with a total of Z * H * M programs.
    # Each program computes quantization for a single token.
    pid = tl.program_id(0)

    # [核心改动] 将一维的pid映射回三维的(z, h, m)索引
    pid_64 = pid.to(tl.int64)
    M_64 = M.to(tl.int64)
    H_64 = H.to(tl.int64)

    m_idx = pid_64 % M_64
    h_idx = (pid_64 // M_64) % H_64
    z_idx = pid_64 // (M_64 * H_64)
    # 定位到当前token的起始地址
    token_input_ptr = input_ptr + z_idx * stride_z + h_idx * stride_h + m_idx * stride_m
    token_quantized_output_ptr = quantized_output_ptr + z_idx * stride_z + h_idx * stride_h + m_idx * stride_m
    
    # --- 2. Find Min/Max of the Token Vector ---
    min_val = float('inf')
    max_val = float('-inf')
    col_offsets = tl.arange(0, BLOCK_SIZE_K)
    
    # 循环遍历整个token向量 (长度为HEAD_DIM)
    for block_start in range(0, HEAD_DIM, BLOCK_SIZE_K):
        current_offsets = block_start + col_offsets
        mask = current_offsets < HEAD_DIM
        
        # 加载token向量的一个分块
        values = tl.load(token_input_ptr + current_offsets * stride_k, mask=mask, other=0.0)
        
        # 更新min/max
        min_val = tl.minimum(min_val, tl.min(tl.where(mask, values, float('inf'))))
        max_val = tl.maximum(max_val, tl.max(tl.where(mask, values, float('-inf'))))

    # --- 3. Compute and Store Scale and Zero-Point for this token ---
    q_max = 127.0
    q_min = -128.0
    scale = (max_val - min_val) / (q_max - q_min)
    scale = tl.where(scale == 0, 1.0, scale)
    
    zero_point_float = q_min - min_val / scale
    zero_point = tl.extra.cuda.libdevice.round(zero_point_float)
    zero_point = tl.maximum(q_min, tl.minimum(q_max, zero_point))

    tl.store(scale_ptr + pid, scale.to(tl.float16))
    tl.store(zero_point_ptr + pid, zero_point.to(tl.int8))

    # --- 4. Quantize and Store the Token Data ---
    for block_start in range(0, HEAD_DIM, BLOCK_SIZE_K):
        current_offsets = block_start + col_offsets
        mask = current_offsets < HEAD_DIM
        
        values = tl.load(token_input_ptr + current_offsets * stride_k, mask=mask, other=0.0)
        
        quantized_values_float = tl.extra.cuda.libdevice.round(values / scale + zero_point)
        quantized_values = tl.maximum(q_min, tl.minimum(q_max, quantized_values_float))
        
        tl.store(token_quantized_output_ptr + current_offsets * stride_k, quantized_values.to(tl.int8), mask=mask)


def asymmetric_quant_per_token(input_tensor: torch.Tensor):
    """
    Quantizes a 4D tensor to INT8 using the per-token Triton kernel.

    Args:
        input_tensor (torch.Tensor): A 4D tensor with shape
                                     (Batch, Num_Heads, Seq_Len, Head_Dim) in FP16 format.

    Returns:
        A tuple containing:
        - quantized_tensor (torch.Tensor): The quantized INT8 tensor of the same shape.
        - scale (torch.Tensor): The FP16 scale factors, shape (Batch, Num_Heads, Seq_Len, 1).
        - zero_point (torch.Tensor): The INT8 zero-points, shape (Batch, Num_Heads, Seq_Len, 1).
    """
    Z, H, M, K = input_tensor.shape
    device = input_tensor.device

    # Create output tensors with the correct per-token shape.
    quantized_output = torch.empty_like(input_tensor, dtype=torch.int8)
    scale = torch.empty(Z, H, M, 1, dtype=torch.float16, device=device)
    zero_point = torch.empty(Z, H, M, 1, dtype=torch.int8, device=device)

    # [核心改动] Grid size is now the total number of tokens.
    grid = (Z * H * M,)
    
    # Kernel expects 1D views of scale and zp for simple indexing with pid.
    scale_1d = scale.view(-1)
    zero_point_1d = zero_point.view(-1)

    _asymmetric_quant_kernel[grid](
        input_tensor,
        quantized_output,
        scale_1d,
        zero_point_1d,
        # Strides are for the original 4D tensor.
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        # Dimensions
        Z, H, M, HEAD_DIM=K,
    )

    return quantized_output, scale, zero_point

# ==============================================================================
# 2. Main Execution Block for Verification and Benchmarking
# ==============================================================================

if __name__ == "__main__":

    # --- Helper function defined inside main scope ---
    def dequantize_per_token_torch(quantized_tensor, scale, zero_point):
        """Helper function to dequantize per-token tensors."""
        return (quantized_tensor.to(torch.float32) - zero_point.to(torch.float32)) * scale.to(torch.float32)

    # --- 1. Precision Verification ---
    def verify_precision():
        print("\n" + "="*60)
        print("Running Per-Token Quantization Precision Verification")
        print("="*60)

        test_shapes = [
            (1, 4, 512, 64),
            (2, 8, 1023, 128),
            (4, 16, 4099, 64),
            (8, 32, 2041, 128),
            (512, 32, 2041, 128),
        ]

        for Z, H, M, K in test_shapes:
            print(f"\n--- Testing Shape (Z, H, M, K): ({Z}, {H}, {M}, {K}) ---")
            
            input_fp16 = torch.randn((Z, H, M, K), dtype=torch.float16, device="cuda")
            
            # Run Triton per-token quantization.
            quant_triton, scale_triton, zp_triton = asymmetric_quant_per_token(input_fp16)
            
            # Dequantize the result back to float.
            dequantized_triton = dequantize_per_token_torch(quant_triton, scale_triton, zp_triton)
            # Calculate the Mean Absolute Error (MAE).
            mae = torch.mean(torch.abs(input_fp16.float() - dequantized_triton.float())).item()
            print(f"✅ Quantization Precision Loss (MAE): {mae:.6f}")
    
    # --- 2. Performance Measurement ---
    def measure_performance():
        print("\n" + "="*60)
        print("Running Per-Token Quantization Performance Measurement")
        print("="*60)

        Z, H, M, K = 8, 32, 4096, 128
        print(f"Benchmark Shape (Z, H, M, K): ({Z}, {H}, {M}, {K})")
        input_fp16 = torch.randn((Z, H, M, K), dtype=torch.float16, device="cuda")

        print("Warming up and running autotuner... (this may take a moment)")
        ms = triton.testing.do_bench(lambda: asymmetric_quant_per_token(input_fp16))
        
        num_tokens = Z * H * M
        ms_per_token = (ms / num_tokens) * 1000 # in microseconds
        
        print(f"✅ Autotuned Triton Kernel Execution Time: {ms:.4f} ms")
        print(f"   -> Average time per token: {ms_per_token:.4f} µs")

    # --- Execute Verification and Benchmarking ---
    verify_precision()
    measure_performance()