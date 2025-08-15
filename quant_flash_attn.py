# -*- coding: utf-8 -*-
"""
A complete and optimized implementation for INT8 per-token quantized flash attention with a backward pass.
"""

import torch
import triton
import triton.language as tl

# ==============================================================================
# KERNELS
# ==============================================================================

COMPREHENSIVE_ATTN_CONFIGS = [
    triton.Config({'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BN in [32, 64, 128, 256]
    for s in [2, 3, 4, 5, 6]
    for w in [4, 8]
]

# --- Quantized Forward Kernel ---
@triton.autotune(
    configs=COMPREHENSIVE_ATTN_CONFIGS, # Apply the new rich configs
    key=['N_CTX_KV', 'HEAD_DIM'],
)
@triton.jit
def _attn_fwd_kernel_quantized(
    Q, KV_quant, KV_scale, KV_zero_point, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kvq_z, stride_kvq_h, stride_kvq_m, stride_kvq_k,
    stride_kvs_z, stride_kvs_h, stride_kvs_m,
    stride_kvz_z, stride_kvz_h, stride_kvz_m,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX_Q, N_CTX_KV, sm_scale,
    HEAD_DIM: tl.constexpr, PADDED_N_CTX_Q: tl.constexpr, BLOCK_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    off_z = (program_id // H).to(tl.int64)
    off_h = (program_id % H).to(tl.int64)
    q_base_ptr = Q + off_z * stride_qz + off_h * stride_qh
    kv_quant_base_ptr = KV_quant + off_z * stride_kvq_z + off_h * stride_kvq_h
    kv_scale_base_ptr = KV_scale + off_z * stride_kvs_z + off_h * stride_kvs_h
    kv_zp_base_ptr = KV_zero_point + off_z * stride_kvz_z + off_h * stride_kvz_h
    out_base_ptr = Out + off_z * stride_oz + off_h * stride_oh
    offs_m = tl.arange(0, PADDED_N_CTX_Q)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = q_base_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m < N_CTX_Q
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    q_dtype = Q.type.element_ty
    q = (q * sm_scale).to(q_dtype)
    acc = tl.zeros([PADDED_N_CTX_Q, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([PADDED_N_CTX_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([PADDED_N_CTX_Q], dtype=tl.float32)
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        current_offs_n = start_n + offs_n
        kv_mask = current_offs_n < N_CTX_KV
        v_quant_ptrs = kv_quant_base_ptr + current_offs_n[:, None] * stride_kvq_m + offs_d[None, :] * stride_kvq_k
        v_quant = tl.load(v_quant_ptrs, mask=kv_mask[:, None], other=0)
        scale_ptrs = kv_scale_base_ptr + current_offs_n[:, None] * stride_kvs_m
        scales_block = tl.load(scale_ptrs, mask=kv_mask[:, None], other=1.0)
        zp_ptrs = kv_zp_base_ptr + current_offs_n[:, None] * stride_kvz_m
        zps_block = tl.load(zp_ptrs, mask=kv_mask[:, None], other=0)
        dequantized_v = (v_quant.to(tl.float32) - zps_block.to(tl.float32)) * scales_block.to(tl.float32)
        v = dequantized_v.to(q_dtype)
        k = tl.trans(v)
        qk = tl.dot(q, k)
        qk = tl.where(q_mask[:, None], qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(q_dtype), v)
        m_i = m_ij
    l_i_inv = 1.0 / l_i
    acc = acc * l_i_inv[:, None]
    m_i += tl.log(l_i)
    m_ptr = M + program_id * PADDED_N_CTX_Q + offs_m
    tl.store(m_ptr, m_i, mask=q_mask)
    out_ptrs = out_base_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(out_ptrs, acc.to(q_dtype), mask=q_mask[:, None])


# --- Backward Preprocess Kernel ---
@triton.jit
def _bwd_preprocess_kernel(O, DO, Delta, stride_oz, stride_oh, stride_om, stride_ok, Z, H, N_CTX_Q, HEAD_DIM: tl.constexpr, PADDED_N_CTX_Q: tl.constexpr):
    program_id = tl.program_id(0)
    o_base_ptr = O + (program_id // H) * stride_oz + (program_id % H) * stride_oh
    do_base_ptr = DO + (program_id // H) * stride_oz + (program_id % H) * stride_oh
    offs_m = tl.arange(0, PADDED_N_CTX_Q)
    offs_k = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_CTX_Q
    o_ptrs = o_base_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    do_ptrs = do_base_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    o = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)
    delta = tl.sum(o * do, axis=1)
    delta_ptr = Delta + program_id * PADDED_N_CTX_Q + offs_m
    tl.store(delta_ptr, delta, mask=m_mask)

# --- Quantized Backward Kernel ---
@triton.autotune(
    configs=COMPREHENSIVE_ATTN_CONFIGS, # Apply the new rich configs
    key=['N_CTX_KV', 'HEAD_DIM'],
)
@triton.jit
def _bwd_kernel_quantized(
    Q, KV_quant, KV_scale, KV_zero_point, LSE, DO, Delta,
    DQ, DKV,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kvq_z, stride_kvq_h, stride_kvq_m, stride_kvq_k,
    stride_kvs_z, stride_kvs_h, stride_kvs_m,
    stride_kvz_z, stride_kvz_h, stride_kvz_m,
    stride_doz, stride_doh, stride_dom, stride_dok,
    Z, H, N_CTX_Q, N_CTX_KV, sm_scale,
    HEAD_DIM: tl.constexpr, PADDED_N_CTX_Q: tl.constexpr, BLOCK_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    off_z = (program_id // H).to(tl.int64)
    off_h = (program_id % H).to(tl.int64)
    q_base_ptr = Q + off_z * stride_qz + off_h * stride_qh
    kv_quant_base_ptr = KV_quant + off_z * stride_kvq_z + off_h * stride_kvq_h
    kv_scale_base_ptr = KV_scale + off_z * stride_kvs_z + off_h * stride_kvs_h
    kv_zp_base_ptr = KV_zero_point + off_z * stride_kvz_z + off_h * stride_kvz_h
    do_base_ptr = DO + off_z * stride_doz + off_h * stride_doh
    dq_base_ptr = DQ + off_z * stride_qz + off_h * stride_qh
    dkv_base_ptr = DKV + off_z * stride_kvq_z + off_h * stride_kvq_h
    offs_m = tl.arange(0, PADDED_N_CTX_Q)
    offs_k = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < N_CTX_Q
    q_ptrs = q_base_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
    do_ptrs = do_base_ptr + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    do = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)
    lse = tl.load(LSE + program_id * PADDED_N_CTX_Q + offs_m, mask=m_mask)
    delta = tl.load(Delta + program_id * PADDED_N_CTX_Q + offs_m, mask=m_mask)
    dq_acc = tl.zeros([PADDED_N_CTX_Q, HEAD_DIM], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        current_offs_n = start_n + offs_n
        n_mask = current_offs_n < N_CTX_KV
        v_quant_ptrs = kv_quant_base_ptr + current_offs_n[:, None] * stride_kvq_m + offs_k[None, :] * stride_kvq_k
        v_quant = tl.load(v_quant_ptrs, mask=n_mask[:, None], other=0)
        scale_ptrs = kv_scale_base_ptr + current_offs_n[:, None] * stride_kvs_m
        scales_block = tl.load(scale_ptrs, mask=n_mask[:, None], other=1.0)
        zp_ptrs = kv_zp_base_ptr + current_offs_n[:, None] * stride_kvz_m
        zps_block = tl.load(zp_ptrs, mask=n_mask[:, None], other=0)
        dequantized_v = (v_quant.to(tl.float32) - zps_block.to(tl.float32)) * scales_block.to(tl.float32)
        v = dequantized_v.to(q.dtype)
        k = tl.trans(v)
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(m_mask[:, None], qk, -float("inf"))
        p = tl.exp(qk - lse[:, None])
        dv_block = tl.dot(p.to(do.dtype).T, do)
        dp_block = tl.dot(do, v.to(do.dtype).T)
        ds_block = p * (dp_block - delta[:, None])
        dq_acc += tl.dot(ds_block.to(v.dtype), v)
        dk_block = tl.dot(ds_block.to(q.dtype).T, q)
        dkv_block = dv_block + dk_block * sm_scale
        dkv_ptrs = dkv_base_ptr + current_offs_n[None, :] * stride_kvq_m + offs_k[:, None] * stride_kvq_k
        tl.store(dkv_ptrs, dkv_block.T, mask=n_mask[None, :])
    dq_acc *= sm_scale
    dq_ptrs = dq_base_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    tl.store(dq_ptrs, dq_acc, mask=m_mask[:, None])


# ==============================================================================
# AUTOGRAD FUNCTION
# ==============================================================================

class QuantFlashAttention(torch.autograd.Function):
    """
    Aautograd Quanted Flash Attention Function.
    """

    @staticmethod
    def forward(ctx, q, kv_fp16, kv_quant, sm_scale):
        Z, H, N_CTX_Q, HEAD_DIM = q.shape
        _, _, N_CTX_KV, _ = kv_fp16.shape
        PADDED_N_CTX_Q = max(triton.next_power_of_2(N_CTX_Q), 16)
        kv_quant, kv_scale, kv_zero_point = kv_quant
        o = torch.empty_like(q)
        lse = torch.empty((Z, H, PADDED_N_CTX_Q), device=q.device, dtype=torch.float32)

        grid = (Z * H,)
        _attn_fwd_kernel_quantized[grid](
            q, kv_quant, kv_scale, kv_zero_point, lse, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            kv_quant.stride(0), kv_quant.stride(1), kv_quant.stride(2), kv_quant.stride(3),
            kv_scale.stride(0), kv_scale.stride(1), kv_scale.stride(2),
            kv_zero_point.stride(0), kv_zero_point.stride(1), kv_zero_point.stride(2),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            Z, H, N_CTX_Q, N_CTX_KV, sm_scale,
            HEAD_DIM=HEAD_DIM, PADDED_N_CTX_Q=PADDED_N_CTX_Q
        )

        ctx.save_for_backward(q, o, lse, kv_quant, kv_scale, kv_zero_point)
        ctx.sm_scale = sm_scale
        ctx.Z, ctx.H, ctx.N_CTX_Q, ctx.HEAD_DIM = Z, H, N_CTX_Q, HEAD_DIM
        ctx.PADDED_N_CTX_Q = PADDED_N_CTX_Q
        
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, o, lse, kv_quant, kv_scale, kv_zero_point = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        Z, H, N_CTX_Q, HEAD_DIM = ctx.Z, ctx.H, ctx.N_CTX_Q, ctx.HEAD_DIM
        PADDED_N_CTX_Q = ctx.PADDED_N_CTX_Q
        
        dq = torch.empty_like(q)
        dkv = torch.empty(kv_quant.shape, dtype=q.dtype, device=q.device)
        
        delta = torch.empty_like(lse)
        grid_pre = (Z * H,)
        _bwd_preprocess_kernel[grid_pre](
            o, grad_output, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            Z, H, N_CTX_Q,
            HEAD_DIM=HEAD_DIM, PADDED_N_CTX_Q=PADDED_N_CTX_Q
        )
        
        grid_main = (Z * H,)
        _bwd_kernel_quantized[grid_main](
            q, kv_quant, kv_scale, kv_zero_point, lse, grad_output, delta, dq, dkv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            kv_quant.stride(0), kv_quant.stride(1), kv_quant.stride(2), kv_quant.stride(3),
            kv_scale.stride(0), kv_scale.stride(1), kv_scale.stride(2),
            kv_zero_point.stride(0), kv_zero_point.stride(1), kv_zero_point.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            Z, H, N_CTX_Q, kv_quant.shape[2], sm_scale,
            HEAD_DIM=HEAD_DIM, PADDED_N_CTX_Q=PADDED_N_CTX_Q,
        )
        
        return dq, dkv, None, None

# Use your requested naming for the final callable object
quant_flash_attn = QuantFlashAttention.apply