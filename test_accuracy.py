# -*- coding: utf-8 -*-
"""
test_accuracy.py

A comprehensive script to test the accuracy of the custom quantized attention
implementation against the native PyTorch FP16 baseline.

This script verifies both the forward pass output and the backward pass gradients
across a variety of tensor shapes.
"""

import torch
import pytest

# ------------------------------------------------------------------
# 1. Import your custom functions
# ------------------------------------------------------------------
try:
    # Import the per-token quantization function
    from asymmetric_quant import asymmetric_quant_per_token
    
    # Import the quantized attention autograd function
    from quant_flash_attn import quant_flash_attn
    print("Successfully imported custom quantization and attention functions.")
except ImportError as e:
    print("--- IMPORT ERROR ---")
    print(f"Failed to import necessary functions. Details: {e}")
    print("Please ensure 'asymmetric_quant.py' and 'quant_flash_attn.py' are in the same directory.")
    exit()

# ------------------------------------------------------------------
# 2. Define a comprehensive set of test parameters
# ------------------------------------------------------------------
# We will test various shapes for (Batch, Heads, Q_Len, KV_Len, Head_Dim)
TEST_SHAPES = [
    # Standard case
    (2, 4, 18, 512, 64),
    # Larger head dimension
    (2, 8, 32, 1024, 128),
    # Larger batch and head count
    (4, 16, 16, 2048, 64),
    # Sequence lengths that are not powers of two or multiples of block size
    (1, 2, 17, 1023, 128),
    (99, 53, 97, 689, 128),
]

@pytest.mark.parametrize("Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM", TEST_SHAPES)
def test_quantized_attention_accuracy(Z, H, N_CTX_Q, N_CTX_KV, HEAD_DIM):
    """
    A comprehensive test that compares the forward and backward passes of the
    custom quantized attention against the PyTorch FP16 reference.
    """
    print("\n" + "="*70)
    print(f"Running Test for Shape (Z,H,Q,KV,D) = ({Z},{H},{N_CTX_Q},{N_CTX_KV},{HEAD_DIM})")
    print("="*70)
    
    # --- Test Setup ---
    torch.manual_seed(20)
    dtype = torch.float16
    device = "cuda"

    # Create reference tensors for the PyTorch path
    q_fp16_ref = torch.randn((Z, H, N_CTX_Q, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    kv_fp16_ref = torch.randn((Z, H, N_CTX_KV, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    
    # Create identical tensors for our custom Triton path
    q_triton = q_fp16_ref.clone().detach().requires_grad_(True)
    kv_triton_fp16 = kv_fp16_ref.clone().detach().requires_grad_(True)
    
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    # Create a random upstream gradient for the backward pass
    grad_output = torch.randn_like(q_fp16_ref)

    # --- 1. PyTorch FP16 Reference Path ---
    print("Step 1: Computing reference forward and backward pass with PyTorch (FP16)...")
    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_fp16_ref, kv_fp16_ref, kv_fp16_ref, is_causal=False, scale=sm_scale
    )
    ref_out.backward(grad_output)
    ref_grad_q = q_fp16_ref.grad.clone()
    ref_grad_kv = kv_fp16_ref.grad.clone()
    print("   ...Done.")

    # --- 2. Custom Triton Quantized Path ---
    print("Step 2: Running custom quantized attention path...")
    
    # Step 2a: Perform quantization externally, as per your design
    kv_quant = asymmetric_quant_per_token(kv_triton_fp16)
    
    # Step 2b: Run the custom forward pass
    # The FP16 `kv_triton_fp16` is passed as a placeholder to receive the gradient.
    # The quantized tensors are used for the actual computation.
    quant_out = quant_flash_attn(
        q_triton, 
        kv_triton_fp16, # Placeholder for gradient
        kv_quant, 
        sm_scale
    )
    
    # Step 2c: Run the custom backward pass
    quant_out.backward(grad_output)
    tri_grad_q = q_triton.grad
    tri_grad_kv = kv_triton_fp16.grad
    print("   ...Done.")

    # --- 3. Report Results ---
    print("\n--- ACCURACY REPORT ---")
    
    # Compare forward pass outputs
    fwd_mae = torch.mean(torch.abs(ref_out - quant_out)).item()
    fwd_max_err = torch.max(torch.abs(ref_out - quant_out)).item()
    print("\n[Forward Pass]")
    print(f"  - Mean Absolute Error (MAE): {fwd_mae:.6f}")
    print(f"  - Maximum Error:           {fwd_max_err:.6f}")
    assert torch.allclose(ref_out, quant_out, atol=1e-1, rtol=0), "Forward pass output deviates too much!"

    # Compare dQ gradients
    dq_mae = torch.mean(torch.abs(ref_grad_q - tri_grad_q)).item()
    dq_max_err = torch.max(torch.abs(ref_grad_q - tri_grad_q)).item()
    print("\n[Backward Pass - dQ Gradient]")
    print(f"  - Mean Absolute Error (MAE): {dq_mae:.6f}")
    print(f"  - Maximum Error:           {dq_max_err:.6f}")
    assert torch.allclose(ref_grad_q, tri_grad_q, atol=1e-2, rtol=0.01), "dQ gradient deviates too much!"

    # Compare dKV gradients
    dkv_mae = torch.mean(torch.abs(ref_grad_kv - tri_grad_kv)).item()
    dkv_max_err = torch.max(torch.abs(ref_grad_kv - tri_grad_kv)).item()
    print("\n[Backward Pass - dKV Gradient]")
    print(f"  - Mean Absolute Error (MAE): {dkv_mae:.6f}")
    print(f"  - Maximum Error:           {dkv_max_err:.6f}")
    assert torch.allclose(ref_grad_kv, tri_grad_kv, atol=1e-2, rtol=0.02), "dKV gradient deviates too much!"
    
    print("\n✅ All checks passed for this shape!")


if __name__ == "__main__":
    # To run this test, you need to install pytest: pip install pytest
    # Then simply run in your terminal: pytest test_accuracy.py
    print("="*70)
    print("This script is designed to be run with pytest.")
    print("To execute the tests, please run the following command in your terminal:")
    print("\n  pytest -v test_accuracy.py\n")
    print("="*70)