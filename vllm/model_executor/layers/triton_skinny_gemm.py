# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton skinny GEMM (few-row activations) for GPUs without vLLM's HIP skinny kernels.

Decode-time linears multiply a few activation rows by a large weight, so they are
bound by weight reads. On gfx908 (MI100), rocBLAS/hipBLASLt pick tile shapes that
reach only 5-30% of HBM bandwidth for these shapes (a 2560->1 gate takes ~0.5 ms),
and wvSplitK/LLMM1 are not built for this arch. These kernels read each weight
element once and split K when there are too few output rows to fill the GPU.
A single row uses multiply-adds; more rows use tl.dot with 16/32-row tiles, which
lower to the 16x16x8 bf16 MFMA (smaller-M bf16 dots are miscompiled on gfx908).
"""

import torch

from vllm.triton_utils import tl, triton

# Rows handled by the multiply-add kernel; up to MAX_ROWS use the tl.dot kernel,
# whose 16/32-row tiles use the 16x16x8 bf16 MFMA (correct on gfx908).
MAX_FMA_ROWS = 1
MAX_ROWS = 32


@triton.jit
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    M,
    K,
    K_PER_SPLIT,
    stride_xn,
    stride_wm,
    stride_os,
    stride_on,
    N_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, N_ROWS)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M
    n_mask = rn < N
    k_start = pid_k * K_PER_SPLIT
    k_end = tl.minimum(k_start + K_PER_SPLIT, K)
    # Accumulate elementwise and reduce over K once at the end: a cross-lane
    # reduction per iteration costs more than the weight load it follows.
    acc = tl.zeros((N_ROWS, BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        kk = k0 + rk
        k_mask = kk < k_end
        w = tl.load(
            w_ptr + rm[:, None] * stride_wm + kk[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        x = tl.load(
            x_ptr + rn[:, None] * stride_xn + kk[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += w.to(tl.float32)[None, :, :] * x.to(tl.float32)[:, None, :]
    tl.store(
        out_ptr + pid_k * stride_os + rn[:, None] * stride_on + rm[None, :],
        tl.sum(acc, axis=2).to(out_ptr.dtype.element_ty),
        mask=n_mask[:, None] & m_mask[None, :],
    )


@triton.jit
def _skinny_dot_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    M,
    K,
    K_PER_SPLIT,
    stride_xn,
    stride_wm,
    stride_os,
    stride_on,
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rr = tl.arange(0, BLOCK_R)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    r_mask = rr < N
    m_mask = rm < M
    k_start = pid_k * K_PER_SPLIT
    k_end = tl.minimum(k_start + K_PER_SPLIT, K)
    acc = tl.zeros((BLOCK_R, BLOCK_M), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        kk = k0 + rk
        k_mask = kk < k_end
        x = tl.load(
            x_ptr + rr[:, None] * stride_xn + kk[None, :],
            mask=r_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + rm[:, None] * stride_wm + kk[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(
        out_ptr + pid_k * stride_os + rr[:, None] * stride_on + rm[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=r_mask[:, None] & m_mask[None, :],
    )


def _split_k(m_programs: int, k: int, block_k: int, target: int) -> tuple[int, int]:
    """Return (split_k, k_per_split) so that about `target` programs run."""
    split_k = 1
    if m_programs < target:
        split_k = max(1, min(triton.cdiv(target, m_programs), triton.cdiv(k, block_k)))
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    return triton.cdiv(k, k_per_split), k_per_split


def _launch_config(n: int, m: int, k: int, num_cus: int) -> tuple[int, int, int, int]:
    """Return (BLOCK_M, BLOCK_K, num_warps, split_k)."""
    block_k = 512 if k >= 512 else triton.next_power_of_2(k)
    block_m = 1 if n == 1 else 2
    num_warps = 2
    programs_m = triton.cdiv(m, block_m)
    target = 4 * num_cus
    split_k = 1
    if programs_m < target:
        split_k = min(triton.cdiv(target, programs_m), triton.cdiv(k, block_k))
    return block_m, block_k, num_warps, max(split_k, 1)


def skinny_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, num_cus: int
) -> torch.Tensor:
    """out = x @ weight.T (+ bias) for x with at most MAX_ROWS rows."""
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    n, k = x2.shape
    m = weight.shape[0]
    if n > MAX_FMA_ROWS:
        return _skinny_dot_gemm(x, x2, weight, bias, num_cus)
    block_m, block_k, num_warps, split_k = _launch_config(n, m, k, num_cus)
    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    split_k = triton.cdiv(k, k_per_split)
    if split_k == 1:
        out = torch.empty((1, n, m), dtype=x.dtype, device=x.device)
    else:
        out = torch.empty((split_k, n, m), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, block_m), split_k)
    _skinny_gemm_kernel[grid](
        x2,
        weight,
        out,
        n,
        m,
        k,
        k_per_split,
        x2.stride(0),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        N_ROWS=max(triton.next_power_of_2(n), 1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    result = out[0] if split_k == 1 else out.sum(0, dtype=torch.float32).to(x.dtype)
    if bias is not None:
        result = result + bias
    return result.reshape(*x.shape[:-1], m)


def _skinny_dot_gemm(
    x: torch.Tensor,
    x2: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_cus: int,
) -> torch.Tensor:
    n, k = x2.shape
    m = weight.shape[0]
    block_r = 16 if n <= 16 else 32
    block_m = 32 if n <= 16 else 16
    block_k = 128 if k >= 128 else max(16, triton.next_power_of_2(k))
    m_programs = triton.cdiv(m, block_m)
    split_k, k_per_split = _split_k(m_programs, k, block_k, 2 * num_cus)
    if split_k == 1:
        out = torch.empty((1, n, m), dtype=x.dtype, device=x.device)
    else:
        out = torch.empty((split_k, n, m), dtype=torch.float32, device=x.device)
    _skinny_dot_kernel[(m_programs, split_k)](
        x2,
        weight,
        out,
        n,
        m,
        k,
        k_per_split,
        x2.stride(0),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_R=block_r,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=2,
        num_stages=1,
    )
    result = out[0] if split_k == 1 else out.sum(0, dtype=torch.float32).to(x.dtype)
    if bias is not None:
        result = result + bias
    return result.reshape(*x.shape[:-1], m)
