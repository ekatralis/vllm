# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton skinny GEMM (few-row activations) for GPUs without vLLM's HIP skinny kernels.

Decode-time linears multiply 1-4 activation rows by a large weight, so they are
bound by weight reads. On gfx908 (MI100), rocBLAS/hipBLASLt pick tile shapes that
reach only 5-30% of HBM bandwidth for these shapes, and wvSplitK/LLMM1 are not
built for this arch. This kernel reads each weight element once, splits K when
there are too few output rows to fill the GPU, and avoids tl.dot (small-M bf16
MFMA is miscompiled on gfx908).
"""

import torch

from vllm.triton_utils import tl, triton

# Largest activation row count routed here; beyond this, BLAS GEMMs win.
MAX_ROWS = 4


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
