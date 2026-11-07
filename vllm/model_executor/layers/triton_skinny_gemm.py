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
def _store_tile(
    acc,
    out_ptr,
    part_ptr,
    cnt_ptr,
    pid_m,
    pid_k,
    split_k,
    rows,
    cols,
    mask,
    stride_on,
    stride_ps,
    stride_pn,
):
    """Store a (rows x cols) fp32 tile. With split-K, each split writes its
    partial sums and the last split to finish adds them up, so no separate
    reduction kernel is needed. The per-tile counters are reset afterwards."""
    if split_k == 1:
        tl.store(
            out_ptr + rows[:, None] * stride_on + cols[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        part = part_ptr + rows[:, None] * stride_pn + cols[None, :]
        tl.store(part + pid_k * stride_ps, acc, mask=mask)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid_m, 1, sem="acq_rel", scope="gpu")
        if done == split_k - 1:
            total = tl.zeros_like(acc)
            for s in range(split_k):
                total += tl.load(
                    part + s * stride_ps, mask=mask, other=0.0, cache_modifier=".cg"
                )
            tl.store(
                out_ptr + rows[:, None] * stride_on + cols[None, :],
                total.to(out_ptr.dtype.element_ty),
                mask=mask,
            )
            tl.store(cnt_ptr + pid_m, 0)


@triton.jit
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    part_ptr,
    cnt_ptr,
    N,
    M,
    K,
    K_PER_SPLIT,
    SPLIT_K,
    stride_xn,
    stride_wm,
    stride_on,
    stride_ps,
    stride_pn,
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
    _store_tile(
        tl.sum(acc, axis=2),
        out_ptr,
        part_ptr,
        cnt_ptr,
        pid_m,
        pid_k,
        SPLIT_K,
        rn,
        rm,
        n_mask[:, None] & m_mask[None, :],
        stride_on,
        stride_ps,
        stride_pn,
    )


@triton.jit
def _skinny_dot_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    part_ptr,
    cnt_ptr,
    N,
    M,
    K,
    K_PER_SPLIT,
    SPLIT_K,
    stride_xn,
    stride_wm,
    stride_on,
    stride_ps,
    stride_pn,
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
    _store_tile(
        acc,
        out_ptr,
        part_ptr,
        cnt_ptr,
        pid_m,
        pid_k,
        SPLIT_K,
        rr,
        rm,
        r_mask[:, None] & m_mask[None, :],
        stride_on,
        stride_ps,
        stride_pn,
    )


# Split-K tile counters, one set per stream: kernels on one stream run one
# after another, but the shared experts run on a second stream at the same
# time. Zero between kernels: the last split of a tile resets its counter, so
# a CUDA graph can replay the kernels.
_COUNTERS: dict[tuple[torch.device, int], torch.Tensor] = {}
_MAX_SPLIT_TILES = 4096


def _buffers(
    x: torch.Tensor, n: int, m: int, split_k: int, m_programs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Output, fp32 split-K partials and tile counters for one call."""
    out = torch.empty((n, m), dtype=x.dtype, device=x.device)
    key = (x.device, torch.cuda.current_stream(x.device).cuda_stream)
    counters = _COUNTERS.get(key)
    if counters is None:
        counters = torch.zeros(_MAX_SPLIT_TILES, dtype=torch.int32, device=x.device)
        _COUNTERS[key] = counters
    if split_k == 1:
        part = torch.empty((1, 1, 1), dtype=torch.float32, device=x.device)
    else:
        assert m_programs <= _MAX_SPLIT_TILES
        part = torch.empty((split_k, n, m), dtype=torch.float32, device=x.device)
    return out, part, counters


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
    grid = (triton.cdiv(m, block_m), split_k)
    out, part, counters = _buffers(x, n, m, split_k, grid[0])
    _skinny_gemm_kernel[grid](
        x2,
        weight,
        out,
        part,
        counters,
        n,
        m,
        k,
        k_per_split,
        split_k,
        x2.stride(0),
        weight.stride(0),
        out.stride(0),
        part.stride(0),
        part.stride(1),
        N_ROWS=max(triton.next_power_of_2(n), 1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], m)


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
    out, part, counters = _buffers(x, n, m, split_k, m_programs)
    _skinny_dot_kernel[(m_programs, split_k)](
        x2,
        weight,
        out,
        part,
        counters,
        n,
        m,
        k,
        k_per_split,
        split_k,
        x2.stride(0),
        weight.stride(0),
        out.stride(0),
        part.stride(0),
        part.stride(1),
        BLOCK_R=block_r,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=2,
        num_stages=1,
    )
    if bias is not None:
        out = out + bias
    return out.reshape(*x.shape[:-1], m)
