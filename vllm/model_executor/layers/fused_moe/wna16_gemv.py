# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-sized int4 (W4A16, symmetric, grouped) MoE GEMMs as GEMVs.

For a handful of tokens, the generic WNA16 MoE kernel pads every routed token to a
16-row tile and runs dequant + tl.dot per tile, which on gfx908 (MI100) reaches
only a few percent of HBM bandwidth. Here each program handles one routed
(token, expert) pair and a block of output rows, streams that expert's packed
weights once, and accumulates with plain multiply-adds (no tl.dot, which also
sidesteps the gfx908 small-M bf16 MFMA miscompile).

Weight layout matches fused_moe_kernel_gptq_awq: B[e, n, k // 2] uint8 with the
even-k value in the low nibble, value = (q - 8) * scale[e, n, k // group_size].
"""

import torch

from vllm.triton_utils import tl, triton

# Largest number of routed (token, expert) pairs handled here.
MAX_PAIRS = 64


@triton.jit
def _moe_wna16_gemv_kernel(
    x_ptr,
    b_ptr,
    scale_ptr,
    out_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    expert_map_ptr,
    N,
    K,
    group_size,
    x_row_div,
    num_ids,
    stride_xm,
    stride_be,
    stride_bn,
    stride_bk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_om,
    HAS_EXPERT_MAP: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pair = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    out_ptrs = out_ptr + pair * stride_om + offs_n

    expert = tl.load(topk_ids_ptr + pair).to(tl.int64)
    # Padded slots (e.g. CUDA graph capture) can hold arbitrary ids; skip them
    # like moe_align_block_size does.
    valid = (expert >= 0) & (expert < num_ids)
    if HAS_EXPERT_MAP:
        expert = tl.load(expert_map_ptr + expert, mask=valid, other=-1).to(tl.int64)
    expert = tl.where(valid, expert, -1)
    if expert < 0:
        # Expert lives on another EP rank: contribute zeros, like the tiled kernel.
        tl.store(out_ptrs, tl.zeros((BLOCK_N,), dtype=out_ptr.dtype.element_ty), mask=n_mask)
        return

    x_row = x_ptr + (pair // x_row_div) * stride_xm
    b_rows = b_ptr + expert * stride_be + offs_n[:, None] * stride_bn
    s_rows = scale_ptr + expert * stride_se + offs_n * stride_sn
    HALF_K: tl.constexpr = BLOCK_K // 2
    offs_h = tl.arange(0, HALF_K)
    acc = tl.zeros((BLOCK_N, HALF_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        h = k0 // 2 + offs_h
        h_mask = h < K // 2
        packed = tl.load(
            b_rows + h[None, :] * stride_bk,
            mask=n_mask[:, None] & h_mask[None, :],
            other=0x88,
        ).to(tl.int32)
        x_even = tl.load(x_row + 2 * h, mask=h_mask, other=0.0).to(tl.float32)
        x_odd = tl.load(x_row + 2 * h + 1, mask=h_mask, other=0.0).to(tl.float32)
        # BLOCK_K divides group_size, so one scale per row covers this block.
        scale = tl.load(s_rows + (k0 // group_size) * stride_sk, mask=n_mask, other=0.0)
        lo = ((packed & 0xF) - 8).to(tl.float32)
        hi = ((packed >> 4) - 8).to(tl.float32)
        acc += scale.to(tl.float32)[:, None] * (lo * x_even[None, :] + hi * x_odd[None, :])
    result = tl.sum(acc, axis=1)
    if MUL_ROUTED_WEIGHT:
        result = result * tl.load(topk_weights_ptr + pair).to(tl.float32)
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty), mask=n_mask)


def moe_wna16_gemv(
    x: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None,
    expert_map: torch.Tensor | None,
    x_row_div: int,
    group_size: int,
    block_n: int = 16,
    num_warps: int = 1,
) -> None:
    """out[p, :] = dequant(b[expert(p)]) @ x[p // x_row_div] (* topk_weights[p]).

    x: [rows, K] activations; b: [E, N, K // 2] packed int4; b_scale: [E, N, K // G];
    out: [P, N] with P = topk_ids.numel(); topk_ids holds global expert ids.
    """
    num_pairs = topk_ids.numel()
    K = x.shape[-1]
    N = b.shape[1]
    block_k = min(group_size, 128)
    assert group_size % block_k == 0 and K % block_k == 0
    # Many single-warp programs keep enough loads in flight on gfx908.
    out2 = out.view(num_pairs, N)
    grid = (num_pairs, triton.cdiv(N, block_n))
    _moe_wna16_gemv_kernel[grid](
        x,
        b,
        b_scale,
        out2,
        topk_ids.reshape(-1),
        topk_weights.reshape(-1) if topk_weights is not None else topk_ids,
        expert_map if expert_map is not None else topk_ids,
        N,
        K,
        group_size,
        x_row_div,
        expert_map.numel() if expert_map is not None else b.shape[0],
        x.stride(0),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out2.stride(0),
        HAS_EXPERT_MAP=expert_map is not None,
        MUL_ROUTED_WEIGHT=topk_weights is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
