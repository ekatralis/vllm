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

The default path (_moe_wna16_dot2_kernel) is bound by dequant ALU work, not memory,
so it keeps that work small: weights are read as int32 words, two nibbles at a time
become fp16 values 1024 + q via (w >> 4s) & 0x000F000F | 0x64006400, and
v_dot2_f32_f16 multiplies them with fp16 activation pairs into an fp32 accumulator.
The 1024 + 8 offset is removed per word with a precomputed sum of its activations.
Activations are first rearranged by _prep_kernel into that pair layout, scaled per
row by a power of two so fp16 can neither overflow nor lose small values.
At 12 tokens on MI100 this is ~1.4x faster than the plain multiply-add kernel
(~700 GB/s vs ~430 GB/s), with the same accuracy.
"""

import torch

from vllm.triton_utils import tl, triton

# Largest number of routed (token, expert) pairs handled here.
MAX_PAIRS = 512


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
        zeros = tl.zeros((BLOCK_N,), dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptrs, zeros, mask=n_mask)
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


@triton.jit
def _dot2(a, b, c):
    # c + a.lo * b.lo + a.hi * b.hi with a, b packed fp16 pairs, fp32 accumulate.
    return tl.inline_asm_elementwise(
        "v_dot2_f32_f16 $0, $1, $2, $3",
        "=v,v,v,v",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prep_kernel(x_ptr, xh_ptr, xs_ptr, rs_ptr, K, stride_xm, KW_P2: tl.constexpr):
    # One program per activation row. Scale by a power of two so max|x| lands in
    # [2^12, 2^13], store fp16 pairs xh[s, w] = (x[8w + s], x[8w + s + 4]), the fp32
    # sum of each word's 8 fp16 values in xs[w], and the inverse scale in rs.
    row = tl.program_id(0).to(tl.int64)
    KW = K // 8
    w = tl.arange(0, KW_P2)
    i = tl.arange(0, 8)
    m = (w < KW)[:, None]
    v = tl.load(
        x_ptr + row * stride_xm + w[:, None] * 8 + i[None, :], mask=m, other=0.0
    ).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(v), axis=1), axis=0)
    e = tl.where(amax > 0, tl.ceil(tl.log2(amax)), 0.0)
    scale = tl.exp2(13.0 - e)
    h = (v * scale).to(tl.float16)
    tl.store(xs_ptr + row * KW + w, tl.sum(h.to(tl.float32), axis=1), mask=w < KW)
    dst = xh_ptr + row * (8 * KW) + (i % 4)[None, :] * (2 * KW)
    tl.store(dst + w[:, None] * 2 + (i // 4)[None, :], h, mask=m)
    tl.store(rs_ptr + row, 1.0 / scale)


@triton.jit
def _moe_wna16_dot2_kernel(
    xh_ptr,
    xs_ptr,
    rs_ptr,
    b_ptr,
    scale_ptr,
    out_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    expert_map_ptr,
    N,
    K,
    x_row_div,
    num_ids,
    stride_be,
    stride_bn,
    stride_se,
    stride_sn,
    stride_om,
    HAS_EXPERT_MAP: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS_PER_ITER: tl.constexpr,
):
    pair = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    out_ptrs = out_ptr + pair * stride_om + offs_n

    expert = tl.load(topk_ids_ptr + pair).to(tl.int64)
    valid = (expert >= 0) & (expert < num_ids)
    if HAS_EXPERT_MAP:
        expert = tl.load(expert_map_ptr + expert, mask=valid, other=-1).to(tl.int64)
    expert = tl.where(valid, expert, -1)
    if expert < 0:
        zeros = tl.zeros((BLOCK_N,), dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptrs, zeros, mask=n_mask)
        return

    GW: tl.constexpr = GROUP // 8  # int32 words per scale group
    WORDS: tl.constexpr = GW * GROUPS_PER_ITER
    KW = K // 8
    offs_w = tl.arange(0, WORDS)
    row = (pair // x_row_div).to(tl.int64)
    xh = xh_ptr + row * (4 * KW)
    xs = xs_ptr + row * KW
    b_rows = (
        b_ptr.to(tl.pointer_type(tl.int32))
        + expert * (stride_be // 4)
        + offs_n[:, None] * (stride_bn // 4)
    )
    s_rows = scale_ptr + expert * stride_se + offs_n[:, None] * stride_sn
    acc = tl.zeros((BLOCK_N, WORDS), dtype=tl.float32)
    for w0 in range(0, KW, WORDS):
        w = tl.load(b_rows + (w0 + offs_w)[None, :], mask=n_mask[:, None], other=0)
        xw = xh + w0 + offs_w
        x0 = tl.load(xw)
        x1 = tl.load(xw + KW)
        x2 = tl.load(xw + 2 * KW)
        x3 = tl.load(xw + 3 * KW)
        # Each value below is 1024 + q, so start from -(1024 + 8) * sum(x).
        t = (-1032.0 * tl.load(xs + w0 + offs_w))[None, :] + tl.zeros(
            (BLOCK_N, WORDS), dtype=tl.float32
        )
        t = _dot2((w & 0x000F000F) | 0x64006400, x0[None, :], t)
        t = _dot2(((w >> 4) & 0x000F000F) | 0x64006400, x1[None, :], t)
        t = _dot2(((w >> 8) & 0x000F000F) | 0x64006400, x2[None, :], t)
        t = _dot2(((w >> 12) & 0x000F000F) | 0x64006400, x3[None, :], t)
        g = (w0 + offs_w) // GW
        sc = tl.load(s_rows + g[None, :], mask=n_mask[:, None], other=0.0)
        acc += sc.to(tl.float32) * t
    result = tl.sum(acc, axis=1) * tl.load(rs_ptr + row)
    if MUL_ROUTED_WEIGHT:
        result = result * tl.load(topk_weights_ptr + pair).to(tl.float32)
    tl.store(out_ptrs, result.to(out_ptr.dtype.element_ty), mask=n_mask)


def _dot2_supported(x: torch.Tensor, b: torch.Tensor, group_size: int) -> bool:
    return (
        group_size % 8 == 0
        and x.shape[-1] % group_size == 0
        and x.stride(-1) == 1
        and b.stride(2) == 1
        and b.stride(1) % 4 == 0
        and b.stride(0) % 4 == 0
        and b.data_ptr() % 4 == 0
    )


def _moe_wna16_gemv_dot2(
    x, b, b_scale, out2, topk_ids, topk_weights, expert_map, x_row_div, group_size
) -> None:
    rows, K = x.shape
    KW = K // 8
    N = b.shape[1]
    xh = torch.empty(rows, 4, KW, dtype=torch.int32, device=x.device)
    xs = torch.empty(rows, KW, dtype=torch.float32, device=x.device)
    rs = torch.empty(rows, dtype=torch.float32, device=x.device)
    _prep_kernel[(rows,)](
        x,
        xh.view(torch.float16),
        xs,
        rs,
        K,
        x.stride(0),
        KW_P2=triton.next_power_of_2(KW),
        num_warps=4,
    )
    # Tuned on MI100 for Qwen3.8 shapes (K=2560 and K=640, group 128) at 1-48 tokens.
    gpi = 2 if K >= 2048 and K % (2 * group_size) == 0 else 1
    block_n = 32 if gpi == 2 else 64
    grid = (out2.shape[0], triton.cdiv(N, block_n))
    _moe_wna16_dot2_kernel[grid](
        xh,
        xs,
        rs,
        b,
        b_scale,
        out2,
        topk_ids,
        topk_weights if topk_weights is not None else topk_ids,
        expert_map if expert_map is not None else topk_ids,
        N,
        K,
        x_row_div,
        expert_map.numel() if expert_map is not None else b.shape[0],
        b.stride(0),
        b.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        out2.stride(0),
        HAS_EXPERT_MAP=expert_map is not None,
        MUL_ROUTED_WEIGHT=topk_weights is not None,
        GROUP=group_size,
        BLOCK_N=block_n,
        GROUPS_PER_ITER=gpi,
        num_warps=2,
    )


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
    out2 = out.view(num_pairs, N)
    if _dot2_supported(x, b, group_size):
        _moe_wna16_gemv_dot2(
            x,
            b,
            b_scale,
            out2,
            topk_ids.reshape(-1),
            topk_weights.reshape(-1) if topk_weights is not None else None,
            expert_map,
            x_row_div,
            group_size,
        )
        return
    block_k = min(group_size, 128)
    assert group_size % block_k == 0 and K % block_k == 0
    # Fallback: plain multiply-add. Many single-warp programs keep enough loads in
    # flight on gfx908.
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
