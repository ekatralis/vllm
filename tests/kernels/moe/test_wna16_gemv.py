# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode int4 MoE GEMV (wna16_gemv.py) against an exact fp64 dequant reference.

Every output element is checked on its own: |got - ref| <= 2^-8 |ref| (bf16 output
rounding) + 1e-4 * sum_k |w_k x_k| (fp32 accumulation, which scales with the dot
product's terms, not with its result). A tolerance relative to the largest output
would hide errors in small outputs.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("gfx908 decode MoE GEMV (v_dot2_f32_f16)", allow_module_level=True)

from vllm.model_executor.layers.fused_moe import wna16_gemv  # noqa: E402

GROUP = 128
TOPK = 10


def _dequant(wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    lo = (wq & 0xF).to(torch.int32) - 8
    hi = (wq >> 4).to(torch.int32) - 8
    w = torch.stack((lo, hi), -1).flatten(-2).double()
    return w * s.double().repeat_interleave(GROUP, -1)


def _reference(x, b, s, topk_ids, topk_weights, expert_map, x_row_div):
    """Exact output and sum_k |w_k x_k| per element; padded slots are 0."""
    ids = topk_ids.reshape(-1).long()
    ref = torch.zeros(ids.numel(), b.shape[1], dtype=torch.float64, device=x.device)
    scale = torch.zeros_like(ref)
    for p, e in enumerate(ids.tolist()):
        if expert_map is not None:
            e = expert_map[e].item() if 0 <= e < expert_map.numel() else -1
        if not 0 <= e < b.shape[0]:
            continue
        w = _dequant(b[e], s[e])
        xv = x[p // x_row_div].double()
        ref[p] = w @ xv
        scale[p] = w.abs() @ xv.abs()
        if topk_weights is not None:
            tw = topk_weights.reshape(-1)[p].double()
            ref[p] *= tw
            scale[p] *= tw.abs()
    return ref, scale


def _check(got: torch.Tensor, ref: torch.Tensor, scale: torch.Tensor) -> None:
    err = (got.double() - ref).abs()
    tol = 2.0**-8 * ref.abs() + 1e-4 * scale
    bad = err > tol
    assert not bad.any(), (
        f"{int(bad.sum())} of {bad.numel()} outputs out of tolerance; worst "
        f"|err| {err[bad].max():.3e} vs tol {tol[bad][err[bad].argmax()]:.3e}"
    )


def _pack(q: torch.Tensor) -> torch.Tensor:
    """int nibbles [..., k] -> uint8 [..., k // 2], even k in the low nibble."""
    return (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)


@pytest.mark.parametrize("num_tokens", [1, 3, 12, 48])
@pytest.mark.parametrize("n,k", [(1280, 2560), (2560, 640)])
@pytest.mark.parametrize("use_expert_map", [False, True])
@pytest.mark.parametrize("x_scale", [1.0, 1e3, 1e-3])
@pytest.mark.parametrize("path", ["dot2", "fallback"])
def test_wna16_gemv(num_tokens, n, k, use_expert_map, x_scale, path, monkeypatch):
    if path == "fallback":
        monkeypatch.setattr(wna16_gemv, "_dot2_supported", lambda *a: False)
    torch.manual_seed(0)
    dev = "cuda"
    e_global, e_local = 64, 32 if use_expert_map else 64
    b = torch.randint(0, 256, (e_local, n, k // 2), dtype=torch.uint8, device=dev)
    s = torch.rand(e_local, n, k // GROUP, dtype=torch.bfloat16, device=dev) * 0.02
    expert_map = None
    if use_expert_map:
        expert_map = torch.full((e_global,), -1, dtype=torch.int32, device=dev)
        expert_map[e_local:] = torch.arange(e_local, dtype=torch.int32, device=dev)
    # w13-style call (rows shared by TOPK pairs) and w2-style call (one row per pair).
    for x_row_div in (TOPK, 1):
        rows = num_tokens if x_row_div == TOPK else num_tokens * TOPK
        x = torch.randn(rows, k, dtype=torch.bfloat16, device=dev) * x_scale
        x[0, 7] *= 50  # outlier
        ids = torch.stack(
            [torch.randperm(e_global, device=dev)[:TOPK] for _ in range(num_tokens)]
        ).to(torch.int32)
        ids[-1, -1] = 12345  # padded slot (e.g. CUDA graph capture): must give zeros
        tw = torch.rand(num_tokens, TOPK, device=dev) if x_row_div == 1 else None
        out = torch.full((num_tokens, TOPK, n), 7.0, dtype=torch.bfloat16, device=dev)
        wna16_gemv.moe_wna16_gemv(x, b, s, out, ids, tw, expert_map, x_row_div, GROUP)
        ref, scale = _reference(x, b, s, ids, tw, expert_map, x_row_div)
        got = out.view(-1, n)
        assert torch.all(got[-1] == 0)
        _check(got, ref, scale)


@pytest.mark.parametrize("n,k", [(1280, 2560), (2560, 640)])
@pytest.mark.parametrize(
    "case", ["zero_weights", "sparse_1pct", "range_1e4", "range_1e6"]
)
@pytest.mark.parametrize("path", ["dot2", "fallback"])
def test_wna16_gemv_hard_cases(n, k, case, path, monkeypatch):
    """Inputs where offset cancellation or fp16 conversion would lose accuracy:
    all-zero weights (the answer is exactly 0), mostly-zero weights, and one huge
    activation on zero weights next to small ones that decide the result. The
    supported dynamic range within a row is ~1e6 (fp16 after per-row scaling)."""
    if path == "fallback":
        monkeypatch.setattr(wna16_gemv, "_dot2_supported", lambda *a: False)
    torch.manual_seed(1)
    dev = "cuda"
    num_tokens, e = 8, 4
    q = torch.randint(0, 16, (e, n, k), dtype=torch.int32, device=dev)
    x = torch.randn(num_tokens, k, dtype=torch.float32, device=dev) * 1e-2
    if case == "zero_weights":
        q.fill_(8)
    elif case == "sparse_1pct":
        keep = torch.rand(q.shape, device=dev) < 0.01
        q = torch.where(keep, q, torch.full_like(q, 8))
    else:
        x[:, 0] = 1e-2 * float(case.split("_")[1])
        q[:, :, 0] = 8
    b = _pack(q)
    s = (torch.rand(e, n, k // GROUP, device=dev) * 0.02 + 0.005).to(torch.bfloat16)
    x = x.to(torch.bfloat16)
    ids = torch.randint(0, e, (num_tokens, 1), dtype=torch.int32, device=dev)
    out = torch.empty(num_tokens, 1, n, dtype=torch.bfloat16, device=dev)
    wna16_gemv.moe_wna16_gemv(x, b, s, out, ids, None, None, 1, GROUP)
    ref, scale = _reference(x, b, s, ids, None, None, 1)
    got = out.view(-1, n)
    if case == "zero_weights":
        assert torch.all(got == 0), f"max |out| {got.abs().max():.3e}, expected 0"
    _check(got, ref, scale)


def test_wna16_gemv_cuda_graph():
    torch.manual_seed(0)
    dev = "cuda"
    b = torch.randint(0, 256, (16, 1280, 1280), dtype=torch.uint8, device=dev)
    s = torch.rand(16, 1280, 20, dtype=torch.bfloat16, device=dev) * 0.02
    x = torch.randn(4, 2560, dtype=torch.bfloat16, device=dev)
    ids = torch.randint(0, 16, (4, TOPK), dtype=torch.int32, device=dev)
    out = torch.empty(4, TOPK, 1280, dtype=torch.bfloat16, device=dev)
    wna16_gemv.moe_wna16_gemv(x, b, s, out, ids, None, None, TOPK, GROUP)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wna16_gemv.moe_wna16_gemv(x, b, s, out, ids, None, None, TOPK, GROUP)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        ids.copy_(torch.randint(0, 16, ids.shape, dtype=torch.int32, device=dev))
        g.replay()
        ref, scale = _reference(x, b, s, ids, None, None, TOPK)
        _check(out.view(-1, 1280), ref, scale)
