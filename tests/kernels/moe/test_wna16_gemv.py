# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode int4 MoE GEMV (wna16_gemv.py) against an exact fp32 dequant reference."""

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
    w = torch.stack((lo, hi), -1).flatten(-2).float()
    return w * s.float().repeat_interleave(GROUP, -1)


def _reference(x, b, s, topk_ids, topk_weights, expert_map, x_row_div):
    ids = topk_ids.reshape(-1).long()
    out = torch.zeros(ids.numel(), b.shape[1], device=x.device)
    for p, e in enumerate(ids.tolist()):
        if expert_map is not None:
            e = expert_map[e].item() if 0 <= e < expert_map.numel() else -1
        if not 0 <= e < b.shape[0]:
            continue
        out[p] = _dequant(b[e], s[e]) @ x[p // x_row_div].float()
        if topk_weights is not None:
            out[p] *= topk_weights.reshape(-1)[p]
    return out


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
        ref = _reference(x, b, s, ids, tw, expert_map, x_row_div)
        got = out.view(-1, n).float()
        assert torch.all(got[-1] == 0)
        tol = 1e-2 * ref.abs().max().item()
        torch.testing.assert_close(got, ref, atol=tol, rtol=0)


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
        ref = _reference(x, b, s, ids, None, None, TOPK)
        tol = 1e-2 * ref.abs().max().item()
        torch.testing.assert_close(out.view(-1, 1280).float(), ref, atol=tol, rtol=0)
