# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny decode GEMM (MI100 path): results, CUDA graph replays and two
streams at once (split-K tile counters must not be shared)."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("skinny GEMM is used on ROCm only", allow_module_level=True)

from vllm.model_executor.layers.triton_skinny_gemm import skinny_gemm  # noqa: E402
from vllm.utils.platform_utils import num_compute_units  # noqa: E402

# (k, m): split-K (large k, small m) and no split (large m).
SHAPES = [(10240, 320), (2560, 512), (2560, 24), (2560, 62080), (160, 2560)]


def _check(out, x, w):
    ref = F.linear(x.float(), w.float())
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 1e-2, err


@pytest.mark.parametrize("n", [1, 3, 12, 24, 36, 48])
@pytest.mark.parametrize("k,m", SHAPES)
def test_skinny_gemm(n, k, m):
    torch.manual_seed(0)
    x = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") / k**0.5
    cu = num_compute_units()
    _check(skinny_gemm(x, w, None, cu), x, w)

    # Replays reuse the tile counters, which each call must leave at zero.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = skinny_gemm(x, w, None, cu)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        g.replay()
        torch.cuda.synchronize()
        _check(out, x, w)


def test_skinny_gemm_two_streams():
    torch.manual_seed(0)
    cu = num_compute_units()
    a = [torch.randn(3, 10240, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    b = [
        torch.randn(320, 10240, dtype=torch.bfloat16, device="cuda") / 100
        for _ in range(2)
    ]
    side = torch.cuda.Stream()
    for _ in range(5):
        # Hold both streams until all kernels are queued, so they overlap.
        torch.cuda._sleep(50_000_000)
        side.wait_stream(torch.cuda.current_stream())
        out0, out1 = [], []
        for _ in range(30):
            out0.append(skinny_gemm(a[0], b[0], None, cu))
            with torch.cuda.stream(side):
                out1.append(skinny_gemm(a[1], b[1], None, cu))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for o in out0:
            _check(o, a[0], b[0])
        for o in out1:
            _check(o, a[1], b[1])
