# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU KV offload host memory on ROCm: size check, env default, exact pinning."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu import rocm_host_memory as rhm

GiB = 1 << 30
# The MI100 node these defaults were derived from: 1007 GiB RAM, TTM limit half
# of it, a 977 GiB pod limit.
NODE = rhm.HostMemoryLimits(
    ttm_limit_bytes=int(503.6 * GiB),
    cgroup_limit_bytes=977 * GiB,
    mem_total_bytes=1007 * GiB,
)


@pytest.mark.parametrize("size_gib", [128, 256, 320, 377])
def test_check_accepts_sizes_that_fit(size_gib: int) -> None:
    rhm.check_offload_size(size_gib * GiB, NODE, driver_owned=True)


@pytest.mark.parametrize("size_gib", [384, 600])
def test_check_rejects_sizes_over_the_ttm_headroom(size_gib: int) -> None:
    with pytest.raises(ValueError, match="TTM limit"):
        rhm.check_offload_size(size_gib * GiB, NODE, driver_owned=True)


def test_ttm_limit_only_applies_to_driver_owned_memory() -> None:
    rhm.check_offload_size(384 * GiB, NODE, driver_owned=False)


def test_check_uses_the_smaller_memory_limit() -> None:
    small_pod = rhm.HostMemoryLimits(None, 200 * GiB, 1007 * GiB)
    rhm.check_offload_size(100 * GiB, small_pod, driver_owned=True)
    with pytest.raises(ValueError, match="memory limit"):
        rhm.check_offload_size(101 * GiB, small_pod, driver_owned=True)


def test_unknown_limits_do_not_block() -> None:
    rhm.check_offload_size(10_000 * GiB, rhm.HostMemoryLimits(None, None, None), True)


def test_configure_sets_driver_owned_default(monkeypatch) -> None:
    monkeypatch.delenv("HSA_USERPTR_FOR_PAGED_MEM", raising=False)
    monkeypatch.delenv("VLLM_ROCM_OFFLOAD_SKIP_MEMORY_CHECK", raising=False)
    monkeypatch.setattr(rhm, "read_host_memory_limits", lambda: NODE)
    rhm.configure_rocm_cpu_offload(256 * GiB)
    import os

    assert os.environ["HSA_USERPTR_FOR_PAGED_MEM"] == "0"
    with pytest.raises(ValueError):
        rhm.configure_rocm_cpu_offload(384 * GiB)


def test_configure_respects_explicit_env_and_skip(monkeypatch) -> None:
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "1")
    monkeypatch.setattr(rhm, "read_host_memory_limits", lambda: NODE)
    rhm.configure_rocm_cpu_offload(384 * GiB)  # not driver-owned: TTM not checked
    import os

    assert os.environ["HSA_USERPTR_FOR_PAGED_MEM"] == "1"
    monkeypatch.setenv("VLLM_ROCM_OFFLOAD_SKIP_MEMORY_CHECK", "1")
    rhm.configure_rocm_cpu_offload(10_000 * GiB)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="hipHostMalloc path")
def test_exact_pinned_tensor_is_exact_pinned_and_zeroed() -> None:
    from vllm.utils.host_memory import alloc_exact_pinned_tensor

    shape = (3, (48 << 20) // 3 + 4096)  # not a power of two
    t = alloc_exact_pinned_tensor(shape, dtype=torch.int8)
    assert t.shape == shape and t.dtype == torch.int8
    assert t.is_pinned()
    assert int(t.abs().max()) == 0
    # The mapping is exactly the requested size, not rounded up.
    ptr, nbytes = t.data_ptr(), t.numel()
    with open("/proc/self/maps") as f:
        sizes = [
            int(b, 16) - int(a, 16)
            for a, b in (line.split()[0].split("-") for line in f)
            if int(a, 16) <= ptr < int(b, 16)
        ]
    assert sizes and sizes[0] < 2 * nbytes and sizes[0] < 1 << (nbytes - 1).bit_length()
    # Usable for async H2D copies.
    t[0, :16] = torch.arange(16, dtype=torch.int8)
    d = t.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(d[0, :16].cpu(), torch.arange(16, dtype=torch.int8))
