# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host memory for the CPU KV offload on ROCm.

By default the ROCm runtime backs pinned host memory with ordinary user pages
that the amdgpu driver only tracks ("userptr"); they are not locked. When the
kernel migrates, compacts or reclaims such a page, the driver suspends every
GPU queue of the owning process and re-validates the whole buffer before
resuming it. For a CPU KV buffer of tens of GiB that takes tens of seconds,
the other workers wait for it in collectives, and a few of these in a row
exceed the engine's RPC timeout: the engine is declared dead under load.

``HSA_USERPTR_FOR_PAGED_MEM=0`` makes the runtime allocate pinned host memory
as driver-owned (GTT) buffers instead, which the kernel does not move. Such
pages count toward TTM's system-wide limit (half of RAM by default, shared by
all GPUs), so the offload size is checked against it at start-up.
"""

import os
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

_USERPTR_ENV = "HSA_USERPTR_FOR_PAGED_MEM"
_SKIP_CHECK_ENV = "VLLM_ROCM_OFFLOAD_SKIP_MEMORY_CHECK"
_TTM_PAGES_LIMIT = "/sys/module/ttm/parameters/pages_limit"
_CGROUP_MEMORY_MAX = "/sys/fs/cgroup/memory.max"
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

# Share of TTM's limit left for every other driver-owned buffer (staging
# buffers, communication buffers, model-specific host tables).
TTM_HEADROOM_FRACTION = 0.25
# Most of the machine (or pod) memory a CPU KV buffer that can never be
# swapped or reclaimed may take.
MAX_MEMORY_FRACTION = 0.5


@dataclass(frozen=True)
class HostMemoryLimits:
    ttm_limit_bytes: int | None
    cgroup_limit_bytes: int | None
    mem_total_bytes: int | None


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            value = f.read().strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None  # "max" means no limit


def read_host_memory_limits() -> HostMemoryLimits:
    ttm_pages = _read_int(_TTM_PAGES_LIMIT)
    mem_total = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    return HostMemoryLimits(
        ttm_limit_bytes=ttm_pages * _PAGE_SIZE if ttm_pages else None,
        cgroup_limit_bytes=_read_int(_CGROUP_MEMORY_MAX),
        mem_total_bytes=mem_total,
    )


def check_offload_size(
    offload_bytes: int, limits: HostMemoryLimits, driver_owned: bool
) -> None:
    """Raise if a CPU KV buffer of `offload_bytes` (all workers) would not fit."""
    gib = 1 << 30
    problems = []
    if driver_owned and limits.ttm_limit_bytes:
        allowed = int(limits.ttm_limit_bytes * (1 - TTM_HEADROOM_FRACTION))
        if offload_bytes > allowed:
            problems.append(
                f"driver-owned pinned memory counts toward the amdgpu TTM limit of "
                f"{limits.ttm_limit_bytes / gib:.0f} GiB shared by all GPUs; the "
                f"offload may use at most {allowed / gib:.0f} GiB of it, leaving "
                f"{TTM_HEADROOM_FRACTION:.0%} for other driver buffers (going over "
                "makes the driver evict buffers and stall the GPUs)"
            )
    memory = [m for m in (limits.cgroup_limit_bytes, limits.mem_total_bytes) if m]
    if memory:
        allowed = int(min(memory) * MAX_MEMORY_FRACTION)
        if offload_bytes > allowed:
            problems.append(
                f"pinned offload memory cannot be swapped or reclaimed; at most "
                f"{MAX_MEMORY_FRACTION:.0%} of the {min(memory) / gib:.0f} GiB "
                f"memory limit ({allowed / gib:.0f} GiB) may be used for it"
            )
    if problems:
        raise ValueError(
            f"--kv-offloading-size of {offload_bytes / gib:.0f} GiB is too large "
            "for this host: " + "; ".join(problems) + f". Set {_SKIP_CHECK_ENV}=1 "
            "to skip this check."
        )


def configure_rocm_cpu_offload(offload_bytes: int) -> None:
    """Make pinned host memory driver-owned and check the offload size.

    Must run before worker processes initialize the HIP runtime; they inherit
    the environment. An explicit ``HSA_USERPTR_FOR_PAGED_MEM`` is respected.
    """
    if _USERPTR_ENV not in os.environ:
        os.environ[_USERPTR_ENV] = "0"
        logger.info(
            "CPU KV offload on ROCm: setting %s=0 so pinned host memory is "
            "driver-owned and not moved by the kernel.",
            _USERPTR_ENV,
        )
    driver_owned = os.environ[_USERPTR_ENV] == "0"
    if not driver_owned:
        logger.warning(
            "CPU KV offload on ROCm with %s=%s: the offload buffers are "
            "kernel-movable user memory; page migration or reclaim can stall the "
            "GPUs for tens of seconds under memory pressure.",
            _USERPTR_ENV,
            os.environ[_USERPTR_ENV],
        )
    if os.environ.get(_SKIP_CHECK_ENV) == "1":
        return
    check_offload_size(offload_bytes, read_host_memory_limits(), driver_owned)
