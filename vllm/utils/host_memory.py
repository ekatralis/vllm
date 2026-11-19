# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host memory registration for CUDA/HIP DMA transfers, and core dump exclusion."""

import math
import mmap

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HOST_REGISTER_CHUNK_BYTES = 256 << 30


def exclude_from_core_dump(mmap_obj: mmap.mmap) -> None:
    """Leave the mapping out of core dumps (MADV_DONTDUMP, Linux 3.4+).

    A CPU KV cache can be hundreds of GiB, so a crashing worker would
    otherwise write all of it to disk. Failure only costs a larger dump.
    """
    madv_dontdump = getattr(mmap, "MADV_DONTDUMP", None)
    if madv_dontdump is None:
        return
    try:
        mmap_obj.madvise(madv_dontdump)
    except OSError as e:
        logger.warning(
            "MADV_DONTDUMP failed; core dumps will include the CPU KV cache: %s", e
        )


def alloc_exact_pinned_tensor(
    shape: tuple[int, ...], dtype: torch.dtype = torch.int8
) -> torch.Tensor:
    """A zeroed pinned host tensor of exactly the requested size.

    ``torch.zeros(..., pin_memory=True)`` goes through the caching host
    allocator, which rounds every allocation up to a power of two: a 48 GiB
    CPU KV buffer takes 64 GiB. On ROCm this allocates with ``hipHostMalloc``
    instead, so the buffer takes what it needs; the memory is freed when the
    last tensor or view of it is gone. Other platforms keep the torch path.
    """
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    from vllm.platforms import current_platform

    if not current_platform.is_rocm() or nbytes == 0:
        return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)

    import ctypes
    import weakref

    from vllm.v1.simple_kv_offload.cuda_mem_ops import _load_hip_runtime

    # torch reports host memory as pinned only once its own HIP context exists.
    torch.cuda.init()
    hip = _load_hip_runtime()
    ptr = ctypes.c_void_p()
    # hipHostMallocDefault: pinned, mapped into every device's address space.
    err = hip.hipHostMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes), 0)
    if err != 0 or not ptr.value:
        raise RuntimeError(
            f"hipHostMalloc of {nbytes / (1 << 30):.2f} GiB failed (error {err})"
        )
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr.value)
    weakref.finalize(buf, hip.hipHostFree, ctypes.c_void_p(ptr.value))
    # The tensor keeps `buf` (and so the allocation) alive through every view.
    tensor = torch.frombuffer(buf, dtype=torch.uint8).view(dtype).view(shape)
    tensor.zero_()
    return tensor


def pin_tensor(tensor: torch.Tensor) -> None:
    """Register one CPU range, bypassing the caching host allocator's rounding."""
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


def unpin_tensor(tensor: torch.Tensor) -> None:
    """Unregister exactly the range passed to pin_tensor."""
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostUnregister(tensor.data_ptr())
    if err.value != 0:
        cudart.cudaGetLastError()
        raise RuntimeError(f"cudaHostUnregister failed (code={err.value}).")


def host_registration_ranges(
    tensor_sizes: list[int],
    num_blocks: int,
    total_size: int,
    max_chunk_bytes: int = HOST_REGISTER_CHUNK_BYTES,
    *,
    page_size: int = mmap.PAGESIZE,
) -> tuple[tuple[int, int], ...]:
    """Split a shared pool without bisecting any tensor's host block.

    Keep registration boundaries aligned to the backing pages. CUDA batch
    copies reject descriptors spanning adjacent registrations.
    """
    if total_size % page_size:
        raise ValueError("Host pool must be page aligned.")
    if max_chunk_bytes < page_size:
        raise ValueError("Host registration chunks must span at least one page.")

    tensors: list[tuple[int, int, int]] = []
    tensor_start = 0
    for tensor_size in tensor_sizes:
        if tensor_size % num_blocks:
            raise ValueError(
                f"Host tensor size {tensor_size} is not divisible by "
                f"{num_blocks} blocks."
            )
        tensor_end = tensor_start + tensor_size
        tensors.append((tensor_start, tensor_end, tensor_size // num_blocks))
        tensor_start = tensor_end
    if tensor_start > total_size:
        raise ValueError("Host tensors exceed the shared host pool.")

    def last_safe_boundary(lower: int, upper: int) -> int | None:
        best = None
        for start, end, block_bytes in tensors:
            first_block = max(0, (lower - start) // block_bytes + 1)
            last_block = min(num_blocks, (min(upper, end) - start) // block_bytes)
            if first_block > last_block:
                continue
            divisor = math.gcd(block_bytes, page_size)
            if start % divisor:
                continue
            period = page_size // divisor
            if period == 1:
                residue = 0
            else:
                # Find block boundaries that are also backing-page boundaries.
                residue = (
                    (-start // divisor) * pow(block_bytes // divisor, -1, period)
                ) % period
            block = last_block - (last_block - residue) % period
            if block >= first_block:
                candidate = start + block * block_bytes
                best = candidate if best is None else max(best, candidate)

        if upper >= tensor_start:
            candidate = upper // page_size * page_size
            if candidate >= tensor_start and candidate > lower:
                best = candidate if best is None else max(best, candidate)
        return best

    ranges: list[tuple[int, int]] = []
    start = 0
    while total_size - start > max_chunk_bytes:
        end = last_safe_boundary(start, start + max_chunk_bytes)
        if end is None:
            raise ValueError(
                "No block and page boundary fits the registration chunk limit."
            )
        ranges.append((start, end))
        start = end
    ranges.append((start, total_size))
    return tuple(ranges)
