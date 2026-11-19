# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug-only byte-level verification of CPU KV offload transfers.

Enabled with `VLLM_KV_OFFLOAD_VERIFY=1`. When a transfer completes (before it
is reported finished, so the scheduler has not reused its blocks), both sides
of every copied range are copied into scratch GPU tensors with HIP, compared
with `torch.equal` and hashed:

* store (GPU->CPU): source and destination must be equal; the CPU-side hash
  is recorded per CPU address.
* load (CPU->GPU): source and destination must be equal, and the CPU source
  must hash to what was recorded when that range was stored.

Counters are logged every `VLLM_KV_OFFLOAD_VERIFY_LOG_S` seconds (default 30).
Slow; never enable in production.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def verify_enabled() -> bool:
    return os.environ.get("VLLM_KV_OFFLOAD_VERIFY", "0") == "1"


class _HipPointerAttribute(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("device", ctypes.c_int),
        ("devicePointer", ctypes.c_void_p),
        ("hostPointer", ctypes.c_void_p),
        ("isManaged", ctypes.c_int),
        ("allocationFlags", ctypes.c_uint),
    ]


_hip = None


def _pointer_device(ptr: int) -> int:
    """Device ordinal owning a GPU pointer (HIP), else the current device."""
    global _hip
    if not torch.version.hip:
        return torch.cuda.current_device()
    if _hip is None:
        from vllm.v1.simple_kv_offload.cuda_mem_ops import _load_hip_runtime

        _hip = _load_hip_runtime()
    attr = _HipPointerAttribute()
    if _hip.hipPointerGetAttributes(ctypes.byref(attr), ctypes.c_void_p(ptr)) != 0:
        return torch.cuda.current_device()
    return attr.device


class OffloadVerifier:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stored: dict[int, tuple[int, int, int]] = {}  # cpu addr -> (size, h1, h2)
        self.c = dict(
            store_ranges=0, store_bytes=0, store_mismatch=0,
            load_ranges=0, load_bytes=0, load_mismatch=0,
            load_vs_stored_mismatch=0, load_unrecorded=0, skipped_unaligned=0,
        )
        self._last_log = time.monotonic()
        self._dumps = 0
        self._log_every = float(os.environ.get("VLLM_KV_OFFLOAD_VERIFY_LOG_S", "30"))

    def check(
        self, src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor, gpu_to_cpu: bool
    ) -> None:
        """Compare each copied range on both sides.

        Both ranges are copied into scratch GPU tensors with HIP on the current
        stream, synchronized, compared with `torch.equal`, and hashed with torch.
        A raw-pointer kernel, or copies on a separate stream, read wrong data for
        KV-cache source pointers on ROCm (2026-11-19); this path is exact.
        """
        from vllm import _custom_ops as ops

        n = sizes.numel()
        if n == 0:
            return
        dev = _pointer_device(int((src if gpu_to_cpu else dst)[0]))
        nbad, cpu_h = 0, []
        with torch.cuda.device(dev):
            for k in range(n):
                sz = int(sizes[k])
                if sz % 8:
                    self.c["skipped_unaligned"] += 1
                    cpu_h.append(None)
                    continue
                bs = torch.empty(sz, dtype=torch.int8, device=f"cuda:{dev}")
                bd = torch.empty(sz, dtype=torch.int8, device=f"cuda:{dev}")
                one = sizes[k : k + 1].clone()
                ops.swap_blocks_batch(src[k : k + 1].clone(), torch.tensor([bs.data_ptr()]), one)
                ops.swap_blocks_batch(dst[k : k + 1].clone(), torch.tensor([bd.data_ptr()]), one)
                torch.cuda.synchronize(dev)
                if not torch.equal(bs, bd):
                    nbad += 1
                    if self._dumps < 3:
                        self._dumps += 1
                        logger.error(
                            "KV offload verify: %s range %d: %d of %d bytes differ "
                            "(src=%#x dst=%#x)",
                            "GPU->CPU" if gpu_to_cpu else "CPU->GPU", k,
                            int((bs != bd).sum()), sz, int(src[k]), int(dst[k]),
                        )
                w = (bd if gpu_to_cpu else bs).view(torch.int64)
                idx = torch.arange(w.numel(), dtype=torch.int64, device=w.device)
                cpu_h.append((int(w.sum()), int((w * (idx * 2654435761 + 1)).sum())))
                del bs, bd, w, idx
        nbytes = int(sizes.sum())
        cpu_addrs = (dst if gpu_to_cpu else src).tolist()
        size_l = sizes.tolist()
        with self._lock:
            kind = "store" if gpu_to_cpu else "load"
            self.c[f"{kind}_ranges"] += n
            self.c[f"{kind}_bytes"] += nbytes
            self.c[f"{kind}_mismatch"] += nbad
            for a, s, h in zip(cpu_addrs, size_l, cpu_h):
                if h is None:
                    continue
                if gpu_to_cpu:
                    self._stored[a] = (s, h[0], h[1])
                    continue
                rec = self._stored.get(a)
                if rec is None or rec[0] != s:
                    self.c["load_unrecorded"] += 1
                elif (rec[1], rec[2]) != h:
                    self.c["load_vs_stored_mismatch"] += 1
            if nbad:
                logger.error(
                    "KV offload verify: %d of %d %s ranges differ between source "
                    "and destination", nbad, n, "GPU->CPU" if gpu_to_cpu else "CPU->GPU",
                )
            now = time.monotonic()
            if now - self._last_log >= self._log_every:
                self._last_log = now
                logger.info("KV offload verify: %s", self.c)


_verifier: OffloadVerifier | None = None
_verifier_lock = threading.Lock()


def get_verifier() -> OffloadVerifier | None:
    global _verifier
    if not verify_enabled():
        return None
    with _verifier_lock:
        if _verifier is None:
            _verifier = OffloadVerifier()
            logger.warning(
                "VLLM_KV_OFFLOAD_VERIFY=1: hashing every CPU offload transfer "
                "(debug only, slow)."
            )
        return _verifier
