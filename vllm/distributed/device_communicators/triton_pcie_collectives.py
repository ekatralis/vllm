# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small all-reduce / all-gather over PCIe peer memory for MI100 (gfx908).

On PCIe-only MI100 nodes RCCL needs ~0.3 ms for any small collective, and a
decode step has several per layer. vLLM's custom all-reduce is only enabled on
XGMI-connected GPUs, and over PCIe it can read a peer's data before that data
has landed (the ready flag and the data take different paths), returning values
from an earlier call.

This uses the "LL" scheme instead: every 4-byte data word is sent together
with the call number in one 8-byte write, so a reader knows a word is current
by looking at it, with no separate ready flag to race against:
  1. each program writes its slice of the input, as (call, word) pairs, into
     every rank's receive buffer (this rank's slot; one of two buffer halves,
     alternating per call). Remote writes over PCIe do not wait for replies.
  2. it then polls the same slice of every slot in its own receive buffer
     until all words carry the current call number,
  3. and sums the slots in rank order (identical results on every rank), or
     copies them (all-gather).
All programs take part in every call, so each program's call counter is the
same on every rank. Kernels on one stream do not overlap, so once a rank has
received call e from a peer, that peer has finished reading call e-1; a rank
writing call e+1 therefore never overwrites data a peer still needs.
The kernel has no per-call host arguments and can be captured in CUDA graphs.
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Fixed number of programs per collective. Measured on 4 MI100s (bf16 x 2560,
# 1/3/12/24 rows): 9/18/82/186 us; larger sizes are PCIe-bandwidth bound.
_NUM_PROGRAMS = 64
_BLOCK = 512
_NUM_WARPS = 4


@triton.jit
def _unpack16(w, DTYPE: tl.constexpr):
    """int32 word -> (low, high) 16-bit floats as float32."""
    lo = w.to(tl.int16).to(DTYPE, bitcast=True).to(tl.float32)
    hi = (w >> 16).to(tl.int16).to(DTYPE, bitcast=True).to(tl.float32)
    return lo, hi


@triton.jit
def _pack16(lo, hi, DTYPE: tl.constexpr):
    lo16 = lo.to(DTYPE).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    hi16 = hi.to(DTYPE).to(tl.int16, bitcast=True).to(tl.int32)
    return lo16 | (hi16 << 16)


@triton.jit
def _pcie_ll_kernel(
    inp_ptr,  # int32 words
    out_ptr,  # int32 words: [n] (reduce) or [W, n] (gather)
    buf_ptrs,  # int64[W]: each rank's receive buffer, int64[2, W, SLOT_WORDS]
    epoch_ptr,  # int32[NUM_PROGRAMS], this rank only
    n,
    rank,
    SLOT_WORDS,
    W: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCK: tl.constexpr,
    GATHER: tl.constexpr,
    DTYPE: tl.constexpr,  # element type for reduce: float32, float16, bfloat16
):
    p = tl.program_id(0)
    e = tl.load(epoch_ptr + p) + 1
    half = (e % 2).to(tl.int64) * (W * SLOT_WORDS)
    chunk = tl.cdiv(tl.cdiv(n, NUM_PROGRAMS), BLOCK) * BLOCK
    start = p * chunk
    end = tl.minimum(start + chunk, n)
    offs = tl.arange(0, BLOCK)
    tag = e.to(tl.int64) << 32

    # 1. Push this rank's slice, as (call, word) pairs, into every rank's
    #    receive buffer (slot `rank`). Remote writes over PCIe do not wait.
    for k in range(start, end, BLOCK):
        idx = k + offs
        m = idx < end
        pair = tag | (tl.load(inp_ptr + idx, mask=m).to(tl.int64) & 0xFFFFFFFF)
        for j in tl.static_range(W):
            dst = tl.load(buf_ptrs + j).to(tl.pointer_type(tl.int64))
            dst = tl.multiple_of(dst + half + rank * SLOT_WORDS, 16)
            tl.store(dst + idx, pair, mask=m)

    # 2./3. Read every slot of the local receive buffer once all its words
    #       carry this call number (polling local memory only).
    my_buf = tl.load(buf_ptrs + rank).to(tl.pointer_type(tl.int64)) + half
    for k in range(start, end, BLOCK):
        idx = k + offs
        m = idx < end
        # Lanes past the end re-read the last word instead of being masked:
        # masked loads in this loop are miscompiled on gfx908 (part of the
        # tail is replaced by the `other` value, so the wait never ends).
        idx_c = tl.minimum(idx, end - 1)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        acc_hi = tl.zeros((BLOCK,), dtype=tl.float32)
        for j in tl.static_range(W):
            src = tl.multiple_of(my_buf + j * SLOT_WORDS, 16)
            v = tl.load(src + idx_c, cache_modifier=".cv")
            while tl.min(((v >> 32).to(tl.int32) == e).to(tl.int32), axis=0) == 0:
                v = tl.load(src + idx_c, cache_modifier=".cv")
            w = v.to(tl.int32)  # low 32 bits
            if GATHER:
                tl.store(out_ptr + j * n + idx, w, mask=m)
            elif DTYPE == tl.float32:
                acc += w.to(tl.float32, bitcast=True)
            else:
                lo, hi = _unpack16(w, DTYPE)
                acc += lo
                acc_hi += hi
        if not GATHER:
            if DTYPE == tl.float32:
                res = acc.to(tl.int32, bitcast=True)
            else:
                res = _pack16(acc, acc_hi, DTYPE)
            tl.store(out_ptr + idx, res, mask=m)

    tl.store(epoch_ptr + p, e)


_TL_DTYPES = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


class PcieCollectives:
    """All-reduce and all-gather of small tensors over PCIe peer memory."""

    def __init__(self, group: ProcessGroup, device: torch.device, max_bytes: int):
        """group: a CPU (gloo) group whose ranks are this node's GPUs."""
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_bytes = max_bytes
        self._slot_words = max_bytes // 4
        with torch.cuda.device(device):
            # Receive buffer: 2 halves x one slot per sender, 8 bytes per word.
            # Zero-filled; call numbers start at 1.
            self._buf_ptrs = CustomAllreduce.create_shared_buffer(
                2 * self.world_size * self._slot_words * 8, group=group
            )
            self._buf_ptr_t = torch.tensor(
                self._buf_ptrs, dtype=torch.int64, device=device
            )
            self._epoch = torch.zeros(_NUM_PROGRAMS, dtype=torch.int32, device=device)

    def should_use(self, inp: torch.Tensor) -> bool:
        nbytes = inp.numel() * inp.element_size()
        return (
            inp.dtype in _TL_DTYPES
            and inp.is_contiguous()
            and 0 < nbytes <= self.max_bytes
            and nbytes % 4 == 0
        )

    def _launch(self, inp: torch.Tensor, out: torch.Tensor, gather: bool) -> None:
        words = inp.view(-1).view(torch.int32)
        _pcie_ll_kernel[(_NUM_PROGRAMS,)](
            words,
            out.view(-1).view(torch.int32),
            self._buf_ptr_t,
            self._epoch,
            words.numel(),
            self.rank,
            self._slot_words,
            W=self.world_size,
            NUM_PROGRAMS=_NUM_PROGRAMS,
            BLOCK=_BLOCK,
            GATHER=gather,
            DTYPE=_TL_DTYPES[inp.dtype],
            num_warps=_NUM_WARPS,
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(inp)
        self._launch(inp, out, gather=False)
        return out

    def all_gather(self, inp: torch.Tensor) -> torch.Tensor:
        """Returns [world_size, *inp.shape], rank-major."""
        out = inp.new_empty((self.world_size,) + inp.shape)
        self._launch(inp, out, gather=True)
        return out

    def close(self) -> None:
        if getattr(self, "_buf_ptrs", None) is not None:
            ops.free_shared_buffer(self._buf_ptrs[self.rank])
            self._buf_ptrs = None
