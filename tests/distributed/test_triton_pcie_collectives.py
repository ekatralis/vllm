# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PcieCollectives / PcieMailbox with 48-bit tags across the tag wrap.

Tags must never repeat in practice (so stale words, e.g. the tail of an earlier
larger message, never match a later call) and must never be 0 (the value of
zero-filled memory). The counters are fast-forwarded to just below the wrap.
"""

import os
import random
import socket
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm PCIe peer-memory protocol", allow_module_level=True)

from vllm.distributed.device_communicators.triton_pcie_collectives import (  # noqa
    _TAG_PERIOD,
    PcieCollectives,
    PcieMailbox,
    _mailbox_recv_kernel,
    _mailbox_send_kernel,
    _pcie_ll_kernel,
)
from vllm.triton_utils import tl  # noqa: E402

from ..utils import multi_gpu_test  # noqa: E402


def _tag(e: int) -> int:
    return (e - 1) % _TAG_PERIOD + 1


@pytest.mark.parametrize("gather", [False, True])
def test_collective_tags_cross_48bit_wrap(gather):
    """One rank (its own buffer as the only peer): results stay exact, the call
    counter is int64, and the tag written next to the data goes P-1, P, 1, 2, ...
    (never 0) while message sizes grow and shrink."""
    programs, slot_words = 64, 1024
    buf = torch.zeros((2, 2 * slot_words), dtype=torch.int64, device="cuda")
    ptrs = torch.tensor([buf.data_ptr()], device="cuda")
    start = _TAG_PERIOD - 2
    epoch = torch.full((programs,), start, dtype=torch.int64, device="cuda")
    for i, n in enumerate([1024, 1, 513, 1024, 7, 1024]):
        e = start + i + 1
        inp = torch.arange(n, dtype=torch.float32, device="cuda") - n / 3
        out = torch.empty((1, n) if gather else (n,), device="cuda")
        _pcie_ll_kernel[(programs,)](
            inp.view(torch.int32),
            out.view(torch.int32),
            ptrs,
            epoch,
            n,
            0,
            slot_words,
            W=1,
            NUM_PROGRAMS=programs,
            BLOCK=512,
            GATHER=gather,
            DTYPE=tl.float32,
        )
        torch.testing.assert_close(out.view(-1), inp, rtol=0, atol=0)
        assert int(epoch[0]) == e and epoch.dtype == torch.int64
        # Both wire words of every data word written in this call carry tag(e).
        half = buf[e & 1]
        for plane in (half[:n], half[slot_words : slot_words + n]):
            tags = (plane >> 16) & ((1 << 48) - 1)
            assert torch.all(tags == _tag(e)) and _tag(e) != 0


@pytest.mark.parametrize("n", [1, 255, 256, 257, 1024])
def test_mailbox_seq_crosses_wraps(n):
    """Sequence numbers past 2^31, 2^32 (the old int32 problems) and the 48-bit
    tag wrap; a local buffer stands in for the receiver's memory."""
    buf = torch.zeros((8, 2 * 1024), dtype=torch.int64, device="cuda")
    ptrs = torch.tensor([buf.data_ptr(), buf.data_ptr()], device="cuda")
    grid = ((n + 255) // 256,)
    for base in (2**31 - 3, 2**32 - 3, _TAG_PERIOD - 3):
        for seq in range(base, base + 6):
            idx = torch.arange(n, dtype=torch.int32, device="cuda")
            inp = (idx * 7919 + seq) & 0x7FFFFFFF
            out = torch.empty_like(inp)
            _mailbox_send_kernel[grid](
                inp, ptrs, n, seq, 0, 1024, NUM_SLOTS=8, W=2, BLOCK=256
            )
            _mailbox_recv_kernel[grid](
                out, ptrs, n, seq, 1, 1024, NUM_SLOTS=8, BLOCK=256
            )
            torch.testing.assert_close(out, inp, rtol=0, atol=0)


@multi_gpu_test(num_gpus=4)
def test_pcie_protocol_across_48bit_wrap():
    # Rendezvous on localhost explicitly: --standalone resolves the hostname,
    # which some containers cannot.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc-per-node=4",
            "--module",
            "tests.distributed.test_triton_pcie_collectives",
        ],
        check=True,
        timeout=600,
    )


def _run_worker():
    """4 ranks: random-size all-reduces (one-shot and two-shot), all-gathers,
    CUDA-graph replays and mailbox messages, all crossing the tag wrap."""
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    group = dist.group.WORLD
    world = dist.get_world_size()
    comm = PcieCollectives(group, device, max_bytes=128 * 1024)
    comm._epoch.fill_(_TAG_PERIOD - 300)
    rng = random.Random(0)  # same sequence on every rank
    sizes = [1, 7, 129, 1000, 4096, 12000, 32768]  # fp32 elements; >= 5120 is two-shot

    def inputs(n, i, dtype):
        base = ((torch.arange(n, device=device) * 7 + i) % 13 + 1).to(dtype)
        return base, base * (rank + 1)

    for i in range(600):
        n, op = rng.choice(sizes), rng.choice(["ar32", "ar16", "ag32"])
        dtype = torch.bfloat16 if op == "ar16" else torch.float32
        if op == "ar16":
            n *= 2  # whole 4-byte words, as should_use() requires
        base, x = inputs(n, i, dtype)
        if op == "ag32":
            expected = torch.stack([base * (r + 1) for r in range(world)])
            torch.testing.assert_close(comm.all_gather(x), expected, rtol=0, atol=0)
        else:
            out = comm.all_reduce(x)
            expected = base * (world * (world + 1) // 2)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    # CUDA graphs: replays keep advancing the counter (no host bookkeeping).
    large_base, large = inputs(32768, 0, torch.float32)
    small_base, small = inputs(129, 0, torch.float32)
    comm.all_reduce(large)
    comm.all_gather(small)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reduced = comm.all_reduce(large)
        gathered = comm.all_gather(small)
    for _ in range(20):
        graph.replay()
        torch.testing.assert_close(
            reduced, large_base * (world * (world + 1) // 2), rtol=0, atol=0
        )
        expected = torch.stack([small_base * (r + 1) for r in range(world)])
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    assert int(comm._epoch.min()) > _TAG_PERIOD  # crossed the tag wrap

    # Mailbox: rank 0 sends, the others receive; random sizes across the wrap.
    mailbox = PcieMailbox(group, device, src=0, max_words=4096, num_slots=8)
    for i, seq in enumerate(range(_TAG_PERIOD - 40, _TAG_PERIOD + 40)):
        size = rng.choice([1, 3, 200, 1000, 4096])
        idx = torch.arange(size, device=device, dtype=torch.int32)
        value = (idx * 3 + seq) % 1000003
        if rank == 0:
            mailbox.send(value, seq)
        else:
            out = torch.empty(size, device=device, dtype=torch.int32)
            mailbox.recv(out, seq)
            torch.testing.assert_close(out, value, rtol=0, atol=0)
        if i % 4 == 3:  # the sender must stay < num_slots messages ahead
            torch.accelerator.synchronize()
            dist.barrier()

    torch.accelerator.synchronize()
    dist.barrier()
    mailbox.close()
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_worker()
