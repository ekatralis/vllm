# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real peer buffers across forced epoch resets and mailbox tag reuse."""

import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from vllm.compilation.collective_cudagraph import CollectiveCUDAGraph
from vllm.distributed.device_communicators.triton_pcie_collectives import (
    PcieCollectives,
    PcieMailbox,
    _mailbox_recv_kernel,
    _mailbox_send_kernel,
)
from vllm.platforms import current_platform

from ..utils import multi_gpu_test


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm mailbox kernel")
@pytest.mark.parametrize("n", [1, 255, 256, 257, 1024])
def test_mailbox_receive_invalidates_every_consumed_word(n):
    # Exercise the actual send/receive kernels with a local allocation standing
    # in for peer memory. No PCIe timing is needed to reproduce the stale tail:
    # before the fix, all received words retain their old nonzero tags.
    buf = torch.zeros((8, 1024), dtype=torch.int64, device="cuda")
    ptrs = torch.tensor([buf.data_ptr(), buf.data_ptr()], device="cuda")
    inp = torch.arange(n, dtype=torch.int32, device="cuda") - n // 2
    out = torch.empty_like(inp)
    seq = 2**31 - 3
    grid = ((n + 255) // 256,)
    _mailbox_send_kernel[grid](inp, ptrs, n, seq, 0, 1024, NUM_SLOTS=8, W=2, BLOCK=256)
    _mailbox_recv_kernel[grid](out, ptrs, n, seq, 1, 1024, NUM_SLOTS=8, BLOCK=256)
    torch.testing.assert_close(out, inp, rtol=0, atol=0)
    torch.testing.assert_close(buf, torch.zeros_like(buf), rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm peer-memory protocol")
@multi_gpu_test(num_gpus=4)
def test_pcie_epoch_reset_and_mailbox_reuse():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "--module",
            "tests.distributed.test_triton_pcie_collectives",
        ],
        check=True,
        timeout=180,
    )


def _run_worker():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    group = dist.group.WORLD
    world = dist.get_world_size()
    # Small limit exercises maintenance in seconds instead of days. The large
    # tensor takes the two-shot path; small tensors and gathers use one-shot.
    comm = PcieCollectives(group, device, max_bytes=32 * 1024, max_epoch=5)
    small = torch.full((129,), rank + 1, device=device, dtype=torch.float32)
    large = torch.full((8192,), rank + 1, device=device, dtype=torch.float32)
    expected_sum = world * (world + 1) // 2

    def check_reduce(out):
        torch.testing.assert_close(
            out, torch.full_like(out, expected_sum), rtol=0, atol=0
        )

    # Warm both kernels before capture.
    check_reduce(comm.all_reduce(small))
    check_reduce(comm.all_reduce(large))
    comm.all_gather(small)
    graph = CollectiveCUDAGraph()
    with torch.cuda.graph(graph):
        reduced = comm.all_reduce(large)
        gathered = comm.all_gather(small)
    for _ in range(12):
        check_reduce(comm.all_reduce(small))
        graph.replay()
        check_reduce(reduced)
        expected = torch.arange(1, world + 1, device=device).float()[:, None]
        torch.testing.assert_close(
            gathered, expected.expand_as(gathered), rtol=0, atol=0
        )

    mailbox = PcieMailbox(group, device, src=0, max_words=1024, num_slots=8)
    period = (2**31 - 2) // 8 * 8
    # Revisit the first message's tag after shorter uses of the same slot.
    for seq, size in [(0, 1024), (8, 1), (period - 1, 17), (period, 1024)]:
        value = seq % 97 + 1
        if rank == 0:
            mailbox.send(
                torch.full((size,), value, device=device, dtype=torch.int32), seq
            )
        else:
            out = torch.empty(size, device=device, dtype=torch.int32)
            mailbox.recv(out, seq)
            torch.testing.assert_close(out, torch.full_like(out, value), rtol=0, atol=0)
        torch.accelerator.synchronize()
        dist.barrier()

    torch.accelerator.synchronize()
    dist.barrier()
    mailbox.close()
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_worker()
