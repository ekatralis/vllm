# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs that account for PCIe collective calls on every replay."""

from typing import Any

import torch

from vllm.compilation.collective_counter import (
    CollectiveCounter,
    CollectiveGraphBudget,
)

# Capture can launch work from microbatch threads, so this is device-local,
# not thread-local. As with stream capture, overlapping captures on one
# device are unsupported.
_captures: dict[int, CollectiveGraphBudget] = {}
_collective_devices: set[int] = set()


def enable_collective_graphs(device: torch.device) -> None:
    assert device.index is not None
    _collective_devices.add(device.index)


def create_cudagraph() -> torch.cuda.CUDAGraph:
    # Other backends keep PyTorch's capture/replay behavior and overhead.
    if (
        _collective_devices
        and torch.accelerator.current_device_index() in _collective_devices
    ):
        return CollectiveCUDAGraph()
    return torch.cuda.CUDAGraph()


def can_capture_collectives(device: torch.device) -> bool:
    return device.index is not None and device.index in _captures


def record_collective(counter: CollectiveCounter, device: torch.device) -> None:
    if torch.cuda.is_current_stream_capturing():
        assert device.index is not None
        capture = _captures.get(device.index)
        if capture is None:
            raise RuntimeError("PCIe collectives require CollectiveCUDAGraph capture")
        capture.record(counter)
    else:
        counter.reserve(1)


class CollectiveCUDAGraph(torch.cuda.CUDAGraph):
    """Reserve all captured collective epochs before launching a graph.

    Participating ranks must use matching capture/replay boundaries, just as
    they must use matching collective ordering. Capture itself consumes no
    device epochs. The same counter is shared by eager calls and all graphs.
    """

    def capture_begin(self, *args: Any, **kwargs: Any) -> None:
        device = torch.accelerator.current_device_index()
        if device in _captures:
            raise RuntimeError("Overlapping collective graph captures")
        self._collective_device = device
        self._collective_budget = CollectiveGraphBudget()
        super().capture_begin(*args, **kwargs)
        _captures[device] = self._collective_budget

    def capture_end(self) -> None:
        try:
            super().capture_end()
        finally:
            _captures.pop(self._collective_device, None)

    def replay(self) -> None:
        if not self._collective_budget.calls:
            super().replay()
            return
        if torch.cuda.is_current_stream_capturing():
            outer = _captures.get(self._collective_device)
            if outer is None:
                raise RuntimeError("Cannot capture PCIe graph in an untracked graph")
            for counter, calls in self._collective_budget.calls.items():
                outer.record(counter, calls)
        else:
            self._collective_budget.reserve()
        super().replay()
