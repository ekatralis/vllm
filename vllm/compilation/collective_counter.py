# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host accounting for collectives whose device tags must not repeat."""

from collections.abc import Callable


class CollectiveCounter:
    def __init__(self, reset: Callable[[], None], limit: int = 2**31 - 1):
        if limit < 1:
            raise ValueError("The collective epoch limit must be positive")
        self.limit = limit
        self.used = 0
        self._reset = reset

    def reserve(self, calls: int) -> None:
        """Reserve a whole eager call or graph before submitting it to the GPU."""
        if not 0 < calls <= self.limit:
            raise ValueError("Collective call count exceeds the epoch budget")
        if self.used + calls > self.limit:
            # Only commit the new accounting after all ranks finish resetting.
            self._reset()
            self.used = 0
        self.used += calls


class CollectiveGraphBudget:
    def __init__(self) -> None:
        self.calls: dict[CollectiveCounter, int] = {}

    def record(self, counter: CollectiveCounter, calls: int = 1) -> None:
        total = self.calls.get(counter, 0) + calls
        if calls <= 0 or total > counter.limit:
            raise ValueError("Captured graph exceeds the collective epoch budget")
        self.calls[counter] = total

    def reserve(self) -> None:
        for counter, calls in self.calls.items():
            counter.reserve(calls)
