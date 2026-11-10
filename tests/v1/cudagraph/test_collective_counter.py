# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of eager/graph accounting before device tag reuse.

Also runnable without the GPU test conftest:
  .venv/bin/python -m unittest tests.v1.cudagraph.test_collective_counter -v
"""

import unittest

from vllm.compilation.collective_counter import (
    CollectiveCounter,
    CollectiveGraphBudget,
)


class TestCollectiveCounter(unittest.TestCase):
    def test_capture_does_not_consume_epochs_but_every_replay_does(self):
        resets = []
        counter = CollectiveCounter(lambda: resets.append(counter.used), limit=7)
        graph = CollectiveGraphBudget()
        graph.record(counter)
        graph.record(counter)
        self.assertEqual(counter.used, 0)
        for _ in range(3):
            graph.reserve()
        self.assertEqual(counter.used, 6)
        self.assertEqual(resets, [])
        graph.reserve()
        self.assertEqual(resets, [6])
        self.assertEqual(counter.used, 2)

    def test_eager_and_different_graphs_share_the_epoch_budget(self):
        resets = []
        counter = CollectiveCounter(lambda: resets.append(counter.used), limit=7)
        small, large = CollectiveGraphBudget(), CollectiveGraphBudget()
        small.record(counter)
        large.record(counter, 3)
        counter.reserve(2)
        small.reserve()
        large.reserve()
        counter.reserve(1)
        self.assertEqual(counter.used, 7)
        self.assertEqual(resets, [])
        large.reserve()
        self.assertEqual(resets, [7])
        self.assertEqual(counter.used, 3)

    def test_reset_completes_before_a_new_generation_is_reserved(self):
        # External device storage represented in memory. A tail carrying tag 1
        # must be invalidated before a new generation can use that tag again.
        halves = [[(1, 42)] * 8, [(2, 43)] * 8]

        def reset():
            self.assertEqual(counter.used, 3)
            for half in halves:
                half[:] = [(0, 0)] * len(half)

        counter = CollectiveCounter(reset, limit=3)
        counter.reserve(3)
        counter.reserve(1)
        self.assertEqual(counter.used, 1)
        self.assertTrue(all(tag == 0 for half in halves for tag, _ in half))

    def test_failed_reset_does_not_authorize_tag_reuse(self):
        def reset():
            raise RuntimeError("peer failed to reach barrier")

        counter = CollectiveCounter(reset, limit=3)
        counter.reserve(3)
        with self.assertRaisesRegex(RuntimeError, "barrier"):
            counter.reserve(1)
        self.assertEqual(counter.used, 3)

    def test_oversized_graph_is_rejected_during_capture(self):
        counter = CollectiveCounter(lambda: self.fail("unexpected reset"), limit=3)
        graph = CollectiveGraphBudget()
        graph.record(counter, 3)
        with self.assertRaisesRegex(ValueError, "budget"):
            graph.record(counter)
        self.assertEqual(counter.used, 0)

    def test_production_limit_resets_before_signed_overflow(self):
        resets = []
        counter = CollectiveCounter(lambda: resets.append(counter.used))
        counter.reserve(2**31 - 2)
        counter.reserve(1)
        counter.reserve(1)
        self.assertEqual(resets, [2**31 - 1])
        self.assertEqual(counter.used, 1)


if __name__ == "__main__":
    unittest.main()
