# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
import unittest

from vggt_ros.buffering import find_timestamp_match


@dataclass(frozen=True)
class Frame:
    timestamp: float
    value: str = ""


class BufferingTest(unittest.TestCase):
    def test_out_of_order_tail_does_not_hide_an_earlier_match(self):
        # A sorted-buffer early exit would inspect the 5.0 tail first and miss 10.0.
        frames = [Frame(10.0), Frame(5.0)]
        self.assertIs(find_timestamp_match(reversed(frames), 10.0, 1e-6), frames[0])

    def test_duplicate_timestamp_uses_most_recent_frame(self):
        frames = [Frame(10.0, "old"), Frame(10.0, "new")]
        self.assertIs(find_timestamp_match(frames, 10.0, 1e-6), frames[-1])


if __name__ == "__main__":
    unittest.main()
