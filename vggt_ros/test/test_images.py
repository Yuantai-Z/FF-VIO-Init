# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass
import unittest

import numpy as np

from vggt_ros.images import decode_rgb_image


@dataclass
class FakeImage:
    height: int
    width: int
    encoding: str
    step: int
    data: bytes


class ImageDecoderTest(unittest.TestCase):
    def test_bgr_rows_with_padding_decode_without_resizing(self):
        # Two BGR pixels plus two padding bytes per row.
        message = FakeImage(
            height=2,
            width=2,
            encoding="bgr8",
            step=8,
            data=bytes(
                [3, 2, 1, 6, 5, 4, 99, 98, 9, 8, 7, 12, 11, 10, 97, 96]
            ),
        )

        decoded = decode_rgb_image(message)

        self.assertEqual(decoded.shape, (2, 2, 3))
        np.testing.assert_array_equal(
            decoded,
            np.array(
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                dtype=np.uint8,
            ),
        )

    def test_mono_is_replicated_and_truncated_storage_is_rejected(self):
        decoded = decode_rgb_image(FakeImage(1, 2, "mono8", 2, bytes([5, 7])))
        np.testing.assert_array_equal(decoded, [[[5, 5, 5], [7, 7, 7]]])
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_rgb_image(FakeImage(2, 2, "rgb8", 6, bytes(11)))


if __name__ == "__main__":
    unittest.main()
