# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

import numpy as np

from vggt_ros.protocol import (
    ROS_POINTFIELD_FLOAT32,
    pack_pointcloud,
    stable_unique_timestamps,
    stamp_parts,
)


class ProtocolTest(unittest.TestCase):
    def test_pointcloud_is_organized_tight_little_endian_and_sanitized(self):
        points = np.array(
            [
                [[1.0, 2.0, 3.0], [np.nan, 1.0, 2.0]],
                [[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]],
            ],
            dtype=np.float64,
        )
        confidence = np.array([[2.0, 3.0], [4.0, -1.0]], dtype=np.float64)

        cloud = pack_pointcloud(points, confidence)

        self.assertEqual((cloud.height, cloud.width), (2, 2))
        self.assertFalse(cloud.is_bigendian)
        self.assertEqual(cloud.point_step, 16)
        self.assertEqual(cloud.row_step, 32)
        self.assertEqual(len(cloud.data), 64)
        self.assertEqual(
            [(field.name, field.offset, field.datatype) for field in cloud.fields],
            [
                ("x", 0, ROS_POINTFIELD_FLOAT32),
                ("y", 4, ROS_POINTFIELD_FLOAT32),
                ("z", 8, ROS_POINTFIELD_FLOAT32),
                ("confidence", 12, ROS_POINTFIELD_FLOAT32),
            ],
        )
        dtype = np.dtype(
            {
                "names": ["x", "y", "z", "confidence"],
                "formats": ["<f4", "<f4", "<f4", "<f4"],
                "offsets": [0, 4, 8, 12],
                "itemsize": 16,
            }
        )
        records = np.frombuffer(cloud.data, dtype=dtype)
        np.testing.assert_allclose(
            [records[1]["x"], records[1]["y"], records[1]["z"], records[1]["confidence"]],
            [0.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            [records[3]["x"], records[3]["y"], records[3]["z"], records[3]["confidence"]],
            [4.0, 5.0, 6.0, 0.0],
        )

    def test_pointcloud_without_confidence_uses_tight_xyz_layout(self):
        points = np.array([[[1.0, 2.0, 3.0], [np.nan, 1.0, 2.0]]], dtype=np.float32)

        cloud = pack_pointcloud(points, None)

        self.assertEqual((cloud.height, cloud.width), (1, 2))
        self.assertEqual(cloud.point_step, 12)
        self.assertEqual(cloud.row_step, 24)
        self.assertEqual(len(cloud.data), 24)
        self.assertEqual([field.name for field in cloud.fields], ["x", "y", "z"])
        records = np.frombuffer(cloud.data, dtype=np.dtype(("<f4", 3)))
        np.testing.assert_allclose(records[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(records[1], [0.0, 0.0, 0.0])

    def test_timestamp_helpers_preserve_order(self):
        timestamps = stable_unique_timestamps([2.25, 1.5, 2.25, float("nan")])
        self.assertEqual(timestamps, (2.25, 1.5))
        self.assertEqual(stamp_parts(10.9999999996), (11, 0))


if __name__ == "__main__":
    unittest.main()
