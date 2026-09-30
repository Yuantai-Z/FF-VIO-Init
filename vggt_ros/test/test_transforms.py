# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

import numpy as np

from vggt_ros.transforms import anchor_local_points, anchor_world_points


class TransformTest(unittest.TestCase):
    def test_vggt_world_is_reanchored_to_first_camera(self):
        points_world = np.zeros((2, 1, 1, 3), dtype=np.float32)
        points_world[:, 0, 0] = [1.0, 2.0, 3.0]
        extrinsics = np.repeat(np.eye(4)[None, :3, :], 2, axis=0)
        extrinsics[0, 0, 3] = 5.0
        extrinsics[1, 0, 3] = 7.0

        points_c0 = anchor_world_points(points_world, extrinsics)

        np.testing.assert_allclose(points_c0[:, 0, 0], [[6.0, 2.0, 3.0], [6.0, 2.0, 3.0]])

    def test_pi3x_local_points_and_c2w_are_reanchored(self):
        local_points = np.zeros((2, 1, 1, 3), dtype=np.float32)
        local_points[:, 0, 0] = [1.0, 0.0, 2.0]
        camera_to_world = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
        camera_to_world[0, 0, 3] = 5.0
        camera_to_world[1, 0, 3] = 7.0

        points_c0 = anchor_local_points(local_points, camera_to_world)

        np.testing.assert_allclose(points_c0[0, 0, 0], [1.0, 0.0, 2.0])
        np.testing.assert_allclose(points_c0[1, 0, 0], [3.0, 0.0, 2.0])

    def test_reanchoring_still_rejects_invalid_model_poses(self):
        world_points = np.zeros((1, 1, 1, 3), dtype=np.float32)
        singular_w2c = np.zeros((1, 3, 4), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "singular"):
            anchor_world_points(world_points, singular_w2c)

        local_points = np.zeros((1, 1, 1, 3), dtype=np.float32)
        invalid_c2w = np.repeat(np.eye(4)[None], 1, axis=0)
        invalid_c2w[0, 3, 3] = 0.0
        with self.assertRaisesRegex(ValueError, "last row"):
            anchor_local_points(local_points, invalid_c2w)


if __name__ == "__main__":
    unittest.main()
