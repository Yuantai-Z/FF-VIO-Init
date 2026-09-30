# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Coordinate transformations into the protocol's first-camera frame."""

import numpy as np


def _homogeneous_from_w2c(extrinsics_w2c: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4) or extrinsics.shape[0] == 0:
        raise ValueError(f"expected non-empty (S, 3, 4) extrinsics, got {extrinsics.shape}")
    if not np.isfinite(extrinsics).all():
        raise ValueError("extrinsics contain non-finite values")
    transforms = np.broadcast_to(np.eye(4, dtype=np.float64), (len(extrinsics), 4, 4)).copy()
    transforms[:, :3, :] = extrinsics
    determinants = np.linalg.det(transforms)
    if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 1e-12):
        raise ValueError("extrinsics contain a singular transform")
    return transforms


def _validated_c2w(camera_to_world: np.ndarray) -> np.ndarray:
    transforms = np.asarray(camera_to_world, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or transforms.shape[0] == 0:
        raise ValueError(f"expected non-empty (S, 4, 4) camera poses, got {transforms.shape}")
    if not np.isfinite(transforms).all():
        raise ValueError("camera poses contain non-finite values")
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(transforms[:, 3, :], expected_last_row, rtol=0.0, atol=1e-6):
        raise ValueError("camera poses have an invalid homogeneous last row")
    determinants = np.linalg.det(transforms)
    if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 1e-12):
        raise ValueError("camera poses contain a singular transform")
    return transforms


def anchor_world_points(
    world_points: np.ndarray, extrinsics_w2c: np.ndarray
) -> np.ndarray:
    """Move model-world points into the first optical camera frame."""

    points = np.asarray(world_points)
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[0] == 0:
        raise ValueError(f"expected non-empty (S, H, W, 3) world points, got {points.shape}")
    transforms_w2c = _homogeneous_from_w2c(extrinsics_w2c)
    if len(points) != len(transforms_w2c):
        raise ValueError("point and pose frame counts differ")

    world_to_anchor = transforms_w2c[0]
    anchor_points = np.einsum(
        "ij,shwj->shwi", world_to_anchor[:3, :3], points, optimize=True
    ) + world_to_anchor[:3, 3]
    return anchor_points.astype(np.float32, copy=False)


def anchor_local_points(
    local_points: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    """Transform per-camera Pi3X points into C0."""

    points = np.asarray(local_points)
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[0] == 0:
        raise ValueError(f"expected non-empty (S, H, W, 3) local points, got {points.shape}")
    c2w = _validated_c2w(camera_to_world)
    if len(points) != len(c2w):
        raise ValueError("point and pose frame counts differ")

    anchor_from_world = np.linalg.inv(c2w[0])
    anchor_from_camera = np.einsum("ij,sjk->sik", anchor_from_world, c2w)
    anchor_from_camera[0] = np.eye(4, dtype=np.float64)
    anchor_points = np.einsum(
        "sij,shwj->shwi", anchor_from_camera[:, :3, :3], points, optimize=True
    ) + anchor_from_camera[:, None, None, :3, 3]
    return anchor_points.astype(np.float32, copy=False)
