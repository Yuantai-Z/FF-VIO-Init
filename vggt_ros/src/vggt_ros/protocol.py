# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure-Python encoding helpers for the released ROS wire protocol."""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple

import numpy as np


ROS_POINTFIELD_FLOAT32 = 7


@dataclass(frozen=True)
class FieldSpec:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass(frozen=True)
class PackedPointCloud:
    height: int
    width: int
    fields: Tuple[FieldSpec, ...]
    point_step: int
    row_step: int
    data: bytes
    is_bigendian: bool = False
    is_dense: bool = False


def stable_unique_timestamps(values: Iterable[float]) -> Tuple[float, ...]:
    """Return finite timestamps once each without changing request order."""

    result = []
    seen = set()
    for value in values:
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp in seen:
            continue
        seen.add(timestamp)
        result.append(timestamp)
    return tuple(result)


def stamp_parts(timestamp: float) -> Tuple[int, int]:
    """Round a floating timestamp to ROS seconds/nanoseconds."""

    timestamp = float(timestamp)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError(f"invalid ROS timestamp {timestamp!r}")
    seconds = math.floor(timestamp)
    nanoseconds = int(round((timestamp - seconds) * 1_000_000_000.0))
    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000
    return int(seconds), nanoseconds


def _sanitize_points(
    points: np.ndarray, confidence: Optional[np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    array = np.asarray(points)
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"expected non-empty (H, W, 3) points, got {array.shape}")

    points_f32 = np.asarray(array, dtype=np.float32).copy()
    finite_geometry = np.isfinite(points_f32).all(axis=2)
    finite_points = np.where(finite_geometry[:, :, None], points_f32, 0.0)
    nonzero_geometry = np.linalg.norm(finite_points, axis=2) > 0.0
    geometry_valid = finite_geometry & nonzero_geometry
    points_f32[~geometry_valid] = 0.0

    if confidence is None:
        return points_f32, None

    confidence_f32 = np.asarray(confidence, dtype=np.float32)
    if confidence_f32.shape != points_f32.shape[:2]:
        raise ValueError(
            f"confidence shape {confidence_f32.shape} does not match points {points_f32.shape[:2]}"
        )
    confidence_f32 = confidence_f32.copy()
    confidence_valid = np.isfinite(confidence_f32) & (confidence_f32 >= 0.0)
    confidence_f32[~(geometry_valid & confidence_valid)] = 0.0
    return points_f32, confidence_f32


def pack_pointcloud(
    points: np.ndarray,
    confidence: Optional[np.ndarray],
) -> PackedPointCloud:
    """Pack one organized cloud using 12-byte xyz or 16-byte xyz+confidence records."""

    points_f32, confidence_f32 = _sanitize_points(points, confidence)
    height, width = points_f32.shape[:2]

    names = ["x", "y", "z"]
    formats = ["<f4", "<f4", "<f4"]
    offsets = [0, 4, 8]
    fields = [
        FieldSpec("x", 0, ROS_POINTFIELD_FLOAT32),
        FieldSpec("y", 4, ROS_POINTFIELD_FLOAT32),
        FieldSpec("z", 8, ROS_POINTFIELD_FLOAT32),
    ]
    point_step = 12
    if confidence_f32 is not None:
        names.append("confidence")
        formats.append("<f4")
        offsets.append(12)
        fields.append(FieldSpec("confidence", 12, ROS_POINTFIELD_FLOAT32))
        point_step = 16

    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step},
        align=False,
    )
    packed = np.zeros(height * width, dtype=dtype)
    flat_points = points_f32.reshape(-1, 3)
    packed["x"] = flat_points[:, 0]
    packed["y"] = flat_points[:, 1]
    packed["z"] = flat_points[:, 2]
    if confidence_f32 is not None:
        packed["confidence"] = confidence_f32.reshape(-1)

    row_step = width * point_step
    data = packed.tobytes(order="C")
    if len(data) != height * row_step:
        raise AssertionError("internal PointCloud2 packing error")
    return PackedPointCloud(
        height=height,
        width=width,
        fields=tuple(fields),
        point_step=point_step,
        row_step=row_step,
        data=data,
    )
