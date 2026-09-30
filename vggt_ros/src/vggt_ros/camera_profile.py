# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Processed-camera profile validation shared by the ROS node and tests."""

from dataclasses import dataclass
import math
from typing import Optional


class ProcessedImageSizeMismatch(ValueError):
    """Raised when an image does not use the configured processed camera grid."""


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or numeric != parsed:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class ProcessedImageSize:
    """Expected width and height of the image after C++ preprocessing."""

    width: int
    height: int

    def validate(self, image) -> None:
        shape = getattr(image, "shape", ())
        if len(shape) < 2:
            raise ValueError(f"processed cam0 image has no spatial dimensions: {shape}")
        actual_height, actual_width = int(shape[0]), int(shape[1])
        if (actual_width, actual_height) != (self.width, self.height):
            raise ProcessedImageSizeMismatch(
                "processed cam0 size does not match the selected camera profile: "
                f"expected {self.width}x{self.height}, got {actual_width}x{actual_height}"
            )


@dataclass(frozen=True)
class ProcessedCameraProfile:
    """Named processed camera geometry consumed by Pi3X."""

    name: str
    image_size: ProcessedImageSize
    fx: float
    fy: float
    cx: float
    cy: float


def parse_processed_image_size(raw) -> Optional[ProcessedImageSize]:
    """Parse an optional ROS dictionary with ``width`` and ``height``."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("~processed_image_size must be a dictionary with width and height")
    try:
        width = _positive_integer(raw["width"], "~processed_image_size/width")
        height = _positive_integer(raw["height"], "~processed_image_size/height")
    except KeyError as exc:
        raise ValueError("~processed_image_size must contain width and height") from exc
    return ProcessedImageSize(width, height)


def parse_processed_camera_profile(
    raw_name, raw_size, raw_intrinsics, required: bool = False
) -> Optional[ProcessedCameraProfile]:
    """Parse one atomic profile, optionally allowing no profile selection."""

    if raw_name is None and raw_size is None:
        if required:
            raise ValueError(
                "Pi3X requires camera_profile_name, processed_image_size, "
                "and camera_intrinsics"
            )
        return None
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("~camera_profile_name must be a non-empty string")
    size = parse_processed_image_size(raw_size)
    if size is None:
        raise ValueError("camera profile requires ~processed_image_size")
    if size.width % 14 != 0 or size.height % 14 != 0:
        raise ValueError(
            "Pi3X processed image width and height must both be divisible by 14: "
            f"got {size.width}x{size.height}"
        )
    if not isinstance(raw_intrinsics, dict):
        raise ValueError("camera profile requires ~camera_intrinsics: {fx, fy, cx, cy}")
    try:
        fx = float(raw_intrinsics["fx"])
        fy = float(raw_intrinsics["fy"])
        cx = float(raw_intrinsics["cx"])
        cy = float(raw_intrinsics["cy"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "camera profile ~camera_intrinsics must contain numeric fx, fy, cx, cy"
        ) from exc
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError("camera profile intrinsics must be finite")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera profile focal lengths must be positive")
    if not 0.0 <= cx < size.width or not 0.0 <= cy < size.height:
        raise ValueError(
            "camera profile principal point must lie inside its processed image: "
            f"K=({fx}, {fy}, {cx}, {cy}), size={size.width}x{size.height}"
        )
    return ProcessedCameraProfile(raw_name.strip(), size, fx, fy, cx, cy)
