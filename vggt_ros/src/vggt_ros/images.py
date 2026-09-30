# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Decode processed ROS Image-like messages without changing their grid."""

import numpy as np


def decode_rgb_image(message) -> np.ndarray:
    """Decode mono/RGB-family uint8 data, respecting per-row padding."""

    encoding = str(message.encoding).lower()
    channels_by_encoding = {
        "mono8": 1,
        "8uc1": 1,
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported processed image encoding '{message.encoding}'")
    channels = channels_by_encoding[encoding]
    height = int(message.height)
    width = int(message.width)
    row_bytes = width * channels
    step = int(message.step)
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError(f"invalid image layout {width}x{height}, step={step}")
    data = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if data.size < required:
        raise ValueError(f"truncated image data: {data.size} < {required}")
    pixels = data[:required].reshape(height, step)[:, :row_bytes]
    if channels == 1:
        gray = pixels.reshape(height, width)
        return np.repeat(gray[:, :, None], 3, axis=2).copy()
    color = pixels.reshape(height, width, channels)
    if encoding in {"bgr8", "bgra8"}:
        return color[:, :, 2::-1].copy()
    return color[:, :, :3].copy()
