# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Timestamp matching that remains correct for out-of-order ROS delivery."""

from typing import Iterable, Optional, TypeVar


FrameT = TypeVar("FrameT")


def find_timestamp_match(
    frames: Iterable[FrameT], target_timestamp: float, tolerance: float
) -> Optional[FrameT]:
    best = None
    best_error = float("inf")
    for frame in frames:
        error = abs(float(frame.timestamp) - target_timestamp)
        # Iteration order is insertion order; replace on a tie so duplicate
        # timestamps resolve to the most recently buffered frame.
        if error <= best_error:
            best = frame
            best_error = error
    return best if best is not None and best_error <= tolerance else None
