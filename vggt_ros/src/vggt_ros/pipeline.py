# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model-independent request processing used by ROS and mock tests."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .prediction import ModelPrediction
from .protocol import PackedPointCloud, pack_pointcloud


@dataclass(frozen=True)
class BatchResult:
    clouds: Tuple[PackedPointCloud, ...]
    prediction: ModelPrediction


class InferencePipeline:
    def __init__(self, backend) -> None:
        self.backend = backend

    def process(self, timestamps: Sequence[float], images_rgb: Sequence[np.ndarray]) -> BatchResult:
        if not timestamps or len(timestamps) != len(images_rgb):
            raise ValueError("timestamps and images must be non-empty and have equal length")
        prediction = self.backend.predict(images_rgb)
        points = np.asarray(prediction.points_c0)
        frame_count = len(timestamps)
        if points.ndim != 4 or points.shape[0] != frame_count or points.shape[-1] != 3:
            raise ValueError(
                f"backend returned points {points.shape}, expected ({frame_count}, H, W, 3)"
            )
        if prediction.confidence is None:
            confidence = None
        else:
            confidence = np.asarray(prediction.confidence)
            if confidence.shape != points.shape[:3]:
                raise ValueError(
                    f"backend confidence {confidence.shape} does not match points {points.shape[:3]}"
                )

        clouds = []
        for index, image in enumerate(images_rgb):
            image_array = np.asarray(image)
            if image_array.shape[:2] != points[index].shape[:2]:
                raise ValueError(
                    "backend changed the estimator's processed image grid: "
                    f"input {image_array.shape[:2]}, output {points[index].shape[:2]}"
                )
            frame_confidence = None if confidence is None else confidence[index]
            clouds.append(pack_pointcloud(points[index], frame_confidence))

        return BatchResult(tuple(clouds), prediction)
