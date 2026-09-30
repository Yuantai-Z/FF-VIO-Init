# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model-independent inference result types."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ModelPrediction:
    """Prediction expressed in the first requested camera frame (C0)."""

    points_c0: np.ndarray
    confidence: Optional[np.ndarray]
    inference_time_ms: float = 0.0
    model_name: str = "unknown"
