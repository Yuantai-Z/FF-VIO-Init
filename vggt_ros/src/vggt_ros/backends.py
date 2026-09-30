# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lazy, external model adapters for VGGT and Pi3X."""

from contextlib import nullcontext
import importlib
import os
from pathlib import Path
import sys
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from .prediction import ModelPrediction
from .transforms import anchor_local_points, anchor_world_points


class BackendError(RuntimeError):
    pass


def canonical_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    if normalized == "vggt":
        return "vggt"
    if normalized in {"pi3x", "pi3"}:
        return "pi3x"
    raise BackendError(f"unsupported model_type '{model_type}'; expected vggt or pi3x")


def activate_pi3x_confidence(logits: np.ndarray, mode: str) -> np.ndarray:
    """Convert Pi3X logits without applying any filtering threshold."""

    logits_f32 = np.asarray(logits, dtype=np.float32)
    normalized = str(mode).strip().lower()
    if normalized == "legacy_exp":
        return (1.0 + np.exp(np.clip(logits_f32, -20.0, 20.0))).astype(np.float32)
    if normalized == "sigmoid":
        clipped = np.clip(logits_f32, -80.0, 80.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
    raise BackendError(
        f"unsupported Pi3X confidence_activation '{mode}'; expected legacy_exp or sigmoid"
    )


def _external_source_path(source_path: str, environment_name: str, package_name: str) -> None:
    candidate = str(source_path).strip() or os.environ.get(environment_name, "").strip()
    if not candidate:
        return
    root = Path(candidate).expanduser().resolve()
    if not root.is_dir() or not (root / package_name).is_dir():
        raise BackendError(
            f"{environment_name}/model_source_path must be a source root containing "
            f"'{package_name}/', got '{root}'"
        )
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)


def _validate_images(images_rgb: Sequence[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    if not images_rgb:
        raise BackendError("at least one processed cam0 image is required")
    images = []
    expected_shape: Optional[Tuple[int, int, int]] = None
    for image in images_rgb:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise BackendError(f"expected uint8 RGB image, got {array.dtype} {array.shape}")
        if expected_shape is None:
            expected_shape = array.shape
            if array.shape[0] % 14 != 0 or array.shape[1] % 14 != 0:
                raise BackendError(
                    f"processed image grid {array.shape[:2]} must use dimensions divisible by 14"
                )
        elif array.shape != expected_shape:
            raise BackendError("all requested processed images must have the same grid")
        images.append(np.ascontiguousarray(array))
    stacked = np.stack(images, axis=0)
    return stacked, stacked.shape[1], stacked.shape[2]


def _to_numpy(value, dtype=np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


class _TorchBackend:
    def __init__(self, device: str, amp_dtype: str, torch_module=None) -> None:
        self.torch = torch_module or importlib.import_module("torch")
        self.device = self.torch.device(device)
        if self.device.type == "cuda" and not self.torch.cuda.is_available():
            raise BackendError("CUDA was requested but is not available")
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)

    def _resolve_amp_dtype(self, requested: str):
        normalized = str(requested).strip().lower()
        if normalized == "auto":
            if self.device.type != "cuda":
                return None
            capability = self.torch.cuda.get_device_capability(self.device)
            return self.torch.bfloat16 if capability[0] >= 8 else self.torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return self.torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return self.torch.float16
        if normalized in {"none", "float32", "fp32"}:
            return None
        raise BackendError(f"unsupported amp_dtype '{requested}'")

    def _autocast(self):
        if self.device.type != "cuda" or self.amp_dtype is None:
            return nullcontext()
        if hasattr(self.torch, "amp") and hasattr(self.torch.amp, "autocast"):
            return self.torch.amp.autocast("cuda", dtype=self.amp_dtype)
        return self.torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _tensorize(self, images_rgb: Sequence[np.ndarray]):
        images, height, width = _validate_images(images_rgb)
        tensor = self.torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
        return tensor, height, width

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(device=self.device)


class VGGTBackend(_TorchBackend):
    def __init__(
        self,
        model_name: str,
        model_revision: str = "",
        model_source_path: str = "",
        device: str = "cuda",
        amp_dtype: str = "auto",
        model=None,
        pose_decoder=None,
        torch_module=None,
    ) -> None:
        super().__init__(device=device, amp_dtype=amp_dtype, torch_module=torch_module)
        _external_source_path(model_source_path, "VGGT_SOURCE_PATH", "vggt")
        if model is None or pose_decoder is None:
            try:
                model_module = importlib.import_module("vggt.models.vggt")
                pose_module = importlib.import_module("vggt.utils.pose_enc")
            except ImportError as exc:
                raise BackendError(
                    "VGGT is not importable; install it or set ~model_source_path/VGGT_SOURCE_PATH"
                ) from exc
            if model is None:
                kwargs = {"revision": model_revision} if model_revision else {}
                model = model_module.VGGT.from_pretrained(model_name, **kwargs)
            if pose_decoder is None:
                pose_decoder = pose_module.pose_encoding_to_extri_intri
        self.model = model.eval().to(self.device)
        self.pose_decoder = pose_decoder
        self.model_name = model_name

    def predict(self, images_rgb: Sequence[np.ndarray]) -> ModelPrediction:
        image_tensor, height, width = self._tensorize(images_rgb)
        images_device = image_tensor.to(self.device, non_blocking=self.device.type == "cuda")
        self._synchronize()
        start = time.perf_counter()
        with self.torch.no_grad():
            with self._autocast():
                output = self.model(images_device)
        self._synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        try:
            points_world = _to_numpy(output["world_points"])[0]
            confidence_value = output.get("world_points_conf")
            confidence = _to_numpy(confidence_value)[0] if confidence_value is not None else None
            extrinsics, _ = self.pose_decoder(output["pose_enc"], (height, width))
            extrinsics_np = _to_numpy(extrinsics, dtype=np.float64)[0]
        except Exception as exc:
            raise BackendError(f"invalid VGGT output: {exc}") from exc

        points_c0 = anchor_world_points(points_world, extrinsics_np)
        return ModelPrediction(
            points_c0=points_c0,
            confidence=confidence,
            inference_time_ms=elapsed_ms,
            model_name="VGGT",
        )


class Pi3XBackend(_TorchBackend):
    def __init__(
        self,
        model_name: str,
        camera_intrinsics: np.ndarray,
        confidence_activation: str = "legacy_exp",
        model_revision: str = "",
        model_source_path: str = "",
        device: str = "cuda",
        amp_dtype: str = "auto",
        model=None,
        torch_module=None,
    ) -> None:
        super().__init__(device=device, amp_dtype=amp_dtype, torch_module=torch_module)
        intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise BackendError("Pi3X requires one finite 3x3 processed-image camera_intrinsics matrix")
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise BackendError("Pi3X processed-image focal lengths must be positive")
        self.camera_intrinsics = intrinsics
        self.confidence_activation = str(confidence_activation).strip().lower()
        activate_pi3x_confidence(np.zeros(1, dtype=np.float32), self.confidence_activation)

        _external_source_path(model_source_path, "PI3X_SOURCE_PATH", "pi3")
        if model is None:
            try:
                model_module = importlib.import_module("pi3.models.pi3x")
            except ImportError as exc:
                raise BackendError(
                    "Pi3X is not importable; install it or set ~model_source_path/PI3X_SOURCE_PATH"
                ) from exc
            kwargs = {"revision": model_revision} if model_revision else {}
            model = model_module.Pi3X.from_pretrained(model_name, **kwargs)
        self.model = model.eval().to(self.device)
        self.model_name = model_name

    def predict(self, images_rgb: Sequence[np.ndarray]) -> ModelPrediction:
        image_tensor, _, _ = self._tensorize(images_rgb)
        frame_count = image_tensor.shape[0]
        images_device = image_tensor.unsqueeze(0).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        intrinsics_np = np.repeat(self.camera_intrinsics[None, :, :], frame_count, axis=0)
        intrinsics_tensor = self.torch.from_numpy(intrinsics_np.astype(np.float32)).unsqueeze(0).to(
            self.device, non_blocking=self.device.type == "cuda"
        )

        self._synchronize()
        start = time.perf_counter()
        with self.torch.no_grad():
            with self._autocast():
                output = self.model(images_device, intrinsics=intrinsics_tensor)
        self._synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        try:
            local_points = _to_numpy(output["local_points"])[0]
            confidence_logits = _to_numpy(output["conf"])[0]
            if confidence_logits.ndim == 4 and confidence_logits.shape[-1] == 1:
                confidence_logits = confidence_logits[..., 0]
            confidence = activate_pi3x_confidence(
                confidence_logits, self.confidence_activation
            )
            camera_to_world = _to_numpy(output["camera_poses"], dtype=np.float64)[0]
        except Exception as exc:
            raise BackendError(f"invalid Pi3X output: {exc}") from exc

        points_c0 = anchor_local_points(local_points, camera_to_world)
        return ModelPrediction(
            points_c0=points_c0,
            confidence=confidence,
            inference_time_ms=elapsed_ms,
            model_name="Pi3X",
        )
