# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

import numpy as np

from vggt_ros.backends import _TorchBackend, activate_pi3x_confidence, canonical_model_type
from vggt_ros.pipeline import InferencePipeline
from vggt_ros.prediction import ModelPrediction


class MockBackend:
    def __init__(self):
        self.received_shapes = None

    def predict(self, images):
        self.received_shapes = [image.shape for image in images]
        frame_count, height, width = len(images), images[0].shape[0], images[0].shape[1]
        points = np.ones((frame_count, height, width, 3), dtype=np.float32)
        confidence = np.full((frame_count, height, width), 2.0, dtype=np.float32)
        return ModelPrediction(
            points_c0=points,
            confidence=confidence,
            inference_time_ms=3.0,
            model_name="mock",
        )


class MockPipelineTest(unittest.TestCase):
    def test_torch_cuda_operations_target_selected_device(self):
        class Device:
            def __init__(self, value):
                self.value = value
                self.type = value.split(":", 1)[0]

        class Cuda:
            def __init__(self):
                self.capability_devices = []
                self.synchronized_devices = []

            @staticmethod
            def is_available():
                return True

            def get_device_capability(self, device):
                self.capability_devices.append(device)
                return (8, 9)

            def synchronize(self, *, device):
                self.synchronized_devices.append(device)

        class Torch:
            bfloat16 = object()
            float16 = object()

            def __init__(self):
                self.cuda = Cuda()

            @staticmethod
            def device(value):
                return Device(value)

        torch = Torch()
        backend = _TorchBackend(device="cuda:1", amp_dtype="auto", torch_module=torch)
        backend._synchronize()

        self.assertIs(torch.cuda.capability_devices[0], backend.device)
        self.assertIs(torch.cuda.synchronized_devices[0], backend.device)
        self.assertEqual(backend.device.value, "cuda:1")

    def test_mock_backend_keeps_processed_grid(self):
        backend = MockBackend()
        pipeline = InferencePipeline(backend)
        images = [np.zeros((14, 28, 3), dtype=np.uint8) for _ in range(2)]

        result = pipeline.process([9.0, 8.0], images)

        self.assertEqual(backend.received_shapes, [(14, 28, 3), (14, 28, 3)])
        self.assertEqual([(cloud.height, cloud.width) for cloud in result.clouds], [(14, 28)] * 2)
        self.assertEqual(result.prediction.model_name, "mock")

    def test_pi3x_names_and_confidence_domains_are_explicit(self):
        self.assertEqual(canonical_model_type("pi3"), "pi3x")
        self.assertEqual(canonical_model_type("Pi3X"), "pi3x")
        logits = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        legacy = activate_pi3x_confidence(logits, "legacy_exp")
        sigmoid = activate_pi3x_confidence(logits, "sigmoid")
        self.assertTrue(np.all(legacy >= 1.0))
        self.assertTrue(np.all((sigmoid >= 0.0) & (sigmoid <= 1.0)))
        self.assertAlmostEqual(float(legacy[1]), 2.0)
        self.assertAlmostEqual(float(sigmoid[1]), 0.5)

    def test_pipeline_rejects_backend_spatial_resize(self):
        class ResizingBackend(MockBackend):
            def predict(self, images):
                prediction = super().predict(images)
                return ModelPrediction(
                    prediction.points_c0[:, :, :-1],
                    prediction.confidence[:, :, :-1],
                )

        pipeline = InferencePipeline(ResizingBackend())
        with self.assertRaisesRegex(ValueError, "changed the estimator's processed image grid"):
            pipeline.process([1.0], [np.zeros((14, 28, 3), dtype=np.uint8)])

    def test_missing_backend_confidence_remains_an_optional_protocol_field(self):
        class NoConfidenceBackend(MockBackend):
            def predict(self, images):
                prediction = super().predict(images)
                return ModelPrediction(
                    prediction.points_c0,
                    None,
                )

        result = InferencePipeline(NoConfidenceBackend()).process(
            [1.0], [np.zeros((14, 28, 3), dtype=np.uint8)]
        )
        cloud = result.clouds[0]
        self.assertEqual([field.name for field in cloud.fields], ["x", "y", "z"])
        self.assertEqual(cloud.point_step, 12)
        self.assertEqual(cloud.row_step, 28 * 12)


if __name__ == "__main__":
    unittest.main()
