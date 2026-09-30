# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from vggt_ros.camera_profile import (
    ProcessedImageSize,
    ProcessedImageSizeMismatch,
    parse_processed_camera_profile,
    parse_processed_image_size,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class CameraProfileTest(unittest.TestCase):
    PROFILES = {
        "euroc": {
            "size": {"width": 518, "height": 280},
            "intrinsics": {
                "fx": 257.221205544979,
                "fy": 255.511586380005,
                "cx": 259.0,
                "cy": 140.0,
            },
        },
        "tumvi": {
            "size": {"width": 518, "height": 392},
            "intrinsics": {
                "fx": 309.816217031999,
                "fy": 309.817448030133,
                "cx": 259.0,
                "cy": 196.0,
            },
        },
    }

    def test_checked_in_profiles_pin_processed_geometry(self):
        for dataset, expected in self.PROFILES.items():
            with self.subTest(dataset=dataset):
                path = PACKAGE_ROOT / "config" / f"pi3x_{dataset}.yaml"
                with path.open("r", encoding="utf-8") as stream:
                    profile = yaml.safe_load(stream)
                self.assertEqual(profile["processed_image_size"], expected["size"])
                self.assertEqual(profile["camera_intrinsics"], expected["intrinsics"])
                parsed = parse_processed_camera_profile(
                    profile["camera_profile_name"],
                    profile["processed_image_size"],
                    profile["camera_intrinsics"],
                )
                self.assertEqual(parsed.image_size.width, expected["size"]["width"])
                self.assertEqual(parsed.image_size.height, expected["size"]["height"])
                self.assertEqual(parsed.fx, expected["intrinsics"]["fx"])
                self.assertEqual(parsed.fy, expected["intrinsics"]["fy"])
                self.assertEqual(parsed.cx, expected["intrinsics"]["cx"])
                self.assertEqual(parsed.cy, expected["intrinsics"]["cy"])

    def test_base_pi3x_config_has_no_placeholder_intrinsics(self):
        path = PACKAGE_ROOT / "config" / "pi3x.yaml"
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertNotIn("camera_intrinsics", config)

    def test_base_configs_do_not_expose_camera_pose_publication(self):
        for name in ("vggt", "pi3x"):
            with self.subTest(name=name):
                path = PACKAGE_ROOT / "config" / f"{name}.yaml"
                with path.open("r", encoding="utf-8") as stream:
                    config = yaml.safe_load(stream)
                self.assertNotIn("pose_topic", config)
                self.assertNotIn("publish_poses", config)

    def test_size_parser_and_runtime_grid_check(self):
        size = parse_processed_image_size({"width": 518, "height": 280})
        self.assertEqual(size, ProcessedImageSize(518, 280))
        size.validate(np.zeros((280, 518, 3), dtype=np.uint8))

        with self.assertRaisesRegex(
            ProcessedImageSizeMismatch, "expected 518x280, got 518x392"
        ):
            size.validate(np.zeros((392, 518, 3), dtype=np.uint8))

        for invalid in (
            {},
            {"width": 518},
            {"width": 0, "height": 280},
            {"width": 518.5, "height": 280},
            {"width": True, "height": 280},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_processed_image_size(invalid)

    def test_profile_is_atomic_and_rejects_intrinsics_outside_the_grid(self):
        intrinsics = {"fx": 300.0, "fy": 301.0, "cx": 259.0, "cy": 196.0}
        self.assertIsNone(parse_processed_camera_profile(None, None, intrinsics))
        with self.assertRaisesRegex(ValueError, "Pi3X requires"):
            parse_processed_camera_profile(None, None, intrinsics, required=True)
        with self.assertRaisesRegex(ValueError, "camera_profile_name"):
            parse_processed_camera_profile(
                None, {"width": 518, "height": 392}, intrinsics
            )
        with self.assertRaisesRegex(ValueError, "processed_image_size"):
            parse_processed_camera_profile("tumvi", None, intrinsics)
        with self.assertRaisesRegex(ValueError, "divisible by 14"):
            parse_processed_camera_profile(
                "tumvi", {"width": 518, "height": 391}, intrinsics
            )
        with self.assertRaisesRegex(ValueError, "principal point"):
            parse_processed_camera_profile(
                "tumvi",
                {"width": 518, "height": 392},
                {"fx": 300.0, "fy": 301.0, "cx": 518.0, "cy": 196.0},
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_processed_camera_profile(
                "tumvi",
                {"width": 518, "height": 392},
                {"fx": float("nan"), "fy": 301.0, "cx": 259.0, "cy": 196.0},
            )

    def test_pi3x_launch_wrappers_select_profiles_without_numeric_overrides(self):
        for dataset in self.PROFILES:
            with self.subTest(dataset=dataset):
                path = PACKAGE_ROOT / "launch" / f"inference_pi3x_{dataset}.launch"
                root = ET.parse(str(path)).getroot()
                self.assertEqual(
                    root.find("arg[@name='camera_topic']").attrib["default"],
                    "/ov_msckf/camera_raw/cam0",
                )
                include = root.find("include")
                self.assertIsNotNone(include)
                arguments = {
                    element.attrib["name"]: element.attrib.get("value")
                    for element in include.findall("arg")
                }
                self.assertEqual(arguments["model"], "pi3x")
                self.assertEqual(arguments["camera_topic"], "$(arg camera_topic)")
                self.assertTrue(arguments["config_file"].endswith("/config/pi3x.yaml"))
                self.assertTrue(
                    arguments["camera_profile_file"].endswith(
                        f"/config/pi3x_{dataset}.yaml"
                    )
                )
                all_argument_names = {
                    element.attrib["name"] for element in root.findall(".//arg")
                }
                self.assertTrue(
                    all_argument_names.isdisjoint(
                        {"fx", "fy", "cx", "cy", "publish_poses"}
                    )
                )

    def test_generic_launch_loads_optional_profile_after_base_config(self):
        path = PACKAGE_ROOT / "launch" / "inference.launch"
        root = ET.parse(str(path)).getroot()
        self.assertIsNotNone(root.find("arg[@name='camera_profile_file']"))
        node = root.find("node")
        self.assertEqual(node.attrib["clear_params"], "true")
        rosparams = node.findall("rosparam")
        self.assertEqual(rosparams[0].attrib["file"], "$(arg config_file)")
        self.assertEqual(rosparams[1].attrib["file"], "$(arg camera_profile_file)")
        self.assertIn("camera_profile_file", rosparams[1].attrib["if"])
        all_argument_names = {element.attrib["name"] for element in root.findall("arg")}
        self.assertTrue(
            all_argument_names.isdisjoint(
                {"fx", "fy", "cx", "cy", "publish_poses"}
            )
        )

    def test_node_treats_profile_grid_mismatch_as_fatal(self):
        # Build only the callback state so this test never starts ROS subscribers or a model.
        from vggt_ros import node as node_module

        subject = node_module.InferenceNode.__new__(node_module.InferenceNode)
        subject.model_label = "mock"
        subject.processed_image_size = ProcessedImageSize(518, 280)
        subject.fatal_error = None
        subject._decode_rgb = lambda unused: np.zeros((392, 518, 3), dtype=np.uint8)
        message = mock.Mock()
        message.header.stamp.to_sec.return_value = 1.0

        with mock.patch.object(node_module.rospy, "logfatal") as logfatal, mock.patch.object(
            node_module.rospy, "signal_shutdown"
        ) as signal_shutdown:
            subject.image_callback(message)

        self.assertIn("expected 518x280, got 518x392", subject.fatal_error)
        logfatal.assert_called_once()
        signal_shutdown.assert_called_once_with(subject.fatal_error)

    def test_pi3x_missing_profile_fails_before_backend_model_loading(self):
        from vggt_ros import node as node_module

        def get_param(name, default=None):
            return "pi3x" if name == "~model_type" else default

        with mock.patch.object(node_module.rospy, "init_node"), mock.patch.object(
            node_module.rospy, "get_param", side_effect=get_param
        ), mock.patch.object(node_module.rospy, "logfatal"), mock.patch.object(
            node_module, "build_backend_from_params"
        ) as build_backend:
            status = node_module.main()

        self.assertEqual(status, 1)
        build_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
