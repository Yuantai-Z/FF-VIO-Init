# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

"""ROS 1 cam0-only request/response node."""

import math
from collections import deque
from dataclasses import dataclass
from queue import Empty
import threading
import time
from typing import Optional, Sequence

import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float64MultiArray

from .backends import BackendError, Pi3XBackend, VGGTBackend, canonical_model_type
from .buffering import find_timestamp_match
from .camera_profile import (
    ProcessedCameraProfile,
    ProcessedImageSizeMismatch,
    parse_processed_camera_profile,
    parse_processed_image_size,
)
from .images import decode_rgb_image
from .pipeline import InferencePipeline
from .protocol import PackedPointCloud, stable_unique_timestamps, stamp_parts
from .request_queue import InferenceRequest, LatestRequestMailbox


DEFAULT_IMAGE_TOPIC = "/ov_msckf/camera_raw/cam0"
DEFAULT_REQUEST_TOPIC = "/vggtinitializer/request_timestamps"
DEFAULT_POINTCLOUD_TOPIC = "/vggtinitializer/pointcloud"


@dataclass(frozen=True)
class BufferedImage:
    timestamp: float
    rgb: np.ndarray


def build_backend_from_params(camera_profile: Optional[ProcessedCameraProfile] = None):
    model_type = canonical_model_type(rospy.get_param("~model_type", "vggt"))
    source_path = rospy.get_param("~model_source_path", "")
    revision = rospy.get_param("~model_revision", "")
    device = rospy.get_param("~device", "cuda")
    amp_dtype = rospy.get_param("~amp_dtype", "auto")
    if model_type == "vggt":
        return VGGTBackend(
            model_name=rospy.get_param("~model_name", "facebook/VGGT-1B"),
            model_revision=revision,
            model_source_path=source_path,
            device=device,
            amp_dtype=amp_dtype,
        )
    if camera_profile is None:
        raise BackendError("Pi3X requires a validated processed camera profile")
    return Pi3XBackend(
        model_name=rospy.get_param("~model_name", "yyfz233/Pi3X"),
        camera_intrinsics=np.array(
            [
                [camera_profile.fx, 0.0, camera_profile.cx],
                [0.0, camera_profile.fy, camera_profile.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        confidence_activation=rospy.get_param("~confidence_activation", "legacy_exp"),
        model_revision=revision,
        model_source_path=source_path,
        device=device,
        amp_dtype=amp_dtype,
    )


class InferenceNode:
    def __init__(
        self, backend, camera_profile: Optional[ProcessedCameraProfile] = None
    ) -> None:
        self.camera_profile = camera_profile
        self.model_label = getattr(backend, "model_name", backend.__class__.__name__)
        self.image_topic = rospy.get_param("~camera_topic", DEFAULT_IMAGE_TOPIC)
        self.request_topic = rospy.get_param("~request_topic", DEFAULT_REQUEST_TOPIC)
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", DEFAULT_POINTCLOUD_TOPIC)
        self.buffer_size = int(rospy.get_param("~buffer_size", 500))
        self.timestamp_tolerance = float(rospy.get_param("~timestamp_tolerance", 1e-6))
        self.request_wait_timeout = float(rospy.get_param("~request_wait_timeout", 5.0))
        self.processed_image_size = (
            camera_profile.image_size
            if camera_profile is not None
            else parse_processed_image_size(rospy.get_param("~processed_image_size", None))
        )
        self.fatal_error = None
        if (
            self.buffer_size <= 0
            or self.timestamp_tolerance < 0.0
            or self.request_wait_timeout < 0.0
        ):
            raise ValueError("buffer and timing parameters must be non-negative (buffer_size > 0)")

        self.pipeline = InferencePipeline(backend)
        self.buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        self.buffer_changed = threading.Condition(self.buffer_lock)
        self.requests = LatestRequestMailbox()

        publisher_queue = int(rospy.get_param("~publisher_queue_size", 20))
        subscriber_queue = int(rospy.get_param("~subscriber_queue_size", 1000))
        self.cloud_publisher = rospy.Publisher(
            self.pointcloud_topic, PointCloud2, queue_size=publisher_queue
        )
        self.image_subscriber = rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=subscriber_queue
        )
        self.request_subscriber = rospy.Subscriber(
            self.request_topic, Float64MultiArray, self.request_callback, queue_size=10
        )

        self.worker = threading.Thread(target=self._request_worker, daemon=True)
        self.worker.start()
        if self.camera_profile is None:
            camera_description = "none, grid=unchecked"
        else:
            camera_description = (
                f"'{self.camera_profile.name}', "
                f"K=({self.camera_profile.fx:.12g}, {self.camera_profile.fy:.12g}, "
                f"{self.camera_profile.cx:.12g}, {self.camera_profile.cy:.12g}), "
                f"grid={self.camera_profile.image_size.width}x"
                f"{self.camera_profile.image_size.height}"
            )
        rospy.loginfo(
            "[%s] ready: cam0='%s', camera_profile=%s, buffer=%d",
            self.model_label,
            self.image_topic,
            camera_description,
            self.buffer_size,
        )

    @staticmethod
    def _decode_rgb(message: Image) -> np.ndarray:
        return decode_rgb_image(message)

    def image_callback(self, message: Image) -> None:
        try:
            timestamp = float(message.header.stamp.to_sec())
            if not math.isfinite(timestamp):
                raise ValueError("non-finite image timestamp")
            image = self._decode_rgb(message)
            if self.processed_image_size is not None:
                self.processed_image_size.validate(image)
        except ProcessedImageSizeMismatch as exc:
            self.fatal_error = str(exc)
            rospy.logfatal("[%s] %s", self.model_label, exc)
            rospy.signal_shutdown(self.fatal_error)
            return
        except Exception as exc:
            rospy.logwarn("[%s] rejected processed cam0 image: %s", self.model_label, exc)
            return
        with self.buffer_changed:
            self.buffer.append(BufferedImage(timestamp, image))
            self.buffer_changed.notify_all()

    def request_callback(self, message: Float64MultiArray) -> None:
        timestamps = stable_unique_timestamps(message.data)
        if timestamps:
            self.requests.submit(timestamps)

    def _find_locked(self, timestamp: float) -> Optional[BufferedImage]:
        return find_timestamp_match(self.buffer, timestamp, self.timestamp_tolerance)

    def _resolve_images(self, request: InferenceRequest):
        deadline = time.monotonic() + self.request_wait_timeout
        with self.buffer_changed:
            while not rospy.is_shutdown():
                if not self.requests.is_current(request.generation):
                    return None
                frames = [self._find_locked(timestamp) for timestamp in request.timestamps]
                if all(frame is not None for frame in frames):
                    return [frame.rgb for frame in frames]
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = [
                        timestamp
                        for timestamp, frame in zip(request.timestamps, frames)
                        if frame is None
                    ]
                    rospy.logwarn(
                        "[%s] request aborted; missing processed cam0 stamps: %s",
                        self.model_label,
                        ", ".join(f"{timestamp:.9f}" for timestamp in missing),
                    )
                    return None
                self.buffer_changed.wait(timeout=min(remaining, 0.1))
        return None

    def _request_worker(self) -> None:
        while not rospy.is_shutdown():
            try:
                request = self.requests.get(timeout=0.1)
            except Empty:
                continue
            try:
                images = self._resolve_images(request)
                if images is not None:
                    self._process_request(request, images)
            except Exception as exc:
                rospy.logerr("[%s] request failed: %s", self.model_label, exc, exc_info=True)
            finally:
                self.requests.task_done()

    def _process_request(
        self, request: InferenceRequest, images: Sequence[np.ndarray]
    ) -> None:
        result = self.pipeline.process(request.timestamps, images)
        published, _ = self.requests.publish_if_current(
            request.generation,
            lambda: self._publish_result(request.timestamps, result),
        )
        if not published:
            rospy.loginfo(
                "[%s] discarded stale generation %d after inference",
                result.prediction.model_name,
                request.generation,
            )

    def _publish_result(self, timestamps, result) -> None:
        for timestamp, packed in zip(timestamps, result.clouds):
            self.cloud_publisher.publish(self._cloud_message(timestamp, packed))
        rospy.loginfo(
            "[%s] inference %.0f ms: published %d organized clouds",
            result.prediction.model_name,
            result.prediction.inference_time_ms,
            len(result.clouds),
        )

    @staticmethod
    def _cloud_message(timestamp: float, packed: PackedPointCloud) -> PointCloud2:
        message = PointCloud2()
        seconds, nanoseconds = stamp_parts(timestamp)
        message.header.stamp = rospy.Time(seconds, nanoseconds)
        message.header.frame_id = "cam0"
        message.height = packed.height
        message.width = packed.width
        message.fields = [
            PointField(
                name=field.name,
                offset=field.offset,
                datatype=field.datatype,
                count=field.count,
            )
            for field in packed.fields
        ]
        message.is_bigendian = packed.is_bigendian
        message.point_step = packed.point_step
        message.row_step = packed.row_step
        message.data = packed.data
        message.is_dense = packed.is_dense
        return message


def main() -> int:
    rospy.init_node("vggt_inference")
    try:
        model_type = canonical_model_type(rospy.get_param("~model_type", "vggt"))
        camera_profile = None
        if model_type == "pi3x":
            camera_profile = parse_processed_camera_profile(
                rospy.get_param("~camera_profile_name", None),
                rospy.get_param("~processed_image_size", None),
                rospy.get_param("~camera_intrinsics", None),
                required=True,
            )
        backend = build_backend_from_params(camera_profile)
        node = InferenceNode(backend, camera_profile)
    except Exception as exc:
        rospy.logfatal("failed to start vggt_ros companion: %s", exc, exc_info=True)
        return 1
    rospy.spin()
    return 1 if node.fatal_error is not None else 0
