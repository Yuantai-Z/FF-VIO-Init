# VGGT Point-Cloud ROS 1 Protocol

This document defines the wire contract consumed by the released `vggt_pt`
initializer. It describes the behavior implemented in
`ov_init/src/vggt/VGGTInitializer.{h,cpp}` and the processed-image publisher in
`ov_msckf/src/ros/ROS1Visualizer.cpp`.

The protocol uses only standard ROS 1 messages and no custom message package.
This repository includes a conforming standalone `vggt_ros` wrapper for
separately installed VGGT and Pi3X model runtimes. The wire contract remains
implementation-independent.

## Participants and sequence

1. The OpenVINS runner receives camera and IMU measurements.
2. In `vggt_pt` mode it rectifies/resizes the cam0 image and publishes it in
   the runner's private namespace.
3. The inference wrapper buffers each processed image by `header.stamp`.
4. When enough feature/IMU history exists, the initializer publishes a batch
   of requested camera timestamps.
5. The inference wrapper returns one organized point cloud for every requested
   timestamp.
6. The initializer matches responses, samples their pixel-aligned 3D points,
   estimates scale/gravity/velocity, and runs nonlinear refinement.

Responses may arrive in any order. A batch is complete only when every unique
requested timestamp has a matching point cloud.

## Topics

| Direction | Topic | Message | Requirement |
| --- | --- | --- | --- |
| estimator to inference | `/<estimator-node>/camera_raw/cam0` | `sensor_msgs/Image` | Required model input; node-private name |
| estimator to inference | `/vggtinitializer/request_timestamps` | `std_msgs/Float64MultiArray` | Required request |
| inference to estimator | `/vggtinitializer/pointcloud` | `sensor_msgs/PointCloud2` | Required response |

The two `/vggtinitializer/...` names are absolute constants in the C++ code.
Standard ROS name remapping can change them, but both participants must be
remapped consistently. There is no depth topic in this release contract, and
neither a model camera-pose response nor `/vggtinitializer/reset` is part of
this protocol. A backend may use its own pose prediction internally to express
all returned points in the required common C0 frame.

The processed-image topic is private to the OpenVINS node:

- `roslaunch ov_msckf serial.launch ...` names the node `/ov_msckf`, producing
  `/ov_msckf/camera_raw/cam0`;
- a direct `rosrun ov_msckf ros1_serial_msckf ...` uses
  `/ros1_serial_msckf/camera_raw/cam0` unless `__name` is overridden.

The inference launch files default to `/ov_msckf/camera_raw/cam0`, matching
the `serial.launch` node name. Override `camera_topic` only when the
estimator is launched under another node name.

## Bundled inference wrapper

Launch the repository's wrapper with canonical `model:=vggt` or
`model:=pi3x`; `model:=pi3` is retained only as a Pi3X compatibility alias.
It does not include, clone, or relicense either upstream model. Supply the
external source root with `model_source_path`, `VGGT_SOURCE_PATH`, or
`PI3X_SOURCE_PATH`, and select a Hugging Face identifier or local weight
snapshot with `model_name`. `model_revision` is forwarded to the upstream
`from_pretrained` loader so an immutable revision can be recorded.
`python_executable` (or `VGGT_ROS_PYTHON`) selects an interpreter containing
both ROS Python modules and the external model runtime.

Pi3X requires an atomic processed-camera profile containing a non-empty
`camera_profile_name`, `processed_image_size: {width, height}`, and
`camera_intrinsics: {fx, fy, cx, cy}`. The values must describe the processed
image on `camera_topic`. The wrapper neither derives nor rescales them. It
validates the profile before loading the model and exits if a received image
does not match the declared grid. Use `inference_pi3x_euroc.launch` or
`inference_pi3x_tumvi.launch` for the checked-in 518-wide profiles. A custom
camera should supply a complete overlay through `camera_profile_file` rather
than hand-entering K on the command line. `confidence_activation` accepts only
`legacy_exp` and `sigmoid`; the wrapper applies no confidence threshold.

The release wrapper is cam0-only and supports neither raw/unprocessed camera
input nor a second camera. It publishes no depth/reset interface and includes
no DA3, HorizonStream, batch, loop, or online-visualization backend.

## Processed image

Message type: `sensor_msgs/Image`.

- `header.stamp` is the camera timestamp later placed in the request array.
- `header.frame_id` is `cam0`.
- The public ROS 1 runners convert camera input to `mono8` before preprocessing,
  so a conforming inference implementation must accept `mono8`. The underlying
  publisher also selects `bgr8` or `bgra8` if it is called with a three- or
  four-channel OpenCV image.
- The image is already undistorted and resized through a square-pixel
  intermediate. Its width is `vggt_target_size` (518 by default), and its
  aspect-preserving height is rounded to a multiple of 14. That final rounding
  can make `fx` and `fy` differ slightly. Equidistant cameras may additionally
  be center-cropped when `init_vggt_crop_to_4_3` is enabled.
- `vggt_target_size` must be a positive multiple of 14.

The inference process must use this published image, not independently
preprocess the original bag image. Otherwise the returned point-cloud grid will
not share the feature tracker's pixel coordinates.

Only cam0 defines the released point-cloud contract. Other camera images can
exist in an OpenVINS configuration, but an inference implementation must not
publish their clouds on the single response topic.

## Timestamp request

Topic: `/vggtinitializer/request_timestamps`

Type: `std_msgs/Float64MultiArray`

`data` is a non-empty array of camera timestamps expressed as seconds in the
same ROS clock domain as the processed image headers. `layout` is not used.
The C++ publisher has queue size 1 and is latched.

Example semantic payload:

```text
data: [1403715520.125000, 1403715520.175000, 1403715520.225000]
```

The estimator waits up to 0.5 seconds for a request subscriber before it
publishes, then waits up to `init_vggt_wait_timeout` for responses. Start the
inference process before the dataset runner so it can buffer images and receive
the first request without consuming that timeout.

Treat the values as opaque identifiers. Do not replace them with frame indices,
wall-clock inference time, or rounded display strings.

## Point-cloud response

Topic: `/vggtinitializer/pointcloud`

Type: `sensor_msgs/PointCloud2`

Publish exactly one message per unique requested timestamp. The cloud must be
organized: `height > 0`, `width > 0`, and the row-major `(u, v)` cell must
correspond to pixel `(u, v)` in the processed image for that response frame.
Its height and width must therefore equal the geometry on which inference was
performed.

The byte layout must be native little-endian and tightly packed by row:
`is_bigendian=false` and `row_step == width * point_step`. The backing `data`
array must contain exactly `row_step * height` bytes. The consumer rejects
messages with row padding, truncated or trailing storage, or a different byte
order rather than risking a silently corrupted point map.

The recognized `FLOAT32` field offsets and `point_step` must be four-byte
aligned. The consumer requires `x=0`, `y=4`, `z=8`, and, when advertised,
`confidence=12`; additional metadata fields are ignored. The included
wrapper uses `point_step=16` when its backend supplies confidence. A backend
without confidence uses a tight xyz-only layout with `point_step=12`; the
consumer assigns 1.0 internally and does not apply the configured confidence
threshold.

### Header and geometry

- `header.stamp` is the corresponding requested timestamp converted to
  `ros::Time` without intentional rounding.
- `header.frame_id` is not inspected by the C++ consumer. Use `cam0` for
  diagnostics, but do not rely on the string to communicate coordinate
  semantics.
- All frames in one requested batch must use one common 3D coordinate frame:
  the optical frame of the first requested camera, denoted C0.
- A point associated with a pixel is expected to obey the pinhole projection
  convention `u_norm = x / z`, `v_norm = y / z`. In the usual optical-axis
  convention this is +x right, +y down, +z forward.
- Scale may be arbitrary, because the initializer estimates it, but it must be
  finite and consistent across the requested batch.

This is a world-point map aligned to the first frame, not an independent
camera-local depth back-projection for each response frame.

### Fields

| Field | PointField datatype | Count | Meaning |
| --- | --- | --- | --- |
| `x` | `FLOAT32` | 1 | C0 x coordinate; required |
| `y` | `FLOAT32` | 1 | C0 y coordinate; required |
| `z` | `FLOAT32` | 1 | C0 z coordinate; required |
| `confidence` | `FLOAT32` | 1 | Optional non-negative model confidence |

`x`, `y`, and `z` are mandatory. The C++ code constructs typed iterators for
them and for confidence when advertised, so names and datatypes must match the
table exactly. Additional fields are ignored.

If `confidence` is omitted, the consumer assigns every point confidence 1.0
and does not apply the configured confidence threshold. Geometry validity
checks still apply. If the field is present:

- zero marks a grid cell invalid;
- positive values are used for sampling and square-root residual weighting;
- when `init_vggt_use_confidence_filter=true`, values below
  `init_vggt_confidence_threshold` are changed to zero;
- values need not be normalized to `[0, 1]`, but they must be finite and
  non-negative.

Every point with positive confidence must have finite `x`, `y`, and `z` and a
non-zero 3D norm. Represent an invalid, non-finite, or zero-vector prediction
with confidence exactly zero. If the field is omitted, geometry checks still
reject invalid cells after they receive the compatibility confidence 1.0.

### Timestamp matching

During an active request batch, the callback accepts a response only if

```text
abs(response.header.stamp - requested_timestamp) < 1e-6 seconds
```

and stores it under the original requested double. Messages received with no
active request are not a supported synchronization mechanism and can be
discarded when the next batch starts. Downstream validation also checks that
all requested frames were received. A convenient inference
implementation should retain the exact requested double as its key and build
the ROS stamp from that value, rather than copying a nearby buffered-image
timestamp.

## Inference conformance checklist

- Subscribe to the processed cam0 topic under the actual estimator node name.
- Buffer images by their full-precision ROS stamp.
- Accept a `Float64MultiArray` timestamp batch and preserve its values.
- Publish one non-empty organized cloud per requested timestamp.
- Use the processed image's exact `H x W` grid.
- Express every cloud in the first requested camera frame C0.
- Publish mandatory `FLOAT32` `x/y/z` fields.
- Either omit confidence or publish finite, non-negative `FLOAT32` confidence;
  use zero for invalid cells.
- Keep all valid coordinates finite and use a consistent arbitrary scale.
- Do not publish depth, another camera's cloud, or per-frame camera-local point
  coordinates as a substitute for this contract.

Useful live checks:

```bash
rostopic info /vggtinitializer/request_timestamps
rostopic info /vggtinitializer/pointcloud
rostopic echo -n 1 /vggtinitializer/request_timestamps
rostopic echo -n 1 /vggtinitializer/pointcloud/header
rostopic echo -n 1 /vggtinitializer/pointcloud/fields
rostopic hz /vggtinitializer/pointcloud
```

For a timeout, first compare the full-precision request and response stamps,
then check cloud count, dimensions, field types, the inference image topic,
and `init_vggt_wait_timeout`.
