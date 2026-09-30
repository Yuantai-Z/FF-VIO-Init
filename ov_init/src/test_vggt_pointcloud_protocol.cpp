/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/sensor_data.h"
#include "vggt/VGGTInitializer.h"

#include <boost/make_shared.hpp>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ov_init {

class VGGTInitializerTestAccess {
public:
  static void feed_pointcloud(VGGTInitializer &initializer, const sensor_msgs::PointCloud2ConstPtr &message) {
    initializer.vggtPointcloudCallback(message);
  }

  static bool received_has_confidence(const VGGTInitializer &initializer, double timestamp) {
    return initializer.vggt_received_pointcloud_has_confidence_.at(timestamp);
  }

  static bool received_pointcloud(const VGGTInitializer &initializer, double timestamp) {
    return initializer.vggt_received_pointclouds_.count(timestamp) != 0;
  }

  static void load_received_pointclouds(VGGTInitializer &initializer) {
    initializer.vggt_point3d_data = initializer.vggt_received_pointclouds_;
    initializer.vggt_pointcloud_has_confidence_ = initializer.vggt_received_pointcloud_has_confidence_;
  }

  static std::pair<int, int> filter_loaded_pointclouds(VGGTInitializer &initializer) {
    return initializer.filter_loaded_pointclouds();
  }

  static const cv::Mat &loaded_pointcloud(const VGGTInitializer &initializer, double timestamp) {
    return initializer.vggt_point3d_data.at(timestamp);
  }

  static void seed_feature_cache(VGGTInitializer &initializer) {
    initializer.features_point3d_c0_[7] = Eigen::Vector3d(1.0, 2.0, 3.0);
  }

  static bool feature_cache_empty(const VGGTInitializer &initializer) {
    return initializer.features_point3d_c0_.empty();
  }

  static void set_loaded_pointcloud(VGGTInitializer &initializer, double timestamp, const cv::Mat &pointcloud) {
    initializer.vggt_point3d_data.clear();
    initializer.vggt_point3d_data[timestamp] = pointcloud.clone();
  }

  static bool get_first_camera_point(VGGTInitializer &initializer, const std::shared_ptr<ov_core::Feature> &feature,
                                     Eigen::Vector3d &point) {
    return initializer.get_pFinC0(feature, point);
  }
};

} // namespace ov_init

namespace {

ov_init::InertialInitializerOptions make_options() {
  ov_init::InertialInitializerOptions options;
  Eigen::VectorXd raw_calibration(8);
  raw_calibration << 12.0, 12.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0;
  const ov_core::VGGTPreprocessPlan plan =
      ov_core::makeVGGTPreprocessPlan(0, "radtan", raw_calibration, cv::Size(14, 14), 14, 0.15, false);
  auto camera = std::make_shared<ov_core::CamRadtan>(plan.output_roi.width, plan.output_roi.height);
  camera->set_value(plan.output_calib);
  options.vggt_preprocess_plans[0] = plan;
  options.camera_intrinsics[0] = camera;
  return options;
}

sensor_msgs::PointCloud2Ptr make_pointcloud(double timestamp, const std::vector<cv::Vec3f> &points,
                                            const std::vector<float> *confidences, const cv::Size &domain) {
  if (domain.width <= 0 || domain.height <= 0 ||
      points.size() != static_cast<size_t>(domain.width) * static_cast<size_t>(domain.height) ||
      (confidences != nullptr && confidences->size() != points.size())) {
    throw std::invalid_argument("test pointcloud data does not match its image domain");
  }
  auto message = boost::make_shared<sensor_msgs::PointCloud2>();
  message->header.stamp = ros::Time(timestamp);
  sensor_msgs::PointCloud2Modifier modifier(*message);
  if (confidences != nullptr) {
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32,
                                 "y", 1, sensor_msgs::PointField::FLOAT32,
                                 "z", 1, sensor_msgs::PointField::FLOAT32,
                                 "confidence", 1, sensor_msgs::PointField::FLOAT32);
  } else {
    modifier.setPointCloud2Fields(3, "x", 1, sensor_msgs::PointField::FLOAT32,
                                 "y", 1, sensor_msgs::PointField::FLOAT32,
                                 "z", 1, sensor_msgs::PointField::FLOAT32);
  }
  modifier.resize(points.size());
  message->width = static_cast<uint32_t>(domain.width);
  message->height = static_cast<uint32_t>(domain.height);
  message->row_step = message->width * message->point_step;

  sensor_msgs::PointCloud2Iterator<float> iter_x(*message, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*message, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*message, "z");
  if (confidences != nullptr) {
    sensor_msgs::PointCloud2Iterator<float> iter_confidence(*message, "confidence");
    for (size_t index = 0; index < points.size(); ++index, ++iter_x, ++iter_y, ++iter_z, ++iter_confidence) {
      *iter_x = points.at(index)[0];
      *iter_y = points.at(index)[1];
      *iter_z = points.at(index)[2];
      *iter_confidence = confidences->at(index);
    }
  } else {
    for (const auto &point : points) {
      *iter_x = point[0];
      *iter_y = point[1];
      *iter_z = point[2];
      ++iter_x;
      ++iter_y;
      ++iter_z;
    }
  }
  return message;
}

sensor_msgs::PointCloud2Ptr make_pointcloud_with_extra_field(double timestamp, const std::vector<cv::Vec3f> &points,
                                                             const std::vector<float> &confidences,
                                                             const cv::Size &domain) {
  if (domain.width <= 0 || domain.height <= 0 ||
      points.size() != static_cast<size_t>(domain.width) * static_cast<size_t>(domain.height) ||
      confidences.size() != points.size()) {
    throw std::invalid_argument("test pointcloud data does not match its image domain");
  }
  auto message = boost::make_shared<sensor_msgs::PointCloud2>();
  message->header.stamp = ros::Time(timestamp);
  sensor_msgs::PointCloud2Modifier modifier(*message);
  modifier.setPointCloud2Fields(5, "x", 1, sensor_msgs::PointField::FLOAT32,
                               "y", 1, sensor_msgs::PointField::FLOAT32,
                               "z", 1, sensor_msgs::PointField::FLOAT32,
                               "confidence", 1, sensor_msgs::PointField::FLOAT32,
                               "diagnostic", 1, sensor_msgs::PointField::UINT8);
  message->point_step = 20;
  modifier.resize(points.size());
  message->width = static_cast<uint32_t>(domain.width);
  message->height = static_cast<uint32_t>(domain.height);
  message->row_step = message->width * message->point_step;

  sensor_msgs::PointCloud2Iterator<float> iter_x(*message, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*message, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*message, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_confidence(*message, "confidence");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_diagnostic(*message, "diagnostic");
  for (size_t index = 0; index < points.size();
       ++index, ++iter_x, ++iter_y, ++iter_z, ++iter_confidence, ++iter_diagnostic) {
    *iter_x = points.at(index)[0];
    *iter_y = points.at(index)[1];
    *iter_z = points.at(index)[2];
    *iter_confidence = confidences.at(index);
    *iter_diagnostic = static_cast<uint8_t>(index % 256);
  }
  return message;
}

bool is_valid(const cv::Mat &pointcloud, int column) {
  return pointcloud.at<cv::Vec4f>(0, column)[3] > 0.0f;
}

} // namespace

int main() {
  ov_init::InertialInitializerOptions options = make_options();
  options.init_vggt_use_confidence_filter = true;
  options.init_vggt_confidence_threshold = 1.1;
  auto database = std::make_shared<ov_core::FeatureDatabase>();
  auto imu_data = std::make_shared<std::vector<ov_core::ImuData>>();
  ov_init::VGGTInitializer initializer(options);

  const cv::Size pointcloud_domain(options.camera_intrinsics.at(0)->w(), options.camera_intrinsics.at(0)->h());
  const size_t point_count = static_cast<size_t>(pointcloud_domain.width) * static_cast<size_t>(pointcloud_domain.height);
  std::vector<cv::Vec3f> points(point_count, cv::Vec3f(1.0f, 0.0f, 2.0f));
  points.at(1) = cv::Vec3f(2.0f, 0.0f, 3.0f);
  std::vector<float> confidences(point_count, 1.1f);
  confidences.at(0) = 1.0f;

  auto xyz_only = make_pointcloud(1.0, points, nullptr, pointcloud_domain);
  if (xyz_only->point_step != 12) {
    std::cerr << "xyz-only test message was not tightly packed\n";
    return 1;
  }
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, xyz_only);

  ov_init::VGGTInitializer schema_initializer(options);
  auto extra_field = make_pointcloud_with_extra_field(1.5, points, confidences, pointcloud_domain);
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(schema_initializer, extra_field);
  if (!ov_init::VGGTInitializerTestAccess::received_pointcloud(schema_initializer, 1.5) ||
      !ov_init::VGGTInitializerTestAccess::received_has_confidence(schema_initializer, 1.5)) {
    std::cerr << "aligned extra PointCloud2 field was not ignored\n";
    return 1;
  }

  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, make_pointcloud(2.0, points, &confidences, pointcloud_domain));
  if (!ov_init::VGGTInitializerTestAccess::received_pointcloud(initializer, 1.0) ||
      !ov_init::VGGTInitializerTestAccess::received_pointcloud(initializer, 2.0) ||
      ov_init::VGGTInitializerTestAccess::received_has_confidence(initializer, 1.0) ||
      !ov_init::VGGTInitializerTestAccess::received_has_confidence(initializer, 2.0)) {
    std::cerr << "xyz-only compatibility or xyz+confidence capability was not preserved\n";
    return 1;
  }

  const cv::Size wrong_domain(static_cast<int>(point_count), 1);
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(
      initializer, make_pointcloud(1.75, points, &confidences, wrong_domain));
  if (ov_init::VGGTInitializerTestAccess::received_pointcloud(initializer, 1.75)) {
    std::cerr << "pointcloud with the right sample count but wrong image domain was accepted\n";
    return 1;
  }

  const auto domain_is_rejected = [&](double timestamp, const cv::Size &domain) {
    const size_t count = static_cast<size_t>(domain.width) * static_cast<size_t>(domain.height);
    const std::vector<cv::Vec3f> domain_points(count, cv::Vec3f(1.0f, 0.0f, 2.0f));
    const std::vector<float> domain_confidences(count, 1.1f);
    ov_init::VGGTInitializerTestAccess::feed_pointcloud(
        initializer, make_pointcloud(timestamp, domain_points, &domain_confidences, domain));
    return !ov_init::VGGTInitializerTestAccess::received_pointcloud(initializer, timestamp);
  };
  if (!domain_is_rejected(1.76, cv::Size(pointcloud_domain.width + 1, pointcloud_domain.height)) ||
      !domain_is_rejected(1.77, cv::Size(pointcloud_domain.width, pointcloud_domain.height + 1))) {
    std::cerr << "wrong-width or wrong-height pointcloud domain was accepted\n";
    return 1;
  }

  auto padded_rows = make_pointcloud(2.1, points, &confidences, pointcloud_domain);
  padded_rows->row_step += 4;
  padded_rows->data.resize(static_cast<size_t>(padded_rows->row_step) * padded_rows->height);
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, padded_rows);

  auto big_endian = make_pointcloud(2.2, points, &confidences, pointcloud_domain);
  big_endian->is_bigendian = true;
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, big_endian);

  auto wrong_field_type = make_pointcloud(2.3, points, &confidences, pointcloud_domain);
  for (auto &field : wrong_field_type->fields) {
    if (field.name == "x") {
      field.datatype = sensor_msgs::PointField::FLOAT64;
    }
  }
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, wrong_field_type);

  auto truncated_data = make_pointcloud(2.4, points, &confidences, pointcloud_domain);
  truncated_data->data.pop_back();
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, truncated_data);

  auto wrong_confidence_offset = make_pointcloud(2.5, points, &confidences, pointcloud_domain);
  for (auto &field : wrong_confidence_offset->fields) {
    if (field.name == "confidence") {
      field.offset = 8;
    }
  }
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, wrong_confidence_offset);

  auto misaligned_points = make_pointcloud(2.6, points, &confidences, pointcloud_domain);
  misaligned_points->point_step = 18;
  misaligned_points->row_step = misaligned_points->width * misaligned_points->point_step;
  misaligned_points->data.resize(static_cast<size_t>(misaligned_points->row_step) * misaligned_points->height);
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, misaligned_points);

  auto confidence_outside_record = make_pointcloud(2.7, points, &confidences, pointcloud_domain);
  confidence_outside_record->point_step = 12;
  confidence_outside_record->row_step = confidence_outside_record->width * confidence_outside_record->point_step;
  confidence_outside_record->data.resize(
      static_cast<size_t>(confidence_outside_record->row_step) * confidence_outside_record->height);
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(initializer, confidence_outside_record);

  for (double rejected_timestamp : {2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7}) {
    if (ov_init::VGGTInitializerTestAccess::received_pointcloud(initializer, rejected_timestamp)) {
      std::cerr << "unsupported PointCloud2 layout was accepted at timestamp " << rejected_timestamp << '\n';
      return 1;
    }
  }

  ov_init::VGGTInitializerTestAccess::load_received_pointclouds(initializer);
  const auto filter_result = ov_init::VGGTInitializerTestAccess::filter_loaded_pointclouds(initializer);
  if (filter_result.second != 1 || filter_result.first != 1) {
    std::cerr << "confidence threshold was applied to the wrong set of frames\n";
    return 1;
  }

  const cv::Mat &without_confidence = ov_init::VGGTInitializerTestAccess::loaded_pointcloud(initializer, 1.0);
  const cv::Mat &with_confidence = ov_init::VGGTInitializerTestAccess::loaded_pointcloud(initializer, 2.0);
  if (!is_valid(without_confidence, 0) || !is_valid(without_confidence, 1) ||
      is_valid(with_confidence, 0) || !is_valid(with_confidence, 1)) {
    std::cerr << "confidence filtering did not follow PointCloud2 field presence\n";
    return 1;
  }

  options.init_vggt_use_confidence_filter = false;
  ov_init::VGGTInitializer unfiltered_initializer(options);
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(
      unfiltered_initializer, make_pointcloud(3.0, points, &confidences, pointcloud_domain));
  ov_init::VGGTInitializerTestAccess::load_received_pointclouds(unfiltered_initializer);
  const auto unfiltered_result = ov_init::VGGTInitializerTestAccess::filter_loaded_pointclouds(unfiltered_initializer);
  const cv::Mat &unfiltered = ov_init::VGGTInitializerTestAccess::loaded_pointcloud(unfiltered_initializer, 3.0);
  if (unfiltered_result.first != 0 || unfiltered_result.second != 0 || !is_valid(unfiltered, 0) || !is_valid(unfiltered, 1)) {
    std::cerr << "disabled confidence filtering changed valid samples\n";
    return 1;
  }

  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<cv::Vec3f> malformed_points(point_count, cv::Vec3f(1.0f, 0.0f, 2.0f));
  malformed_points.at(1) = cv::Vec3f(nan, 0.0f, 2.0f);
  malformed_points.at(2) = cv::Vec3f(1.0f, 0.0f, infinity);
  malformed_points.at(3) = cv::Vec3f(0.0f, 0.0f, 0.0f);
  std::vector<float> malformed_confidences(point_count, 1.0f);
  malformed_confidences.at(4) = 0.0f;
  malformed_confidences.at(5) = -1.0f;
  malformed_confidences.at(6) = nan;
  malformed_confidences.at(7) = infinity;
  ov_init::VGGTInitializerTestAccess::feed_pointcloud(
      unfiltered_initializer, make_pointcloud(4.0, malformed_points, &malformed_confidences, pointcloud_domain));
  ov_init::VGGTInitializerTestAccess::load_received_pointclouds(unfiltered_initializer);
  const auto malformed_result = ov_init::VGGTInitializerTestAccess::filter_loaded_pointclouds(unfiltered_initializer);
  const cv::Mat &malformed = ov_init::VGGTInitializerTestAccess::loaded_pointcloud(unfiltered_initializer, 4.0);
  if (malformed_result.first != 7 || malformed_result.second != 0 || !is_valid(malformed, 0)) {
    std::cerr << "malformed pointcloud sample count was not rejected as expected\n";
    return 1;
  }
  for (int column = 1; column < 8; ++column) {
    if (is_valid(malformed, column)) {
      std::cerr << "malformed pointcloud sample remained valid at column " << column << '\n';
      return 1;
    }
  }

  ov_init::VGGTInitializerTestAccess::seed_feature_cache(initializer);
  double initialization_timestamp = 0.0;
  Eigen::MatrixXd covariance;
  std::vector<std::shared_ptr<ov_type::Type>> order;
  std::shared_ptr<ov_type::IMU> imu;
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> clones;
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> landmarks;
  ov_init::InitializationDiagnostics diagnostics;
  if (initializer.initialize_pointcloud(initialization_timestamp, covariance, order, imu, database, imu_data, clones, landmarks,
                                        diagnostics) ||
      !ov_init::VGGTInitializerTestAccess::feature_cache_empty(initializer)) {
    std::cerr << "failed initialization attempt retained its feature-point cache\n";
    return 1;
  }

  cv::Mat camera_zero_cloud(1, 2, CV_32FC4);
  camera_zero_cloud.at<cv::Vec4f>(0, 0) = cv::Vec4f(1.0f, 2.0f, 3.0f, 1.0f);
  camera_zero_cloud.at<cv::Vec4f>(0, 1) = cv::Vec4f(4.0f, 5.0f, 6.0f, 1.0f);
  ov_init::VGGTInitializerTestAccess::set_loaded_pointcloud(initializer, 5.0, camera_zero_cloud);

  auto stereo_feature = std::make_shared<ov_core::Feature>();
  Eigen::VectorXf camera_one_uv(2);
  camera_one_uv << 0.0f, 0.0f;
  stereo_feature->timestamps[1] = {4.0};
  stereo_feature->uvs[1] = {camera_one_uv};
  Eigen::VectorXf camera_zero_uv(2);
  camera_zero_uv << 1.0f, 0.0f;
  stereo_feature->timestamps[0] = {5.0};
  stereo_feature->uvs[0] = {camera_zero_uv};

  Eigen::Vector3d camera_zero_point = Eigen::Vector3d::Zero();
  if (!ov_init::VGGTInitializerTestAccess::get_first_camera_point(initializer, stereo_feature, camera_zero_point) ||
      !camera_zero_point.isApprox(Eigen::Vector3d(4.0, 5.0, 6.0))) {
    std::cerr << "stereo feature was not associated with the camera-0 pointcloud\n";
    return 1;
  }

  auto camera_one_only = std::make_shared<ov_core::Feature>();
  camera_one_only->timestamps[1] = {5.0};
  camera_one_only->uvs[1] = {camera_one_uv};
  if (ov_init::VGGTInitializerTestAccess::get_first_camera_point(initializer, camera_one_only, camera_zero_point)) {
    std::cerr << "camera-1-only feature was accepted for a camera-0 pointcloud\n";
    return 1;
  }

  auto malformed_uv = std::make_shared<ov_core::Feature>();
  malformed_uv->timestamps[0] = {5.0};
  malformed_uv->uvs[0] = {Eigen::VectorXf::Zero(1)};
  if (ov_init::VGGTInitializerTestAccess::get_first_camera_point(initializer, malformed_uv, camera_zero_point)) {
    std::cerr << "undersized camera-0 pixel coordinate was accepted\n";
    return 1;
  }

  auto mismatched_measurements = std::make_shared<ov_core::Feature>();
  mismatched_measurements->timestamps[0] = {5.0, 5.1};
  mismatched_measurements->uvs[0] = {camera_zero_uv};
  if (ov_init::VGGTInitializerTestAccess::get_first_camera_point(initializer, mismatched_measurements, camera_zero_point)) {
    std::cerr << "mismatched camera-0 timestamp and pixel arrays were accepted\n";
    return 1;
  }

  auto out_of_bounds = std::make_shared<ov_core::Feature>();
  Eigen::VectorXf out_of_bounds_uv(2);
  out_of_bounds_uv << 10.0f, 0.0f;
  out_of_bounds->timestamps[0] = {5.0};
  out_of_bounds->uvs[0] = {out_of_bounds_uv};
  if (ov_init::VGGTInitializerTestAccess::get_first_camera_point(initializer, out_of_bounds, camera_zero_point)) {
    std::cerr << "out-of-bounds camera-0 pixel coordinate was accepted\n";
    return 1;
  }

  return 0;
}
