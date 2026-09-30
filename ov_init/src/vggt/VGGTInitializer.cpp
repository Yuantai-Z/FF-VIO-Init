/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "VGGTInitializer.h"

#include "ceres/Factor_GenericPrior.h"
#include "ceres/Factor_ImageReprojCalib.h"
#include "ceres/Factor_ImageReprojScale.h"
#include "ceres/Factor_ImuCPIv1.h"
#include "ceres/Factor_PointCloudScale.h"
#include "ceres/Factor_ScalePrior.h"
#include "ceres/Factor_ScaleSmoothness.h"
#include "vggt/NonlinearMode.h"
#include "vggt/ScaleLayout.h"
#include "ceres/State_JPLQuatLocal.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"
#include "utils/helper.h"

#include <algorithm>
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <unordered_set>
#include <ros/ros.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float64MultiArray.h>
using namespace ov_core;
using namespace ov_type;
using namespace ov_init;

VGGTInitializer::VGGTInitializer(const InertialInitializerOptions &params_) : params(params_) {

  // Load confidence filtering parameters from config
  use_confidence_filter_ = params.init_vggt_use_confidence_filter;
  confidence_threshold_ = static_cast<float>(params.init_vggt_confidence_threshold);
  const auto nonlinear_method = vggt_nonlinear_method_from_flags(params.init_vggt_use_scale_optimization,
                                                                  params.init_vggt_feature_pc_constraint_weight);
  PRINT_INFO("[VGGT-INIT]: nonlinear method=%s, region_n=%d\n", vggt_nonlinear_method_name(nonlinear_method),
             params.init_vggt_scale_region_n);

  min_num_meas_to_optimize = static_cast<int>(params.init_window_time);
  min_valid_features = 8;
  size_pointcloud_scale = 1;
  size_velocity = 3;
  size_gravity = 3;
  system_size = size_pointcloud_scale + size_velocity + size_gravity;
  if (ros::isInitialized()) {
    ros_node_ = std::make_shared<ros::NodeHandle>();
    vggt_request_pub_ = ros_node_->advertise<std_msgs::Float64MultiArray>(vggt_request_topic_, 1, true);
    vggt_pointcloud_sub_ = ros_node_->subscribe(vggt_pointcloud_topic_, 10, &VGGTInitializer::vggtPointcloudCallback, this);
    ros_topics_ready_ = true;
  } else {
    ros_topics_ready_ = false;
    PRINT_WARNING(YELLOW "[VGGT-INIT]: ROS not initialized, VGGT data acquisition disabled!\n" RESET);
  }
}

bool VGGTInitializer::initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                                std::shared_ptr<ov_type::IMU> &_imu,
                                const std::shared_ptr<ov_core::FeatureDatabase> &feature_db,
                                const std::shared_ptr<std::vector<ov_core::ImuData>> &imu_data,
                                std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
                                std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM,
                                InitializationDiagnostics &diagnostics) {
  const auto method = vggt_nonlinear_method_from_flags(params.init_vggt_use_scale_optimization,
                                                        params.init_vggt_feature_pc_constraint_weight);
  PRINT_DEBUG("[VGGT-INIT] attempt (method=%s, region_n=%d)\n", vggt_nonlinear_method_token(method),
              params.init_vggt_scale_region_n);

  return initialize_pointcloud(timestamp, covariance, order, _imu, feature_db, imu_data, _clones_IMU, _features_SLAM,
                               diagnostics);
}

bool VGGTInitializer::initialize_pointcloud(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                                            std::shared_ptr<ov_type::IMU> &_imu,
                                            const std::shared_ptr<ov_core::FeatureDatabase> &feature_db,
                                            const std::shared_ptr<std::vector<ov_core::ImuData>> &imu_data,
                                            std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
                                            std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM,
                                            InitializationDiagnostics &diagnostics) {
  const VggtNonlinearMethod nonlinear_method =
      vggt_nonlinear_method_from_flags(params.init_vggt_use_scale_optimization,
                                       params.init_vggt_feature_pc_constraint_weight);
  const bool use_ff_graph = nonlinear_method == VggtNonlinearMethod::FF;
  features_point3d_c0_.clear();
  // Get the newest and oldest timestamps we will try to initialize between!
  auto rT1 = boost::posix_time::microsec_clock::local_time();
  double newest_cam_time = -1;
  for (auto const &feat : feature_db->get_internal_data()) {
    for (auto const &camtimepair : feat.second->timestamps) {
      for (auto const &time : camtimepair.second) {
        newest_cam_time = std::max(newest_cam_time, time);
      }
    }
  }
  double oldest_time = newest_cam_time - params.init_window_time;
  if (newest_cam_time < 0 || oldest_time < 0) {
    return false;
  }

  // Remove all measurements that are older than our initialization window
  // Then we will try to use all features that are in the feature database!
  feature_db->cleanup_measurements(oldest_time);
  bool have_old_imu_readings = false;
  auto it_imu = imu_data->begin();
  while (it_imu != imu_data->end() && it_imu->timestamp < oldest_time + params.calib_camimu_dt) {
    have_old_imu_readings = true;
    it_imu = imu_data->erase(it_imu);
  }
  // FF has no landmark state; SC and FEATURE_ONLY require tracked landmarks.
  if (!use_ff_graph) {
    if (feature_db->get_internal_data().size() < 0.75 * params.init_max_features) {
      PRINT_WARNING(RED "[VGGT-INIT]: only %zu valid features of required (%.0f thresh)!!\n" RESET,
                    feature_db->get_internal_data().size(),
                    0.95 * params.init_max_features);
      return false;
    }
  }
  if (imu_data->size() < 2 || !have_old_imu_readings) {
    return false;
  }
  if (!std::isfinite(diagnostics.first_attempt_oldest_time)) {
    diagnostics.first_attempt_oldest_time = oldest_time;
  }

  // Now we will make a copy of our features here
  // We do this to ensure that the feature database can continue to have new
  // measurements appended to it in an async-manor so this initialization
  // can be performed in a secondary thread while feature tracking is still performed.
  std::unordered_map<size_t, std::shared_ptr<Feature>> features;
  for (const auto &feat : feature_db->get_internal_data()) {
    auto feat_new = std::make_shared<Feature>();
    feat_new->featid = feat.second->featid;
    feat_new->uvs = feat.second->uvs;
    feat_new->uvs_norm = feat.second->uvs_norm;
    feat_new->timestamps = feat.second->timestamps;
    features.insert({feat.first, feat_new});
  }

  // ======================================================
  // ======================================================



  bool have_stereo = false;
  std::map<size_t, int> map_features_num_meas;      // feature_id -> number of measurements for each feature
  int measurements_size = 0;
  double oldest_camera_time = INFINITY;
  std::map<double, bool> map_camera_times;          // timestamp -> whether this camera frame is used for initialization
  std::map<size_t, bool> map_camera_ids;

  if (use_ff_graph) {
    if (!select_features_and_keyframes(features, newest_cam_time, have_stereo,
                                     map_features_num_meas, measurements_size, oldest_camera_time,
                                     map_camera_times, map_camera_ids)) {
      return false;
    }
  } else {
    if (!select_features_and_keyframes_firstanchored(features, newest_cam_time, have_stereo,
                                     map_features_num_meas, measurements_size, oldest_camera_time,
                                     map_camera_times, map_camera_ids)) {
      return false;
    }
  }
  // IMU preintegration
  // map_camera_cpi_I0toIi: Ii w.r.t I0, used in linear system
  // map_camera_cpi_IitoIi1: Ii+1 w.r.t Ii, used i·n MLE optimization
  std::map<double, std::shared_ptr<ov_core::CpiV1>> map_camera_cpi_I0toIi, map_camera_cpi_IitoIi1;
  if (!compute_camera_imu_preintegration(map_camera_times,
                                          /*oldest*/ oldest_camera_time,
                                          params.init_dyn_bias_g,
                                          params.init_dyn_bias_a,
                                          imu_data,
                                          map_camera_cpi_I0toIi,
                                          map_camera_cpi_IitoIi1)) {
    return false;
  }

  if (!load_vggt_data(map_camera_times, oldest_camera_time)) {
    return false;
  }

  // Always invalidate malformed geometry. Apply the configured threshold only
  // to frames whose PointCloud2 message actually supplied a confidence field.
  const auto filter_result = filter_loaded_pointclouds();
  PRINT_INFO(CYAN "[VGGT-INIT]: Invalidated %d pointcloud samples; confidence threshold %.2f applied to %d/%zu frames\n" RESET,
             filter_result.first, confidence_threshold_, filter_result.second, vggt_point3d_data.size());

  auto rT2 = boost::posix_time::microsec_clock::local_time();

  // ======================================================
  // ======================================================

  // Make sure we have enough measurements to fully constrain the system
  if (measurements_size < system_size) {
    PRINT_WARNING(YELLOW "[VGGT-INIT]: not enough feature measurements (%d meas vs %d state size)!\n" RESET, measurements_size, system_size);
    return false;
  }


  // Loop through each feature observation and append it!
  // State ordering is: [point-cloud scale, velocity, gravity]
  Eigen::VectorXd x_hat = Eigen::VectorXd::Zero(system_size);
  Eigen::Vector3d v_I0inI0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d gravity_inI0 = Eigen::Vector3d::Zero();

  // Sampled points for scale-based optimization
  std::map<double, std::vector<Eigen::Vector2d>> sampled_pixels_per_time;
  std::map<double, std::vector<Eigen::Vector3d>> sampled_points_per_time;
  std::map<double, std::vector<float>> sampled_confidences_per_time;

  // Use member variables to store first frame 3D information for each feature
  if (!build_and_solve_linear_system_pointcloud(features,
                                      map_features_num_meas,
                                      map_camera_times,
                                      map_camera_cpi_I0toIi,
                                      oldest_camera_time,
                                      measurements_size,
                                      x_hat, v_I0inI0, gravity_inI0,
                                      sampled_pixels_per_time, sampled_points_per_time, sampled_confidences_per_time)) {
    return false;
  }

  auto rT3 = boost::posix_time::microsec_clock::local_time();

  auto rT4 = boost::posix_time::microsec_clock::local_time();

  // ======================================================
  // ======================================================

  // Recover state in global frame
  std::map<double, Eigen::VectorXd> ori_GtoIi, pos_IiinG, vel_IiinG;
  std::map<size_t, Eigen::Vector3d> features_inG;
  if (!recover_states_in_global(features, map_features_num_meas, map_camera_times,
                                map_camera_cpi_I0toIi, x_hat, v_I0inI0, gravity_inI0,
                                ori_GtoIi, pos_IiinG, vel_IiinG, features_inG)) {
    return false;
  }

  // ======================================================
  // FF optimizes direct point-cloud reprojection; SC/FEATURE_ONLY use the landmark graph.
  // ======================================================

  if (use_ff_graph) {
    PRINT_INFO(CYAN "[VGGT-INIT]: Running scale-based nonlinear optimization...\n" RESET);
    if (!run_nonlinear_optimization_with_scale(map_camera_times,
                                    map_camera_cpi_IitoIi1,
                                    ori_GtoIi, pos_IiinG, vel_IiinG,
                                    params.init_dyn_bias_g, params.init_dyn_bias_a,
                                    sampled_pixels_per_time, sampled_points_per_time, sampled_confidences_per_time,
                                    map_camera_ids,
                                    /*newest*/ newest_cam_time,
                                    _imu, _clones_IMU,
                                    timestamp, covariance, order,
                                    rT1, rT2, rT3, rT4, diagnostics)) {
      return false;
    }
  } else {
    PRINT_INFO(CYAN "[VGGT-INIT]: Running feature-based nonlinear optimization...\n" RESET);
    if (!run_nonlinear_optimization_with_feature(map_camera_times,
                                    map_camera_cpi_IitoIi1,
                                    ori_GtoIi, pos_IiinG, vel_IiinG,
                                    params.init_dyn_bias_g, params.init_dyn_bias_a,
                                    features, map_features_num_meas, features_inG,
                                    map_camera_ids,
                                    /*newest*/ newest_cam_time,
                                    /*oldest*/ oldest_camera_time,
                                    _imu, _clones_IMU, _features_SLAM,
                                    timestamp, covariance, order,
                                    rT1, rT2, rT3, rT4, diagnostics)) {
      return false;
    }
  }

  const auto linear = ori_GtoIi.rbegin();
  diagnostics.success_oldest_time = oldest_camera_time;
  diagnostics.init_window_time = params.init_window_time;
  diagnostics.linear_state = make_initialization_state(linear->first, linear->second, pos_IiinG.at(linear->first),
                                                        vel_IiinG.at(linear->first));
  diagnostics.has_timing = true;
  diagnostics.has_linear_state = true;
  return true;
}

bool VGGTInitializer::load_vggt_data(const std::map<double, bool> &map_camera_times, double oldest_camera_time) {
  if (!ros_topics_ready_) {
    PRINT_ERROR(RED "[VGGT-INIT]: ROS topics not ready, cannot load VGGT data\n" RESET);
    return false;
  }
  {
    std::vector<double> requested_times;
    vggt_point3d_data.clear();
    vggt_pointcloud_has_confidence_.clear();
    for (const auto &timepair : map_camera_times) {
      if (vggt_point3d_data.find(timepair.first) == vggt_point3d_data.end()) {
        requested_times.push_back(timepair.first);
      }
    }
    if (!requested_times.empty()) {
      std::unordered_set<double> requested_set(requested_times.begin(), requested_times.end());
      {
        std::lock_guard<std::mutex> lock(vggt_ros_mutex_);
        vggt_requested_timestamps_ = requested_set;
        vggt_received_pointclouds_.clear();
        vggt_received_pointcloud_has_confidence_.clear();
      }
      {
        ros::WallRate wait_rate(200);
        ros::WallTime publish_start = ros::WallTime::now();
        while (ros::ok() && vggt_request_pub_.getNumSubscribers() == 0 &&
          (ros::WallTime::now() - publish_start).toSec() < 0.5) {
          wait_rate.sleep();
        }
        std_msgs::Float64MultiArray request_msg;
        request_msg.data.assign(requested_times.begin(), requested_times.end());

        // Record inference start time
        ros::WallTime inference_start = ros::WallTime::now();

        vggt_request_pub_.publish(request_msg);

        const double wait_timeout_sec = params.init_vggt_wait_timeout;
        ros::WallTime wait_start = ros::WallTime::now();
        while (ros::ok()) {
          {
            std::lock_guard<std::mutex> lock(vggt_ros_mutex_);
            const bool pointcloud_ready = !params.init_vggt_use_pointcloud ||
                                          vggt_received_pointclouds_.size() >= requested_set.size();
            if (pointcloud_ready) {
              break;
            }
          }
          if ((ros::WallTime::now() - wait_start).toSec() > wait_timeout_sec) {
            break;
          }
          ros::spinOnce();
          wait_rate.sleep();
        }

        // Record inference time (from request to response)
        vggt_inference_time_ = (ros::WallTime::now() - inference_start).toSec();
        PRINT_INFO(CYAN "[VGGT-INIT]: ROS round-trip (request->all data received): %.4f sec for %zu frames\n" RESET,
                   vggt_inference_time_, requested_set.size());
      }
      ros::spinOnce();

      std::map<double, cv::Mat> received_pt_copy;
      std::map<double, bool> received_pt_confidence_copy;
      {
        std::lock_guard<std::mutex> lock(vggt_ros_mutex_);
        received_pt_copy = vggt_received_pointclouds_;
        received_pt_confidence_copy = vggt_received_pointcloud_has_confidence_;
        vggt_requested_timestamps_.clear();
        vggt_received_pointclouds_.clear();
        vggt_received_pointcloud_has_confidence_.clear();
      }

      for (const auto &entry : received_pt_copy) {
        vggt_point3d_data[entry.first] = entry.second.clone();
        const auto confidence_it = received_pt_confidence_copy.find(entry.first);
        vggt_pointcloud_has_confidence_[entry.first] =
            confidence_it != received_pt_confidence_copy.end() && confidence_it->second;
      }
      if (params.init_vggt_use_pointcloud && received_pt_copy.size() != requested_set.size()) {
        PRINT_WARNING(YELLOW "[VGGT-INIT]: received %zu/%zu pointcloud frames via ROS topics\n" RESET,
                      received_pt_copy.size(), requested_set.size());
      }
    }
  }

  // Validate data availability based on enabled features
  if (params.init_vggt_use_pointcloud && vggt_point3d_data.empty()) {
    PRINT_ERROR(RED "[VGGT-INIT]: no VGGT pointcloud data available\n" RESET);
    return false;
  }
  // Validate point-cloud timestamp matching with tolerance when point-cloud input is enabled.
  if (params.init_vggt_use_pointcloud) {
    for (const auto &timepair : map_camera_times) {
      auto it = vggt_point3d_data.lower_bound(timepair.first);
      bool matched = false;

      if (it != vggt_point3d_data.end() && std::abs(it->first - timepair.first) <= 1e-3) {
        matched = true;
      } else if (it != vggt_point3d_data.begin()) {
        auto prev_it = std::prev(it);
        if (std::abs(prev_it->first - timepair.first) <= 1e-3) {
          matched = true;
        }
      }
      if (!matched) {
        double candidate_time = (it == vggt_point3d_data.end())
                                    ? vggt_point3d_data.rbegin()->first
                                    : it->first;
        PRINT_ERROR(RED "[VGGT-INIT]: VGGT pointcloud timestamp mismatch, camera=%.6f, closest_vggt=%.6f\n" RESET,
                    timepair.first, candidate_time);
        return false;
      }
    }

    auto vggt_oldest_it = vggt_point3d_data.lower_bound(oldest_camera_time);
    if (vggt_oldest_it == vggt_point3d_data.end() ||
        std::abs(vggt_oldest_it->first - oldest_camera_time) > 1e-2) {
      double reported_time = (vggt_oldest_it == vggt_point3d_data.end())
                                 ? vggt_point3d_data.rbegin()->first
                                 : vggt_oldest_it->first;
      PRINT_ERROR(RED "[VGGT-INIT]: oldest_camera_time = %.6f, closest_vggt_pointcloud_time = %.6f\n" RESET,
                  oldest_camera_time, reported_time);
      return false;
    }
  }
  return true;
}

void VGGTInitializer::vggtPointcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg) {
  if (!msg) {
    return;
  }
  double timestamp_ros = msg->header.stamp.toSec();

  // Snap to matching requested timestamp within 1us tolerance (defends against
  // ~1ns drift from Python rospy.Time.from_sec truncation vs C++ round). After
  // snapping we store under the ORIGINAL requested timestamp so downstream
  // .find/.at keyed by map_camera_times work.
  bool accept_message = false;
  double matched_ts = timestamp_ros;
  {
    std::lock_guard<std::mutex> lock(vggt_ros_mutex_);
    if (vggt_requested_timestamps_.empty()) {
      accept_message = true;
    } else {
      for (double rt : vggt_requested_timestamps_) {
        if (std::abs(rt - timestamp_ros) < 1e-6) {
          accept_message = true;
          matched_ts = rt;
          break;
        }
      }
    }
  }
  if (!accept_message || msg->height == 0 || msg->width == 0)
    return;

  // Organized point-cloud pixels must be in the exact processed cam0 image domain used by the tracker and optimizer.
  constexpr size_t model_cam_id = 0;
  const auto plan_it = params.vggt_preprocess_plans.find(model_cam_id);
  if (plan_it == params.vggt_preprocess_plans.end()) {
    PRINT_WARNING(YELLOW "[VGGT-INIT] PointcloudCallback: processed cam0 has no preprocessing plan\n" RESET);
    return;
  }
  const cv::Size expected_size = plan_it->second.output_size();
  if (msg->width != static_cast<uint32_t>(expected_size.width) || msg->height != static_cast<uint32_t>(expected_size.height)) {
    PRINT_WARNING(YELLOW "[VGGT-INIT] PointcloudCallback: pointcloud domain %ux%u does not match processed cam0 %dx%d\n" RESET,
                  msg->width, msg->height, expected_size.width, expected_size.height);
    return;
  }

  const auto find_field = [&msg](const std::string &name) -> const sensor_msgs::PointField * {
    for (const auto &field : msg->fields) {
      if (field.name == name) {
        return &field;
      }
    }
    return nullptr;
  };
  const auto valid_float_field = [&msg](const sensor_msgs::PointField *field, uint32_t expected_offset) {
    return field != nullptr && field->datatype == sensor_msgs::PointField::FLOAT32 &&
           field->count == 1 && field->offset == expected_offset &&
           static_cast<uint64_t>(field->offset) + sizeof(float) <= msg->point_step;
  };

  const sensor_msgs::PointField *field_x = find_field("x");
  const sensor_msgs::PointField *field_y = find_field("y");
  const sensor_msgs::PointField *field_z = find_field("z");
  const sensor_msgs::PointField *field_confidence = find_field("confidence");
  const bool has_confidence = field_confidence != nullptr;

  constexpr uint32_t minimum_point_step = 3 * sizeof(float);
  const uint64_t packed_row_step = static_cast<uint64_t>(msg->width) * msg->point_step;
  const uint64_t required_data_size = static_cast<uint64_t>(msg->row_step) * msg->height;
  if (msg->is_bigendian || msg->point_step < minimum_point_step || msg->point_step % sizeof(float) != 0 ||
      packed_row_step != msg->row_step ||
      required_data_size != msg->data.size()) {
    PRINT_WARNING(YELLOW "[VGGT-INIT] PointcloudCallback: unsupported PointCloud2 byte layout\n" RESET);
    return;
  }
  if (!valid_float_field(field_x, 0) || !valid_float_field(field_y, sizeof(float)) ||
      !valid_float_field(field_z, 2 * sizeof(float)) ||
      (has_confidence && !valid_float_field(field_confidence, 3 * sizeof(float)))) {
    PRINT_WARNING(YELLOW "[VGGT-INIT] PointcloudCallback: invalid PointCloud2 field schema\n" RESET);
    return;
  }

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

  // Create CV_32FC4 organized point cloud: [x, y, z, confidence]
  cv::Mat organized(msg->height, msg->width, CV_32FC4);
  if (has_confidence) {
    sensor_msgs::PointCloud2ConstIterator<float> iter_confidence(*msg, "confidence");
    for (uint32_t v = 0; v < msg->height; ++v) {
      for (uint32_t u = 0; u < msg->width; ++u, ++iter_x, ++iter_y, ++iter_z, ++iter_confidence) {
        organized.at<cv::Vec4f>(v, u) = cv::Vec4f(*iter_x, *iter_y, *iter_z, *iter_confidence);
      }
    }
  } else {
    for (uint32_t v = 0; v < msg->height; ++v) {
      for (uint32_t u = 0; u < msg->width; ++u, ++iter_x, ++iter_y, ++iter_z) {
        organized.at<cv::Vec4f>(v, u) = cv::Vec4f(*iter_x, *iter_y, *iter_z, 1.0f);
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(vggt_ros_mutex_);
    vggt_received_pointclouds_[matched_ts] = organized;
    vggt_received_pointcloud_has_confidence_[matched_ts] = has_confidence;
    PRINT_DEBUG("[VGGT-INIT] PointcloudCallback: received timestamp=%.6f (drift=%.2e), size=%dx%d, confidence=%d, buffer size=%zu\n",
                matched_ts, matched_ts - timestamp_ros, msg->width, msg->height, has_confidence,
                vggt_received_pointclouds_.size());
  }
}

// IMU integretion + preintegration
bool VGGTInitializer::compute_camera_imu_preintegration(
    const std::map<double, bool> &map_camera_times,
    double oldest_camera_time,
    const Eigen::Vector3d &gyroscope_bias,
    const Eigen::Vector3d &accelerometer_bias,
    const std::shared_ptr<std::vector<ov_core::ImuData>> &imu_data,
    std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_I0toIi,
    std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_IitoIi1) {
  // map_camera_cpi_I0toIi: Ii w.r.t I0, used in linear system
  // map_camera_cpi_IitoIi1: Ii+1 w.r.t Ii, used in MLE optimization
  assert(!map_camera_times.empty());

  // Check that we have some angular velocity / orientation change
  double newest_cam_time = map_camera_times.rbegin()->first;
  double accel_inI_norm = 0.0;
  double theta_inI_norm = 0.0;
  double time0_in_imu = oldest_camera_time + params.calib_camimu_dt;
  double time1_in_imu = newest_cam_time + params.calib_camimu_dt;
  std::vector<ov_core::ImuData> readings = InitializerHelper::select_imu_readings(*imu_data, time0_in_imu, time1_in_imu);
  assert(readings.size() > 2);
  for (size_t k = 0; k < readings.size() - 1; k++) {
    auto imu0 = readings.at(k);
    auto imu1 = readings.at(k + 1);
    double dt = imu1.timestamp - imu0.timestamp;
    Eigen::Vector3d wm = 0.5 * (imu0.wm + imu1.wm) - gyroscope_bias;
    Eigen::Vector3d am = 0.5 * (imu0.am + imu1.am) - accelerometer_bias;
    theta_inI_norm += (-wm * dt).norm();
    accel_inI_norm += am.norm();
  }
  accel_inI_norm /= (double)(readings.size() - 1);
  if (180.0 / M_PI * theta_inI_norm < params.init_dyn_min_deg) {
    PRINT_WARNING(YELLOW "[VGGT-INIT]: gyroscope only %.2f degree change (%.2f thresh)\n" RESET, 180.0 / M_PI * theta_inI_norm,
                  params.init_dyn_min_deg);
    return false;
  }
  PRINT_DEBUG("[VGGT-INIT]: |theta_I| = %.4f deg and |accel| = %.4f\n", 180.0 / M_PI * theta_inI_norm, accel_inI_norm);


  double last_camera_timestamp = 0.0;

  for (auto const &timepair : map_camera_times) {

    double current_time = timepair.first;
    // No preintegration at the first timestamp
    if (current_time == oldest_camera_time) {
      map_camera_cpi_I0toIi.insert({current_time, nullptr});
      map_camera_cpi_IitoIi1.insert({current_time, nullptr});
      last_camera_timestamp = current_time;
      continue;
    }

    // I0 -> Ii imu mechanism for linear system
    double cpiI0toIi1_time0_in_imu = oldest_camera_time + params.calib_camimu_dt;
    double cpiI0toIi1_time1_in_imu = current_time      + params.calib_camimu_dt;
    auto cpiI0toIi1 = std::make_shared<ov_core::CpiV1>(params.sigma_w, params.sigma_wb,
                                                       params.sigma_a, params.sigma_ab, true);
    cpiI0toIi1->setLinearizationPoints(gyroscope_bias, accelerometer_bias);
    std::vector<ov_core::ImuData> cpiI0toIi1_readings =
        InitializerHelper::select_imu_readings(*imu_data, cpiI0toIi1_time0_in_imu, cpiI0toIi1_time1_in_imu);
    if (cpiI0toIi1_readings.size() < 2) {
      PRINT_DEBUG(YELLOW "[VGGT-INIT]: camera %.2f in has %zu IMU readings!\n" RESET,
                  (cpiI0toIi1_time1_in_imu - cpiI0toIi1_time0_in_imu), cpiI0toIi1_readings.size());
      return false;
    }
    double cpiI0toIi1_dt_imu = cpiI0toIi1_readings.at(cpiI0toIi1_readings.size() - 1).timestamp - cpiI0toIi1_readings.at(0).timestamp;
    if (std::abs(cpiI0toIi1_dt_imu - (cpiI0toIi1_time1_in_imu - cpiI0toIi1_time0_in_imu)) > 0.01) {
      PRINT_DEBUG(YELLOW "[VGGT-INIT]: camera IMU was only propagated %.3f of %.3f\n" RESET,
                  cpiI0toIi1_dt_imu, (cpiI0toIi1_time1_in_imu - cpiI0toIi1_time0_in_imu));
      return false;
    }
    for (size_t k = 0; k < cpiI0toIi1_readings.size() - 1; k++) {
      auto imu0 = cpiI0toIi1_readings.at(k);
      auto imu1 = cpiI0toIi1_readings.at(k + 1);
      cpiI0toIi1->feed_IMU(imu0.timestamp, imu1.timestamp, imu0.wm, imu0.am, imu1.wm, imu1.am);
    }

    // Ii -> Ii+1 preintegration for nonlinear system
    double cpiIitoIi1_time0_in_imu = last_camera_timestamp + params.calib_camimu_dt;
    double cpiIitoIi1_time1_in_imu = current_time         + params.calib_camimu_dt;
    auto cpiIitoIi1 = std::make_shared<ov_core::CpiV1>(params.sigma_w, params.sigma_wb,
                                                       params.sigma_a, params.sigma_ab, true);
    cpiIitoIi1->setLinearizationPoints(gyroscope_bias, accelerometer_bias);
    std::vector<ov_core::ImuData> cpiIitoIi1_readings =
        InitializerHelper::select_imu_readings(*imu_data, cpiIitoIi1_time0_in_imu, cpiIitoIi1_time1_in_imu);
    if (cpiIitoIi1_readings.size() < 2) {
      PRINT_DEBUG(YELLOW "[VGGT-INIT]: camera %.2f in has %zu IMU readings!\n" RESET,
                  (cpiIitoIi1_time1_in_imu - cpiIitoIi1_time0_in_imu), cpiIitoIi1_readings.size());
      return false;
    }
    double cpiIitoIi1_dt_imu = cpiIitoIi1_readings.at(cpiIitoIi1_readings.size() - 1).timestamp - cpiIitoIi1_readings.at(0).timestamp;

    if (std::abs(cpiIitoIi1_dt_imu - (cpiIitoIi1_time1_in_imu - cpiIitoIi1_time0_in_imu)) > 0.01) {
      PRINT_DEBUG(YELLOW "[VGGT-INIT]: camera IMU was only propagated %.3f of %.3f\n" RESET,
                  cpiIitoIi1_dt_imu, (cpiIitoIi1_time1_in_imu - cpiIitoIi1_time0_in_imu));
      return false;
    }
    for (size_t k = 0; k < cpiIitoIi1_readings.size() - 1; k++) {
      auto imu0 = cpiIitoIi1_readings.at(k);
      auto imu1 = cpiIitoIi1_readings.at(k + 1);
      cpiIitoIi1->feed_IMU(imu0.timestamp, imu1.timestamp, imu0.wm, imu0.am, imu1.wm, imu1.am);
    }

    // store
    map_camera_cpi_I0toIi.insert({current_time, cpiI0toIi1});
    map_camera_cpi_IitoIi1.insert({current_time, cpiIitoIi1});
    last_camera_timestamp = current_time;
  }

  return true;
}

bool VGGTInitializer::recover_states_in_global(
    const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
    const std::map<size_t, int> &map_features_num_meas,
    const std::map<double, bool> &map_camera_times,
    const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_I0toIi,
    const Eigen::VectorXd &x_hat,
    const Eigen::Vector3d &v_I0inI0,
    const Eigen::Vector3d &gravity_inI0,
    std::map<double, Eigen::VectorXd> &ori_GtoIi,
    std::map<double, Eigen::VectorXd> &pos_IiinG,
    std::map<double, Eigen::VectorXd> &vel_IiinG,
    std::map<size_t, Eigen::Vector3d> &features_inG) {

  // ======================================================
  // ======================================================

  // 1. Extract imu state elements, recover state in the first IMU frame
  std::map<double, Eigen::VectorXd> ori_I0toIi, pos_IiinI0, vel_IiinI0;
  for (auto const &timepair : map_camera_times) {

    // Timestamp of this pose
    double time = timepair.first;

    // Get our CPI integration values
    double DT = 0.0;
    Eigen::MatrixXd R_I0toIk = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd alpha_I0toIk = Eigen::MatrixXd::Zero(3, 1);
    Eigen::MatrixXd beta_I0toIk = Eigen::MatrixXd::Zero(3, 1);
    if (map_camera_cpi_I0toIi.find(time) != map_camera_cpi_I0toIi.end() && map_camera_cpi_I0toIi.at(time) != nullptr) {
      auto cpi = map_camera_cpi_I0toIi.at(time);
      DT = cpi->DT;
      R_I0toIk = cpi->R_k2tau;
      alpha_I0toIk = cpi->alpha_tau;
      beta_I0toIk = cpi->beta_tau;
    }

    // Integrate to get the relative to the current timestamp
    Eigen::Vector3d p_IkinI0 = v_I0inI0 * DT - 0.5 * gravity_inI0 * DT * DT + alpha_I0toIk;
    Eigen::Vector3d v_IkinI0 = v_I0inI0 - gravity_inI0 * DT + beta_I0toIk;

    // Record the values all transformed to the I0 frame
    ori_I0toIi.insert({time, rot_2_quat(R_I0toIk)});
    pos_IiinI0.insert({time, p_IkinI0});
    vel_IiinI0.insert({time, v_IkinI0});
  }

  // 2. Recover the features in the first IMU frame
  count_valid_features = 0;
  std::map<size_t, Eigen::Vector3d> features_inI0;
  if (!params.init_vggt_use_scale_optimization) {
    double scale_pointcloud = x_hat(0) > 0.5 ? x_hat(0) : 3.0;
    for (auto const &feat : features) {
      // Safety check: ensure feature exists in measurement map
      // This can happen if feature was not visible in first frame or was filtered out earlier
      if (map_features_num_meas.find(feat.first) == map_features_num_meas.end()) {
        continue;
      }
      if (map_features_num_meas.at(feat.first) < min_num_meas_to_optimize)
        continue;
      // VGGT pointclouds and their image coordinates are defined for camera 0.
      constexpr size_t cam_id = 0;
      Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(cam_id).block(0, 0, 4, 1);
      Eigen::Vector3d p_IinC = params.camera_extrinsics.at(cam_id).block(4, 0, 3, 1);
      Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);

      Eigen::Vector3d point3d_c0 = Eigen::Vector3d::Zero();
      if (!get_pFinC0(feat.second, point3d_c0)) {
        continue;
      }
      Eigen::Vector3d p_FinC0 = scale_pointcloud * point3d_c0;
      Eigen::Vector3d p_FinI0 = R_ItoC.transpose() * p_FinC0 - R_ItoC.transpose() * p_IinC;

      // check if in front of camera
      if (p_FinC0.allFinite() && p_FinI0.allFinite() && p_FinC0(2) > 0) {
        // This cache belongs to the current attempt and must overwrite any
        // value previously associated with the feature ID.
        features_point3d_c0_[feat.first] = point3d_c0;
        features_inI0.insert({feat.first, p_FinI0});
        count_valid_features++;
      }
    }
    if (count_valid_features < min_valid_features) {
      PRINT_ERROR(YELLOW "[VGGT-INIT]: not enough features for our mle (%d < %d)!\n" RESET, count_valid_features, min_valid_features);
      return false;
    }
  }
  // 3. Convert our states to be a gravity aligned global frame of reference
  // Here we say that the I0 frame is at 0,0,0 and shared the global origin
  InitializerHelper::gram_schmidt(gravity_inI0, R_GtoI0);
  Eigen::Vector4d q_GtoI0 = rot_2_quat(R_GtoI0);
  for (auto const &timepair : map_camera_times) {
    ori_GtoIi[timepair.first] = quat_multiply(ori_I0toIi.at(timepair.first), q_GtoI0);
    pos_IiinG[timepair.first] = R_GtoI0.transpose() * pos_IiinI0.at(timepair.first);
    vel_IiinG[timepair.first] = R_GtoI0.transpose() * vel_IiinI0.at(timepair.first);
  }
  for (auto const &feat : features_inI0) {
    features_inG[feat.first] = R_GtoI0.transpose() * feat.second;
  }
  return true;
}
// Linear System Solving
bool VGGTInitializer::build_and_solve_linear_system_pointcloud(
    const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
    const std::map<size_t, int> &map_features_num_meas,
    const std::map<double, bool> &map_camera_times,
    const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_I0toIi,
    double oldest_camera_time,
    int measurements_size,
    Eigen::VectorXd &x_hat,
    Eigen::Vector3d &v_I0inI0,
    Eigen::Vector3d &gravity_inI0,
    std::map<double, std::vector<Eigen::Vector2d>> &sampled_pixels_per_time,
    std::map<double, std::vector<Eigen::Vector3d>> &sampled_points_per_time,
    std::map<double, std::vector<float>> &sampled_confidences_per_time) {

  // Only use monocular
  size_t cam_id = features.begin()->second->timestamps.begin()->first;
  // size_t cam_id = 0;
  Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(cam_id).block(0, 0, 4, 1);
  Eigen::Vector3d p_IinC = params.camera_extrinsics.at(cam_id).block(4, 0, 3, 1);
  Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);

  int total_valid_points = 0;

  for (const auto &timepair : map_camera_times) {

    if (!timepair.second)
      continue;
    double time = timepair.first;
    // Skip the first timestamp (oldest_camera_time)
    if (time == oldest_camera_time) continue;
    std::vector<Eigen::Vector2d> frame_pixels;
    std::vector<Eigen::Vector3d> frame_points;
    std::vector<float> frame_confidences;
    if (!sample_pointcloud(time, frame_pixels, frame_points, frame_confidences)) {
      PRINT_WARNING(YELLOW "[VGGT-INIT]: failed to sample VGGT pointcloud at time %.6f\n" RESET, time);
      continue;
    }

    sampled_pixels_per_time[time] = frame_pixels;
    sampled_points_per_time[time] = frame_points;
    sampled_confidences_per_time[time] = frame_confidences;
    total_valid_points += frame_pixels.size();

    if (frame_points.size() > 0) {
      double avg_z = 0.0;
      for (const auto &pt : frame_points) {
        avg_z += pt(2);
      }
      avg_z /= frame_points.size();
      PRINT_DEBUG("[VGGT-INIT]: Frame %.6f - Sampled %zu points, avg z=%.2fm\n", time, frame_points.size(), avg_z);
    }
  }

  PRINT_INFO("[VGGT-INIT]: Total valid points = %d across %zu frames\n", total_valid_points, sampled_pixels_per_time.size());

  // Reject if total valid points is less than min(expected/2, 50)
  if (params.init_vggt_use_pointcloud) {
    int expected_total_points = params.init_vggt_points_per_frame * params.init_dyn_num_pose;
    int min_required_points = std::min(expected_total_points / 2, 50);
    if (total_valid_points < min_required_points) {
      PRINT_WARNING(YELLOW "[VGGT-INIT]: Not enough valid points: %d < %d (min of %d/2 and 50)\n" RESET,
                    total_valid_points, min_required_points, expected_total_points);
      return false;
    }
  }

  // Step 2: Build complete A and b matrices first
  total_valid_points = params.init_vggt_use_pointcloud ? total_valid_points : 0;
  int actual_measurements_size = total_valid_points * 2;
  Eigen::MatrixXd A_full = Eigen::MatrixXd::Zero(actual_measurements_size, system_size);
  Eigen::VectorXd b_full = Eigen::VectorXd::Zero(actual_measurements_size);

  // Record row index ranges for each timestamp (for RANSAC)
  std::map<double, std::pair<int, int>> timestamp_row_ranges;  // timestamp -> (start_row, end_row) for point cloud constraints
  int index_meas = 0;

  if (params.init_vggt_use_pointcloud)
    for (const auto &timepair : sampled_pixels_per_time) {
      double time = timepair.first;
      const std::vector<Eigen::Vector2d> &frame_pixels = timepair.second;
      const std::vector<Eigen::Vector3d> &frame_points = sampled_points_per_time.at(time);
      const std::vector<float> &frame_confidences = sampled_confidences_per_time.at(time);

      // Build linear system for this frame's points
      Eigen::MatrixXd A_frame;
      Eigen::VectorXd b_frame;
      int num_meas = build_linear_system_from_points(time, frame_pixels, frame_points, frame_confidences, cam_id, R_ItoC, p_IinC, map_camera_cpi_I0toIi,
                                                     system_size, A_frame, b_frame);

      // Copy to global A and b
      if (num_meas > 0) {
        A_full.block(index_meas, 0, num_meas, A_full.cols()) = A_frame;
        b_full.segment(index_meas, num_meas) = b_frame;
        index_meas += num_meas;
        timestamp_row_ranges[time] = {index_meas - num_meas, index_meas};

      }
    }
  PRINT_DEBUG("[VGGT-INIT]: Built full A (%dx%d) and b (%d)\n",
              A_full.rows(), A_full.cols(), b_full.rows());

  // Choose between RANSAC and standard solving based on configuration
  // RANSAC is only used when we have point-cloud constraints.
  if (params.init_vggt_use_ransac && !timestamp_row_ranges.empty()) {
    // ========== RANSAC PATH ==========
    PRINT_INFO("[VGGT-INIT]: Using RANSAC for robust initialization\n");

    // Strategy: sample points from every timestamp
    int num_sample_timestamps = (int)timestamp_row_ranges.size();
    int points_per_timestamp = (params.init_vggt_ransac_min_sample_size > 0)
                                   ? params.init_vggt_ransac_min_sample_size
                                   : 3;

    if (num_sample_timestamps == 0) {
      PRINT_ERROR(RED "[VGGT-INIT]: Not enough timestamps for RANSAC (%d)\n" RESET,
                  num_sample_timestamps);
      return false;
    }

    // Collect all timestamps for sampling (from point cloud constraints)
    std::vector<double> all_timestamps;
    for (const auto &entry : timestamp_row_ranges) {
      all_timestamps.push_back(entry.first);
    }

    PRINT_DEBUG("[VGGT-INIT]: RANSAC with %d point-cloud timestamps, sampling up to %d points each timestamp\n",
                (int)all_timestamps.size(), points_per_timestamp);

    // RANSAC main loop
    int best_inlier_count = 0;
    std::vector<int> best_inlier_point_rows;
    Eigen::VectorXd best_x_hat;
    Eigen::Vector3d best_v_I0inI0, best_gravity_inI0;

    std::random_device rd;
    std::mt19937 gen(rd());
    for (int iter = 0; iter < params.init_vggt_ransac_max_iterations; iter++) {
      // Step 1: For each timestamp, randomly select points and collect their first rows.
      std::vector<int> sampled_point_rows;  // stores only the first row (even row) of each sampled point from point cloud

      for (double ts : all_timestamps) {
        // Sample point cloud constraints if available
        if (timestamp_row_ranges.count(ts)) {
          const auto &pixels = sampled_pixels_per_time.at(ts);
          const auto &row_range = timestamp_row_ranges.at(ts);
          int num_points = pixels.size();

          if (num_points > 0) {
            std::uniform_int_distribution<> point_dis(0, num_points - 1);
            std::set<int> sampled_point_indices;

            int points_to_sample = std::min(points_per_timestamp, num_points);
            while ((int)sampled_point_indices.size() < points_to_sample) {
              int point_idx = point_dis(gen);
              if (sampled_point_indices.find(point_idx) == sampled_point_indices.end()) {
                sampled_point_indices.insert(point_idx);
                // Store only the first row (base_row) for each point
                int base_row = row_range.first + point_idx * 2;
                sampled_point_rows.push_back(base_row);
              }
            }
          }
        }
      }

      if (sampled_point_rows.empty()) {
        continue; // No valid samples
      }

      // Step 2: Build and solve A_sample and b_sample by extracting both rows for each sampled point.
      int sample_measurements_size = sampled_point_rows.size() * 2;
      Eigen::MatrixXd A_sample = Eigen::MatrixXd::Zero(sample_measurements_size, system_size);
      Eigen::VectorXd b_sample = Eigen::VectorXd::Zero(sample_measurements_size);

      int sample_idx = 0;

      // Add point cloud constraints
      for (int i = 0; i < (int)sampled_point_rows.size(); i++) {
        int base_row = sampled_point_rows[i];  // first row of the point
        A_sample.row(sample_idx) = A_full.row(base_row);
        A_sample.row(sample_idx + 1) = A_full.row(base_row + 1);
        b_sample(sample_idx) = b_full(base_row);
        b_sample(sample_idx + 1) = b_full(base_row + 1);
        sample_idx += 2;
      }

      // Solve with constraint
      Eigen::VectorXd x_hat_temp = Eigen::VectorXd::Zero(system_size);;
      Eigen::Vector3d v_temp, g_temp;
      if (!LLSsolver_with_gravity_constraint(A_sample, b_sample, x_hat_temp, v_temp, g_temp)) {
        continue;  // Solve failed, try next iteration
      }

      // Check basic validity
      if (x_hat_temp(0) <= 0) {
        continue;  // Invalid scale
      }

      // Step 3: Compute inliers on all point-cloud constraints.
      std::vector<int> inlier_point_rows;  // stores only the first row (even row) of each inlier point from point cloud
      int total_points = 0;

      // Compute point cloud inliers
      for (double ts : all_timestamps) {
        if (!timestamp_row_ranges.count(ts)) continue;

        const auto &row_range = timestamp_row_ranges.at(ts);
        const auto &pixels = sampled_pixels_per_time.at(ts);
        const auto &confidences = sampled_confidences_per_time.at(ts);
        int num_points = pixels.size();
        total_points += num_points;

        for (int point_idx = 0; point_idx < num_points; point_idx++) {
          // Each point has 2 rows (base_row and base_row+1)
          int base_row = row_range.first + point_idx * 2;
          Eigen::VectorXd A_row_u = A_full.row(base_row);
          Eigen::VectorXd A_row_v = A_full.row(base_row + 1);
          double b_u = b_full(base_row);
          double b_v = b_full(base_row + 1);

          // Compute residual for this point
          double residual_u = A_row_u.dot(x_hat_temp) - b_u;
          double residual_v = A_row_v.dot(x_hat_temp) - b_v;
          if (point_idx < (int)confidences.size() && confidences[point_idx] > 0.0f) {
            double weight = std::sqrt(confidences[point_idx]);
            if (weight > 0.0) {
              residual_u /= weight;
              residual_v /= weight;
            }
          }
          double error = std::sqrt(residual_u * residual_u + residual_v * residual_v);

          if (error < params.init_vggt_ransac_inlier_threshold) {
            inlier_point_rows.push_back(base_row);  // only store the first row (even index)
          }
        }
      }

      int inlier_point_count = inlier_point_rows.size();
      // Update the best model according to point-cloud inliers.
      if (inlier_point_count > best_inlier_count) {
        best_inlier_count = inlier_point_count;
        best_inlier_point_rows = inlier_point_rows;
        best_x_hat = x_hat_temp;
        best_v_I0inI0 = v_temp;
        best_gravity_inI0 = g_temp;
      }

      // Early termination if inlier ratio is very high
      if ((double)best_inlier_count / total_points > 0.95) {
        PRINT_DEBUG("[VGGT-INIT]: RANSAC early termination at iter %d\n", iter);
        break;
      }
    }

    // Check if RANSAC succeeded
    int total_points_count = total_valid_points;
    double inlier_ratio = (double)best_inlier_count / total_points_count;
    PRINT_INFO("[VGGT-INIT]: RANSAC finished: %d/%d point inliers (%.2f%%)\n",
               best_inlier_count, total_points_count, inlier_ratio * 100.0);

    if (inlier_ratio < params.init_vggt_ransac_min_inlier_ratio) {
      PRINT_ERROR(RED "[VGGT-INIT]: RANSAC failed: inlier ratio %.2f < %.2f\n" RESET,
                  inlier_ratio, params.init_vggt_ransac_min_inlier_ratio);
      return false;
    }

    // Refine with all inliers (extract both rows for each inlier point).
    int inlier_measurements_size = best_inlier_point_rows.size() * 2;
    Eigen::MatrixXd A_inliers = Eigen::MatrixXd::Zero(inlier_measurements_size, system_size);
    Eigen::VectorXd b_inliers = Eigen::VectorXd::Zero(inlier_measurements_size);

    int inlier_idx = 0;

    // Add point cloud inliers
    for (int i = 0; i < (int)best_inlier_point_rows.size(); i++) {
      int base_row = best_inlier_point_rows[i];  // first row of the point
      A_inliers.row(inlier_idx) = A_full.row(base_row);
      A_inliers.row(inlier_idx + 1) = A_full.row(base_row + 1);
      b_inliers(inlier_idx) = b_full(base_row);
      b_inliers(inlier_idx + 1) = b_full(base_row + 1);
      inlier_idx += 2;
    }

    // Final solve with inliers
    if (!LLSsolver_with_gravity_constraint(A_inliers, b_inliers, x_hat, v_I0inI0, gravity_inI0)) {
      PRINT_ERROR(RED "[VGGT-INIT]: Final solve with inliers failed\n" RESET);
      return false;
    }

    // Remove outliers from sampled data (so nonlinear optimization only uses inliers)
    std::set<int> inlier_row_set(best_inlier_point_rows.begin(), best_inlier_point_rows.end());
    int total_removed = 0;

    for (auto &ts_entry : sampled_pixels_per_time) {
      double ts = ts_entry.first;
      if (!timestamp_row_ranges.count(ts)) continue;

      const auto &row_range = timestamp_row_ranges.at(ts);
      std::vector<Eigen::Vector2d> filtered_pixels;
      std::vector<Eigen::Vector3d> filtered_points;
      std::vector<float> filtered_confidences;

      auto &pixels = ts_entry.second;
      auto &points = sampled_points_per_time.at(ts);
      auto &confidences = sampled_confidences_per_time.at(ts);

      for (size_t i = 0; i < pixels.size(); i++) {
        int base_row = row_range.first + i * 2;
        if (inlier_row_set.count(base_row)) {
          filtered_pixels.push_back(pixels[i]);
          filtered_points.push_back(points[i]);
          filtered_confidences.push_back(confidences[i]);
        }
      }

      total_removed += pixels.size() - filtered_pixels.size();
      pixels = filtered_pixels;
      points = filtered_points;
      confidences = filtered_confidences;
    }

    PRINT_INFO("[VGGT-INIT]: Removed %d outlier points after RANSAC\n", total_removed);

  } else {
    // ========== STANDARD PATH (No RANSAC) ==========
    // A_full and b_full are already constructed above, directly solve with them
    PRINT_INFO("[VGGT-INIT]: Using standard initialization (no RANSAC)\n");

    // Solve with gravity constraint using the full matrices
    if (!LLSsolver_with_gravity_constraint(A_full, b_full, x_hat, v_I0inI0, gravity_inI0)) {
      PRINT_ERROR(RED "[VGGT-INIT]: LLSsolver_with_gravity_constraint failed\n" RESET);
      return false;
    }
  }
  // Posterior analysis of LLS
  Eigen::VectorXd residual = A_full * x_hat - b_full;
  double residual_norm = residual.norm();
  double residual_max = residual.cwiseAbs().maxCoeff();
  double residual_mean = residual.cwiseAbs().mean();
  double residual_std = std::sqrt((residual.array() - residual.mean()).square().sum() / residual.size());
  PRINT_INFO("[VGGT-INIT]: Post-solve residual | norm=%.6e | max=%.6e | mean=%.6e | std=%.6e\n",
              residual_norm, residual_max, residual_mean, residual_std);
  // Check linear residual threshold
  if (params.init_vggt_max_linear_avg_residual > 0 && residual_mean > params.init_vggt_max_linear_avg_residual) {
    PRINT_WARNING(YELLOW "[VGGT-INIT]: high linear residual: %.6e > threshold=%.2f\n" RESET,
                  residual_mean, params.init_vggt_max_linear_avg_residual);
    return false;
  }

  // Check convergence: the point-cloud scale should be positive.
  if (x_hat(0) <= 0) {
    PRINT_WARNING("[VGGT-INIT]: negative scale, scale = %.6f <= 0\n", x_hat(0));
    // return false;
  }
  // Check gravity magnitude to see if converged
  double init_max_grav_difference = 1e-3;
  if (std::abs(gravity_inI0.norm() - params.gravity_mag) > init_max_grav_difference) {
    PRINT_WARNING(YELLOW "[VGGT-INIT]: gravity did not converge (%.3f > %.3f)\n" RESET,
                  std::abs(gravity_inI0.norm() - params.gravity_mag), init_max_grav_difference);
    return false;
  }

  PRINT_INFO("[VGGT-INIT]: State estimation results:\n");
  PRINT_INFO("  - Point cloud scale: %.6f\n", x_hat(0));
  PRINT_INFO("[VGGT-INIT]: velocity in I0 was %.3f,%.3f,%.3f and |v| = %.4f\n",
             v_I0inI0(0), v_I0inI0(1), v_I0inI0(2), v_I0inI0.norm());

  PRINT_INFO("[VGGT-INIT]: gravity in I0 was %.3f,%.3f,%.3f and |g| = %.4f\n",
             gravity_inI0(0), gravity_inI0(1), gravity_inI0(2), gravity_inI0.norm());

  return true;
}


bool VGGTInitializer::run_nonlinear_optimization_with_feature(
    const std::map<double, bool> &map_camera_times,
    const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_IitoIi1,
    const std::map<double, Eigen::VectorXd> &ori_GtoIi,
    const std::map<double, Eigen::VectorXd> &pos_IiinG,
    const std::map<double, Eigen::VectorXd> &vel_IiinG,
    const Eigen::Vector3d &gyroscope_bias,
    const Eigen::Vector3d &accelerometer_bias,
    const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
    const std::map<size_t,int> &map_features_num_meas,
    const std::map<size_t, Eigen::Vector3d> &features_inG,
    const std::map<size_t, bool> &map_camera_ids,
    double newest_cam_time,
    double oldest_camera_time,
    std::shared_ptr<ov_type::IMU> &_imu,
    std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
    std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM,
    double &timestamp,
    Eigen::MatrixXd &covariance,
    std::vector<std::shared_ptr<ov_type::Type>> &order,
    const boost::posix_time::ptime &rT1,
    const boost::posix_time::ptime &rT2,
    const boost::posix_time::ptime &rT3,
    const boost::posix_time::ptime &rT4,
    InitializationDiagnostics &diagnostics) {

// Ceres problem stuff
  // NOTE: By default the problem takes ownership of the memory
  ceres::Problem problem;

  // Our system states (map from time to index)
  std::map<double, int> map_states;
  std::vector<double *> ceres_vars_ori;
  std::vector<double *> ceres_vars_pos;
  std::vector<double *> ceres_vars_vel;
  std::vector<double *> ceres_vars_bias_g;
  std::vector<double *> ceres_vars_bias_a;

  // Feature states (3dof p_FinG)
  std::map<size_t, int> map_features;
  std::vector<double *> ceres_vars_feat;

  // Setup extrinsic calibration q_ItoC, p_IinC (map from camera id to index)
  std::map<size_t, int> map_calib_cam2imu;
  std::vector<double *> ceres_vars_calib_cam2imu_ori;
  std::vector<double *> ceres_vars_calib_cam2imu_pos;

  // Setup intrinsic calibration focal, center, distortion (map from camera id to index)
  std::map<size_t, int> map_calib_cam;
  std::vector<double *> ceres_vars_calib_cam_intrinsics;

  // Scale states for point cloud constraint (global or region-based)
  double *ceres_var_global_scale = nullptr;
  std::vector<double *> ceres_vars_scale;
  int region_n = params.init_vggt_scale_region_n;
  const VggtScaleLayout scale_layout = make_vggt_scale_layout(region_n);

  // PCA variables for region mode
  Eigen::Vector3d pca_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d pca_axis1 = Eigen::Vector3d::UnitX();
  Eigen::Vector3d pca_axis2 = Eigen::Vector3d::UnitY();
  double pca_min1 = 0, pca_max1 = 1, pca_min2 = 0, pca_max2 = 1;

  // Helper lambda: compute region index from 3D point (in C0 frame) using PCA projection
  auto get_region_index = [&](const Eigen::Vector3d &point) -> int {
    if (region_n <= 0) return -1;
    Eigen::Vector3d diff = point - pca_centroid;
    double proj1 = diff.dot(pca_axis1);
    double proj2 = diff.dot(pca_axis2);
    double norm1 = std::max(0.0, std::min(0.999999, (proj1 - pca_min1) / (pca_max1 - pca_min1)));
    double norm2 = std::max(0.0, std::min(0.999999, (proj2 - pca_min2) / (pca_max2 - pca_min2)));
    return static_cast<int>(norm1 * region_n) + static_cast<int>(norm2 * region_n) * region_n;
  };

  Eigen::Vector3d gravity;
  gravity << 0.0, 0.0, params.gravity_mag;
  // Helper lambda that will free any memory we have allocated
  auto free_state_memory = [&]() {
    for (auto ptr : ceres_vars_ori) delete[] ptr;
    for (auto ptr : ceres_vars_pos) delete[] ptr;
    for (auto ptr : ceres_vars_vel) delete[] ptr;
    for (auto ptr : ceres_vars_bias_g) delete[] ptr;
    for (auto ptr : ceres_vars_bias_a) delete[] ptr;
    for (auto ptr : ceres_vars_feat) delete[] ptr;
    for (auto ptr : ceres_vars_calib_cam2imu_ori) delete[] ptr;
    for (auto ptr : ceres_vars_calib_cam2imu_pos) delete[] ptr;
    for (auto ptr : ceres_vars_calib_cam_intrinsics) delete[] ptr;
    for (auto ptr : ceres_vars_scale) delete[] ptr;
    if (ceres_var_global_scale != nullptr) delete[] ceres_var_global_scale;
  };

  // Set the optimization settings
  // NOTE: We use dense schur since after eliminating features we have a dense problem
  // NOTE: http://ceres-solver.org/solving_faqs.html#solving
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  // options.linear_solver_type = ceres::SPARSE_SCHUR;
  // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  // options.preconditioner_type = ceres::SCHUR_JACOBI;
  // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.num_threads = params.init_dyn_mle_max_threads;
  options.max_solver_time_in_seconds = params.init_dyn_mle_max_time;
  options.max_num_iterations = params.init_dyn_mle_max_iter;
  // options.minimizer_progress_to_stdout = true;
  // options.linear_solver_ordering = ordering;
  options.function_tolerance = 1e-5;
  options.gradient_tolerance = 1e-4 * options.function_tolerance;

  // Loop through each CPI integration and add its measurement to the problem
  double timestamp_k = -1;
  for (auto const &timepair : map_camera_times) {

    // Get our predicted state at the requested camera timestep
    double timestamp_k1 = timepair.first;
    std::shared_ptr<ov_core::CpiV1> cpi = map_camera_cpi_IitoIi1.at(timestamp_k1);
    Eigen::Matrix<double, 16, 1> state_k1;
    state_k1.block(0, 0, 4, 1) = ori_GtoIi.at(timestamp_k1);
    state_k1.block(4, 0, 3, 1) = pos_IiinG.at(timestamp_k1);
    state_k1.block(7, 0, 3, 1) = vel_IiinG.at(timestamp_k1);
    state_k1.block(10, 0, 3, 1) = gyroscope_bias;
    state_k1.block(13, 0, 3, 1) = accelerometer_bias;

    // ================================================================
    //  ADDING GRAPH STATE / ESTIMATES!
    // ================================================================

    // Load our state variables into our allocated state pointers
    auto *var_ori = new double[4];
    for (int j = 0; j < 4; j++) {
      var_ori[j] = state_k1(0 + j, 0);
    }
    auto *var_pos = new double[3];
    auto *var_vel = new double[3];
    auto *var_bias_g = new double[3];
    auto *var_bias_a = new double[3];
    for (int j = 0; j < 3; j++) {
      var_pos[j] = state_k1(4 + j, 0);
      var_vel[j] = state_k1(7 + j, 0);
      var_bias_g[j] = state_k1(10 + j, 0);
      var_bias_a[j] = state_k1(13 + j, 0);
    }

    // Now actually create the parameter block in the ceres problem
    auto ceres_jplquat = new State_JPLQuatLocal();
    problem.AddParameterBlock(var_ori, 4, ceres_jplquat);
    problem.AddParameterBlock(var_pos, 3);
    problem.AddParameterBlock(var_vel, 3);
    problem.AddParameterBlock(var_bias_g, 3);
    problem.AddParameterBlock(var_bias_a, 3);

    // Fix this first ever pose to constrain the problem
    // NOTE: If we don't do this, then the problem won't be full rank
    // NOTE: Since init is over a small window, we are likely to be degenerate
    // NOTE: Thus we need to fix these parameters
    if (map_states.empty()) {

      // Construct state and prior
      Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(13, 1);
      for (int j = 0; j < 4; j++) {
        x_lin(0 + j) = var_ori[j];
      }
      for (int j = 0; j < 3; j++) {
        x_lin(4 + j) = var_pos[j];
        x_lin(7 + j) = var_bias_g[j];
        x_lin(10 + j) = var_bias_a[j];
      }
      Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(10, 1);
      Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(10, 10);
      prior_Info.block(0, 0, 4, 4) *= 1.0 / std::pow(1e-5, 2); // 4dof unobservable yaw and position
      prior_Info.block(4, 4, 3, 3) *= 1.0 / std::pow(0.05, 2); // bias_g prior
      prior_Info.block(7, 7, 3, 3) *= 1.0 / std::pow(0.10, 2); // bias_a prior

      // Construct state type and ceres parameter pointers
      std::vector<std::string> x_types;
      std::vector<double *> factor_params;
      factor_params.push_back(var_ori);
      x_types.emplace_back("quat_yaw");
      factor_params.push_back(var_pos);
      x_types.emplace_back("vec3");
      factor_params.push_back(var_bias_g);
      x_types.emplace_back("vec3");
      factor_params.push_back(var_bias_a);
      x_types.emplace_back("vec3");

      // Append it to the problem
      auto *factor_prior = new Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad);
      problem.AddResidualBlock(factor_prior, nullptr, factor_params);
    }

    // Append to our historical vector of states
    map_states.insert({timestamp_k1, (int)ceres_vars_ori.size()});
    ceres_vars_ori.push_back(var_ori);
    ceres_vars_pos.push_back(var_pos);
    ceres_vars_vel.push_back(var_vel);
    ceres_vars_bias_g.push_back(var_bias_g);
    ceres_vars_bias_a.push_back(var_bias_a);

    // ================================================================
    //  ADDING GRAPH FACTORS!
    // ================================================================

    // Append the new IMU factor
    if (cpi != nullptr) {
      assert(timestamp_k != -1);
      auto state_k_it = map_states.find(timestamp_k);
      auto state_k1_it = map_states.find(timestamp_k1);
      if (state_k_it == map_states.end() || state_k1_it == map_states.end()) {
        timestamp_k = timestamp_k1;
        continue;
      }
      int idx_k = state_k_it->second;
      int idx_k1 = state_k1_it->second;
      std::vector<double *> factor_params;
      factor_params.push_back(ceres_vars_ori.at(idx_k));
      factor_params.push_back(ceres_vars_bias_g.at(idx_k));
      factor_params.push_back(ceres_vars_vel.at(idx_k));
      factor_params.push_back(ceres_vars_bias_a.at(idx_k));
      factor_params.push_back(ceres_vars_pos.at(idx_k));
      factor_params.push_back(ceres_vars_ori.at(idx_k1));
      factor_params.push_back(ceres_vars_bias_g.at(idx_k1));
      factor_params.push_back(ceres_vars_vel.at(idx_k1));
      factor_params.push_back(ceres_vars_bias_a.at(idx_k1));
      factor_params.push_back(ceres_vars_pos.at(idx_k1));
      auto *factor_imu = new Factor_ImuCPIv1(cpi->DT, gravity, cpi->alpha_tau, cpi->beta_tau, cpi->q_k2tau, cpi->b_a_lin, cpi->b_w_lin,
                                             cpi->J_q, cpi->J_b, cpi->J_a, cpi->H_b, cpi->H_a, cpi->P_meas);
      problem.AddResidualBlock(factor_imu, nullptr, factor_params);
    }

    // Move time forward
    timestamp_k = timestamp_k1;
  }

  // First make sure we have calibration states added
  for (auto const &idpair : map_camera_ids) {
    size_t cam_id = idpair.first;
    if (map_calib_cam2imu.find(cam_id) == map_calib_cam2imu.end()) {
      auto *var_calib_ori = new double[4];
      for (int j = 0; j < 4; j++) {
        var_calib_ori[j] = params.camera_extrinsics.at(cam_id)(0 + j, 0);
      }
      auto *var_calib_pos = new double[3];
      for (int j = 0; j < 3; j++) {
        var_calib_pos[j] = params.camera_extrinsics.at(cam_id)(4 + j, 0);
      }
      auto ceres_calib_jplquat = new State_JPLQuatLocal();
      problem.AddParameterBlock(var_calib_ori, 4, ceres_calib_jplquat);
      problem.AddParameterBlock(var_calib_pos, 3);
      map_calib_cam2imu.insert({cam_id, (int)ceres_vars_calib_cam2imu_ori.size()});
      ceres_vars_calib_cam2imu_ori.push_back(var_calib_ori);
      ceres_vars_calib_cam2imu_pos.push_back(var_calib_pos);

      // Construct state and prior
      Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(7, 1);
      for (int j = 0; j < 4; j++) {
        x_lin(0 + j) = var_calib_ori[j];
      }
      for (int j = 0; j < 3; j++) {
        x_lin(4 + j) = var_calib_pos[j];
      }
      Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(6, 1);
      Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(6, 6);
      prior_Info.block(0, 0, 3, 3) *= 1.0 / std::pow(0.001, 2);
      prior_Info.block(3, 3, 3, 3) *= 1.0 / std::pow(0.01, 2);

      // Construct state type and ceres parameter pointers
      std::vector<std::string> x_types;
      std::vector<double *> factor_params;
      factor_params.push_back(var_calib_ori);
      x_types.emplace_back("quat");
      factor_params.push_back(var_calib_pos);
      x_types.emplace_back("vec3");
      auto *factor_prior = new Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad);
      problem.AddResidualBlock(factor_prior, nullptr, factor_params);
      if (!params.init_dyn_mle_opt_calib) {
        problem.SetParameterBlockConstant(var_calib_ori);
        problem.SetParameterBlockConstant(var_calib_pos);
      }
    }
    if (map_calib_cam.find(cam_id) == map_calib_cam.end()) {
      auto *var_calib_cam = new double[8];
      for (int j = 0; j < 8; j++) {
        var_calib_cam[j] = params.camera_intrinsics.at(cam_id)->get_value()(j, 0);
      }
      problem.AddParameterBlock(var_calib_cam, 8);
      map_calib_cam.insert({cam_id, (int)ceres_vars_calib_cam_intrinsics.size()});
      ceres_vars_calib_cam_intrinsics.push_back(var_calib_cam);

      // Construct state and prior
      Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(8, 1);
      for (int j = 0; j < 8; j++) {
        x_lin(0 + j) = var_calib_cam[j];
      }
      Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(8, 1);
      Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(8, 8);
      prior_Info.block(0, 0, 4, 4) *= 1.0 / std::pow(1.0, 2);
      prior_Info.block(4, 4, 4, 4) *= 1.0 / std::pow(0.005, 2);

      // Construct state type and ceres parameter pointers
      std::vector<std::string> x_types;
      std::vector<double *> factor_params;
      factor_params.push_back(var_calib_cam);
      x_types.emplace_back("vec8");
      auto *factor_prior = new Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad);
      problem.AddResidualBlock(factor_prior, nullptr, factor_params);
      if (!params.init_dyn_mle_opt_calib) {
        problem.SetParameterBlockConstant(var_calib_cam);
      }
    }
  }
  assert(map_calib_cam2imu.size() == map_calib_cam.size());


  // Initialize scale parameters for point cloud constraint
  if (params.init_vggt_feature_pc_constraint_weight > 0.0 && !features_point3d_c0_.empty()) {
    // Determine scale initial value (optionally softplus parameterized)
    double scale_init = 1.0;
    if (params.init_vggt_scale_use_softplus) {
      // inverse_softplus(1.0) ≈ log(exp(1.0 - 1e-5) - 1.0)
      scale_init = std::log(std::exp(1.0 - 1e-5) - 1.0);
    }

    if (scale_layout.is_region()) {
      // REGION MODE: compute PCA from C0 frame points and create n×n scale parameters
      PRINT_INFO("[VGGT-INIT]: Setting up REGION scale mode (n=%d, %d regions)\n", region_n, region_n * region_n);

      // Step 1: Collect all C0 points and compute centroid
      std::vector<Eigen::Vector3d> all_points;
      for (const auto &pt : features_point3d_c0_) {
        all_points.push_back(pt.second);
      }
      if (all_points.size() < 3) {
        PRINT_WARNING(YELLOW "[VGGT-INIT]: Not enough C0 points for PCA (%zu), disabling region mode\n" RESET, all_points.size());
      } else {
        pca_centroid = Eigen::Vector3d::Zero();
        for (const auto &pt : all_points) {
          pca_centroid += pt;
        }
        pca_centroid /= all_points.size();

        // Step 2: Compute covariance matrix
        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (const auto &pt : all_points) {
          Eigen::Vector3d diff = pt - pca_centroid;
          cov += diff * diff.transpose();
        }
        cov /= all_points.size();

        // Step 3: Eigenvalue decomposition (eigenvalues sorted ascending)
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        Eigen::Matrix3d eigenvectors = solver.eigenvectors();

        // Principal axes: col(2) = largest variance, col(1) = second largest
        pca_axis1 = eigenvectors.col(2);
        pca_axis2 = eigenvectors.col(1);

        // Step 4: Compute bounds along principal axes
        pca_min1 = std::numeric_limits<double>::max();
        pca_max1 = std::numeric_limits<double>::lowest();
        pca_min2 = std::numeric_limits<double>::max();
        pca_max2 = std::numeric_limits<double>::lowest();
        for (const auto &pt : all_points) {
          Eigen::Vector3d diff = pt - pca_centroid;
          double proj1 = diff.dot(pca_axis1);
          double proj2 = diff.dot(pca_axis2);
          pca_min1 = std::min(pca_min1, proj1);
          pca_max1 = std::max(pca_max1, proj1);
          pca_min2 = std::min(pca_min2, proj2);
          pca_max2 = std::max(pca_max2, proj2);
        }
        // Add margin
        double margin1 = (pca_max1 - pca_min1) * 0.01;
        double margin2 = (pca_max2 - pca_min2) * 0.01;
        pca_min1 -= margin1; pca_max1 += margin1;
        pca_min2 -= margin2; pca_max2 += margin2;

        // Step 5: Create scale parameter for each region (n×n)
        for (std::size_t r = 0; r < scale_layout.scale_count; ++r) {
          auto *var_scale = new double[1];
          var_scale[0] = scale_init;
          problem.AddParameterBlock(var_scale, 1);
          ceres_vars_scale.push_back(var_scale);

          // Add prior constraint
          if (params.init_vggt_scale_prior_weight > 0) {
            auto *factor_scale_prior = new Factor_ScalePrior(3.0, params.init_vggt_scale_prior_weight, params.init_vggt_scale_use_softplus);
            problem.AddResidualBlock(factor_scale_prior, nullptr, var_scale);
          }
        }

        // Step 6: Add smoothness constraints between adjacent regions (2D grid)
        if (params.init_vggt_scale_smoothness_weight > 0) {
          for (const auto &edge : scale_layout.smoothness_edges) {
            auto *factor_smooth = new Factor_ScaleSmoothness(
                params.init_vggt_scale_smoothness_weight, params.init_vggt_scale_use_softplus);
            problem.AddResidualBlock(factor_smooth, nullptr, ceres_vars_scale[edge.first], ceres_vars_scale[edge.second]);
          }
          PRINT_DEBUG("[VGGT-INIT]: Added %zu region smoothness constraints\n", scale_layout.smoothness_edges.size());
        }
      }
    } else {
      // GLOBAL MODE: single shared scale
      PRINT_INFO("[VGGT-INIT]: Setting up GLOBAL scale mode\n");
      ceres_var_global_scale = new double[1];
      ceres_var_global_scale[0] = scale_init;
      problem.AddParameterBlock(ceres_var_global_scale, 1);

      if (params.init_vggt_scale_prior_weight > 0) {
        auto *factor_scale_prior = new Factor_ScalePrior(3.0, params.init_vggt_scale_prior_weight, params.init_vggt_scale_use_softplus);
        problem.AddResidualBlock(factor_scale_prior, nullptr, ceres_var_global_scale);
      }
    }
  }

  // Then, append new feature observations factors seen from all cameras
  for (auto const &feat : features) {
    // Skip features that don't have enough measurements
    auto meas_it = map_features_num_meas.find(feat.first);
    if (meas_it == map_features_num_meas.end() || meas_it->second < min_num_meas_to_optimize)
      continue;
    // Features can be removed if behind the camera!
    if (features_inG.find(feat.first) == features_inG.end())
      continue;
    // Finally loop through each raw uv observation and append it as a factor
    for (auto const &camtime : feat.second->timestamps) {

      // Get our ids and if the camera is a fisheye or not
      size_t feat_id = feat.first;
      size_t cam_id = camtime.first;
      bool is_fisheye = (std::dynamic_pointer_cast<ov_core::CamEqui>(params.camera_intrinsics.at(cam_id)) != nullptr);

      // Loop through each observation
      for (size_t i = 0; i < camtime.second.size(); i++) {

        // Skip measurements we don't have poses for
        double time = feat.second->timestamps.at(cam_id).at(i);
        if (map_camera_times.find(time) == map_camera_times.end())
          continue;

        // Our measurement
        Eigen::Vector2d uv_raw = feat.second->uvs.at(cam_id).at(i).block(0, 0, 2, 1).cast<double>();

        // If we don't have the feature state we should create that parameter block
        // The initial guess of the features are the scaled feature map from the SFM
        if (map_features.find(feat_id) == map_features.end()) {
          auto feat_inG_it = features_inG.find(feat_id);
          if (feat_inG_it == features_inG.end()) {
            continue;
          }
          auto *var_feat = new double[3];
          for (int j = 0; j < 3; j++) {
            var_feat[j] = feat_inG_it->second(j);
          }
          problem.AddParameterBlock(var_feat, 3);
          map_features.insert({feat_id, (int)ceres_vars_feat.size()});
          ceres_vars_feat.push_back(var_feat);
        }

        // Then lets add the factors
        auto state_idx_it = map_states.find(time);
        auto feat_idx_it = map_features.find(feat_id);
        auto calib_ori_it = map_calib_cam2imu.find(cam_id);
        auto calib_int_it = map_calib_cam.find(cam_id);
        if (state_idx_it == map_states.end() || feat_idx_it == map_features.end() || calib_ori_it == map_calib_cam2imu.end() ||
            calib_int_it == map_calib_cam.end()) {
          continue;
        }
        int state_idx = state_idx_it->second;
        int feat_idx = feat_idx_it->second;
        int calib_idx = calib_ori_it->second;
        int calib_int_idx = calib_int_it->second;
        std::vector<double *> factor_params;
        factor_params.push_back(ceres_vars_ori.at(state_idx));
        factor_params.push_back(ceres_vars_pos.at(state_idx));
        factor_params.push_back(ceres_vars_feat.at(feat_idx));
        factor_params.push_back(ceres_vars_calib_cam2imu_ori.at(calib_idx));
        factor_params.push_back(ceres_vars_calib_cam2imu_pos.at(calib_idx));
        factor_params.push_back(ceres_vars_calib_cam_intrinsics.at(calib_int_idx));
        auto *factor_pinhole = new Factor_ImageReprojCalib(uv_raw, params.sigma_pix, is_fisheye);
        // ceres::LossFunction *loss_function = nullptr;
        ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(factor_pinhole, loss_function, factor_params);
      }
    }
  }

  // Add point cloud scale constraints
  if (params.init_vggt_feature_pc_constraint_weight > 0.0 &&
      (!ceres_vars_scale.empty() || ceres_var_global_scale != nullptr)) {
    // Compute transformation from C0 to G frame
    // v_G = R_I0toG * R_CtoI * v_C = R_GtoI0^T * R_ItoC^T * v_C
    // So: R_C0toG = R_GtoI0^T * R_ItoC^T
    // t_C0toG = -R_C0toG * p_IinC (translation offset from extrinsics)
    size_t cam_id = map_camera_ids.begin()->first;
    Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(cam_id).block(0, 0, 4, 1);
    Eigen::Vector3d p_IinC = params.camera_extrinsics.at(cam_id).block(4, 0, 3, 1);
    Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);
    Eigen::Matrix3d R_C0toG = R_GtoI0.transpose() * R_ItoC.transpose();
    Eigen::Vector3d t_C0toG = -R_C0toG * p_IinC;

    int pc_count = 0;
    for (auto const &featpair : map_features) {
      size_t feat_id = featpair.first;
      int feat_idx = featpair.second;

      // Get point cloud in C0 frame
      auto c0_it = features_point3d_c0_.find(feat_id);
      if (c0_it == features_point3d_c0_.end()) continue;

      // Determine which scale parameter to use
      double *scale_ptr = nullptr;
      if (scale_layout.is_region() && !ceres_vars_scale.empty()) {
        int region_idx = get_region_index(c0_it->second);
        if (region_idx < 0 || region_idx >= (int)ceres_vars_scale.size()) continue;
        scale_ptr = ceres_vars_scale[region_idx];
      } else {
        scale_ptr = ceres_var_global_scale;
      }

      if (scale_ptr == nullptr) continue;

      std::vector<double *> factor_params;
      factor_params.push_back(ceres_vars_feat.at(feat_idx));
      factor_params.push_back(scale_ptr);

      auto *factor = new Factor_PointCloudScale(
          c0_it->second,  // p_FinC0
          R_C0toG,
          t_C0toG,
          params.init_vggt_feature_pc_constraint_weight,
          params.init_vggt_scale_use_softplus);

      problem.AddResidualBlock(factor, new ceres::CauchyLoss(1.0), factor_params);
      pc_count++;
    }
    PRINT_INFO("[VGGT-INIT]: Added %d point cloud scale constraints (weight=%.2f)\n",
               pc_count, params.init_vggt_feature_pc_constraint_weight);
  }


  assert(ceres_vars_ori.size() == ceres_vars_bias_g.size());
  assert(ceres_vars_ori.size() == ceres_vars_vel.size());
  assert(ceres_vars_ori.size() == ceres_vars_bias_a.size());
  assert(ceres_vars_ori.size() == ceres_vars_pos.size());
  auto rT5 = boost::posix_time::microsec_clock::local_time();

  // Optimize the ceres graph
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  PRINT_INFO("[VGGT-INIT]: %d iterations | %zu states, %zu feats (%zu valid) | %d param and %d res | cost %.4e => %.4e\n",
             (int)summary.iterations.size(), map_states.size(), map_features.size(), count_valid_features, summary.num_parameters,
             summary.num_residuals, summary.initial_cost, summary.final_cost);
  PRINT_DEBUG("[VGGT-INIT]: %s\n", summary.message.c_str());
  auto rT6 = boost::posix_time::microsec_clock::local_time();

  // Return if we have failed!
  timestamp = newest_cam_time;
  if (params.init_dyn_mle_max_iter != 0 && summary.termination_type != ceres::CONVERGENCE) {
    PRINT_WARNING(YELLOW "[VGGT-INIT]: opt failed: %s!\n" RESET, summary.message.c_str());
    free_state_memory();
    return false;
  }

  // Record average residual and check threshold
  if (summary.num_residuals > 0) {
    const double nonlinear_avg_residual = summary.final_cost / summary.num_residuals;
    PRINT_DEBUG("[VGGT-INIT]: avg_residual=%.4f\n", nonlinear_avg_residual);
    if (params.init_dyn_max_nonlinear_avg_residual > 0 && nonlinear_avg_residual > params.init_dyn_max_nonlinear_avg_residual) {
      PRINT_WARNING(YELLOW "[VGGT-INIT]: high final cost: avg_residual=%.4f > threshold=%.4f\n" RESET,
                    nonlinear_avg_residual, params.init_dyn_max_nonlinear_avg_residual);
      free_state_memory();
      return false;
    }
  }

  // Log optimized scale values.
  if (params.init_vggt_feature_pc_constraint_weight > 0.0) {
    // Helper to convert softplus parameterized scale to actual scale
    auto convert_scale = [&](double s_param) -> double {
      if (params.init_vggt_scale_use_softplus) {
        double t = std::max(s_param, 0.0);
        return std::log1p(std::exp(-std::abs(s_param))) + t + 1e-5;
      }
      return s_param;
    };

    if (scale_layout.is_region() && !ceres_vars_scale.empty()) {
      // Region mode: compute and report the average scale.
      double min_scale = std::numeric_limits<double>::max();
      double max_scale = std::numeric_limits<double>::lowest();
      double sum_scale = 0.0;
      for (size_t i = 0; i < ceres_vars_scale.size(); i++) {
        double scale_value = convert_scale(ceres_vars_scale[i][0]);
        min_scale = std::min(min_scale, scale_value);
        max_scale = std::max(max_scale, scale_value);
        sum_scale += scale_value;
      }
      double avg_scale = sum_scale / ceres_vars_scale.size();
      PRINT_INFO("[VGGT-INIT]: Region scales (%zu regions): min=%.4f, max=%.4f, avg=%.4f\n",
                 ceres_vars_scale.size(), min_scale, max_scale, avg_scale);
    } else if (ceres_var_global_scale != nullptr) {
      // Global mode: single shared scale
      double global_scale = convert_scale(ceres_var_global_scale[0]);
      PRINT_INFO("[VGGT-INIT]: Optimized global scale = %.4f\n", global_scale);
    }
  }


  //======================================================
  //======================================================

  // Helper function to get the IMU pose value from our ceres problem
  auto get_pose = [&](double timestamp) {
    Eigen::VectorXd state_imu = Eigen::VectorXd::Zero(16);
    for (int i = 0; i < 4; i++) {
      state_imu(0 + i) = ceres_vars_ori[map_states[timestamp]][i];
    }
    for (int i = 0; i < 3; i++) {
      state_imu(4 + i) = ceres_vars_pos[map_states[timestamp]][i];
      state_imu(7 + i) = ceres_vars_vel[map_states[timestamp]][i];
      state_imu(10 + i) = ceres_vars_bias_g[map_states[timestamp]][i];
      state_imu(13 + i) = ceres_vars_bias_a[map_states[timestamp]][i];
    }
    return state_imu;
  };

  // Our most recent state is the IMU state!
  assert(map_states.find(newest_cam_time) != map_states.end());
  if (_imu == nullptr) {
    _imu = std::make_shared<ov_type::IMU>();
  }
  Eigen::VectorXd imu_state = get_pose(newest_cam_time);
  _imu->set_value(imu_state);
  _imu->set_fej(imu_state);

  // Append our IMU clones (includes most recent)
  for (auto const &statepair : map_states) {
    Eigen::VectorXd pose = get_pose(statepair.first);
    auto clone_it = _clones_IMU.find(statepair.first);
    if (clone_it == _clones_IMU.end()) {
      auto _pose = std::make_shared<ov_type::PoseJPL>();
      _pose->set_value(pose.block(0, 0, 7, 1));
      _pose->set_fej(pose.block(0, 0, 7, 1));
      _clones_IMU.insert({statepair.first, _pose});
    } else {
      clone_it->second->set_value(pose.block(0, 0, 7, 1));
      clone_it->second->set_fej(pose.block(0, 0, 7, 1));
    }
  }

  // Append features as SLAM features!
  for (auto const &featpair : map_features) {
    Eigen::Vector3d feature;
    feature << ceres_vars_feat[featpair.second][0], ceres_vars_feat[featpair.second][1], ceres_vars_feat[featpair.second][2];
    auto slam_it = _features_SLAM.find(featpair.first);
    if (slam_it == _features_SLAM.end()) {
      auto _feature = std::make_shared<ov_type::Landmark>(3);
      _feature->_featid = featpair.first;
      _feature->_feat_representation = LandmarkRepresentation::Representation::GLOBAL_3D;
      _feature->set_from_xyz(feature, false);
      _feature->set_from_xyz(feature, true);
      _features_SLAM.insert({featpair.first, _feature});
    } else {
      slam_it->second->_featid = featpair.first;
      slam_it->second->_feat_representation = LandmarkRepresentation::Representation::GLOBAL_3D;
      slam_it->second->set_from_xyz(feature, false);
      slam_it->second->set_from_xyz(feature, true);
    }
  }

  // If we optimized calibration, we should also save it to our state
  if (params.init_dyn_mle_opt_calib) {
    // TODO: append our calibration states too if we are doing calibration!
    // TODO: (if we are not doing calibration do not calibrate them....)
    // TODO: std::shared_ptr<ov_type::Vec> _calib_dt_CAMtoIMU,
    // TODO: std::unordered_map<size_t, std::shared_ptr<ov_type::PoseJPL>> &_calib_IMUtoCAM,
    // TODO: std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> &_cam_intrinsics
  }

  // Recover the covariance of the optimized IMU state
  int state_index = map_states[newest_cam_time];
  if (!recover_covariance(problem, state_index, ceres_vars_ori, ceres_vars_pos, ceres_vars_vel,
                          ceres_vars_bias_g, ceres_vars_bias_a, map_calib_cam2imu,
                          _imu, covariance, order)) {
    free_state_memory();
    return false;
  }

  // Set our position to be zero
  Eigen::MatrixXd x = _imu->value();
  x.block(4, 0, 3, 1).setZero();
  _imu->set_value(x);
  _imu->set_fej(x);

  // Wall-clock diagnostics; this is distinct from the sensor-time initialization duration.
  auto rT7 = boost::posix_time::microsec_clock::local_time();
  double t_prelim = (rT2 - rT1).total_microseconds() * 1e-6;
  double t_prelim_no_rtt = (vggt_inference_time_ > 0) ? (t_prelim - vggt_inference_time_) : t_prelim;
  double t_linsys_setup = (rT3 - rT2).total_microseconds() * 1e-6;
  double t_linsys = (rT4 - rT3).total_microseconds() * 1e-6;
  double t_ceres_setup = (rT5 - rT4).total_microseconds() * 1e-6;
  double t_ceres_opt = (rT6 - rT5).total_microseconds() * 1e-6;
  double t_cov = (rT7 - rT6).total_microseconds() * 1e-6;
  double t_total = (rT7 - rT1).total_microseconds() * 1e-6;
  PRINT_INFO(CYAN "[WALL-TIME] initializer_compute_total=%.3fs | inference_rtt=%.3fs prelim_no_rtt=%.3fs linsys_setup=%.3fs "
                  "linsys=%.3fs ceres_setup=%.3fs ceres_opt=%.3fs cov=%.3fs\n" RESET,
             t_total, vggt_inference_time_, t_prelim_no_rtt, t_linsys_setup, t_linsys, t_ceres_setup, t_ceres_opt, t_cov);

  const double nonlinear_timestamp = map_states.begin()->first;
  const Eigen::VectorXd nonlinear = get_pose(nonlinear_timestamp);
  diagnostics.nonlinear_state =
      make_initialization_state(nonlinear_timestamp, nonlinear.block<4, 1>(0, 0), nonlinear.block<3, 1>(4, 0),
                                nonlinear.block<3, 1>(7, 0));
  diagnostics.has_nonlinear_state = true;

  free_state_memory();
  return true;
}
bool VGGTInitializer::run_nonlinear_optimization_with_scale(
    const std::map<double, bool> &map_camera_times, const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_IitoIi1,
    const std::map<double, Eigen::VectorXd> &ori_GtoIi, const std::map<double, Eigen::VectorXd> &pos_IiinG,
    const std::map<double, Eigen::VectorXd> &vel_IiinG, const Eigen::Vector3d &gyroscope_bias, const Eigen::Vector3d &accelerometer_bias,
    const std::map<double, std::vector<Eigen::Vector2d>> &sampled_pixels_per_time,
    const std::map<double, std::vector<Eigen::Vector3d>> &sampled_points_per_time,
    const std::map<double, std::vector<float>> &sampled_confidences_per_time, const std::map<size_t, bool> &map_camera_ids,
    double newest_cam_time, std::shared_ptr<ov_type::IMU> &_imu,
    std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU, double &timestamp, Eigen::MatrixXd &covariance,
    std::vector<std::shared_ptr<ov_type::Type>> &order, const boost::posix_time::ptime &rT1, const boost::posix_time::ptime &rT2,
    const boost::posix_time::ptime &rT3, const boost::posix_time::ptime &rT4,
    InitializationDiagnostics &diagnostics) {

  // Ceres problem stuff
  // NOTE: By default the problem takes ownership of the memory
  ceres::Problem problem;

  // Our system states (map from time to index)
  std::map<double, int> map_states;
  std::vector<double *> ceres_vars_ori;
  std::vector<double *> ceres_vars_pos;
  std::vector<double *> ceres_vars_vel;
  std::vector<double *> ceres_vars_bias_g;
  std::vector<double *> ceres_vars_bias_a;

  // Scale states: one global scale or an n-by-n regional grid.
  double *ceres_var_global_scale = nullptr;
  std::vector<double *> ceres_vars_scale;

  // Region-based scale parameter
  int region_n = params.init_vggt_scale_region_n;
  const VggtScaleLayout scale_layout = make_vggt_scale_layout(region_n);

  // Setup extrinsic calibration q_ItoC, p_IinC (map from camera id to index)
  std::map<size_t, int> map_calib_cam2imu;
  std::vector<double *> ceres_vars_calib_cam2imu_ori;
  std::vector<double *> ceres_vars_calib_cam2imu_pos;

  // Setup intrinsic calibration focal, center, distortion (map from camera id to index)
  std::map<size_t, int> map_calib_cam;
  std::vector<double *> ceres_vars_calib_cam_intrinsics;
  Eigen::Vector3d gravity;
  gravity << 0.0, 0.0, params.gravity_mag;

  // Helper lambda that will free any memory we have allocated
  auto free_state_memory = [&]() {
    for (auto ptr : ceres_vars_ori) delete[] ptr;
    for (auto ptr : ceres_vars_pos) delete[] ptr;
    for (auto ptr : ceres_vars_vel) delete[] ptr;
    for (auto ptr : ceres_vars_bias_g) delete[] ptr;
    for (auto ptr : ceres_vars_bias_a) delete[] ptr;
    for (auto ptr : ceres_vars_scale) delete[] ptr;
    for (auto ptr : ceres_vars_calib_cam2imu_ori) delete[] ptr;
    for (auto ptr : ceres_vars_calib_cam2imu_pos) delete[] ptr;
    for (auto ptr : ceres_vars_calib_cam_intrinsics) delete[] ptr;
    if (ceres_var_global_scale != nullptr) delete[] ceres_var_global_scale;
  };

  // Set the optimization settings
  // NOTE: We use dense schur since after eliminating features we have a dense problem
  // NOTE: http://ceres-solver.org/solving_faqs.html#solving
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  // options.linear_solver_type = ceres::SPARSE_SCHUR;
  // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  // options.preconditioner_type = ceres::SCHUR_JACOBI;
  // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.num_threads = params.init_dyn_mle_max_threads;
  options.max_solver_time_in_seconds = params.init_dyn_mle_max_time;
  options.max_num_iterations = params.init_dyn_mle_max_iter;
  // options.minimizer_progress_to_stdout = true;
  // options.linear_solver_ordering = ordering;
  options.function_tolerance = 1e-5;
  options.gradient_tolerance = 1e-4 * options.function_tolerance;

  // Scale prior parameters
  const double scale_prior_value = 3.0;
  const double scale_prior_weight = params.init_vggt_scale_prior_weight;

  // Compute initial parameter value based on softplus setting
  // If softplus: actual_scale = softplus(param) + 1e-5, so param = inverse_softplus(actual_scale - 1e-5)
  double scale_init_param = (scale_prior_weight > 0) ? scale_prior_value : 1.0;
  if (params.init_vggt_scale_use_softplus) {
    // inverse_softplus(y) = log(exp(y) - 1) for y > 0
    scale_init_param = std::log(std::exp(scale_init_param - 1e-5) - 1.0);
  }

  // PCA-based region variables (computed in region mode, initialized to avoid warnings)
  Eigen::Vector3d pca_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d pca_axis1 = Eigen::Vector3d::UnitX();
  Eigen::Vector3d pca_axis2 = Eigen::Vector3d::UnitY();
  double pca_min1 = 0, pca_max1 = 1, pca_min2 = 0, pca_max2 = 1;

  // Helper lambda: compute region index from 3D point using PCA projection
  auto get_region_index = [&](const Eigen::Vector3d &point) -> int {
    if (region_n <= 0) return -1;
    // Project point onto the two principal axes
    Eigen::Vector3d diff = point - pca_centroid;
    double proj1 = diff.dot(pca_axis1);
    double proj2 = diff.dot(pca_axis2);
    // Normalize to [0, 1]
    double norm1 = (proj1 - pca_min1) / (pca_max1 - pca_min1);
    double norm2 = (proj2 - pca_min2) / (pca_max2 - pca_min2);
    // Clamp to [0, 1)
    norm1 = std::max(0.0, std::min(0.999999, norm1));
    norm2 = std::max(0.0, std::min(0.999999, norm2));
    // Compute 2D grid indices
    int i1 = static_cast<int>(norm1 * region_n);
    int i2 = static_cast<int>(norm2 * region_n);
    // Flatten to 1D index: i1 + i2 * n
    return i1 + i2 * region_n;
  };

  // Initialize either the regional grid or the single global scale.
  if (scale_layout.is_region()) {
    // REGION-BASED SCALE MODE with PCA (n×n regions along principal axes)
    PRINT_INFO("[VGGT-INIT]: Using REGION scale mode (n=%d, %d regions)\n", region_n, region_n * region_n);

    // Step 1: Collect all points and compute centroid
    std::vector<Eigen::Vector3d> all_points;
    for (const auto &points_pair : sampled_points_per_time) {
      for (const auto &pt : points_pair.second) {
        all_points.push_back(pt);
      }
    }
    pca_centroid = Eigen::Vector3d::Zero();
    for (const auto &pt : all_points) {
      pca_centroid += pt;
    }
    pca_centroid /= all_points.size();

    // Step 2: Compute covariance matrix
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto &pt : all_points) {
      Eigen::Vector3d diff = pt - pca_centroid;
      cov += diff * diff.transpose();
    }
    cov /= all_points.size();

    // Step 3: Eigenvalue decomposition (eigenvalues sorted ascending)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d eigenvalues = solver.eigenvalues();
    Eigen::Matrix3d eigenvectors = solver.eigenvectors();

    // Principal axes: col(2) = largest variance, col(1) = second largest, col(0) = smallest
    pca_axis1 = eigenvectors.col(2);  // Primary direction (largest variance)
    pca_axis2 = eigenvectors.col(1);  // Secondary direction
    Eigen::Vector3d pca_axis3 = eigenvectors.col(0);  // Smallest variance (ignored for partitioning)

    PRINT_DEBUG("[VGGT-INIT]: PCA eigenvalues: %.4f, %.4f, %.4f\n",
                eigenvalues(0), eigenvalues(1), eigenvalues(2));
    PRINT_DEBUG("[VGGT-INIT]: Primary axis (%.3f, %.3f, %.3f), Secondary axis (%.3f, %.3f, %.3f)\n",
                pca_axis1.x(), pca_axis1.y(), pca_axis1.z(),
                pca_axis2.x(), pca_axis2.y(), pca_axis2.z());
    PRINT_DEBUG("[VGGT-INIT]: Ignored axis (%.3f, %.3f, %.3f) - smallest variance\n",
                pca_axis3.x(), pca_axis3.y(), pca_axis3.z());

    // Step 4: Compute bounds along principal axes
    pca_min1 = std::numeric_limits<double>::max();
    pca_max1 = std::numeric_limits<double>::lowest();
    pca_min2 = std::numeric_limits<double>::max();
    pca_max2 = std::numeric_limits<double>::lowest();
    for (const auto &pt : all_points) {
      Eigen::Vector3d diff = pt - pca_centroid;
      double proj1 = diff.dot(pca_axis1);
      double proj2 = diff.dot(pca_axis2);
      pca_min1 = std::min(pca_min1, proj1);
      pca_max1 = std::max(pca_max1, proj1);
      pca_min2 = std::min(pca_min2, proj2);
      pca_max2 = std::max(pca_max2, proj2);
    }
    // Add margin
    double margin1 = (pca_max1 - pca_min1) * 0.01;
    double margin2 = (pca_max2 - pca_min2) * 0.01;
    pca_min1 -= margin1; pca_max1 += margin1;
    pca_min2 -= margin2; pca_max2 += margin2;

    PRINT_DEBUG("[VGGT-INIT]: Region bounds: axis1=[%.2f, %.2f], axis2=[%.2f, %.2f]\n",
                pca_min1, pca_max1, pca_min2, pca_max2);

    // Step 5: Create scale parameter for each region (n×n)
    for (std::size_t r = 0; r < scale_layout.scale_count; ++r) {
      auto *var_scale = new double[1];
      var_scale[0] = scale_init_param;
      problem.AddParameterBlock(var_scale, 1);
      ceres_vars_scale.push_back(var_scale);

      // Add prior constraint
      if (scale_prior_weight > 0) {
        auto *factor_scale_prior = new Factor_ScalePrior(scale_prior_value, scale_prior_weight, params.init_vggt_scale_use_softplus);
        problem.AddResidualBlock(factor_scale_prior, nullptr, var_scale);
      }
    }

    // Step 6: Add smoothness constraints between adjacent regions (2D grid)
    if (params.init_vggt_scale_smoothness_weight > 0) {
      for (const auto &edge : scale_layout.smoothness_edges) {
        auto *factor_smooth = new Factor_ScaleSmoothness(
            params.init_vggt_scale_smoothness_weight, params.init_vggt_scale_use_softplus);
        problem.AddResidualBlock(factor_smooth, nullptr, ceres_vars_scale[edge.first], ceres_vars_scale[edge.second]);
      }
      PRINT_INFO("[VGGT-INIT]: Added %zu region smoothness constraints\n", scale_layout.smoothness_edges.size());
    }
  } else {
    // GLOBAL SCALE MODE
    PRINT_INFO("[VGGT-INIT]: Using GLOBAL scale mode\n");
    ceres_var_global_scale = new double[1];
    ceres_var_global_scale[0] = scale_init_param;
    problem.AddParameterBlock(ceres_var_global_scale, 1);

    // Add soft prior constraint on global scale
    if (scale_prior_weight > 0) {
      auto *factor_scale_prior = new Factor_ScalePrior(scale_prior_value, scale_prior_weight, params.init_vggt_scale_use_softplus);
      problem.AddResidualBlock(factor_scale_prior, nullptr, ceres_var_global_scale);
      PRINT_INFO("[VGGT-INIT]: Added global scale prior: value=%.2f, weight=%.2f, softplus=%d\n",
                 scale_prior_value, scale_prior_weight, params.init_vggt_scale_use_softplus);
    }
  }

  // Loop through each CPI integration and add its measurement to the problem
  double timestamp_k = -1;
  for (auto const &timepair : map_camera_times) {

    // Get our predicted state at the requested camera timestep
    double timestamp_k1 = timepair.first;
    std::shared_ptr<ov_core::CpiV1> cpi = map_camera_cpi_IitoIi1.at(timestamp_k1);
    Eigen::Matrix<double, 16, 1> state_k1;
    state_k1.block(0, 0, 4, 1) = ori_GtoIi.at(timestamp_k1);
    state_k1.block(4, 0, 3, 1) = pos_IiinG.at(timestamp_k1);
    state_k1.block(7, 0, 3, 1) = vel_IiinG.at(timestamp_k1);
    state_k1.block(10, 0, 3, 1) = gyroscope_bias;
    state_k1.block(13, 0, 3, 1) = accelerometer_bias;

    // ================================================================
    //  ADDING GRAPH STATE / ESTIMATES!
    // ================================================================

    // Load our state variables into our allocated state pointers
    auto *var_ori = new double[4];
    for (int j = 0; j < 4; j++) {
      var_ori[j] = state_k1(0 + j, 0);
    }
    auto *var_pos = new double[3];
    auto *var_vel = new double[3];
    auto *var_bias_g = new double[3];
    auto *var_bias_a = new double[3];
    for (int j = 0; j < 3; j++) {
      var_pos[j] = state_k1(4 + j, 0);
      var_vel[j] = state_k1(7 + j, 0);
      var_bias_g[j] = state_k1(10 + j, 0);
      var_bias_a[j] = state_k1(13 + j, 0);
    }

    // Now actually create the parameter block in the ceres problem
    auto ceres_jplquat = new State_JPLQuatLocal();
    problem.AddParameterBlock(var_ori, 4, ceres_jplquat);
    problem.AddParameterBlock(var_pos, 3);
    problem.AddParameterBlock(var_vel, 3);
    problem.AddParameterBlock(var_bias_g, 3);
    problem.AddParameterBlock(var_bias_a, 3);

    // Fix this first ever pose to constrain the problem
    // NOTE: If we don't do this, then the problem won't be full rank
    // NOTE: Since init is over a small window, we are likely to be degenerate
    // NOTE: Thus we need to fix these parameters
    if (map_states.empty()) {

      // Construct state and prior
      Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(13, 1);
      for (int j = 0; j < 4; j++) {
        x_lin(0 + j) = var_ori[j];
      }
      for (int j = 0; j < 3; j++) {
        x_lin(4 + j) = var_pos[j];
        x_lin(7 + j) = var_bias_g[j];
        x_lin(10 + j) = var_bias_a[j];
      }
      Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(10, 1);
      Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(10, 10);
      prior_Info.block(0, 0, 4, 4) *= 1.0 / std::pow(1e-5, 2); // 4dof unobservable yaw and position
      prior_Info.block(4, 4, 3, 3) *= 1.0 / std::pow(0.05, 2); // bias_g prior
      prior_Info.block(7, 7, 3, 3) *= 1.0 / std::pow(0.10, 2); // bias_a prior

      // Construct state type and ceres parameter pointers
      std::vector<std::string> x_types;
      std::vector<double *> factor_params;
      factor_params.push_back(var_ori);
      x_types.emplace_back("quat_yaw");
      factor_params.push_back(var_pos);
      x_types.emplace_back("vec3");
      factor_params.push_back(var_bias_g);
      x_types.emplace_back("vec3");
      factor_params.push_back(var_bias_a);
      x_types.emplace_back("vec3");

      // Append it to the problem
      auto *factor_prior = new Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad);
      problem.AddResidualBlock(factor_prior, nullptr, factor_params);
    }

    // Append to our historical vector of states
    map_states.insert({timestamp_k1, (int)ceres_vars_ori.size()});
    ceres_vars_ori.push_back(var_ori);
    ceres_vars_pos.push_back(var_pos);
    ceres_vars_vel.push_back(var_vel);
    ceres_vars_bias_g.push_back(var_bias_g);
    ceres_vars_bias_a.push_back(var_bias_a);

    // ================================================================
    //  ADDING GRAPH FACTORS!
    // ================================================================

    // Append the new IMU factor
    if (cpi != nullptr) {
      assert(timestamp_k != -1);
      auto state_k_it = map_states.find(timestamp_k);
      auto state_k1_it = map_states.find(timestamp_k1);
      if (state_k_it == map_states.end() || state_k1_it == map_states.end()) {
        timestamp_k = timestamp_k1;
        continue;
      }
      int idx_k = state_k_it->second;
      int idx_k1 = state_k1_it->second;
      std::vector<double *> factor_params;
      factor_params.push_back(ceres_vars_ori.at(idx_k));
      factor_params.push_back(ceres_vars_bias_g.at(idx_k));
      factor_params.push_back(ceres_vars_vel.at(idx_k));
      factor_params.push_back(ceres_vars_bias_a.at(idx_k));
      factor_params.push_back(ceres_vars_pos.at(idx_k));
      factor_params.push_back(ceres_vars_ori.at(idx_k1));
      factor_params.push_back(ceres_vars_bias_g.at(idx_k1));
      factor_params.push_back(ceres_vars_vel.at(idx_k1));
      factor_params.push_back(ceres_vars_bias_a.at(idx_k1));
      factor_params.push_back(ceres_vars_pos.at(idx_k1));
      auto *factor_imu = new Factor_ImuCPIv1(cpi->DT, gravity, cpi->alpha_tau, cpi->beta_tau, cpi->q_k2tau, cpi->b_a_lin, cpi->b_w_lin,
                                             cpi->J_q, cpi->J_b, cpi->J_a, cpi->H_b, cpi->H_a, cpi->P_meas);
      problem.AddResidualBlock(factor_imu, nullptr, factor_params);
    }

    // Move time forward
    timestamp_k = timestamp_k1;
  }

  // First make sure we have calibration states added
  for (auto const &idpair : map_camera_ids) {
    size_t cam_id = idpair.first;
    if (map_calib_cam2imu.find(cam_id) == map_calib_cam2imu.end()) {
      auto *var_calib_ori = new double[4];
      for (int j = 0; j < 4; j++) {
        var_calib_ori[j] = params.camera_extrinsics.at(cam_id)(0 + j, 0);
      }
      auto *var_calib_pos = new double[3];
      for (int j = 0; j < 3; j++) {
        var_calib_pos[j] = params.camera_extrinsics.at(cam_id)(4 + j, 0);
      }
      auto ceres_calib_jplquat = new State_JPLQuatLocal();
      problem.AddParameterBlock(var_calib_ori, 4, ceres_calib_jplquat);
      problem.AddParameterBlock(var_calib_pos, 3);
      map_calib_cam2imu.insert({cam_id, (int)ceres_vars_calib_cam2imu_ori.size()});
      ceres_vars_calib_cam2imu_ori.push_back(var_calib_ori);
      ceres_vars_calib_cam2imu_pos.push_back(var_calib_pos);

      // Construct state and prior
      Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(7, 1);
      for (int j = 0; j < 4; j++) {
        x_lin(0 + j) = var_calib_ori[j];
      }
      for (int j = 0; j < 3; j++) {
        x_lin(4 + j) = var_calib_pos[j];
      }
      Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(6, 1);
      Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(6, 6);
      prior_Info.block(0, 0, 3, 3) *= 1.0 / std::pow(0.001, 2);
      prior_Info.block(3, 3, 3, 3) *= 1.0 / std::pow(0.01, 2);

      // Construct state type and ceres parameter pointers
      std::vector<std::string> x_types;
      std::vector<double *> factor_params;
      factor_params.push_back(var_calib_ori);
      x_types.emplace_back("quat");
      factor_params.push_back(var_calib_pos);
      x_types.emplace_back("vec3");
      auto *factor_prior = new Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad);
      problem.AddResidualBlock(factor_prior, nullptr, factor_params);
      if (!params.init_dyn_mle_opt_calib) {
        problem.SetParameterBlockConstant(var_calib_ori);
        problem.SetParameterBlockConstant(var_calib_pos);
      }
    }
    if (map_calib_cam.find(cam_id) == map_calib_cam.end()) {
      auto *var_calib_cam = new double[8];
      for (int j = 0; j < 8; j++) {
        var_calib_cam[j] = params.camera_intrinsics.at(cam_id)->get_value()(j, 0);
      }
      problem.AddParameterBlock(var_calib_cam, 8);
      map_calib_cam.insert({cam_id, (int)ceres_vars_calib_cam_intrinsics.size()});
      ceres_vars_calib_cam_intrinsics.push_back(var_calib_cam);

      // Construct state and prior
      Eigen::MatrixXd x_lin = Eigen::MatrixXd::Zero(8, 1);
      for (int j = 0; j < 8; j++) {
        x_lin(0 + j) = var_calib_cam[j];
      }
      Eigen::MatrixXd prior_grad = Eigen::MatrixXd::Zero(8, 1);
      Eigen::MatrixXd prior_Info = Eigen::MatrixXd::Identity(8, 8);
      prior_Info.block(0, 0, 4, 4) *= 1.0 / std::pow(1.0, 2);
      prior_Info.block(4, 4, 4, 4) *= 1.0 / std::pow(0.005, 2);

      // Construct state type and ceres parameter pointers
      std::vector<std::string> x_types;
      std::vector<double *> factor_params;
      factor_params.push_back(var_calib_cam);
      x_types.emplace_back("vec8");
      auto *factor_prior = new Factor_GenericPrior(x_lin, x_types, prior_Info, prior_grad);
      problem.AddResidualBlock(factor_prior, nullptr, factor_params);
      if (!params.init_dyn_mle_opt_calib) {
        problem.SetParameterBlockConstant(var_calib_cam);
      }
    }
  }
  assert(map_calib_cam2imu.size() == map_calib_cam.size());

  // Then, append scale-based reprojection factors using VGGT sampled points
  int num_scale_factors = 0;
  std::map<int, int> region_point_counts;  // Debug: count points per region
  for (auto const &sampled_pair : sampled_pixels_per_time) {
    double time = sampled_pair.first;
    const std::vector<Eigen::Vector2d> &pixels = sampled_pair.second;

    // Get corresponding 3D points, confidences & state index
    auto points_it = sampled_points_per_time.find(time);
    auto confidences_it = sampled_confidences_per_time.find(time);
    auto state_idx_it = map_states.find(time);
    if (points_it == sampled_points_per_time.end() || confidences_it == sampled_confidences_per_time.end() || state_idx_it == map_states.end())
      continue;
    const std::vector<Eigen::Vector3d> &points = points_it->second;
    const std::vector<float> &confidences = confidences_it->second;
    int state_idx = state_idx_it->second;

    // Get camera info (assuming cam0)
    size_t cam_id = map_camera_ids.begin()->first;

    bool is_fisheye = (std::dynamic_pointer_cast<ov_core::CamEqui>(params.camera_intrinsics.at(cam_id)) != nullptr);
    auto calib_ori_it = map_calib_cam2imu.find(cam_id);
    auto calib_int_it = map_calib_cam.find(cam_id);
    if (calib_ori_it == map_calib_cam2imu.end() || calib_int_it == map_calib_cam.end())
      continue;
    int calib_idx = calib_ori_it->second;
    int calib_int_idx = calib_int_it->second;

    // Get camera intrinsics values
    Eigen::Matrix<double, 8, 1> camera_vals;
    for (int j = 0; j < 8; j++) {
      camera_vals(j) = ceres_vars_calib_cam_intrinsics.at(calib_int_idx)[j];
    }

    // Loop through each sampled point
    for (size_t i = 0; i < pixels.size(); i++) {
      const Eigen::Vector2d &uv_meas = pixels[i];
      const Eigen::Vector3d &point_c0 = points[i];
      float confidence = confidences[i];

      // Determine scale pointer based on mode
      double *scale_ptr = ceres_var_global_scale;
      if (scale_layout.is_region()) {
        // Region mode: get scale based on 3D point position
        int region_idx = get_region_index(point_c0);
        region_point_counts[region_idx]++;  // Debug: count
        scale_ptr = ceres_vars_scale.at(region_idx);
      }
      assert(scale_ptr != nullptr);

      // Create scale-based reprojection factor with confidence
      auto *factor_scale = new Factor_ImageReprojScale(uv_meas, point_c0, params.init_vggt_scale_sigma_pix, confidence, camera_vals, R_GtoI0, params.init_vggt_scale_use_softplus, is_fisheye);

      // Set up factor parameters: [q_GtoIk, p_IkinG, lambda, q_ItoC, p_IinC]
      std::vector<double *> factor_params;
      factor_params.push_back(ceres_vars_ori.at(state_idx));
      factor_params.push_back(ceres_vars_pos.at(state_idx));
      factor_params.push_back(scale_ptr);
      factor_params.push_back(ceres_vars_calib_cam2imu_ori.at(calib_idx));
      factor_params.push_back(ceres_vars_calib_cam2imu_pos.at(calib_idx));

      // Add factor to problem
      ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
      problem.AddResidualBlock(factor_scale, loss_function, factor_params);
      num_scale_factors++;
    }
  }
  PRINT_INFO("[VGGT-INIT]: Added %d scale-based reprojection factors\n", num_scale_factors);

  // Debug: print region distribution
  if (region_n > 0 && !region_point_counts.empty()) {
    PRINT_DEBUG("[VGGT-INIT]: Region point distribution (n=%d):\n", region_n);
    for (int r = 0; r < region_n * region_n; r++) {
      int i2 = r / region_n;
      int i1 = r % region_n;
      int count = region_point_counts.count(r) ? region_point_counts[r] : 0;
      PRINT_DEBUG("  region[%d,%d]: %d points\n", i1, i2, count);
    }
  }

  assert(ceres_vars_ori.size() == ceres_vars_bias_g.size());
  assert(ceres_vars_ori.size() == ceres_vars_vel.size());
  assert(ceres_vars_ori.size() == ceres_vars_bias_a.size());
  assert(ceres_vars_ori.size() == ceres_vars_pos.size());
  auto rT5 = boost::posix_time::microsec_clock::local_time();

  // Optimize the ceres graph
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  const std::size_t num_scales = scale_layout.scale_count;
  PRINT_INFO("[VGGT-INIT]: %d iterations | %zu states, %zu scales | %d param and %d res | cost %.4e => %.4e\n",
             (int)summary.iterations.size(), map_states.size(), num_scales, summary.num_parameters,
             summary.num_residuals, summary.initial_cost, summary.final_cost);
  PRINT_DEBUG("[VGGT-INIT]: %s\n", summary.message.c_str());
  auto rT6 = boost::posix_time::microsec_clock::local_time();

  // Return if we have failed!
  timestamp = newest_cam_time;
  if (params.init_dyn_mle_max_iter != 0 && summary.termination_type != ceres::CONVERGENCE) {
    PRINT_WARNING(YELLOW "[VGGT-INIT]: opt failed: %s!\n" RESET, summary.message.c_str());
    free_state_memory();
    return false;
  }

  // Record and check final cost (average residual)
  if (summary.num_residuals > 0) {
    const double nonlinear_avg_residual = summary.final_cost / summary.num_residuals;
    PRINT_DEBUG("[VGGT-INIT]: avg_residual=%.4f\n", nonlinear_avg_residual);
    if (params.init_vggt_max_nonlinear_avg_residual > 0 && nonlinear_avg_residual > params.init_vggt_max_nonlinear_avg_residual) {
      PRINT_WARNING(YELLOW "[VGGT-INIT]: high final cost: avg_residual=%.4f > threshold=%.4f\n" RESET,
                    nonlinear_avg_residual, params.init_vggt_max_nonlinear_avg_residual);
      free_state_memory();
      return false;
    }
  }

  // Check consistency across regional scales.
  if (scale_layout.is_region() && params.init_vggt_max_scale_std > 0 && ceres_vars_scale.size() > 1) {
    // Helper function to convert the optimization parameter to its scale value.
    auto convert_scale_check = [&](double s_param) -> double {
      if (params.init_vggt_scale_use_softplus) {
        return std::log(1 + std::exp(s_param)) + 1e-5;
      }
      return s_param;
    };

    // Compute mean and std of scales
    double scale_sum = 0.0;
    std::vector<double> scale_values;
    for (size_t i = 0; i < ceres_vars_scale.size(); i++) {
      double scale_val = convert_scale_check(ceres_vars_scale[i][0]);
      scale_values.push_back(scale_val);
      scale_sum += scale_val;
    }
    double scale_mean = scale_sum / scale_values.size();
    double scale_var = 0.0;
    for (double s : scale_values) {
      scale_var += (s - scale_mean) * (s - scale_mean);
    }
    double scale_std = std::sqrt(scale_var / scale_values.size());

    if (scale_std > params.init_vggt_max_scale_std) {
      PRINT_WARNING(YELLOW "[VGGT-INIT]: scale variance too large: std=%.4f > threshold=%.4f (mean=%.4f)\n" RESET,
                    scale_std, params.init_vggt_max_scale_std, scale_mean);
      free_state_memory();
      return false;
    }
    PRINT_DEBUG("[VGGT-INIT]: scale_std=%.4f, scale_mean=%.4f (threshold=%.4f)\n",
                scale_std, scale_mean, params.init_vggt_max_scale_std);
  }

  //======================================================
  //======================================================

  // Helper function to get the IMU pose value from our ceres problem
  auto get_pose = [&](double timestamp) {
    Eigen::VectorXd state_imu = Eigen::VectorXd::Zero(16);
    for (int i = 0; i < 4; i++) {
      state_imu(0 + i) = ceres_vars_ori[map_states[timestamp]][i];
    }
    for (int i = 0; i < 3; i++) {
      state_imu(4 + i) = ceres_vars_pos[map_states[timestamp]][i];
      state_imu(7 + i) = ceres_vars_vel[map_states[timestamp]][i];
      state_imu(10 + i) = ceres_vars_bias_g[map_states[timestamp]][i];
      state_imu(13 + i) = ceres_vars_bias_a[map_states[timestamp]][i];
    }
    return state_imu;
  };

  // Our most recent state is the IMU state!
  assert(map_states.find(newest_cam_time) != map_states.end());
  if (_imu == nullptr) {
    _imu = std::make_shared<ov_type::IMU>();
  }
  Eigen::VectorXd imu_state = get_pose(newest_cam_time);
  _imu->set_value(imu_state);
  _imu->set_fej(imu_state);

  // Append our IMU clones (includes most recent)
  for (auto const &statepair : map_states) {
    Eigen::VectorXd pose = get_pose(statepair.first);
    auto clone_it = _clones_IMU.find(statepair.first);
    if (clone_it == _clones_IMU.end()) {
      auto _pose = std::make_shared<ov_type::PoseJPL>();
      _pose->set_value(pose.block(0, 0, 7, 1));
      _pose->set_fej(pose.block(0, 0, 7, 1));
      _clones_IMU.insert({statepair.first, _pose});
    } else {
      clone_it->second->set_value(pose.block(0, 0, 7, 1));
      clone_it->second->set_fej(pose.block(0, 0, 7, 1));
    }
  }

  // Extract optimized scale values
  // Helper function to convert the optimization parameter to its scale value.
  auto convert_scale = [&](double s_param) -> double {
    if (params.init_vggt_scale_use_softplus) {
      return std::log(1 + std::exp(s_param)) + 1e-5;
    }
    return s_param;
  };

  if (scale_layout.is_region()) {
    // Region mode: extract each region's scale and report the average.
    PRINT_DEBUG("[VGGT-INIT]: Optimized region scales (n=%d, %d regions):\n", region_n, region_n * region_n);
    double scale_sum = 0.0;
    for (int r = 0; r < (int)ceres_vars_scale.size(); r++) {
      double s_param = ceres_vars_scale[r][0];
      double scale_value = convert_scale(s_param);
      scale_sum += scale_value;
      // Convert 1D index back to 2D coordinates for printing
      int i2 = r / region_n;
      int i1 = r % region_n;
      PRINT_DEBUG("  region[%d,%d]: %.6f (param: %.6f)\n", i1, i2, scale_value, s_param);
    }
    double avg_scale = scale_sum / ceres_vars_scale.size();
    PRINT_INFO("[VGGT-INIT]: Region scale average = %.6f\n", avg_scale);
  } else {
    // Global mode: all frames share the same scale
    double s_param = ceres_var_global_scale[0];
    double global_scale = convert_scale(s_param);
    PRINT_INFO("[VGGT-INIT]: Optimized global scale = %.6f (param: %.6f)\n", global_scale, s_param);
  }

  // If we optimized calibration, we should also save it to our state
  if (params.init_dyn_mle_opt_calib) {
    // TODO: append our calibration states too if we are doing calibration!
    // TODO: (if we are not doing calibration do not calibrate them....)
    // TODO: std::shared_ptr<ov_type::Vec> _calib_dt_CAMtoIMU,
    // TODO: std::unordered_map<size_t, std::shared_ptr<ov_type::PoseJPL>> &_calib_IMUtoCAM,
    // TODO: std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> &_cam_intrinsics
  }

  // Recover the covariance of the optimized IMU state
  int state_index = map_states[newest_cam_time];
  if (!recover_covariance(problem, state_index, ceres_vars_ori, ceres_vars_pos, ceres_vars_vel,
                          ceres_vars_bias_g, ceres_vars_bias_a, map_calib_cam2imu,
                          _imu, covariance, order)) {
    free_state_memory();
    return false;
  }

  // Set our position to be zero
  Eigen::MatrixXd x = _imu->value();
  x.block(4, 0, 3, 1).setZero();
  _imu->set_value(x);
  _imu->set_fej(x);

  // Wall-clock diagnostics; this is distinct from the sensor-time initialization duration.
  auto rT7 = boost::posix_time::microsec_clock::local_time();
  double t_prelim = (rT2 - rT1).total_microseconds() * 1e-6;
  double t_prelim_no_rtt = (vggt_inference_time_ > 0) ? (t_prelim - vggt_inference_time_) : t_prelim;
  double t_linsys_setup = (rT3 - rT2).total_microseconds() * 1e-6;
  double t_linsys = (rT4 - rT3).total_microseconds() * 1e-6;
  double t_ceres_setup = (rT5 - rT4).total_microseconds() * 1e-6;
  double t_ceres_opt = (rT6 - rT5).total_microseconds() * 1e-6;
  double t_cov = (rT7 - rT6).total_microseconds() * 1e-6;
  double t_total = (rT7 - rT1).total_microseconds() * 1e-6;
  PRINT_INFO(CYAN "[WALL-TIME] initializer_compute_total=%.3fs | inference_rtt=%.3fs prelim_no_rtt=%.3fs linsys_setup=%.3fs "
                  "linsys=%.3fs ceres_setup=%.3fs ceres_opt=%.3fs cov=%.3fs\n" RESET,
             t_total, vggt_inference_time_, t_prelim_no_rtt, t_linsys_setup, t_linsys, t_ceres_setup, t_ceres_opt, t_cov);

  const double nonlinear_timestamp = map_states.begin()->first;
  const Eigen::VectorXd nonlinear = get_pose(nonlinear_timestamp);
  diagnostics.nonlinear_state =
      make_initialization_state(nonlinear_timestamp, nonlinear.block<4, 1>(0, 0), nonlinear.block<3, 1>(4, 0),
                                nonlinear.block<3, 1>(7, 0));
  diagnostics.has_nonlinear_state = true;

  free_state_memory();
  return true;
}

int VGGTInitializer::filter_pointcloud_by_confidence(cv::Mat &pointcloud, float threshold) {
  // Validate input
  if (pointcloud.empty() || pointcloud.type() != CV_32FC4) {
    PRINT_ERROR(RED "[VGGT-INIT]: Invalid pointcloud for filtering (must be CV_32FC4)\n" RESET);
    return 0;
  }

  int filtered_count = 0;
  const int rows = pointcloud.rows;
  const int cols = pointcloud.cols;

  // Loop through all pixels
  for (int v = 0; v < rows; ++v) {
    for (int u = 0; u < cols; ++u) {
      cv::Vec4f &point = pointcloud.at<cv::Vec4f>(v, u);

      const bool invalid_geometry = !std::isfinite(point[0]) || !std::isfinite(point[1]) ||
                                    !std::isfinite(point[2]) || !std::isfinite(point[3]) ||
                                    point[0] * point[0] + point[1] * point[1] + point[2] * point[2] <= 1e-12f;

      // Confidence is also the validity flag used by downstream samplers.
      if (invalid_geometry || point[3] <= 0.0f || point[3] < threshold) {
        // Mark as invalid by setting 4th channel to 0
        point[3] = 0.0f;
        filtered_count++;
      }
      // If confidence >= threshold, keep original confidence value as validity flag
    }
  }

  return filtered_count;
}

std::pair<int, int> VGGTInitializer::filter_loaded_pointclouds() {
  int total_filtered = 0;
  int frames_thresholded = 0;
  for (auto &entry : vggt_point3d_data) {
    const auto confidence_it = vggt_pointcloud_has_confidence_.find(entry.first);
    const bool has_confidence =
        confidence_it != vggt_pointcloud_has_confidence_.end() && confidence_it->second;
    const bool apply_threshold = use_confidence_filter_ && has_confidence;
    const float threshold =
        apply_threshold ? confidence_threshold_ : -std::numeric_limits<float>::infinity();
    total_filtered += filter_pointcloud_by_confidence(entry.second, threshold);
    frames_thresholded += apply_threshold ? 1 : 0;
  }
  return {total_filtered, frames_thresholded};
}

bool VGGTInitializer::get_point3d(double timestamp, double u, double v, Eigen::Vector3f &point3d, float *confidence) {
  if (vggt_point3d_data.empty()) {
    return false;
  }

  // Find the closest timestamp in the data
  auto it = vggt_point3d_data.lower_bound(timestamp);
  double closest_timestamp;

  if (it == vggt_point3d_data.end()) {
    it = std::prev(vggt_point3d_data.end());
    closest_timestamp = it->first;
  } else if (it != vggt_point3d_data.begin()) {
    auto prev_it = std::prev(it);
    if (std::abs(timestamp - prev_it->first) < std::abs(timestamp - it->first))
      it = prev_it;
    closest_timestamp = it->first;
  } else {
    closest_timestamp = it->first;
  }

  // Check if the time difference is too large (threshold: 0.1 seconds)
  const double time_threshold = 0.1;
  if (std::abs(timestamp - closest_timestamp) > time_threshold) {
    return false;
  }

  const cv::Mat &point3d_image = it->second;

  // Check if input is effectively an integer (no interpolation needed)
  int u0 = (int)std::floor(u);
  int v0 = (int)std::floor(v);
  double du = u - u0;
  double dv = v - v0;

  // If input is integer or very close to integer, use direct lookup
  if (du < 1e-6 && dv < 1e-6) {
    if (u0 < 0 || v0 < 0 || u0 >= point3d_image.cols || v0 >= point3d_image.rows) {
      return false;
    }
    cv::Vec4f pt4d = point3d_image.at<cv::Vec4f>(v0, u0);
    if (pt4d[3] == 0.0f) {
      return false;
    }
    point3d = Eigen::Vector3f(pt4d[0], pt4d[1], pt4d[2]);
    if (confidence != nullptr) {
      *confidence = pt4d[3];
    }
    return true;
  }

  // Bilinear interpolation for sub-pixel coordinates
  int u1 = u0 + 1;
  int v1 = v0 + 1;

  // Boundary check for all 4 corners
  if (u0 < 0 || v0 < 0 || u1 >= point3d_image.cols || v1 >= point3d_image.rows) {
    return false;
  }

  // Get the 4 corner points [x, y, z, confidence]
  cv::Vec4f pt00 = point3d_image.at<cv::Vec4f>(v0, u0);
  cv::Vec4f pt01 = point3d_image.at<cv::Vec4f>(v0, u1);
  cv::Vec4f pt10 = point3d_image.at<cv::Vec4f>(v1, u0);
  cv::Vec4f pt11 = point3d_image.at<cv::Vec4f>(v1, u1);

  // Check if all 4 corners are valid (confidence > 0)
  bool valid00 = (pt00[3] > 0.0f);
  bool valid01 = (pt01[3] > 0.0f);
  bool valid10 = (pt10[3] > 0.0f);
  bool valid11 = (pt11[3] > 0.0f);

  if (valid00 && valid01 && valid10 && valid11) {
    // All corners valid: bilinear interpolation
    double w00 = (1.0 - du) * (1.0 - dv);
    double w01 = du * (1.0 - dv);
    double w10 = (1.0 - du) * dv;
    double w11 = du * dv;

    point3d = Eigen::Vector3f(
        w00 * pt00[0] + w01 * pt01[0] + w10 * pt10[0] + w11 * pt11[0],
        w00 * pt00[1] + w01 * pt01[1] + w10 * pt10[1] + w11 * pt11[1],
        w00 * pt00[2] + w01 * pt01[2] + w10 * pt10[2] + w11 * pt11[2]);

    if (confidence != nullptr) {
      *confidence = w00 * pt00[3] + w01 * pt01[3] + w10 * pt10[3] + w11 * pt11[3];
    }
    return true;
  }

  // Fallback: use nearest valid neighbor
  int u_nearest = (int)std::round(u);
  int v_nearest = (int)std::round(v);
  u_nearest = std::max(0, std::min(u_nearest, point3d_image.cols - 1));
  v_nearest = std::max(0, std::min(v_nearest, point3d_image.rows - 1));

  cv::Vec4f pt_nearest = point3d_image.at<cv::Vec4f>(v_nearest, u_nearest);
  if (pt_nearest[3] == 0.0f) {
    return false;
  }

  point3d = Eigen::Vector3f(pt_nearest[0], pt_nearest[1], pt_nearest[2]);
  if (confidence != nullptr) {
    *confidence = pt_nearest[3];
  }
  return true;
}
int VGGTInitializer::build_linear_system_from_points(
    double timestamp,
    const std::vector<Eigen::Vector2d> &frame_pixels,
    const std::vector<Eigen::Vector3d> &frame_points,
    const std::vector<float> &frame_confidences,
    size_t cam_id,
    const Eigen::Matrix3d &R_ItoC,
    const Eigen::Vector3d &p_IinC,
    const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_I0toIi,
    int system_size,
    Eigen::MatrixXd &A,
    Eigen::VectorXd &b) {

  // Validate input
  if (frame_pixels.size() != frame_points.size() || frame_pixels.size() != frame_confidences.size()) {
    PRINT_ERROR(RED "[VGGT-INIT]: frame_pixels, frame_points, and frame_confidences size mismatch!\n" RESET);
    return 0;
  }

  if (frame_pixels.empty()) {
    return 0;
  }

  int measurements_size = frame_pixels.size() * 2;  // Each point contributes 2 measurements
  A = Eigen::MatrixXd::Zero(measurements_size, system_size);
  b = Eigen::VectorXd::Zero(measurements_size);

  double DT = 0.0;
  Eigen::MatrixXd R_I0toIk = Eigen::MatrixXd::Identity(3, 3);
  Eigen::MatrixXd alpha_I0toIk = Eigen::MatrixXd::Zero(3, 1);
  if (map_camera_cpi_I0toIi.count(timestamp) && map_camera_cpi_I0toIi.at(timestamp) != nullptr) {
    DT          = map_camera_cpi_I0toIi.at(timestamp)->DT;
    R_I0toIk    = map_camera_cpi_I0toIi.at(timestamp)->R_k2tau;
    alpha_I0toIk= map_camera_cpi_I0toIi.at(timestamp)->alpha_tau;
  }

  int index_meas = 0;
  for (size_t i = 0; i < frame_pixels.size(); i++) {
    auto uv = frame_pixels[i];
    auto point3d_c0 = frame_points[i];

    // The point cloud and tracked pixels share the estimator's processed image grid.
    Eigen::Vector2d uv_norm = params.camera_intrinsics.at(cam_id)->undistort_d(uv);

    // Create the linear system based on the feature reprojection
    // [ 1 0 -u ] p_FinCi = [ 0 ]
    // [ 0 1 -v ]           [ 0 ]
    // where
    // p_FinCi = R_ItoC * R_I0toIk * (p_FinI0 - p_IiinI0) + p_IinC
    //         = R_ItoC * R_I0toIk * (p_FinI0 - v_I0inI0 * dt - 0.5 * grav_inI0 * dt^2 - alpha) + p_IinC
    // Substituting in p_FinI0 = s * R_CtoI * [u, v, 1] - R_ItoC^T*p_IinC
    // AKA p_FinI0 = s * bearing - R_ItoC^T*p_IinC
    Eigen::MatrixXd H_proj = Eigen::MatrixXd::Zero(2, 3);
    H_proj << 1, 0, -uv_norm(0),
              0, 1, -uv_norm(1);
    Eigen::MatrixXd Y  = H_proj * R_ItoC * R_I0toIk;
    Eigen::MatrixXd H_i = Eigen::MatrixXd::Zero(2, system_size);
    Eigen::MatrixXd b_i = Y * alpha_I0toIk - H_proj * p_IinC + Y * R_ItoC.transpose() * p_IinC;
    H_i.block(0, 0, 2, 1) = Y * R_ItoC.transpose() * point3d_c0; // scale 2x1
    H_i.block(0, size_pointcloud_scale, 2, 3) = -DT * Y; // velocity 2x3
    H_i.block(0, size_pointcloud_scale + size_velocity, 2, 3) = 0.5 * DT * DT * Y; // gravity 2x3

    // Apply confidence weighting (sqrt for information matrix weighting)
    float weight = std::sqrt(frame_confidences[i]);
    A.block(index_meas, 0, 2, A.cols()) = H_i * weight;
    b.block(index_meas, 0, 2, 1)        = b_i * weight;
    index_meas += 2;
  }

  return measurements_size;
}

bool VGGTInitializer::LLSsolver_with_gravity_constraint(
    const Eigen::MatrixXd &A,
    const Eigen::VectorXd &b,
    Eigen::VectorXd &x_hat,
    Eigen::Vector3d &v_I0inI0,
    Eigen::Vector3d &gravity_inI0) {

  // ======================================================
  // ======================================================

  // Solve the linear system without constraint
  // Eigen::MatrixXd AtA = A.transpose() * A;
  // Eigen::MatrixXd Atb = A.transpose() * b;
  // Eigen::MatrixXd x_hat = AtA.colPivHouseholderQr().solve(Atb);
  const int size_gravity = 3;
  Eigen::MatrixXd A1 = A.block(0, 0, A.rows(), A.cols() - size_gravity);
  Eigen::MatrixXd A2 = A.block(0, A.cols() - size_gravity, A.rows(), size_gravity);
  Eigen::MatrixXd A1A1_inv =
      (A1.transpose() * A1).llt().solve(Eigen::MatrixXd::Identity(A1.cols(), A1.cols()));
  Eigen::MatrixXd Temp = A2.transpose() *
                         (Eigen::MatrixXd::Identity(A1.rows(), A1.rows()) - A1 * A1A1_inv * A1.transpose());
  Eigen::MatrixXd D = Temp * A2;
  Eigen::MatrixXd d = Temp * b;

  Eigen::Matrix<double, 7, 1> coeff = InitializerHelper::compute_dongsi_coeff(D, d, params.gravity_mag);
  // Create companion matrix of our polynomial
  // https://en.wikipedia.org/wiki/Companion_matrix
  assert(coeff(0) == 1);
  Eigen::Matrix<double, 6, 6> companion_matrix = Eigen::Matrix<double, 6, 6>::Zero(coeff.rows() - 1, coeff.rows() - 1);
  companion_matrix.diagonal(-1).setOnes();
  companion_matrix.col(companion_matrix.cols() - 1) = -coeff.reverse().head(coeff.rows() - 1);
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd0(companion_matrix);
  if (svd0.rank() != companion_matrix.rows()) {
    PRINT_ERROR(RED "[VGGT-INIT]: eigenvalue decomposition not full rank!!\n" RESET);
    return false;
  }
  // Find its eigenvalues (can be complex)
  Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> solver(companion_matrix, false);
  if (solver.info() != Eigen::Success) {
    PRINT_ERROR(RED "[VGGT-INIT]: failed to compute the eigenvalue decomposition!!\n" RESET);
    return false;
  }
  // Find the smallest real eigenvalue
  // NOTE: we find the one that gives us minimal constraint cost
  // NOTE: not sure if the best, but one that gives the correct mag should be good?
  bool lambda_found = false;
  double lambda_min = -1, cost_min = INFINITY;
  Eigen::MatrixXd I_dd = Eigen::MatrixXd::Identity(D.rows(), D.rows());
  for (int i = 0; i < solver.eigenvalues().size(); i++) {
    auto val = solver.eigenvalues()(i);
    if (val.imag() == 0) {
      double lambda = val.real();
      Eigen::MatrixXd D_lambdaI_inv = (D - lambda * I_dd).llt().solve(I_dd);
      Eigen::VectorXd state_grav = D_lambdaI_inv * d;
      double cost = std::abs(state_grav.norm() - params.gravity_mag);
      if (!lambda_found || cost < cost_min) {
        lambda_found = true; lambda_min = lambda; cost_min = cost;
      }
    }
  }
  if (!lambda_found) {
    PRINT_ERROR(RED "[VGGT-INIT]: failed to find a real eigenvalue!!!\n" RESET);
    return false;
  }

  // Recover our gravity from the constraint!
  // Eigen::MatrixXd D_lambdaI_inv = (D - lambda_min * I_dd).inverse();
  Eigen::MatrixXd D_lambdaI_inv = (D - lambda_min * I_dd).llt().solve(I_dd);
  Eigen::VectorXd state_grav = D_lambdaI_inv * d;
  Eigen::VectorXd state_scale_vel =
      -A1A1_inv * A1.transpose() * A2 * state_grav + A1A1_inv * A1.transpose() * b;
  // Overwrite our state: [scale, velocity, gravity]
  x_hat.setZero();
  x_hat.block(0, 0, size_pointcloud_scale + size_velocity, 1) = state_scale_vel;
  x_hat.block(size_pointcloud_scale + size_velocity, 0, 3, 1) = state_grav;
  v_I0inI0 = x_hat.block(size_pointcloud_scale, 0, 3, 1);
  gravity_inI0 = state_grav;

  return true;
}

bool VGGTInitializer::sample_pointcloud(
    double timestamp,
    std::vector<Eigen::Vector2d> &pixels_uv,
    std::vector<Eigen::Vector3d> &points_3d,
    std::vector<float> &confidences) {

  pixels_uv.clear();
  points_3d.clear();
  confidences.clear();

  if (vggt_point3d_data.empty()) {
    return false;
  }

  auto it = vggt_point3d_data.lower_bound(timestamp);
  if (it == vggt_point3d_data.end()) {
    it = std::prev(vggt_point3d_data.end());
  } else if (it != vggt_point3d_data.begin()) {
    auto prev_it = std::prev(it);
    if (std::abs(timestamp - prev_it->first) < std::abs(timestamp - it->first)) {
      it = prev_it;
    }
  }

  const double closest_timestamp = it->first;
  constexpr double kTimeThreshold = 0.1;
  if (std::abs(timestamp - closest_timestamp) > kTimeThreshold) {
    return false;
  }

  const cv::Mat &pointcloud = it->second;
  const int rows = pointcloud.rows;
  const int cols = pointcloud.cols;
  const int desired_samples = params.init_vggt_points_per_frame;

  const int grid_cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(desired_samples))));
  const int grid_rows = static_cast<int>(std::ceil(static_cast<double>(desired_samples) / grid_cols));

  const int stride_u = static_cast<int>(std::round(static_cast<double>(cols) / grid_cols));
  const int stride_v = static_cast<int>(std::round(static_cast<double>(rows) / grid_rows));

  for (int v = stride_v / 2; v < rows && static_cast<int>(pixels_uv.size()) < desired_samples; v += stride_v) {
    for (int u = stride_u / 2; u < cols && static_cast<int>(pixels_uv.size()) < desired_samples; u += stride_u) {
      const int patch_u_start = std::max(0, u - stride_u / 2);
      const int patch_u_end = std::min(cols, u + stride_u / 2 + 1);
      const int patch_v_start = std::max(0, v - stride_v / 2);
      const int patch_v_end = std::min(rows, v + stride_v / 2 + 1);

      Eigen::Vector3f best_point3d;
      float best_confidence = -std::numeric_limits<float>::infinity();
      int best_u = -1, best_v = -1;
      bool found = false;

      for (int pv = patch_v_start; pv < patch_v_end; ++pv) {
        for (int pu = patch_u_start; pu < patch_u_end; ++pu) {
          Eigen::Vector3f point3d_f;
          float confidence_f = 0.0f;
          if (get_point3d(timestamp, pu, pv, point3d_f, &confidence_f)) {
            if (confidence_f > best_confidence) {
              best_confidence = confidence_f;
              best_point3d = point3d_f;
              best_u = pu;
              best_v = pv;
              found = true;
            }
          }
        }
      }

      if (!found) {
        continue;
      }

      pixels_uv.emplace_back(static_cast<double>(best_u), static_cast<double>(best_v));
      points_3d.emplace_back(best_point3d.cast<double>());
      confidences.emplace_back(best_confidence);
    }
  }

  return !pixels_uv.empty();
}

bool VGGTInitializer::select_features_and_keyframes(const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
                                                  double newest_cam_time, bool &have_stereo,
                                                  std::map<size_t, int> &map_features_num_meas, int &measurements_size,
                                                  double &oldest_camera_time, std::map<double, bool> &map_camera_times,
                                                  std::map<size_t, bool> &map_camera_ids) {

  // Step 1: Collect all available camera timestamps from features
  std::set<double> all_times_set;
  for (auto const &feat : features) {
    for (auto const &camtime : feat.second->timestamps) {
      map_camera_ids[camtime.first] = true;
      for (double time : camtime.second) {
        all_times_set.insert(time);
      }
    }
    map_features_num_meas[feat.first] = 0;
    for (auto const &camtime : feat.second->timestamps) {
      map_features_num_meas[feat.first] += (int)camtime.second.size();
    }
  }

  // Convert to sorted vector
  std::vector<double> all_times(all_times_set.begin(), all_times_set.end());
  std::sort(all_times.begin(), all_times.end());

  // Return if we do not have enough frames
  if ((int)all_times.size() < params.init_dyn_num_pose) {
    return false;
  }

  // Step 2: Uniformly select init_dyn_num_pose frames (always include first and last)
  std::set<double> selected_times;
  selected_times.insert(all_times.front()); // oldest
  selected_times.insert(all_times.back());  // newest

  // Uniformly select remaining frames
  int remaining = params.init_dyn_num_pose - 2;
  if (remaining > 0 && all_times.size() > 2) {
    double step = (double)(all_times.size() - 1) / (params.init_dyn_num_pose - 1);
    for (int i = 1; i <= remaining; i++) {
      int idx = (int)std::round(i * step);
      idx = std::min(idx, (int)all_times.size() - 2);
      idx = std::max(idx, 1);
      selected_times.insert(all_times[idx]);
    }
  }

  // Step 3: Build output map_camera_times
  for (double t : selected_times) {
    map_camera_times[t] = true;
  }
  oldest_camera_time = all_times.front();

  // Check for stereo
  if (map_camera_ids.size() > 1) {
    have_stereo = true;
  }

  measurements_size = 2 * params.init_vggt_points_per_frame * params.init_vggt_use_pointcloud * (int)map_camera_times.size();

  return true;
}

bool VGGTInitializer::select_features_and_keyframes_firstanchored(const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
                                                             double newest_cam_time, bool &have_stereo,
                                                             std::map<size_t, int> &map_features_num_meas, int &measurements_size,
                                                             double &oldest_camera_time, std::map<double, bool> &map_camera_times,
                                                             std::map<size_t, bool> &map_camera_ids) {
  // This keeps feature anchors tied to the first point-cloud frame.
  const int min_num_meas_to_optimize = (int)params.init_window_time;
  const int min_valid_features = 8;
  count_valid_features = 0;

  // Initialize camera times with newest frame
  map_camera_times[newest_cam_time] = true;

  // First, find the first frame timestamp
  double first_frame_time = INFINITY;
  for (auto const &feat : features) {
    for (auto const &camtime : feat.second->timestamps) {
      for (double time : camtime.second) {
        first_frame_time = std::min(first_frame_time, time);
      }
    }
  }

  if (first_frame_time == INFINITY) {
    return false; // No features found
  }

  oldest_camera_time = first_frame_time;

  double pose_dt_avg = params.init_window_time / (double)(params.init_dyn_num_pose + 1);
  for (auto const &feat : features) {
    // Check if this feature is visible in the first frame
    bool visible_in_first_frame = false;
    for (auto const &camtime : feat.second->timestamps) {
      for (double time : camtime.second) {
        if (std::abs(time - first_frame_time) < 1e-6) {  // tolerance for time comparison
          visible_in_first_frame = true;
          break;
        }
      }
      if (visible_in_first_frame) break;
    }

    // Skip features not visible in first frame
    if (!visible_in_first_frame) {
      continue;
    }

    // Loop through each timestamp and make sure it is a valid pose
    std::vector<double> times;
    std::map<size_t, bool> camids;
    for (auto const &camtime : feat.second->timestamps) {
      for (double time : camtime.second) {
        double time_dt = INFINITY;
        for (auto const &tmp : map_camera_times) {
          time_dt = std::min(time_dt, std::abs(time - tmp.first));
        }
        for (auto const &tmp : times) {
          time_dt = std::min(time_dt, std::abs(time - tmp));
        }
        // either this pose is a new one at the desired frequency
        // or it is a timestamp that we already have, thus can use for free
        if (time_dt >= pose_dt_avg || time_dt == 0.0) {
          times.push_back(time);
          camids[camtime.first] = true;
        }
      }
    }

    // This isn't a feature we should use if there are not enough measurements
    map_features_num_meas[feat.first] = (int)times.size();
    if (map_features_num_meas[feat.first] < min_num_meas_to_optimize)
      continue;

    // If we have enough measurements we should append this feature!
    for (auto const &tmp : times) {
      map_camera_times[tmp] = true;
      measurements_size += 2;
    }
    for (auto const &tmp : camids) {
      map_camera_ids[tmp.first] = true;
    }
    if (camids.size() > 1) {
      have_stereo = true;
    }
    count_valid_features++;
  }

  if ((int)map_camera_times.size() < params.init_dyn_num_pose) {
    PRINT_WARNING(RED "[VGGT-INIT]: only %zu first-frame features of required %d!!\n" RESET, count_valid_features, min_valid_features);
    return false;
  }
  if (count_valid_features < min_valid_features) {
    PRINT_WARNING(RED "[VGGT-INIT]: only %zu first-frame features of required %d!!\n" RESET, count_valid_features, min_valid_features);
    return false;
  }

  // Truncate to exactly init_dyn_num_pose frames if we have more
  // Keep oldest and newest, select uniformly from the rest
  if ((int)map_camera_times.size() > params.init_dyn_num_pose) {
    std::vector<double> all_times;
    for (const auto &t : map_camera_times) {
      all_times.push_back(t.first);
    }
    std::sort(all_times.begin(), all_times.end());

    // Select init_dyn_num_pose times uniformly (always include first and last)
    std::set<double> selected_times;
    selected_times.insert(all_times.front()); // oldest
    selected_times.insert(all_times.back());  // newest

    // Uniformly select remaining frames
    int remaining = params.init_dyn_num_pose - 2;
    if (remaining > 0 && all_times.size() > 2) {
      double step = (double)(all_times.size() - 1) / (params.init_dyn_num_pose - 1);
      for (int i = 1; i <= remaining; i++) {
        int idx = (int)std::round(i * step);
        idx = std::min(idx, (int)all_times.size() - 2);
        idx = std::max(idx, 1);
        selected_times.insert(all_times[idx]);
      }
    }

    // Rebuild map_camera_times with only selected times
    std::map<double, bool> new_map_camera_times;
    for (double t : selected_times) {
      new_map_camera_times[t] = true;
    }
    map_camera_times = new_map_camera_times;
    oldest_camera_time = all_times.front();
  }

  return true;
}

bool VGGTInitializer::get_pFinC0(const std::shared_ptr<ov_core::Feature> &feat,
                                Eigen::Vector3d &features_point3d_c0_) {
  // The model pointcloud is produced for camera 0. Never associate a
  // measurement from another camera with its pixel grid.
  const auto timestamps_it = feat->timestamps.find(0);
  const auto uvs_it = feat->uvs.find(0);
  if (timestamps_it == feat->timestamps.end() || uvs_it == feat->uvs.end() ||
      timestamps_it->second.empty() || timestamps_it->second.size() != uvs_it->second.size()) {
    return false;
  }

  const auto oldest_it = std::min_element(timestamps_it->second.begin(), timestamps_it->second.end());
  const size_t index = static_cast<size_t>(std::distance(timestamps_it->second.begin(), oldest_it));
  if (uvs_it->second.at(index).size() < 2) {
    return false;
  }
  const Eigen::Vector2f uv = uvs_it->second.at(index).block<2, 1>(0, 0);
  Eigen::Vector3f point3d;
  if (!get_point3d(*oldest_it, uv(0), uv(1), point3d) || !point3d.allFinite()) {
    return false;
  }
  features_point3d_c0_ = point3d.cast<double>();
  return true;
}


bool VGGTInitializer::recover_covariance(ceres::Problem &problem, int state_index,
                                        const std::vector<double *> &ceres_vars_ori,
                                        const std::vector<double *> &ceres_vars_pos,
                                        const std::vector<double *> &ceres_vars_vel,
                                        const std::vector<double *> &ceres_vars_bias_g,
                                        const std::vector<double *> &ceres_vars_bias_a,
                                        const std::map<size_t, int> &map_calib_cam2imu,
                                        std::shared_ptr<ov_type::IMU> &_imu,
                                        Eigen::MatrixXd &covariance,
                                        std::vector<std::shared_ptr<ov_type::Type>> &order) {

  // Prepare covariance blocks for the IMU state
  std::vector<std::pair<const double *, const double *>> covariance_blocks;

  // diagonals
  covariance_blocks.push_back(std::make_pair(ceres_vars_ori[state_index], ceres_vars_ori[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_pos[state_index], ceres_vars_pos[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_vel[state_index], ceres_vars_vel[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_bias_g[state_index], ceres_vars_bias_g[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_bias_a[state_index], ceres_vars_bias_a[state_index]));

  // orientation cross-terms
  covariance_blocks.push_back(std::make_pair(ceres_vars_ori[state_index], ceres_vars_pos[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_ori[state_index], ceres_vars_vel[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_ori[state_index], ceres_vars_bias_g[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_ori[state_index], ceres_vars_bias_a[state_index]));

  // position cross-terms
  covariance_blocks.push_back(std::make_pair(ceres_vars_pos[state_index], ceres_vars_vel[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_pos[state_index], ceres_vars_bias_g[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_pos[state_index], ceres_vars_bias_a[state_index]));

  // velocity cross-terms
  covariance_blocks.push_back(std::make_pair(ceres_vars_vel[state_index], ceres_vars_bias_g[state_index]));
  covariance_blocks.push_back(std::make_pair(ceres_vars_vel[state_index], ceres_vars_bias_a[state_index]));

  // bias_g cross-term
  covariance_blocks.push_back(std::make_pair(ceres_vars_bias_g[state_index], ceres_vars_bias_a[state_index]));

  // Compute covariance using Ceres
  ceres::Covariance::Options options_cov;
  options_cov.null_space_rank = (!params.init_dyn_mle_opt_calib) * ((int)map_calib_cam2imu.size() * (6 + 8));
  options_cov.min_reciprocal_condition_number = params.init_dyn_min_rec_cond;
  options_cov.apply_loss_function = true;
  options_cov.num_threads = params.init_dyn_mle_max_threads;

  ceres::Covariance problem_cov(options_cov);
  bool success = problem_cov.Compute(covariance_blocks, &problem);
  if (!success) {
    PRINT_WARNING(YELLOW "[init]: covariance recovery failed...\n" RESET);
    return false;
  }

  // Construct the covariance matrix
  order.clear();
  order.push_back(_imu);
  covariance = Eigen::MatrixXd::Zero(_imu->size(), _imu->size());
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> covtmp = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

  // Extract block diagonal elements
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_ori[state_index], ceres_vars_ori[state_index], covtmp.data()));
  covariance.block(0, 0, 3, 3) = covtmp.eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_pos[state_index], ceres_vars_pos[state_index], covtmp.data()));
  covariance.block(3, 3, 3, 3) = covtmp.eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_vel[state_index], ceres_vars_vel[state_index], covtmp.data()));
  covariance.block(6, 6, 3, 3) = covtmp.eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_bias_g[state_index], ceres_vars_bias_g[state_index], covtmp.data()));
  covariance.block(9, 9, 3, 3) = covtmp.eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_bias_a[state_index], ceres_vars_bias_a[state_index], covtmp.data()));
  covariance.block(12, 12, 3, 3) = covtmp.eval();

  // Extract orientation cross-terms
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_ori[state_index], ceres_vars_pos[state_index], covtmp.data()));
  covariance.block(0, 3, 3, 3) = covtmp.eval();
  covariance.block(3, 0, 3, 3) = covtmp.transpose();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_ori[state_index], ceres_vars_vel[state_index], covtmp.data()));
  covariance.block(0, 6, 3, 3) = covtmp.eval();
  covariance.block(6, 0, 3, 3) = covtmp.transpose().eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_ori[state_index], ceres_vars_bias_g[state_index], covtmp.data()));
  covariance.block(0, 9, 3, 3) = covtmp.eval();
  covariance.block(9, 0, 3, 3) = covtmp.transpose().eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_ori[state_index], ceres_vars_bias_a[state_index], covtmp.data()));
  covariance.block(0, 12, 3, 3) = covtmp.eval();
  covariance.block(12, 0, 3, 3) = covtmp.transpose().eval();

  // Extract position cross-terms
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_pos[state_index], ceres_vars_vel[state_index], covtmp.data()));
  covariance.block(3, 6, 3, 3) = covtmp.eval();
  covariance.block(6, 3, 3, 3) = covtmp.transpose().eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_pos[state_index], ceres_vars_bias_g[state_index], covtmp.data()));
  covariance.block(3, 9, 3, 3) = covtmp.eval();
  covariance.block(9, 3, 3, 3) = covtmp.transpose().eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_pos[state_index], ceres_vars_bias_a[state_index], covtmp.data()));
  covariance.block(3, 12, 3, 3) = covtmp.eval();
  covariance.block(12, 3, 3, 3) = covtmp.transpose().eval();

  // Extract velocity cross-terms
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_vel[state_index], ceres_vars_bias_g[state_index], covtmp.data()));
  covariance.block(6, 9, 3, 3) = covtmp.eval();
  covariance.block(9, 6, 3, 3) = covtmp.transpose().eval();
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_vel[state_index], ceres_vars_bias_a[state_index], covtmp.data()));
  covariance.block(6, 12, 3, 3) = covtmp.eval();
  covariance.block(12, 6, 3, 3) = covtmp.transpose().eval();

  // Extract bias_g cross-term
  CHECK(problem_cov.GetCovarianceBlockInTangentSpace(ceres_vars_bias_g[state_index], ceres_vars_bias_a[state_index], covtmp.data()));
  covariance.block(9, 12, 3, 3) = covtmp.eval();
  covariance.block(12, 9, 3, 3) = covtmp.transpose().eval();

  // Inflate covariance as needed
  covariance.block(0, 0, 3, 3) *= params.init_dyn_inflation_orientation;
  covariance.block(6, 6, 3, 3) *= params.init_dyn_inflation_velocity;
  covariance.block(9, 9, 3, 3) *= params.init_dyn_inflation_bias_gyro;
  covariance.block(12, 12, 3, 3) *= params.init_dyn_inflation_bias_accel;

  // Make symmetric and print debug info
  covariance = 0.5 * (covariance + covariance.transpose());
  Eigen::Vector3d sigmas_vel = covariance.block(6, 6, 3, 3).diagonal().transpose().cwiseSqrt();
  Eigen::Vector3d sigmas_bg = covariance.block(9, 9, 3, 3).diagonal().transpose().cwiseSqrt();
  Eigen::Vector3d sigmas_ba = covariance.block(12, 12, 3, 3).diagonal().transpose().cwiseSqrt();
  PRINT_DEBUG("[init]: vel priors = %.3f, %.3f, %.3f\n", sigmas_vel(0), sigmas_vel(1), sigmas_vel(2));
  PRINT_DEBUG("[init]: bg priors = %.3f, %.3f, %.3f\n", sigmas_bg(0), sigmas_bg(1), sigmas_bg(2));
  PRINT_DEBUG("[init]: ba priors = %.3f, %.3f, %.3f\n", sigmas_ba(0), sigmas_ba(1), sigmas_ba(2));

  return true;
}
