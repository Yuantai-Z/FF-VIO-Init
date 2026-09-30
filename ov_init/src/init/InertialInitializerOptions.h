/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 * Modifications Copyright (C) 2025-2026 Yuantai-Z
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

#ifndef OV_INIT_INERTIALINITIALIZEROPTIONS_H
#define OV_INIT_INERTIALINITIALIZEROPTIONS_H

#include <Eigen/Eigen>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "feat/FeatureInitializerOptions.h"
#include "track/TrackBase.h"
#include "utils/colors.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "vggt/NonlinearMode.h"
#include "vggt/ScaleLayout.h"
#include "utils/quat_ops.h"
#include "utils/vggt_preprocessing.h"

namespace ov_init {

/**
 * @brief Initialization method selection.
 *
 * YAML key: init_mode (case-insensitive string). Accepted values:
 *   - "static"        : StaticInitializer (acceleration-disparity gated)
 *   - "dynamic"       : DynamicInitializer (MLE / Ceres)
 *   - "vggt_pt"       : VGGTInitializer using predicted 3D points
 */
enum class InitMode {
  STATIC = 0,
  DYNAMIC = 1,
  VGGT_PT = 2,
};

inline bool init_mode_is_vggt(InitMode m) { return m == InitMode::VGGT_PT; }

inline std::string init_mode_to_string(InitMode m) {
  switch (m) {
    case InitMode::STATIC: return "static";
    case InitMode::DYNAMIC: return "dynamic";
    case InitMode::VGGT_PT: return "vggt_pt";
  }
  return "static";
}

inline InitMode string_to_init_mode(const std::string &s) {
  std::string lower;
  lower.reserve(s.size());
  for (char c : s) lower.push_back(std::tolower(static_cast<unsigned char>(c)));
  if (lower == "static") return InitMode::STATIC;
  if (lower == "dynamic" || lower == "dyn") return InitMode::DYNAMIC;
  if (lower == "vggt_pt" || lower == "vggt-pt" || lower == "vggt_point" || lower == "vggt_point3d" ||
      lower == "vggt_pointcloud") return InitMode::VGGT_PT;
  PRINT_ERROR(RED "[init]: unknown init_mode '%s' (expected static, dynamic, or vggt_pt)\n" RESET, s.c_str());
  std::exit(EXIT_FAILURE);
}

/**
 * @brief Struct which stores all options needed for state estimation.
 *
 * This is broken into initializer, tracker, noise, and state options.
 * If you are going to add a parameter here you will need to add it to the parsers.
 * You will also need to add it to the print statement at the bottom of each.
 */
struct InertialInitializerOptions {

  /**
   * @brief Load and print all initializer parameters.
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    print_and_load_initializer(parser);
    print_and_load_noise(parser);
    print_and_load_state(parser);
  }

  // INITIALIZATION ============================

  /// Amount of time we will initialize over (seconds)
  double init_window_time = 1.0;

  /// Variance threshold on our acceleration to be classified as moving
  double init_imu_thresh = 1.0;

  /// Max disparity we will consider the unit to be stationary
  double init_max_disparity = 1.0;

  /// Number of features we should try to track
  int init_max_features = 50;

  /// Which initialization method to use.
  InitMode init_mode = InitMode::STATIC;

  /// Required point-cloud input selector for vggt_pt.
  bool init_vggt_use_pointcloud = true;

  /// If we should optimize and recover the calibration in our MLE
  bool init_dyn_mle_opt_calib = false;

  /// Max number of MLE iterations for dynamic initialization
  int init_dyn_mle_max_iter = 20;

  /// Max number of MLE threads for dynamic initialization
  int init_dyn_mle_max_threads = 20;

  /// Max time for MLE optimization (seconds)
  double init_dyn_mle_max_time = 5.0;

  /// Number of poses to use during initialization (max should be cam freq * window)
  int init_dyn_num_pose = 5;

  /// Minimum degrees we need to rotate before we try to init (sum of norm)
  double init_dyn_min_deg = 45.0;

  /// Magnitude we will inflate initial covariance of orientation
  double init_dyn_inflation_orientation = 10.0;

  /// Magnitude we will inflate initial covariance of velocity
  double init_dyn_inflation_velocity = 10.0;

  /// Magnitude we will inflate initial covariance of gyroscope bias
  double init_dyn_inflation_bias_gyro = 100.0;

  /// Magnitude we will inflate initial covariance of accelerometer bias
  double init_dyn_inflation_bias_accel = 100.0;

  /// Minimum reciprocal condition number acceptable for our covariance recovery (min_sigma / max_sigma <
  /// sqrt(min_reciprocal_condition_number))
  double init_dyn_min_rec_cond = 1e-15;

  /// Initial IMU gyroscope bias values for dynamic initialization (will be optimized)
  Eigen::Vector3d init_dyn_bias_g = Eigen::Vector3d::Zero();

  /// Initial IMU accelerometer bias values for dynamic initialization (will be optimized)
  Eigen::Vector3d init_dyn_bias_a = Eigen::Vector3d::Zero();

  /// Nonlinear residual limit for DYNAMIC and the vggt_pt landmark graph (0 = disabled); inactive for FF
  double init_dyn_max_nonlinear_avg_residual = 0.0;

  // VGGT INITIALIZATION =======================

  /// Timeout when waiting for VGGT ROS responses (seconds)
  double init_vggt_wait_timeout = 30.0;

  /// Number of VGGT 3D samples to draw per frame
  int init_vggt_points_per_frame = 20;

  /// Processed-image target width (the output height is derived)
  int vggt_target_size = 518;

  /// FOV scale for fisheye undistortion (< 1.0 zooms in, > 1.0 zooms out)
  double init_vggt_fisheye_fov_scale = 0.15;

  /// Whether to crop fisheye image height to 4:3 aspect ratio
  bool init_vggt_crop_to_4_3 = true;

  /// Enable confidence filtering for model point clouds
  bool init_vggt_use_confidence_filter = false;

  /// Confidence threshold for model point clouds (points below threshold are marked invalid)
  double init_vggt_confidence_threshold = 0.5;

  /// Enable RANSAC for VGGT linear system solving
  bool init_vggt_use_ransac = false;

  /// Maximum number of RANSAC iterations
  int init_vggt_ransac_max_iterations = 200;

  /// RANSAC inlier threshold (reprojection error in pixels)
  double init_vggt_ransac_inlier_threshold = 3.0;

  /// Minimum inlier ratio to accept RANSAC result
  double init_vggt_ransac_min_inlier_ratio = 0.6;

  /// Minimum sample size for RANSAC (if -1, auto-calculated as 2*system_size)
  int init_vggt_ransac_min_sample_size = -1;

  /// Legacy nonlinear-method selector: true selects FF; false enters the landmark graph
  bool init_vggt_use_scale_optimization = true;

  /// Weight for neighboring-region scale smoothness (0 = no constraint)
  double init_vggt_scale_smoothness_weight = 100.0;

  /// Use softplus parameterization for scale factor (ensures scale > 0)
  bool init_vggt_scale_use_softplus = false;

  /// Weight for scale prior constraint (0 = no prior)
  double init_vggt_scale_prior_weight = 0.0;

  /// Maximum average residual for linear system acceptance (0 = disabled)
  double init_vggt_max_linear_avg_residual = 0.0;

  /// Effective maximum average nonlinear residual for FF (0 = disabled); inactive for SC and FEATURE_ONLY
  double init_vggt_max_nonlinear_avg_residual = 0.0;

  /// Maximum standard deviation across FF regional scales (0 = disabled)
  double init_vggt_max_scale_std = 0.0;

  /// Scale layout: 0 = one global scale; n > 0 = n-by-n PCA regions
  /// Each region has its own scale parameter.
  /// Smoothness constraint uses init_vggt_scale_smoothness_weight
  int init_vggt_scale_region_n = 0;

  /// Landmark-graph point-cloud constraint weight: false selector + >0 selects SC; false + 0 selects FEATURE_ONLY
  /// A positive value adds p_FinG = scale * p_FinG_vggt in run_nonlinear_optimization_with_feature
  double init_vggt_feature_pc_constraint_weight = 0.0;

  /// Pixel sigma for FF scale-based reprojection factors
  double init_vggt_scale_sigma_pix = 1.0;

  /**
   * @brief This function will load print out all initializer settings loaded.
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_initializer(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    PRINT_DEBUG("INITIALIZATION SETTINGS:\n");
    if (parser != nullptr) {
      parser->parse_config("init_window_time", init_window_time);
      parser->parse_config("init_imu_thresh", init_imu_thresh);
      parser->parse_config("init_max_disparity", init_max_disparity);
      parser->parse_config("init_max_features", init_max_features);
      // Unified mode selector.
      std::string init_mode_str = init_mode_to_string(init_mode);
      parser->parse_config("init_mode", init_mode_str);
      init_mode = string_to_init_mode(init_mode_str);
      parser->parse_config("init_vggt_use_pointcloud", init_vggt_use_pointcloud);
      parser->parse_config("init_dyn_mle_opt_calib", init_dyn_mle_opt_calib);
      parser->parse_config("init_dyn_mle_max_iter", init_dyn_mle_max_iter);
      parser->parse_config("init_dyn_mle_max_threads", init_dyn_mle_max_threads);
      parser->parse_config("init_dyn_mle_max_time", init_dyn_mle_max_time);
      parser->parse_config("init_dyn_num_pose", init_dyn_num_pose);
      parser->parse_config("init_dyn_min_deg", init_dyn_min_deg);
      parser->parse_config("init_dyn_inflation_ori", init_dyn_inflation_orientation);
      parser->parse_config("init_dyn_inflation_vel", init_dyn_inflation_velocity);
      parser->parse_config("init_dyn_inflation_bg", init_dyn_inflation_bias_gyro);
      parser->parse_config("init_dyn_inflation_ba", init_dyn_inflation_bias_accel);
      parser->parse_config("init_dyn_min_rec_cond", init_dyn_min_rec_cond);
      std::vector<double> bias_g = {0, 0, 0};
      std::vector<double> bias_a = {0, 0, 0};
      parser->parse_config("init_dyn_bias_g", bias_g);
      parser->parse_config("init_dyn_bias_a", bias_a);
      init_dyn_bias_g << bias_g.at(0), bias_g.at(1), bias_g.at(2);
      init_dyn_bias_a << bias_a.at(0), bias_a.at(1), bias_a.at(2);
      parser->parse_config("init_dyn_max_nonlinear_avg_residual", init_dyn_max_nonlinear_avg_residual, false);
      parser->parse_config("init_vggt_wait_timeout", init_vggt_wait_timeout);
      parser->parse_config("init_vggt_points_per_frame", init_vggt_points_per_frame);
      parser->parse_config("vggt_target_size", vggt_target_size);
      parser->parse_config("init_vggt_fisheye_fov_scale", init_vggt_fisheye_fov_scale);
      parser->parse_config("init_vggt_crop_to_4_3", init_vggt_crop_to_4_3);
      parser->parse_config("init_vggt_use_confidence_filter", init_vggt_use_confidence_filter);
      parser->parse_config("init_vggt_confidence_threshold", init_vggt_confidence_threshold);
      parser->parse_config("init_vggt_use_ransac", init_vggt_use_ransac);
      parser->parse_config("init_vggt_ransac_max_iterations", init_vggt_ransac_max_iterations);
      parser->parse_config("init_vggt_ransac_inlier_threshold", init_vggt_ransac_inlier_threshold);
      parser->parse_config("init_vggt_ransac_min_inlier_ratio", init_vggt_ransac_min_inlier_ratio);
      parser->parse_config("init_vggt_ransac_min_sample_size", init_vggt_ransac_min_sample_size);
      parser->parse_config("init_vggt_use_scale_optimization", init_vggt_use_scale_optimization);
      parser->parse_config("init_vggt_scale_smoothness_weight", init_vggt_scale_smoothness_weight);
      parser->parse_config("init_vggt_scale_use_softplus", init_vggt_scale_use_softplus);
      parser->parse_config("init_vggt_scale_prior_weight", init_vggt_scale_prior_weight);
      parser->parse_config("init_vggt_max_linear_avg_residual", init_vggt_max_linear_avg_residual);
      parser->parse_config("init_vggt_max_nonlinear_avg_residual", init_vggt_max_nonlinear_avg_residual);
      parser->parse_config("init_vggt_max_scale_std", init_vggt_max_scale_std);
      parser->parse_config("init_vggt_scale_region_n", init_vggt_scale_region_n, false);
      parser->parse_config("init_vggt_feature_pc_constraint_weight", init_vggt_feature_pc_constraint_weight, false);
      parser->parse_config("init_vggt_scale_sigma_pix", init_vggt_scale_sigma_pix, false);
    }
    PRINT_DEBUG("  - init_window_time: %.2f\n", init_window_time);
    PRINT_DEBUG("  - init_imu_thresh: %.2f\n", init_imu_thresh);
    PRINT_DEBUG("  - init_max_disparity: %.2f\n", init_max_disparity);
    PRINT_DEBUG("  - init_max_features: %.2f\n", init_max_features);
    if (init_max_features < 15) {
      PRINT_ERROR(RED "number of requested feature tracks to init not enough!!\n" RESET);
      PRINT_ERROR(RED "  init_max_features = %d\n" RESET, init_max_features);
      std::exit(EXIT_FAILURE);
    }
    if (init_imu_thresh <= 0.0 && init_mode == InitMode::STATIC) {
      PRINT_ERROR(RED "need to have an IMU threshold for static initialization!\n" RESET);
      PRINT_ERROR(RED "  init_imu_thresh = %.3f\n" RESET, init_imu_thresh);
      PRINT_ERROR(RED "  init_mode = %s\n" RESET, init_mode_to_string(init_mode).c_str());
      std::exit(EXIT_FAILURE);
    }
    if (init_max_disparity <= 0.0 && init_mode == InitMode::STATIC) {
      PRINT_ERROR(RED "need to have an DISPARITY threshold for static initialization!\n" RESET);
      PRINT_ERROR(RED "  init_max_disparity = %.3f\n" RESET, init_max_disparity);
      PRINT_ERROR(RED "  init_mode = %s\n" RESET, init_mode_to_string(init_mode).c_str());
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - init_mode: %s\n", init_mode_to_string(init_mode).c_str());
    if (init_mode == InitMode::VGGT_PT && !init_vggt_use_pointcloud) {
      PRINT_ERROR(RED "init_mode=vggt_pt requires init_vggt_use_pointcloud=true in this release!\n" RESET);
      std::exit(EXIT_FAILURE);
    }
    if (init_mode == InitMode::VGGT_PT && (vggt_target_size <= 0 || vggt_target_size % 14 != 0)) {
      PRINT_ERROR(RED "vggt_target_size must be a positive multiple of 14 (got %d)!\n" RESET, vggt_target_size);
      std::exit(EXIT_FAILURE);
    }
    if (init_mode == InitMode::VGGT_PT &&
        (init_vggt_scale_region_n < 0 || init_vggt_scale_region_n > kMaxVggtScaleRegionN)) {
      PRINT_ERROR(RED "init_vggt_scale_region_n must be in [0, %d] (got %d)!\n" RESET,
                  kMaxVggtScaleRegionN, init_vggt_scale_region_n);
      std::exit(EXIT_FAILURE);
    }
    if (init_mode == InitMode::VGGT_PT &&
        !is_valid_vggt_nonlinear_flags(init_vggt_use_scale_optimization, init_vggt_feature_pc_constraint_weight)) {
      PRINT_ERROR(RED "invalid vggt_pt nonlinear method flags: point-cloud constraint weight must be finite and non-negative, "
                      "and init_vggt_use_scale_optimization=true cannot be combined with a positive weight!\n" RESET);
      PRINT_ERROR(RED "  FF requires true + 0; SC requires false + >0; FEATURE_ONLY requires false + 0.\n" RESET);
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - init_vggt_use_pointcloud: %d\n", init_vggt_use_pointcloud);
    PRINT_DEBUG("  - init_dyn_mle_opt_calib: %d\n", init_dyn_mle_opt_calib);
    PRINT_DEBUG("  - init_dyn_mle_max_iter: %d\n", init_dyn_mle_max_iter);
    PRINT_DEBUG("  - init_dyn_mle_max_threads: %d\n", init_dyn_mle_max_threads);
    PRINT_DEBUG("  - init_dyn_mle_max_time: %.2f\n", init_dyn_mle_max_time);
    PRINT_DEBUG("  - init_dyn_num_pose: %d\n", init_dyn_num_pose);
    PRINT_DEBUG("  - init_dyn_min_deg: %.2f\n", init_dyn_min_deg);
    PRINT_DEBUG("  - init_dyn_inflation_ori: %.2e\n", init_dyn_inflation_orientation);
    PRINT_DEBUG("  - init_dyn_inflation_vel: %.2e\n", init_dyn_inflation_velocity);
    PRINT_DEBUG("  - init_dyn_inflation_bg: %.2e\n", init_dyn_inflation_bias_gyro);
    PRINT_DEBUG("  - init_dyn_inflation_ba: %.2e\n", init_dyn_inflation_bias_accel);
    PRINT_DEBUG("  - init_dyn_min_rec_cond: %.2e\n", init_dyn_min_rec_cond);
    if (init_dyn_num_pose < 4) {
      PRINT_ERROR(RED "number of requested frames to init not enough!!\n" RESET);
      PRINT_ERROR(RED "  init_dyn_num_pose = %d (4 min)\n" RESET, init_dyn_num_pose);
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - init_dyn_bias_g: %.2f, %.2f, %.2f\n", init_dyn_bias_g(0), init_dyn_bias_g(1), init_dyn_bias_g(2));
    PRINT_DEBUG("  - init_dyn_bias_a: %.2f, %.2f, %.2f\n", init_dyn_bias_a(0), init_dyn_bias_a(1), init_dyn_bias_a(2));
    PRINT_DEBUG("  - init_dyn_max_nonlinear_avg_residual: %.2f\n", init_dyn_max_nonlinear_avg_residual);
    PRINT_DEBUG("  - init_vggt_wait_timeout: %.2f\n", init_vggt_wait_timeout);
    PRINT_DEBUG("  - init_vggt_points_per_frame: %d\n", init_vggt_points_per_frame);
    PRINT_DEBUG("  - vggt_target_size: %d\n", vggt_target_size);
    PRINT_DEBUG("  - init_vggt_fisheye_fov_scale: %.2f\n", init_vggt_fisheye_fov_scale);
    PRINT_DEBUG("  - init_vggt_crop_to_4_3: %d\n", init_vggt_crop_to_4_3);
    PRINT_DEBUG("  - init_vggt_use_confidence_filter: %d\n", init_vggt_use_confidence_filter);
    PRINT_DEBUG("  - init_vggt_confidence_threshold: %.2f\n", init_vggt_confidence_threshold);
    PRINT_DEBUG("  - init_vggt_use_ransac: %d\n", init_vggt_use_ransac);
    PRINT_DEBUG("  - init_vggt_ransac_max_iterations: %d\n", init_vggt_ransac_max_iterations);
    PRINT_DEBUG("  - init_vggt_ransac_inlier_threshold: %.2f\n", init_vggt_ransac_inlier_threshold);
    PRINT_DEBUG("  - init_vggt_ransac_min_inlier_ratio: %.2f\n", init_vggt_ransac_min_inlier_ratio);
    PRINT_DEBUG("  - init_vggt_ransac_min_sample_size: %d\n", init_vggt_ransac_min_sample_size);
    PRINT_DEBUG("  - init_vggt_use_scale_optimization: %d\n", init_vggt_use_scale_optimization);
    PRINT_DEBUG("  - init_vggt_scale_smoothness_weight: %.2f\n", init_vggt_scale_smoothness_weight);
    PRINT_DEBUG("  - init_vggt_scale_use_softplus: %d\n", init_vggt_scale_use_softplus);
    PRINT_DEBUG("  - init_vggt_scale_prior_weight: %.2f\n", init_vggt_scale_prior_weight);
    PRINT_DEBUG("  - init_vggt_max_linear_avg_residual: %.2f\n", init_vggt_max_linear_avg_residual);
    PRINT_DEBUG("  - init_vggt_max_nonlinear_avg_residual: %.2f\n", init_vggt_max_nonlinear_avg_residual);
    PRINT_DEBUG("  - init_vggt_max_scale_std: %.2f\n", init_vggt_max_scale_std);
    PRINT_DEBUG("  - init_vggt_scale_region_n: %d\n", init_vggt_scale_region_n);
    PRINT_DEBUG("  - init_vggt_feature_pc_constraint_weight: %.2f\n", init_vggt_feature_pc_constraint_weight);
    if (init_mode == InitMode::VGGT_PT) {
      const VggtNonlinearMethod nonlinear_method =
          vggt_nonlinear_method_from_flags(init_vggt_use_scale_optimization, init_vggt_feature_pc_constraint_weight);
      PRINT_DEBUG("  - init_vggt_nonlinear_method: %s\n", vggt_nonlinear_method_name(nonlinear_method));
    }
    PRINT_DEBUG("  - init_vggt_scale_sigma_pix: %.2f\n", init_vggt_scale_sigma_pix);
  }

  // NOISE / CHI2 ============================

  /// Gyroscope white noise (rad/s/sqrt(hz))
  double sigma_w = 1.6968e-04;

  /// Gyroscope random walk (rad/s^2/sqrt(hz))
  double sigma_wb = 1.9393e-05;

  /// Accelerometer white noise (m/s^2/sqrt(hz))
  double sigma_a = 2.0000e-3;

  /// Accelerometer random walk (m/s^3/sqrt(hz))
  double sigma_ab = 3.0000e-03;

  /// Noise sigma for our raw pixel measurements
  double sigma_pix = 1;

  /**
   * @brief This function will load print out all noise parameters loaded.
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_noise(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    PRINT_DEBUG("NOISE PARAMETERS:\n");
    if (parser != nullptr) {
      parser->parse_external("relative_config_imu", "imu0", "gyroscope_noise_density", sigma_w);
      parser->parse_external("relative_config_imu", "imu0", "gyroscope_random_walk", sigma_wb);
      parser->parse_external("relative_config_imu", "imu0", "accelerometer_noise_density", sigma_a);
      parser->parse_external("relative_config_imu", "imu0", "accelerometer_random_walk", sigma_ab);
      parser->parse_config("up_slam_sigma_px", sigma_pix);
    }
    PRINT_DEBUG("  - gyroscope_noise_density: %.6f\n", sigma_w);
    PRINT_DEBUG("  - accelerometer_noise_density: %.5f\n", sigma_a);
    PRINT_DEBUG("  - gyroscope_random_walk: %.7f\n", sigma_wb);
    PRINT_DEBUG("  - accelerometer_random_walk: %.6f\n", sigma_ab);
    PRINT_DEBUG("  - sigma_pix: %.2f\n", sigma_pix);
  }

  // STATE DEFAULTS ==========================

  /// Gravity magnitude in the global frame (i.e. should be 9.81 typically)
  double gravity_mag = 9.81;

  /// Number of distinct cameras that we will observe features in
  int num_cameras = 1;

  /// If we should process two cameras are being stereo or binocular. If binocular, we do monocular feature tracking on each image.
  bool use_stereo = true;

  /// Will half the resolution all tracking image (aruco will be 1/4 instead of halved if dowsize_aruoc also enabled)
  bool downsample_cameras = false;

  /// Time offset between camera and IMU (t_imu = t_cam + t_off)
  double calib_camimu_dt = 0.0;

  /// Map between camid and camera intrinsics (fx, fy, cx, cy, d1...d4, cam_w, cam_h)
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> camera_intrinsics;

  /// Map between camid and camera extrinsics (q_ItoC, p_IinC).
  std::map<size_t, Eigen::VectorXd> camera_extrinsics;

  /// Single preprocessing plan per camera. VioManager reuses these plans and the processed camera models below.
  std::map<size_t, ov_core::VGGTPreprocessPlan> vggt_preprocess_plans;

  /**
   * @brief This function will load and print all state parameters (e.g. sensor extrinsics)
   * This allows for visual checking that everything was loaded properly from ROS/CMD parsers.
   *
   * @param parser If not null, this parser will be used to load our parameters
   */
  void print_and_load_state(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("gravity_mag", gravity_mag);
      parser->parse_config("max_cameras", num_cameras); // might be redundant
      parser->parse_config("use_stereo", use_stereo);
      parser->parse_config("downsample_cameras", downsample_cameras);
      for (int i = 0; i < num_cameras; i++) {

        // Time offset (use the first one)
        // TODO: support multiple time offsets between cameras
        if (i == 0) {
          parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "timeshift_cam_imu", calib_camimu_dt, false);
        }

        // Distortion model
        std::string dist_model = "radtan";
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_model", dist_model);

        // Distortion parameters
        std::vector<double> cam_calib1 = {1, 1, 0, 0};
        std::vector<double> cam_calib2 = {0, 0, 0, 0};
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "intrinsics", cam_calib1);
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_coeffs", cam_calib2);
        Eigen::VectorXd cam_calib = Eigen::VectorXd::Zero(8);
        cam_calib << cam_calib1.at(0), cam_calib1.at(1), cam_calib1.at(2), cam_calib1.at(3), cam_calib2.at(0), cam_calib2.at(1),
            cam_calib2.at(2), cam_calib2.at(3);
        // FOV / resolution - get original resolution first
        std::vector<int> matrix_wh = {1, 1};
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "resolution", matrix_wh);

        PRINT_DEBUG("[CAM%d] Original intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, resolution=%dx%d\n",
                   i, cam_calib(0), cam_calib(1), cam_calib(2), cam_calib(3), matrix_wh.at(0), matrix_wh.at(1));
        PRINT_DEBUG("[CAM%d] Original distortion: d1=%.6f, d2=%.6f, d3=%.6f, d4=%.6f\n",
                   i, cam_calib(4), cam_calib(5), cam_calib(6), cam_calib(7));

        // Build the processed camera grid used by the point-cloud initializer.
        if (init_mode_is_vggt(init_mode)) {
          ov_core::VGGTPreprocessPlan plan = ov_core::makeVGGTPreprocessPlan(
              i, dist_model, cam_calib, cv::Size(matrix_wh.at(0), matrix_wh.at(1)), vggt_target_size,
              init_vggt_fisheye_fov_scale, init_vggt_crop_to_4_3);
          vggt_preprocess_plans[i] = plan;
          cam_calib = plan.output_calib;
          matrix_wh.at(0) = plan.output_roi.width;
          matrix_wh.at(1) = plan.output_roi.height;
        } else {
          // Standard downsampling
          if (downsample_cameras) {
            PRINT_DEBUG("[CAM%d] Standard 2x downsampling enabled\n", i);
            cam_calib(0) /= 2.0;
            cam_calib(1) /= 2.0;
            cam_calib(2) /= 2.0;
            cam_calib(3) /= 2.0;
            matrix_wh.at(0) /= 2.0;
            matrix_wh.at(1) /= 2.0;
            PRINT_DEBUG("[CAM%d] After standard downsampling: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, resolution=%dx%d\n",
                       i, cam_calib(0), cam_calib(1), cam_calib(2), cam_calib(3), matrix_wh.at(0), matrix_wh.at(1));
          } else {
            PRINT_DEBUG("[CAM%d] No downsampling applied (using original intrinsics)\n", i);
          }
        }

        // Extrinsics
        Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "T_imu_cam", T_CtoI);

        // Load these into our state
        Eigen::Matrix<double, 7, 1> cam_eigen;
        cam_eigen.block(0, 0, 4, 1) = ov_core::rot_2_quat(T_CtoI.block(0, 0, 3, 3).transpose());
        cam_eigen.block(4, 0, 3, 1) = -T_CtoI.block(0, 0, 3, 3).transpose() * T_CtoI.block(0, 3, 3, 1);

        // Create intrinsics model
        // When VGGT initialization is used, the image is undistorted (post-rect),
        // so always use CamRadtan (pinhole). cam_calib was overwritten to post-rect
        // values with zero distortion above. Using CamEqui here would make the init
        // optimizer treat features as fisheye-projected (wrong projection model).
        if (!init_mode_is_vggt(init_mode) && dist_model == "equidistant") {
          camera_intrinsics.insert({i, std::make_shared<ov_core::CamEqui>(matrix_wh.at(0), matrix_wh.at(1))});
          camera_intrinsics.at(i)->set_value(cam_calib);
        } else {
          camera_intrinsics.insert({i, std::make_shared<ov_core::CamRadtan>(matrix_wh.at(0), matrix_wh.at(1))});
          camera_intrinsics.at(i)->set_value(cam_calib);
        }
        camera_extrinsics.insert({i, cam_eigen});
      }
    }
    PRINT_DEBUG("STATE PARAMETERS:\n");
    PRINT_DEBUG("  - gravity_mag: %.4f\n", gravity_mag);
    PRINT_DEBUG("  - gravity: %.3f, %.3f, %.3f\n", 0.0, 0.0, gravity_mag);
    PRINT_DEBUG("  - num_cameras: %d\n", num_cameras);
    PRINT_DEBUG("  - use_stereo: %d\n", use_stereo);
    PRINT_DEBUG("  - downsize cameras: %d\n", downsample_cameras);
    if (num_cameras != (int)camera_intrinsics.size() || num_cameras != (int)camera_extrinsics.size()) {
      PRINT_ERROR(RED "[INIT]: camera calib size does not match max cameras...\n" RESET);
      PRINT_ERROR(RED "[INIT]: got %d but expected %d max cameras (camera_intrinsics)\n" RESET, (int)camera_intrinsics.size(), num_cameras);
      PRINT_ERROR(RED "[INIT]: got %d but expected %d max cameras (camera_extrinsics)\n" RESET, (int)camera_extrinsics.size(), num_cameras);
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - calib_camimu_dt: %.4f\n", calib_camimu_dt);
    for (int n = 0; n < num_cameras; n++) {
      std::stringstream ss;
      ss << "cam_" << n << "_fisheye:" << (std::dynamic_pointer_cast<ov_core::CamEqui>(camera_intrinsics.at(n)) != nullptr) << std::endl;
      ss << "cam_" << n << "_wh:" << std::endl << camera_intrinsics.at(n)->w() << " x " << camera_intrinsics.at(n)->h() << std::endl;
      ss << "cam_" << n << "_intrinsic(0:3):" << std::endl
         << camera_intrinsics.at(n)->get_value().block(0, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_intrinsic(4:7):" << std::endl
         << camera_intrinsics.at(n)->get_value().block(4, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_extrinsic(0:3):" << std::endl << camera_extrinsics.at(n).block(0, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_extrinsic(4:6):" << std::endl << camera_extrinsics.at(n).block(4, 0, 3, 1).transpose() << std::endl;
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = ov_core::quat_2_Rot(camera_extrinsics.at(n).block(0, 0, 4, 1)).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * camera_extrinsics.at(n).block(4, 0, 3, 1);
      ss << "T_C" << n << "toI:" << std::endl << T_CtoI << std::endl << std::endl;
      PRINT_DEBUG(ss.str().c_str());
    }
  }

};

} // namespace ov_init

#endif // OV_INIT_INERTIALINITIALIZEROPTIONS_H
