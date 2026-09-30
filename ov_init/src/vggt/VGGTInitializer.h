/*
 * VGGT point-cloud initialization for OpenVINS
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */

#ifndef OV_INIT_VGGTINITIALIZER_H
#define OV_INIT_VGGTINITIALIZER_H

#include "init/InitializationDiagnostics.h"
#include "init/InertialInitializerOptions.h"
#include "cpi/CpiV1.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <unordered_set>
#include <ceres/ceres.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
namespace ov_core {
class FeatureDatabase;
struct ImuData;
} // namespace ov_core
namespace ov_type {
class Type;
class IMU;
class PoseJPL;
class Landmark;
class Vec;
} // namespace ov_type

namespace ov_init {
class VGGTInitializerTestAccess;

/**
 * @brief Model-independent visual-inertial initializer consuming organized C0 point clouds.
 */
class VGGTInitializer {
public:
  /**
   * @brief Default constructor
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  explicit VGGTInitializer(const InertialInitializerOptions &params_);

  /**
   * @brief Main entry point for point-cloud initialization
   */
  bool initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                  std::shared_ptr<ov_type::IMU> &_imu,
                  const std::shared_ptr<ov_core::FeatureDatabase> &feature_db,
                  const std::shared_ptr<std::vector<ov_core::ImuData>> &imu_data,
                  std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
                  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM,
                  InitializationDiagnostics &diagnostics);

  /**
   * @brief Try to get the initialized system with VGGT pointcloud data
   *
   * @param[out] timestamp Timestamp we have initialized the state at (last imu state)
   * @param[out] covariance Calculated covariance of the returned state
   * @param[out] order Order of the covariance matrix
   * @param _imu Pointer to the "active" IMU state (q_GtoI, p_IinG, v_IinG, bg, ba)
   * @param _clones_IMU Map between imaging times and clone poses (q_GtoIi, p_IiinG)
   * @param _features_SLAM Our current set of SLAM features (3d positions)
   * @return True if we have successfully initialized our system
   */
  bool initialize_pointcloud(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                             std::shared_ptr<ov_type::IMU> &_imu,
                             const std::shared_ptr<ov_core::FeatureDatabase> &feature_db,
                             const std::shared_ptr<std::vector<ov_core::ImuData>> &imu_data,
                             std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
                             std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> &_features_SLAM,
                             InitializationDiagnostics &diagnostics);
private:
  friend class VGGTInitializerTestAccess;

  /**
   * @brief Recover covariance of optimized IMU state from Ceres optimization problem
   * @param problem Ceres optimization problem (after solving)
   * @param state_index Index of the state to recover covariance for
   * @param ceres_vars_ori Orientation state variables
   * @param ceres_vars_pos Position state variables
   * @param ceres_vars_vel Velocity state variables
   * @param ceres_vars_bias_g Gyroscope bias state variables
   * @param ceres_vars_bias_a Accelerometer bias state variables
   * @param map_calib_cam2imu Camera-IMU calibration map (for null space rank computation)
   * @param _imu IMU state object to add to order
   * @param covariance Output covariance matrix (15x15)
   * @param order Output state ordering vector
   * @return True if covariance recovery succeeded, false otherwise
   */
  bool recover_covariance(ceres::Problem &problem, int state_index,
                         const std::vector<double *> &ceres_vars_ori,
                         const std::vector<double *> &ceres_vars_pos,
                         const std::vector<double *> &ceres_vars_vel,
                         const std::vector<double *> &ceres_vars_bias_g,
                         const std::vector<double *> &ceres_vars_bias_a,
                         const std::map<size_t, int> &map_calib_cam2imu,
                         std::shared_ptr<ov_type::IMU> &_imu,
                         Eigen::MatrixXd &covariance,
                         std::vector<std::shared_ptr<ov_type::Type>> &order);

  /**
   * @brief Load VGGT auxiliary data (ROS) needed for initialization
   * @param map_camera_times Camera timestamps involved in the initialization window
   * @param oldest_camera_time Oldest camera timestamp in the window
   * @return True if data successfully prepared
   */
  bool load_vggt_data(const std::map<double, bool> &map_camera_times, double oldest_camera_time);
  /**
   * @brief Filter point cloud by confidence threshold
   * @param pointcloud CV_32FC4 point cloud [x, y, z, confidence/validity]
   * @param threshold Confidence threshold (points below this will be marked invalid)
   * @return Number of points marked as invalid
   *
   * This function sets the 4th channel to 0 for points with confidence < threshold.
   * The 4th channel acts as a validity flag: 0 = invalid, non-zero = valid
   */
  int filter_pointcloud_by_confidence(cv::Mat &pointcloud, float threshold);

  /// Sanitize loaded pointclouds and report {invalidated samples, thresholded frames}.
  std::pair<int, int> filter_loaded_pointclouds();

  /**
   * @brief Get 3D point coordinates at specific pixel coordinates for given timestamp
   * @param timestamp Target timestamp
   * @param u Pixel coordinate u (horizontal)
   * @param v Pixel coordinate v (vertical)
   * @param point3d Output 3D point coordinates (X, Y, Z)
   * @param confidence Output confidence value (optional, from 4th channel)
   * @return True if 3D point retrieval was successful (returns false if 4th channel is 0)
   */
  bool get_point3d(double timestamp, double u, double v, Eigen::Vector3f &point3d, float *confidence = nullptr);

  /**
   * @brief Sample the highest-confidence valid point from each image patch.
   * @param timestamp Target timestamp
   * @param pixels_uv Output pixel coordinates (u, v) of the selected samples
   * @param points_3d Output vector of corresponding 3D points (X, Y, Z)
   * @param confidences Output vector of confidence values for each point
   * @return True if at least one valid sample was retrieved
   */
  bool sample_pointcloud(double timestamp, std::vector<Eigen::Vector2d> &pixels_uv,
                         std::vector<Eigen::Vector3d> &points_3d, std::vector<float> &confidences);

  /**
   * @brief Build linear system matrix A and vector b from sampled points at a given timestamp
   * @param timestamp Timestamp for this frame
   * @param frame_pixels Pixel coordinates (u, v) for sampled points
   * @param frame_points 3D point coordinates (X, Y, Z) in C0
   * @param cam_id Camera ID
   * @param R_ItoC Rotation matrix from IMU to camera frame
   * @param p_IinC Translation from IMU to camera frame
   * @param map_camera_cpi_I0toIi Preintegration data map
   * @param system_size Total number of unknowns in the system
   * @param A Output matrix A (will be resized to [2*num_points x system_size])
   * @param b Output vector b (will be resized to [2*num_points x 1])
   * @return Number of measurements added (2 * number of valid points)
   */
  int build_linear_system_from_points(
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
      Eigen::VectorXd &b);
  /**
   * @brief Solve linear system with gravity magnitude constraint |g| = gravity_mag
   * @param A Linear system matrix [num_measurements x system_size]
   * @param b Linear system vector [num_measurements x 1]
   * @param x_hat Output state vector [system_size x 1]
   * @param v_I0inI0 Output velocity in I0 frame
   * @param gravity_inI0 Output gravity in I0 frame
   * @return True if solve succeeded with valid solution
   */
  bool LLSsolver_with_gravity_constraint(
      const Eigen::MatrixXd &A,
      const Eigen::VectorXd &b,
      Eigen::VectorXd &x_hat,
      Eigen::Vector3d &v_I0inI0,
      Eigen::Vector3d &gravity_inI0);

  /**
   * @brief Validate features and select keyframes for initialization
   * @param features Feature database to validate
   * @param newest_cam_time Latest camera timestamp
   * @param have_stereo Output flag indicating if stereo features exist
   * @param map_features_num_meas Output map of feature measurement counts
   * @param num_measurements Output total number of measurements
   * @param oldest_camera_time Output oldest camera timestamp
   * @param map_camera_times Output map of camera timestamps
   * @param map_camera_ids Output map of camera IDs
   * @return True if sufficient features for initialization
   */
  bool select_features_and_keyframes(const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
                                   double newest_cam_time, bool &have_stereo,
                                   std::map<size_t, int> &map_features_num_meas, int &num_measurements,
                                   double &oldest_camera_time, std::map<double, bool> &map_camera_times,
                                   std::map<size_t, bool> &map_camera_ids);

  /**
   * @brief Validate features visible in first frame and select keyframes for initialization
   * @param features Feature database to validate
   * @param newest_cam_time Latest camera timestamp
   * @param have_stereo Output flag indicating if stereo features exist
   * @param map_features_num_meas Output map of feature measurement counts
   * @param num_measurements Output total number of measurements
   * @param oldest_camera_time Output oldest camera timestamp (first frame time)
   * @param map_camera_times Output map of camera timestamps
   * @param map_camera_ids Output map of camera IDs
   * @return True if sufficient features for initialization
   */
  bool select_features_and_keyframes_firstanchored(const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
                                               double newest_cam_time, bool &have_stereo,
                                               std::map<size_t, int> &map_features_num_meas, int &num_measurements,
                                               double &oldest_camera_time, std::map<double, bool> &map_camera_times,
                                               std::map<size_t, bool> &map_camera_ids);

  /**
   * @brief Get first frame 3D point in C0 frame for a feature (point cloud mode)
   * @param feat Feature data
   * @param features_point3d_c0_ Output 3D point in C0 frame
   * @return True if 3D point was successfully obtained
   */
  bool get_pFinC0(const std::shared_ptr<ov_core::Feature> &feat,
                  Eigen::Vector3d &features_point3d_c0_);

  /**
   * Linear least squares with gravity-norm constraint.
   * Constructs A/b, applies the Dong-SI polynomial constraint, and solves for
   * x_hat / v_I0inI0 / g_inI0.
   */
  bool build_and_solve_linear_system_pointcloud(
      const std::unordered_map<size_t, std::shared_ptr<ov_core::Feature>> &features,
      const std::map<size_t, int> &map_features_num_meas,
      const std::map<double, bool> &map_camera_times,
      const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_I0toIi,
      double oldest_camera_time,
      int num_measurements,
      Eigen::VectorXd &x_hat,
      Eigen::Vector3d &v_I0inI0,
      Eigen::Vector3d &gravity_inI0,
      std::map<double, std::vector<Eigen::Vector2d>> &sampled_pixels_per_time,
      std::map<double, std::vector<Eigen::Vector3d>> &sampled_points_per_time,
      std::map<double, std::vector<float>> &sampled_confidences_per_time);
  /**
   * @brief Extract IMU states and transform features to global frame
   * @param features Feature database
   * @param map_features_num_meas Feature measurement counts
   * @param map_camera_times Camera timestamps
   * @param map_camera_cpi_I0toIi IMU preintegration from I0 to Ii
   * @param x_hat State estimate from linear system
   * @param v_I0inI0 Velocity in I0 frame
   * @param gravity_inI0 Gravity in I0 frame
   * @param ori_GtoIi Output: orientation from global to Ii
   * @param pos_IiinG Output: position of Ii in global
   * @param vel_IiinG Output: velocity of Ii in global
   * @param features_inG Output: features in global frame
   * @return True if successful
   */
  bool recover_states_in_global(
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
      std::map<size_t, Eigen::Vector3d> &features_inG);

  /**
   * Nonlinear Ceres optimization with covariance recovery.
   * Builds states and factors, solves, writes results back to
   * _imu/_clones/_features_SLAM and recovers the IMU covariance
   * including prior inflation. The rT1..rT4 timestamps preserve phase timing.
   */
  bool run_nonlinear_optimization_with_feature(
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
      InitializationDiagnostics &diagnostics);

  /**
   * @brief Run nonlinear optimization with scale-based reprojection (instead of feature-based)
   *
   * This function optimizes IMU states (pose, velocity, bias) and global or regional point-cloud scales
   * using VGGT point cloud reprojection constraints. Unlike run_nonlinear_optimization_with_feature,
   * this method does not estimate 3D feature positions but instead estimates point-cloud scale factors.
   *
   * @param map_camera_times Map of camera timestamps to validity flags
   * @param map_camera_cpi_IitoIi1 Map of IMU preintegration between consecutive frames
   * @param ori_GtoIi Initial orientations (quaternions) for each frame
   * @param pos_IiinG Initial positions for each frame
   * @param vel_IiinG Initial velocities for each frame
   * @param gyroscope_bias Initial gyroscope bias
   * @param accelerometer_bias Initial accelerometer bias
   * @param sampled_pixels_per_time Map from timestamp to sampled pixel coordinates
   * @param sampled_points_per_time Map from timestamp to VGGT 3D points (in C0 frame)
   * @param map_camera_ids Map of camera IDs to validity flags
   * @param newest_cam_time Newest camera timestamp
   * @param _imu Output IMU state
   * @param _clones_IMU Output IMU pose clones
   * @param timestamp Output final timestamp
   * @param covariance Output covariance matrix
   * @param order Output state ordering
   * @param rT1-rT4 Timing checkpoints
   * @return True if optimization succeeded
   */
  bool run_nonlinear_optimization_with_scale(
      const std::map<double, bool> &map_camera_times,
      const std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_IitoIi1,
      const std::map<double, Eigen::VectorXd> &ori_GtoIi,
      const std::map<double, Eigen::VectorXd> &pos_IiinG,
      const std::map<double, Eigen::VectorXd> &vel_IiinG,
      const Eigen::Vector3d &gyroscope_bias,
      const Eigen::Vector3d &accelerometer_bias,
      const std::map<double, std::vector<Eigen::Vector2d>> &sampled_pixels_per_time,
      const std::map<double, std::vector<Eigen::Vector3d>> &sampled_points_per_time,
      const std::map<double, std::vector<float>> &sampled_confidences_per_time,
      const std::map<size_t, bool> &map_camera_ids,
      double newest_cam_time,
      std::shared_ptr<ov_type::IMU> &_imu,
      std::map<double, std::shared_ptr<ov_type::PoseJPL>> &_clones_IMU,
      double &timestamp,
      Eigen::MatrixXd &covariance,
      std::vector<std::shared_ptr<ov_type::Type>> &order,
      const boost::posix_time::ptime &rT1,
      const boost::posix_time::ptime &rT2,
      const boost::posix_time::ptime &rT3,
      const boost::posix_time::ptime &rT4,
      InitializationDiagnostics &diagnostics);

  /// Initialization parameters
  InertialInitializerOptions params;

  /// Confidence filtering parameters (loaded from config in constructor)
  float confidence_threshold_;
  bool use_confidence_filter_;

  /// VGGT inference time (request to response, in seconds)
  double vggt_inference_time_ = -1.0;

  int size_pointcloud_scale = 1;
  int size_velocity = 3;
  int size_gravity = 3;
  int system_size = 0;
  int min_num_meas_to_optimize = 0;
  int min_valid_features = 8;
  int count_valid_features = 0;

  /// Rotation from gravity-aligned global frame to first IMU frame (I0)
  Eigen::Matrix3d R_GtoI0 = Eigen::Matrix3d::Identity();
  void vggtPointcloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
  std::shared_ptr<ros::NodeHandle> ros_node_;
  ros::Publisher vggt_request_pub_;
  ros::Subscriber vggt_pointcloud_sub_;
  bool ros_topics_ready_ = false;
  mutable std::mutex vggt_ros_mutex_;
  std::unordered_set<double> vggt_requested_timestamps_;
  std::map<double, cv::Mat> vggt_received_pointclouds_;  // CV_32FC4 format
  std::map<double, bool> vggt_received_pointcloud_has_confidence_;

  /// Loaded VGGT 3D point data: timestamp -> 4D point image (CV_32FC4)
  /// Channel layout: [x, y, z, confidence]
  /// confidence is finite and non-negative; zero marks an invalid sample
  std::map<double, cv::Mat> vggt_point3d_data;
  std::map<double, bool> vggt_pointcloud_has_confidence_;

  /// C0-frame points used by region PCA and point-cloud scale constraints.
  /// Key: feature ID, Value: 3D position in C0 frame
  std::map<size_t, Eigen::Vector3d> features_point3d_c0_;

  const std::string vggt_request_topic_ = "/vggtinitializer/request_timestamps";
  const std::string vggt_pointcloud_topic_ = "/vggtinitializer/pointcloud";

  /**
   * @brief Compute IMU preintegration for initialization
   * @param map_camera_times Camera timestamp map
   * @param oldest_camera_time Oldest camera timestamp
   * @param gyroscope_bias Gyroscope bias
   * @param accelerometer_bias Accelerometer bias
   * @param imu_data IMU measurements
   * @param map_camera_cpi_I0toIi Output: preintegration from I0 to Ii
   * @param map_camera_cpi_IitoIi1 Output: preintegration from Ii to Ii+1
   * @return True if successful, false otherwise
   */
  bool compute_camera_imu_preintegration(const std::map<double, bool> &map_camera_times, double oldest_camera_time,
                                         const Eigen::Vector3d &gyroscope_bias, const Eigen::Vector3d &accelerometer_bias,
                                         const std::shared_ptr<std::vector<ov_core::ImuData>> &imu_data,
                                         std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_I0toIi,
                                         std::map<double, std::shared_ptr<ov_core::CpiV1>> &map_camera_cpi_IitoIi1);
};

} // namespace ov_init

#endif // OV_INIT_VGGTINITIALIZER_H
