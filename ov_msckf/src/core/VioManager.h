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

#ifndef OV_MSCKF_VIOMANAGER_H
#define OV_MSCKF_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "VioManagerOptions.h"
#include "init/InitializationDiagnostics.h"
#include "utils/sensor_data.h"

namespace ov_core {
class TrackBase;
class FeatureInitializer;
} // namespace ov_core
namespace ov_init {
class InertialInitializer;
} // namespace ov_init
namespace ov_type {
class IMU;
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterZeroVelocity;
class Propagator;

/**
 * @brief Core class that manages the entire system
 *
 * This class contains the state and other algorithms needed for the MSCKF to work.
 * We feed in measurements into this class and send them to their respective algorithms.
 * If we have measurements to propagate or update with, this class will call on our state to do that.
 */
class VioManager {

public:
  struct InitializationTiming {
    uint64_t segment_id = 0;
    std::string reason = "startup";
    ov_init::InitializationDiagnostics diagnostics;
  };

  struct PipelineSnapshot {
    uint64_t segment_id = 0;
    std::shared_ptr<State> state;
    std::shared_ptr<Propagator> propagator;
    bool initialized = false;
  };

  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  /// Join asynchronous initialization workers before destroying the manager.
  ~VioManager();

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(const ov_core::CameraData &message) { track_image_and_update(message); }

  /**
   * @brief Feed function for a synchronized simulated cameras
   * @param timestamp Time that this image was collected
   * @param camids Camera ids that we have simulated measurements for
   * @param feats Raw uv simulated measurements
   */
  void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                   const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);

  /**
   * @brief Given a state, this will initialize our IMU state.
   * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
   */
  void initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate);

  /// If we are initialized or not
  bool initialized() const {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    return is_initialized_vio.load() && timelastupdate.load() != -1.0;
  }

  /// Timestamp that the system was initialized at
  double initialized_time() const { return startup_time.load(); }

  /// Monotonic ID of the current local-coordinate estimator segment
  uint64_t segment_id() const { return estimator_generation.load(); }

  /// Snapshot all successful sensor-time initialization records
  std::vector<InitializationTiming> get_initialization_timings() const;

  /// Atomically snapshot the state, propagator, segment ID, and initialized flag
  PipelineSnapshot get_pipeline_snapshot() const;

  /// Try to compute high-rate odometry without racing a camera state update.
  bool try_fast_state_propagate(double timestamp, Eigen::Matrix<double, 13, 1> &state_plus,
                                Eigen::Matrix<double, 12, 12> &covariance,
                                std::map<size_t, Eigen::Matrix<double, 7, 1>> &camera_extrinsics);

  /// Current camera-to-IMU time offset, synchronized with state updates.
  double camera_to_imu_offset() const;

  /// Wait until every asynchronous initialization worker has exited.
  void wait_for_initialization_worker();

  /// Accessor for current system parameters
  VioManagerOptions get_params() { return params; }

  /// Accessor to get the current state
  std::shared_ptr<State> get_state() const {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    return state;
  }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator() const {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    return propagator;
  }

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image(double &timestamp);

  /// Returns 3d SLAM features in the global frame
  std::vector<Eigen::Vector3d> get_features_SLAM();

  /// Returns 3d ARUCO features in the global frame
  std::vector<Eigen::Vector3d> get_features_ARUCO();

  /// Returns 3d features used in the last update in global frame
  std::vector<Eigen::Vector3d> get_good_features_MSCKF() { return good_features_MSCKF; }

  /// Return the image used when projecting the active tracks
  void get_active_image(double &timestamp, cv::Mat &image) {
    timestamp = active_tracks_time;
    image = active_image;
  }

  /// Returns active tracked features in the current frame
  void get_active_tracks(double &timestamp, std::unordered_map<size_t, Eigen::Vector3d> &feat_posinG,
                         std::unordered_map<size_t, Eigen::Vector3d> &feat_tracks_uvd) {
    timestamp = active_tracks_time;
    feat_posinG = active_tracks_posinG;
    feat_tracks_uvd = active_tracks_uvd;
  }

  /// Callback for images on the configured VGGT processed grid.
  void set_processed_camera_callback(std::function<void(const ov_core::CameraData&)> callback) {
    processed_camera_callback = callback;
  }

protected:
  /**
   * @brief Given a new set of camera images, this will track them.
   *
   * If we are having stereo tracking, we should call stereo tracking functions.
   * Otherwise we will try to track on each of the images passed.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(const ov_core::CameraData &message);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(const ov_core::CameraData &message);

  /**
   * @brief This function will try to initialize the state.
   *
   * This should call on our initializer and try to init the state.
   * In the future we should call the structure-from-motion code from here.
   * This function could also be repurposed to re-initialize the system after failure.
   *
   * @param message Contains our timestamp, images, and camera ids
   * @return True if we have successfully initialized
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /// Replace the estimator pipeline and begin a new local-coordinate segment
  bool reset_system(double timestamp);

  /// Return true when the current segment reached the configured reset interval
  bool periodic_reset_due(double timestamp) const;

  /// Store one IMU sample in the ordered reset replay buffer. Caller holds lifecycle_mtx.
  void buffer_imu_for_reset_locked(const ov_core::ImuData &message);

  /// Select the boundary predecessor and all buffered samples at/after the reset time.
  std::vector<ov_core::ImuData> select_reset_imu_replay_locked(double timestamp) const;

  /// Join and remove initialization workers that have already exited.
  void reap_finished_initialization_workers();

  /// Isolate initializer writes from the live estimator state until commit.
  static std::shared_ptr<ov_type::IMU> clone_imu_for_initialization(const std::shared_ptr<ov_type::IMU> &active_imu);

  /// Copy an accepted initializer result into the live IMU and remap its covariance ordering.
  static bool commit_initialized_imu(const std::shared_ptr<ov_type::IMU> &temporary_imu,
                                     const std::shared_ptr<ov_type::IMU> &active_imu,
                                     const Eigen::MatrixXd &covariance,
                                     std::vector<std::shared_ptr<ov_type::Type>> &order);

  /**
   * @brief This function will will re-triangulate all features in the current frame
   *
   * For all features that are currently being tracked by the system, this will re-triangulate them.
   * This is useful for downstream applications which need the current pointcloud of points (e.g. loop closure).
   * This will try to triangulate *all* points, not just ones that have been used in the update.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void retriangulate_active_tracks(const ov_core::CameraData &message);

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object :D
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our sparse feature tracker (klt or descriptor)
  std::shared_ptr<ov_core::TrackBase> trackFEATS;

  /// VGGT-specific feature tracker using preprocessed images and matching intrinsics.
  std::shared_ptr<ov_core::TrackBase> trackFEATS_vggt;

  /// Our aruoc tracker
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// State initializer
  std::shared_ptr<ov_init::InertialInitializer> initializer;

  /// Boolean if we are initialized or not
  std::atomic<bool> is_initialized_vio{false};

  /// Our MSCKF feature updater
  std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

  /// Our SLAM/ARUCO feature updater
  std::shared_ptr<UpdaterSLAM> updaterSLAM;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// This is the queue of measurement times that have come in since we starting doing initialization
  /// After we initialize, we will want to prop & update to the latest timestamp quickly
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // Timing statistic file and variables
  std::ofstream of_statistics;
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  /// Callback for images on the configured VGGT processed grid.
  std::function<void(const ov_core::CameraData&)> processed_camera_callback;

  // Track how much distance we have traveled
  std::atomic<double> timelastupdate{-1.0};
  double distance = 0;

  // Camera-thread watermark used to prune IMU buffers without reading State
  // containers concurrently from the IMU callback.
  std::atomic<double> imu_retention_timestamp{-1.0};

  // Startup time of the filter
  std::atomic<double> startup_time{-1.0};

  // A reset and an initialization result commit may not modify the pipeline together.
  std::mutex init_commit_mtx;

  // Short-lived lock for consistent pipeline pointer snapshots and segment metadata.
  mutable std::mutex lifecycle_mtx;

  // Serializes State and Propagator cache writes with high-rate odometry reads.
  mutable std::mutex state_update_mtx;

  // Monotonic generations prevent an old initialization worker from
  // committing into a newer segment or clearing the newer thread's state.
  static constexpr uint64_t no_init_generation = std::numeric_limits<uint64_t>::max();
  std::atomic<uint64_t> estimator_generation{0};
  std::atomic<uint64_t> thread_init_running_generation{no_init_generation};
  std::atomic<uint64_t> thread_init_success_generation{no_init_generation};
  struct InitializationWorker {
    std::thread thread;
    std::shared_ptr<std::atomic<bool>> finished;
  };
  std::mutex init_workers_mtx;
  std::vector<InitializationWorker> init_workers;
  bool accept_init_workers = true;

  double segment_initialized_sensor_timestamp = std::numeric_limits<double>::quiet_NaN();
  std::vector<InitializationTiming> initialization_timing_history;

  // Ordered rolling IMU history used to seed a replacement pipeline at reset.
  std::vector<ov_core::ImuData> reset_imu_history;

  // If we did a zero velocity update
  std::atomic<bool> did_zupt_update{false};
  std::atomic<bool> has_moved_since_zupt{false};

  // Good features that where used in the last update (used in visualization)
  std::vector<Eigen::Vector3d> good_features_MSCKF;

  // Re-triangulated features 3d positions seen from the current frame (used in visualization)
  // For each feature we have a linear system A * p_FinG = b we create and increment their costs
  double active_tracks_time = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_image;
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
  std::map<size_t, int> active_feat_linsys_count;

};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGER_H
