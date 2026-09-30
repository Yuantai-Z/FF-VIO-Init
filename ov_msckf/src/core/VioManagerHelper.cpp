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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "types/LandmarkRepresentation.h"
#include "utils/print.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"

#include <system_error>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

std::shared_ptr<ov_type::IMU> VioManager::clone_imu_for_initialization(const std::shared_ptr<ov_type::IMU> &active_imu) {
  if (active_imu == nullptr) {
    return nullptr;
  }
  auto temporary_imu = std::make_shared<ov_type::IMU>();
  temporary_imu->set_value(active_imu->value());
  temporary_imu->set_fej(active_imu->fej());
  return temporary_imu;
}

bool VioManager::commit_initialized_imu(const std::shared_ptr<ov_type::IMU> &temporary_imu,
                                        const std::shared_ptr<ov_type::IMU> &active_imu,
                                        const Eigen::MatrixXd &covariance,
                                        std::vector<std::shared_ptr<ov_type::Type>> &order) {
  if (temporary_imu == nullptr || active_imu == nullptr) {
    return false;
  }

  const auto temporary_type = std::static_pointer_cast<ov_type::Type>(temporary_imu);
  if (order.size() != 1 || order.front().get() != temporary_type.get() || covariance.rows() != temporary_imu->size() ||
      covariance.cols() != temporary_imu->size() || !covariance.allFinite() || temporary_imu->value().rows() != 16 ||
      temporary_imu->value().cols() != 1 || !temporary_imu->value().allFinite() || temporary_imu->fej().rows() != 16 ||
      temporary_imu->fej().cols() != 1 || !temporary_imu->fej().allFinite()) {
    return false;
  }

  active_imu->set_value(temporary_imu->value());
  active_imu->set_fej(temporary_imu->fej());
  order.front() = active_imu;
  return true;
}

void VioManager::initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate) {

  std::lock_guard<std::mutex> commit_lock(init_commit_mtx);
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx);
  std::lock_guard<std::mutex> state_update_lock(state_update_mtx);

  // Initialize the system
  state->_imu->set_value(imustate.block(1, 0, 16, 1));
  state->_imu->set_fej(imustate.block(1, 0, 16, 1));

  // Fix the global yaw and position gauge freedoms
  // TODO: Why does this break out simulation consistency metrics?
  std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu};
  Eigen::MatrixXd Cov = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(state->_imu->size(), state->_imu->size());
  Cov.block(0, 0, 3, 3) = std::pow(0.017, 2) * Eigen::Matrix3d::Identity(); // q
  Cov.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();  // p
  Cov.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity();  // v (static)
  StateHelper::set_initial_covariance(state, Cov, order);

  // Set the state time
  state->_timestamp = imustate(0, 0);
  startup_time.store(imustate(0, 0));
  is_initialized_vio.store(true);

  const uint64_t generation = estimator_generation.load();
  const double timestamp_in_imu = imustate(0, 0) + params.calib_camimu_dt;
  InitializationTiming timing;
  timing.segment_id = generation;
  timing.reason = generation == 0 ? "startup" : "periodic_reset";
  if (std::isfinite(timestamp_in_imu)) {
    initialization_timing_history.push_back(timing);
    segment_initialized_sensor_timestamp = timestamp_in_imu;
    PRINT_INFO(GREEN "[init]: segment %llu initialized from ground truth\n" RESET,
               static_cast<unsigned long long>(generation));
  }

  // Cleanup any features older then the initialization time
  trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
  }

  // Print what we init'ed with
  PRINT_DEBUG(GREEN "[INIT]: INITIALIZED FROM GROUNDTRUTH FILE!!!!!\n" RESET);
  PRINT_DEBUG(GREEN "[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
              state->_imu->quat()(2), state->_imu->quat()(3));
  PRINT_DEBUG(GREEN "[INIT]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
              state->_imu->bias_g()(2));
  PRINT_DEBUG(GREEN "[INIT]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
  PRINT_DEBUG(GREEN "[INIT]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
              state->_imu->bias_a()(2));
  PRINT_DEBUG(GREEN "[INIT]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
}

bool VioManager::try_to_initialize(const ov_core::CameraData &message) {

  if (params.use_multi_threading_subs) {
    reap_finished_initialization_workers();
  }

  // Debug output to verify VGGT parameter transmission
  PRINT_DEBUG(BLUE "[INIT]: init_mode = %s (use_vggt_init = %s)\n" RESET,
              ov_init::init_mode_to_string(params.init_options.init_mode).c_str(),
              params.use_vggt_init() ? "true" : "false");

  uint64_t generation = estimator_generation.load();

  // Queue this camera time while the current generation is initializing. Recheck
  // under the queue lock because the thread may finish between the two reads.
  if (thread_init_running_generation.load() == generation) {
    std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
    if (thread_init_running_generation.load() == generation && estimator_generation.load() == generation) {
      camera_queue_init.push_back(message.timestamp);
      return false;
    }
  }

  if (thread_init_success_generation.load() == generation) {
    return true;
  }

  std::shared_ptr<ov_init::InertialInitializer> initializer_ptr;
  std::shared_ptr<State> state_ptr;
  std::shared_ptr<ov_core::TrackBase> track_feats_ptr;
  std::shared_ptr<ov_core::TrackBase> track_aruco_ptr;
  std::shared_ptr<Propagator> propagator_ptr;
  std::shared_ptr<UpdaterZeroVelocity> updater_zupt_ptr;
  std::shared_ptr<ov_type::IMU> initialization_imu_ptr;
  {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    generation = estimator_generation.load();
    if (thread_init_running_generation.load() == generation) {
      std::lock_guard<std::mutex> queue_lock(camera_queue_init_mtx);
      camera_queue_init.push_back(message.timestamp);
      return false;
    }
    if (thread_init_success_generation.load() == generation) {
      return true;
    }
    thread_init_running_generation.store(generation);
    initializer_ptr = initializer;
    state_ptr = state;
    track_feats_ptr = trackFEATS;
    track_aruco_ptr = trackARUCO;
    propagator_ptr = propagator;
    updater_zupt_ptr = updaterZUPT;
    initialization_imu_ptr = clone_imu_for_initialization(state_ptr->_imu);
  }

  // Captured shared pointers keep an old pipeline alive until its worker exits.
  // Only a matching generation is allowed to commit the result.
  auto initialize_pipeline = [this, generation, initializer_ptr, state_ptr, track_feats_ptr, track_aruco_ptr, propagator_ptr,
                              updater_zupt_ptr, initialization_imu_ptr] {
    // Returns from our initializer
    double timestamp;
    Eigen::MatrixXd covariance;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    ov_init::InitializationDiagnostics diagnostics;

    // Try to initialize the system
    // We will wait for a jerk if we do not have the zero velocity update enabled
    // Otherwise we can initialize right away as the zero velocity will handle the stationary case
    bool wait_for_jerk = (updater_zupt_ptr == nullptr);
    bool success = initializer_ptr->initialize(timestamp, covariance, order, initialization_imu_ptr, diagnostics, wait_for_jerk);

    // Reset and result commit are serialized. A generation mismatch means every
    // object captured above belongs to an obsolete local-coordinate segment.
    std::unique_lock<std::mutex> commit_lock(init_commit_mtx);
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx);
    std::unique_lock<std::mutex> state_update_lock(state_update_mtx);
    bool pipeline_matches = estimator_generation.load() == generation;
    if (pipeline_matches) {
      pipeline_matches = state.get() == state_ptr.get() && initializer.get() == initializer_ptr.get() &&
                         trackFEATS.get() == track_feats_ptr.get() && propagator.get() == propagator_ptr.get();
    }
    if (!pipeline_matches) {
      PRINT_WARNING(YELLOW "[init]: discarded result from obsolete estimator segment %llu\n" RESET,
                    static_cast<unsigned long long>(generation));
      uint64_t expected = generation;
      thread_init_running_generation.compare_exchange_strong(expected, no_init_generation);
      return;
    }

    // The initializer only writes its private IMU. Publish that result after the
    // pipeline/generation check and before using its covariance ordering.
    if (success && (!std::isfinite(timestamp) ||
                    !commit_initialized_imu(initialization_imu_ptr, state_ptr->_imu, covariance, order))) {
      PRINT_WARNING(YELLOW "[init]: rejected initialization result with an invalid IMU covariance order\n" RESET);
      success = false;
    }

    // If we have initialized successfully we will set the covariance and state elements as needed
    // TODO: set the clones and SLAM features here so we can start updating right away...
    if (success) {

      // Set our covariance (state should already be set in the initializer)
      StateHelper::set_initial_covariance(state_ptr, covariance, order);

      // Set the state time
      state_ptr->_timestamp = timestamp;
      startup_time.store(timestamp);

      // Cleanup any features older than the initialization time
      // Also increase the number of features to the desired amount during estimation
      // NOTE: we will split the total number of features over all cameras uniformly
      track_feats_ptr->get_feature_database()->cleanup_measurements(state_ptr->_timestamp);
      track_feats_ptr->set_num_features(std::floor((double)params.num_pts / (double)params.state_options.num_cameras));
      if (track_aruco_ptr != nullptr) {
        track_aruco_ptr->get_feature_database()->cleanup_measurements(state_ptr->_timestamp);
      }

      // If we are moving then don't do zero velocity update4
      if (state_ptr->_imu->vel().norm() > params.zupt_max_velocity) {
        has_moved_since_zupt.store(true);
      }

      double initialization_timestamp;
      initialization_timestamp = timestamp + params.calib_camimu_dt;
      InitializationTiming timing;
      timing.segment_id = generation;
      timing.reason = generation == 0 ? "startup" : "periodic_reset";
      timing.diagnostics = diagnostics;
      if (std::isfinite(initialization_timestamp)) {
        initialization_timing_history.push_back(timing);
        segment_initialized_sensor_timestamp = initialization_timestamp;
      }

      if (diagnostics.timing_valid()) {
        const double evaluation_time = diagnostics.success_oldest_time - diagnostics.first_attempt_oldest_time +
                                       diagnostics.init_window_time;
        PRINT_INFO(GREEN "[init]: segment %llu evaluation_time=%.5f (first_oldest=%.5f success_oldest=%.5f window=%.5f)\n" RESET,
                   static_cast<unsigned long long>(generation), evaluation_time, diagnostics.first_attempt_oldest_time,
                   diagnostics.success_oldest_time, diagnostics.init_window_time);
      } else {
        PRINT_DEBUG("[init]: segment %llu has no evaluation timing metadata\n", static_cast<unsigned long long>(generation));
      }
      PRINT_INFO(GREEN "[init]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state_ptr->_imu->quat()(0), state_ptr->_imu->quat()(1),
                 state_ptr->_imu->quat()(2), state_ptr->_imu->quat()(3));
      PRINT_INFO(GREEN "[init]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state_ptr->_imu->bias_g()(0), state_ptr->_imu->bias_g()(1),
                 state_ptr->_imu->bias_g()(2));
      PRINT_INFO(GREEN "[init]: velocity = %.4f, %.4f, %.4f\n" RESET, state_ptr->_imu->vel()(0), state_ptr->_imu->vel()(1),
                 state_ptr->_imu->vel()(2));
      PRINT_INFO(GREEN "[init]: bias accel = %.4f, %.4f, %.4f\n" RESET, state_ptr->_imu->bias_a()(0), state_ptr->_imu->bias_a()(1),
                 state_ptr->_imu->bias_a()(2));
      PRINT_INFO(GREEN "[init]: position = %.4f, %.4f, %.4f\n" RESET, state_ptr->_imu->pos()(0), state_ptr->_imu->pos()(1),
                 state_ptr->_imu->pos()(2));

      // Remove any camera times that are order then the initialized time
      // This can happen if the initialization has taken a while to perform
      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      std::vector<double> camera_timestamps_to_init;
      for (size_t i = 0; i < camera_queue_init.size(); i++) {
        if (camera_queue_init.at(i) > timestamp) {
          camera_timestamps_to_init.push_back(camera_queue_init.at(i));
        }
      }

      // Now we have initialized we will propagate the state to the current timestep
      // In general this should be ok as long as the initialization didn't take too long to perform
      // Propagating over multiple seconds will become an issue if the initial biases are bad
      size_t clone_rate = (size_t)((double)camera_timestamps_to_init.size() / (double)params.state_options.max_clone_size) + 1;
      for (size_t i = 0; i < camera_timestamps_to_init.size(); i += clone_rate) {
        propagator_ptr->propagate_and_clone(state_ptr, camera_timestamps_to_init.at(i));
        StateHelper::marginalize_old_clone(state_ptr);
      }
      PRINT_DEBUG(YELLOW "[init]: moved the state forward %.2f seconds\n" RESET, state_ptr->_timestamp - timestamp);
      thread_init_success_generation.store(generation);
      camera_queue_init.clear();
      uint64_t expected = generation;
      thread_init_running_generation.compare_exchange_strong(expected, no_init_generation);

    } else {
      PRINT_DEBUG(YELLOW "[init]: initialization attempt did not produce a state\n" RESET);
      thread_init_success_generation.store(no_init_generation);
      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      camera_queue_init.clear();
      uint64_t expected = generation;
      thread_init_running_generation.compare_exchange_strong(expected, no_init_generation);
    }
  };

  // If we are single threaded, then run single threaded
  // Otherwise retain the worker so shutdown can join it safely.
  if (!params.use_multi_threading_subs) {
    initialize_pipeline();
  } else {
    std::lock_guard<std::mutex> worker_lock(init_workers_mtx);
    if (!accept_init_workers) {
      uint64_t expected = generation;
      thread_init_running_generation.compare_exchange_strong(expected, no_init_generation);
      return false;
    }
    const auto finished = std::make_shared<std::atomic<bool>>(false);
    auto worker_body = [initialize_pipeline = std::move(initialize_pipeline), finished]() mutable {
      struct CompletionGuard {
        std::shared_ptr<std::atomic<bool>> flag;
        ~CompletionGuard() { flag->store(true, std::memory_order_release); }
      } completion{finished};
      initialize_pipeline();
    };
    try {
      init_workers.emplace_back();
      init_workers.back().finished = finished;
      init_workers.back().thread = std::thread(std::move(worker_body));
    } catch (const std::system_error &error) {
      init_workers.pop_back();
      uint64_t expected = generation;
      thread_init_running_generation.compare_exchange_strong(expected, no_init_generation);
      PRINT_ERROR(RED "[init]: unable to start initialization worker: %s\n" RESET, error.what());
      return false;
    }
  }
  return false;
}

void VioManager::retriangulate_active_tracks(const ov_core::CameraData &message) {

  // Start timing
  boost::posix_time::ptime retri_rT1, retri_rT2, retri_rT3;
  retri_rT1 = boost::posix_time::microsec_clock::local_time();

  // Clear old active track data
  assert(state->_clones_IMU.find(message.timestamp) != state->_clones_IMU.end());
  active_tracks_time = message.timestamp;
  active_image = cv::Mat();
  trackFEATS->display_active(active_image, 255, 255, 255, 255, 255, 255, " ");
  if (!active_image.empty()) {
    active_image = active_image(cv::Rect(0, 0, message.images.at(0).cols, message.images.at(0).rows));
  }
  active_tracks_posinG.clear();
  active_tracks_uvd.clear();

  // Current active tracks in our frontend
  // TODO: should probably assert here that these are at the message time...
  auto last_obs = trackFEATS->get_last_obs();
  auto last_ids = trackFEATS->get_last_ids();

  // New set of linear systems that only contain the latest track info
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A_new;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b_new;
  std::map<size_t, int> active_feat_linsys_count_new;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG_new;

  // Append our new observations for each camera
  std::map<size_t, cv::Point2f> feat_uvs_in_cam0;
  for (auto const &cam_id : message.sensor_ids) {

    // IMU historical clone
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(active_tracks_time)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(active_tracks_time)->pos();

    // Calibration for this cam_id
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();

    // Convert current CAMERA position relative to global
    Eigen::Matrix3d R_GtoCi = R_ItoC * R_GtoI;
    Eigen::Vector3d p_CiinG = p_IinG - R_GtoCi.transpose() * p_IinC;

    // Loop through each measurement
    assert(last_obs.find(cam_id) != last_obs.end());
    assert(last_ids.find(cam_id) != last_ids.end());
    for (size_t i = 0; i < last_obs.at(cam_id).size(); i++) {

      // Record this feature uv if is seen from cam0
      size_t featid = last_ids.at(cam_id).at(i);
      cv::Point2f pt_d = last_obs.at(cam_id).at(i).pt;
      if (cam_id == 0) {
        feat_uvs_in_cam0[featid] = pt_d;
      }

      // Skip this feature if it is a SLAM feature (the state estimate takes priority)
      if (state->_features_SLAM.find(featid) != state->_features_SLAM.end()) {
        continue;
      }

      // Get the UV coordinate normal
      cv::Point2f pt_n = state->_cam_intrinsics_cameras.at(cam_id)->undistort_cv(pt_d);
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << pt_n.x, pt_n.y, 1;
      b_i = R_GtoCi.transpose() * b_i;
      b_i = b_i / b_i.norm();
      Eigen::Matrix3d Bperp = skew_x(b_i);

      // Append to our linear system
      Eigen::Matrix3d Ai = Bperp.transpose() * Bperp;
      Eigen::Vector3d bi = Ai * p_CiinG;
      if (active_feat_linsys_A.find(featid) == active_feat_linsys_A.end()) {
        active_feat_linsys_A_new.insert({featid, Ai});
        active_feat_linsys_b_new.insert({featid, bi});
        active_feat_linsys_count_new.insert({featid, 1});
      } else {
        active_feat_linsys_A_new[featid] = Ai + active_feat_linsys_A[featid];
        active_feat_linsys_b_new[featid] = bi + active_feat_linsys_b[featid];
        active_feat_linsys_count_new[featid] = 1 + active_feat_linsys_count[featid];
      }

      // For this feature, recover its 3d position if we have enough observations!
      if (active_feat_linsys_count_new.at(featid) > 3) {

        // Recover feature estimate
        Eigen::Matrix3d A = active_feat_linsys_A_new[featid];
        Eigen::Vector3d b = active_feat_linsys_b_new[featid];
        Eigen::MatrixXd p_FinG = A.colPivHouseholderQr().solve(b);
        Eigen::MatrixXd p_FinCi = R_GtoCi * (p_FinG - p_CiinG);

        // Check A and p_FinCi
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
        Eigen::MatrixXd singularValues;
        singularValues.resize(svd.singularValues().rows(), 1);
        singularValues = svd.singularValues();
        double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

        // If we have a bad condition number, or it is too close
        // Then set the flag for bad (i.e. set z-axis to nan)
        if (std::abs(condA) <= params.featinit_options.max_cond_number && p_FinCi(2, 0) >= params.featinit_options.min_dist &&
            p_FinCi(2, 0) <= params.featinit_options.max_dist && !std::isnan(p_FinCi.norm())) {
          active_tracks_posinG_new[featid] = p_FinG;
        }
      }
    }
  }
  size_t total_triangulated = active_tracks_posinG.size();

  // Update active set of linear systems
  active_feat_linsys_A = active_feat_linsys_A_new;
  active_feat_linsys_b = active_feat_linsys_b_new;
  active_feat_linsys_count = active_feat_linsys_count_new;
  active_tracks_posinG = active_tracks_posinG_new;
  retri_rT2 = boost::posix_time::microsec_clock::local_time();

  // Return if no features
  if (active_tracks_posinG.empty() && state->_features_SLAM.empty())
    return;

  // Append our SLAM features we have
  for (const auto &feat : state->_features_SLAM) {
    Eigen::Vector3d p_FinG = feat.second->get_xyz(false);
    if (LandmarkRepresentation::is_relative_representation(feat.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(feat.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->Rot();
      Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->Rot();
      Eigen::Vector3d p_IinG = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feat.second->get_xyz(false) - p_IinC) + p_IinG;
    }
    active_tracks_posinG[feat.second->_featid] = p_FinG;
  }

  // Calibration of the first camera (cam0)
  std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(0);
  std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0);
  Eigen::Matrix<double, 3, 3> R_ItoC = calibration->Rot();
  Eigen::Matrix<double, 3, 1> p_IinC = calibration->pos();

  // Get current IMU clone state
  std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(active_tracks_time);
  Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
  Eigen::Vector3d p_IiinG = clone_Ii->pos();

  // 4. Next we can update our variable with the global position
  //    We also will project the features into the current frame
  for (const auto &feat : active_tracks_posinG) {

    // For now skip features not seen from current frame
    // TODO: should we publish other features not tracked in cam0??
    if (feat_uvs_in_cam0.find(feat.first) == feat_uvs_in_cam0.end())
      continue;

    // Calculate the depth of the feature in the current frame
    // Project SLAM feature and non-cam0 features into the current frame of reference
    Eigen::Vector3d p_FinIi = R_GtoIi * (feat.second - p_IiinG);
    Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
    double depth = p_FinCi(2);
    Eigen::Vector2d uv_dist;
    if (feat_uvs_in_cam0.find(feat.first) != feat_uvs_in_cam0.end()) {
      uv_dist << (double)feat_uvs_in_cam0.at(feat.first).x, (double)feat_uvs_in_cam0.at(feat.first).y;
    } else {
      Eigen::Vector2d uv_norm;
      uv_norm << p_FinCi(0) / depth, p_FinCi(1) / depth;
      uv_dist = state->_cam_intrinsics_cameras.at(0)->distort_d(uv_norm);
    }

    // Skip if not valid (i.e. negative depth, or outside of image)
    if (depth < 0.1) {
      continue;
    }

    // Skip if not valid (i.e. negative depth, or outside of image)
    int width = state->_cam_intrinsics_cameras.at(0)->w();
    int height = state->_cam_intrinsics_cameras.at(0)->h();
    if (uv_dist(0) < 0 || (int)uv_dist(0) >= width || uv_dist(1) < 0 || (int)uv_dist(1) >= height) {
      // PRINT_DEBUG("feat %zu -> depth = %.2f | u_d = %.2f | v_d = %.2f\n",(*it2)->featid,depth,uv_dist(0),uv_dist(1));
      continue;
    }

    // Finally construct the uv and depth
    Eigen::Vector3d uvd;
    uvd << uv_dist, depth;
    active_tracks_uvd.insert({feat.first, uvd});
  }
  retri_rT3 = boost::posix_time::microsec_clock::local_time();

  // Timing information
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds for triangulation (%zu tri of %zu active)\n" RESET,
            (retri_rT2 - retri_rT1).total_microseconds() * 1e-6, total_triangulated, active_feat_linsys_A.size());
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds for re-projection into current\n" RESET, (retri_rT3 - retri_rT2).total_microseconds() * 1e-6);
  PRINT_ALL(CYAN "[RETRI-TIME]: %.4f seconds total\n" RESET, (retri_rT3 - retri_rT1).total_microseconds() * 1e-6);
}

cv::Mat VioManager::get_historical_viz_image(double &timestamp) {

  std::shared_ptr<State> state_ptr;
  std::shared_ptr<ov_core::TrackBase> track_feats_ptr;
  std::shared_ptr<ov_core::TrackBase> track_aruco_ptr;
  uint64_t generation;
  {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    generation = estimator_generation.load();
    state_ptr = state;
    track_feats_ptr = trackFEATS;
    track_aruco_ptr = trackARUCO;
  }
  std::lock_guard<std::mutex> update_lock(state_update_mtx);
  if (estimator_generation.load() != generation) {
    return cv::Mat();
  }

  // Return if not ready yet
  if (state_ptr == nullptr || track_feats_ptr == nullptr)
    return cv::Mat();
  timestamp = state_ptr->_timestamp;

  // Build an id-list of what features we should highlight (i.e. SLAM)
  std::vector<size_t> highlighted_ids;
  for (const auto &feat : state_ptr->_features_SLAM) {
    highlighted_ids.push_back(feat.first);
  }

  // Text we will overlay if needed
  std::string overlay = (did_zupt_update) ? "zvupt" : "tracking";
  overlay = (!is_initialized_vio) ? "init" : overlay;
  // Get the current active tracks
  cv::Mat img_history;
  if (is_initialized_vio) {
    track_feats_ptr->display_history(img_history, 255, 255, 0, 255, 255, 255, highlighted_ids, overlay);
    if (track_aruco_ptr != nullptr) {
      track_aruco_ptr->display_history(img_history, 0, 255, 255, 255, 255, 255, highlighted_ids, overlay);
    }
  } else {
    track_feats_ptr->display_active(img_history, 255, 255, 0, 255, 255, 255, overlay);
  }

  // Finally return the image
  return img_history;
}

std::vector<Eigen::Vector3d> VioManager::get_features_SLAM() {
  const PipelineSnapshot pipeline = get_pipeline_snapshot();
  std::lock_guard<std::mutex> update_lock(state_update_mtx);
  if (estimator_generation.load() != pipeline.segment_id) {
    return {};
  }
  const std::shared_ptr<State> state_ptr = pipeline.state;
  std::vector<Eigen::Vector3d> slam_feats;
  for (auto &f : state_ptr->_features_SLAM) {
    if ((int)f.first <= 4 * state_ptr->_options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(f.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix<double, 3, 3> R_ItoC = state_ptr->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state_ptr->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix<double, 3, 3> R_GtoI = state_ptr->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state_ptr->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      slam_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    } else {
      slam_feats.push_back(f.second->get_xyz(false));
    }
  }
  return slam_feats;
}

std::vector<Eigen::Vector3d> VioManager::get_features_ARUCO() {
  const PipelineSnapshot pipeline = get_pipeline_snapshot();
  std::lock_guard<std::mutex> update_lock(state_update_mtx);
  if (estimator_generation.load() != pipeline.segment_id) {
    return {};
  }
  const std::shared_ptr<State> state_ptr = pipeline.state;
  std::vector<Eigen::Vector3d> aruco_feats;
  for (auto &f : state_ptr->_features_SLAM) {
    if ((int)f.first > 4 * state_ptr->_options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(f.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix<double, 3, 3> R_ItoC = state_ptr->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state_ptr->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix<double, 3, 3> R_GtoI = state_ptr->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state_ptr->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      aruco_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    } else {
      aruco_feats.push_back(f.second->get_xyz(false));
    }
  }
  return aruco_feats;
}
