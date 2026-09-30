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
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

#include <cmath>
#include <stdexcept>
#include <string>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

constexpr uint64_t VioManager::no_init_generation;

VioManager::VioManager(VioManagerOptions &params_) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Set the IMU intrinsics
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  trackFEATS->set_viz_params(params.viz_font_scale, params.viz_font_thickness);

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
    trackARUCO->set_viz_params(params.viz_font_scale, params.viz_font_thickness);
  }

  // VGGT tracking uses preprocessed/undistorted images and their matching intrinsics.
  if (params.use_vggt_init()) {
    if (params.use_klt) {
      trackFEATS_vggt = std::shared_ptr<TrackBase>(new TrackKLT(params.vggt_camera_intrinsics, init_max_features,
                                                                 state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                                 params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
    } else {
      trackFEATS_vggt = std::shared_ptr<TrackBase>(new TrackDescriptor(
          params.vggt_camera_intrinsics, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
          params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
    }
    PRINT_DEBUG("[VioManager]: VGGT tracker created with VGGT intrinsics for init\n");
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  // When VGGT init is enabled, use VGGT tracker's feature database for initialization
  // This ensures VGGTInitializer uses features tracked on preprocessed images (matching VGGT coordinates)
  if (params.use_vggt_init() && trackFEATS_vggt != nullptr) {
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS_vggt->get_feature_database());
    PRINT_DEBUG("[VioManager]: Initializer using VGGT (rectified) feature database\n");
  } else {
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
  }

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }
}

VioManager::~VioManager() { wait_for_initialization_worker(); }

bool VioManager::periodic_reset_due(double timestamp) const {
  if (params.auto_reset_interval <= 0.0 || !std::isfinite(timestamp)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(lifecycle_mtx);
  if (!is_initialized_vio.load()) {
    return false;
  }
  const double timestamp_in_imu = timestamp + params.calib_camimu_dt;
  if (!std::isfinite(segment_initialized_sensor_timestamp) ||
      timestamp_in_imu - segment_initialized_sensor_timestamp < params.auto_reset_interval) {
    return false;
  }

  // Keep the replay history bounded. If a live camera queue falls farther
  // behind than that history, defer reset until a camera is bracketed again.
  return !reset_imu_history.empty() && reset_imu_history.front().timestamp <= timestamp_in_imu &&
         reset_imu_history.back().timestamp > timestamp_in_imu;
}

void VioManager::buffer_imu_for_reset_locked(const ov_core::ImuData &message) {
  if (!std::isfinite(message.timestamp)) {
    return;
  }

  const auto position = std::lower_bound(
      reset_imu_history.begin(), reset_imu_history.end(), message.timestamp,
      [](const ov_core::ImuData &sample, double timestamp) { return sample.timestamp < timestamp; });
  if (position != reset_imu_history.end() && position->timestamp == message.timestamp) {
    *position = message;
  } else {
    reset_imu_history.insert(position, message);
  }

  // Retain the full initialization horizon and one predecessor for boundary
  // interpolation. The public reset interval may be shorter than this horizon.
  const double newest = reset_imu_history.back().timestamp;
  const double keep_after = newest - params.init_options.init_window_time - 0.25;
  auto first_kept = std::lower_bound(
      reset_imu_history.begin(), reset_imu_history.end(), keep_after,
      [](const ov_core::ImuData &sample, double timestamp) { return sample.timestamp < timestamp; });
  if (first_kept != reset_imu_history.begin()) {
    --first_kept;
  }
  reset_imu_history.erase(reset_imu_history.begin(), first_kept);
}

std::vector<ov_core::ImuData> VioManager::select_reset_imu_replay_locked(double timestamp) const {
  std::vector<ov_core::ImuData> replay;
  if (!std::isfinite(timestamp) || reset_imu_history.empty()) {
    return replay;
  }

  auto first = std::lower_bound(
      reset_imu_history.begin(), reset_imu_history.end(), timestamp,
      [](const ov_core::ImuData &sample, double boundary) { return sample.timestamp < boundary; });
  if (first != reset_imu_history.begin()) {
    --first;
  }
  replay.assign(first, reset_imu_history.end());
  return replay;
}

std::vector<VioManager::InitializationTiming> VioManager::get_initialization_timings() const {
  std::lock_guard<std::mutex> lock(lifecycle_mtx);
  return initialization_timing_history;
}

VioManager::PipelineSnapshot VioManager::get_pipeline_snapshot() const {
  std::lock_guard<std::mutex> lock(lifecycle_mtx);
  PipelineSnapshot snapshot;
  snapshot.segment_id = estimator_generation.load();
  snapshot.state = state;
  snapshot.propagator = propagator;
  snapshot.initialized = is_initialized_vio.load() && timelastupdate.load() != -1.0;
  return snapshot;
}

bool VioManager::try_fast_state_propagate(double timestamp, Eigen::Matrix<double, 13, 1> &state_plus,
                                          Eigen::Matrix<double, 12, 12> &covariance,
                                          std::map<size_t, Eigen::Matrix<double, 7, 1>> &camera_extrinsics) {
  const PipelineSnapshot pipeline = get_pipeline_snapshot();
  if (!pipeline.initialized) {
    return false;
  }
  std::unique_lock<std::mutex> update_lock(state_update_mtx, std::try_to_lock);
  if (!update_lock.owns_lock() || estimator_generation.load() != pipeline.segment_id) {
    return false;
  }
  if (!pipeline.propagator->fast_state_propagate(pipeline.state, timestamp, state_plus, covariance)) {
    return false;
  }
  camera_extrinsics.clear();
  for (const auto &calibration : pipeline.state->_calib_IMUtoCAM) {
    camera_extrinsics.emplace(calibration.first, calibration.second->value());
  }
  return estimator_generation.load() == pipeline.segment_id;
}

double VioManager::camera_to_imu_offset() const {
  std::lock_guard<std::mutex> update_lock(state_update_mtx);
  return state->_calib_dt_CAMtoIMU->value()(0);
}

void VioManager::wait_for_initialization_worker() {
  std::vector<InitializationWorker> workers;
  {
    std::lock_guard<std::mutex> lock(init_workers_mtx);
    accept_init_workers = false;
    workers.swap(init_workers);
  }
  for (InitializationWorker &worker : workers) {
    if (worker.thread.joinable()) {
      worker.thread.join();
    }
  }
}

void VioManager::reap_finished_initialization_workers() {
  std::vector<std::thread> completed;
  {
    std::lock_guard<std::mutex> lock(init_workers_mtx);
    auto worker = init_workers.begin();
    while (worker != init_workers.end()) {
      if (worker->finished != nullptr && worker->finished->load(std::memory_order_acquire)) {
        completed.emplace_back(std::move(worker->thread));
        worker = init_workers.erase(worker);
      } else {
        ++worker;
      }
    }
  }
  for (std::thread &worker : completed) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

bool VioManager::reset_system(double timestamp) {

  // Block IMU ingestion until the replacement pipeline is ready. Constructing
  // outside this critical section would feed samples to the retiring pipeline
  // and silently drop them at the generation swap.
  std::unique_lock<std::mutex> commit_lock(init_commit_mtx);
  std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx);
  std::unique_lock<std::mutex> state_update_lock(state_update_mtx);
  const double timestamp_in_imu = timestamp + params.calib_camimu_dt;
  if (!is_initialized_vio.load() || !std::isfinite(segment_initialized_sensor_timestamp) ||
      timestamp_in_imu - segment_initialized_sensor_timestamp < params.auto_reset_interval || reset_imu_history.empty() ||
      reset_imu_history.front().timestamp > timestamp_in_imu ||
      reset_imu_history.back().timestamp <= timestamp_in_imu) {
    return false;
  }

  auto new_state = std::make_shared<State>(params.state_options);
  new_state->_calib_imu_dw->set_value(params.vec_dw);
  new_state->_calib_imu_dw->set_fej(params.vec_dw);
  new_state->_calib_imu_da->set_value(params.vec_da);
  new_state->_calib_imu_da->set_fej(params.vec_da);
  new_state->_calib_imu_tg->set_value(params.vec_tg);
  new_state->_calib_imu_tg->set_fej(params.vec_tg);
  new_state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  new_state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  new_state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  new_state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  Eigen::VectorXd temp_camimu_dt(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  new_state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  new_state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  new_state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < new_state->_options.num_cameras; i++) {
    new_state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    new_state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    new_state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    new_state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  const int init_max_features =
      std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  std::shared_ptr<TrackBase> new_track_feats;
  if (params.use_klt) {
    new_track_feats = std::make_shared<TrackKLT>(
        new_state->_cam_intrinsics_cameras, init_max_features, new_state->_options.max_aruco_features, params.use_stereo,
        params.histogram_method, params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist);
  } else {
    new_track_feats = std::make_shared<TrackDescriptor>(
        new_state->_cam_intrinsics_cameras, init_max_features, new_state->_options.max_aruco_features, params.use_stereo,
        params.histogram_method, params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio);
  }
  new_track_feats->set_viz_params(params.viz_font_scale, params.viz_font_thickness);

  std::shared_ptr<TrackBase> new_track_aruco;
  if (params.use_aruco) {
    new_track_aruco = std::make_shared<TrackAruco>(new_state->_cam_intrinsics_cameras, new_state->_options.max_aruco_features,
                                                   params.use_stereo, params.histogram_method, params.downsize_aruco);
    new_track_aruco->set_viz_params(params.viz_font_scale, params.viz_font_thickness);
  }

  std::shared_ptr<TrackBase> new_track_feats_vggt;
  if (params.use_vggt_init()) {
    if (params.use_klt) {
      new_track_feats_vggt = std::make_shared<TrackKLT>(
          params.vggt_camera_intrinsics, init_max_features, new_state->_options.max_aruco_features, params.use_stereo,
          params.histogram_method, params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist);
    } else {
      new_track_feats_vggt = std::make_shared<TrackDescriptor>(
          params.vggt_camera_intrinsics, init_max_features, new_state->_options.max_aruco_features, params.use_stereo,
          params.histogram_method, params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio);
    }
    new_track_feats_vggt->set_viz_params(params.viz_font_scale, params.viz_font_thickness);
  }

  auto new_propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);
  std::shared_ptr<ov_init::InertialInitializer> new_initializer;
  if (params.use_vggt_init() && new_track_feats_vggt != nullptr) {
    new_initializer =
        std::make_shared<ov_init::InertialInitializer>(params.init_options, new_track_feats_vggt->get_feature_database());
  } else {
    new_initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, new_track_feats->get_feature_database());
  }

  auto new_updater_msckf = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  auto new_updater_slam = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);
  std::shared_ptr<UpdaterZeroVelocity> new_updater_zupt;
  if (params.try_zupt) {
    new_updater_zupt = std::make_shared<UpdaterZeroVelocity>(
        params.zupt_options, params.imu_noises, new_track_feats->get_feature_database(), new_propagator, params.gravity_mag,
        params.zupt_max_velocity, params.zupt_noise_multiplier, params.zupt_max_disparity);
  }

  const std::vector<ov_core::ImuData> reset_imu_replay = select_reset_imu_replay_locked(timestamp_in_imu);
  for (const ov_core::ImuData &sample : reset_imu_replay) {
    new_propagator->feed_imu(sample, -1.0);
    new_initializer->feed_imu(sample, -1.0);
    if (new_updater_zupt != nullptr) {
      new_updater_zupt->feed_imu(sample, -1.0);
    }
  }

  const uint64_t new_generation = estimator_generation.load() + 1;
  estimator_generation.store(new_generation);
  state = std::move(new_state);
  trackFEATS = std::move(new_track_feats);
  trackFEATS_vggt = std::move(new_track_feats_vggt);
  trackARUCO = std::move(new_track_aruco);
  propagator = std::move(new_propagator);
  initializer = std::move(new_initializer);
  updaterMSCKF = std::move(new_updater_msckf);
  updaterSLAM = std::move(new_updater_slam);
  updaterZUPT = std::move(new_updater_zupt);

  is_initialized_vio.store(false);
  thread_init_running_generation.store(no_init_generation);
  thread_init_success_generation.store(no_init_generation);
  did_zupt_update.store(false);
  has_moved_since_zupt.store(false);
  timelastupdate.store(-1.0);
  imu_retention_timestamp.store(-1.0);
  startup_time.store(-1.0);
  distance = 0.0;
  segment_initialized_sensor_timestamp = std::numeric_limits<double>::quiet_NaN();

  active_tracks_time = -1.0;
  good_features_MSCKF.clear();
  active_tracks_posinG.clear();
  active_tracks_uvd.clear();
  active_feat_linsys_A.clear();
  active_feat_linsys_b.clear();
  active_feat_linsys_count.clear();
  active_image.release();

  {
    std::lock_guard<std::mutex> queue_lock(camera_queue_init_mtx);
    camera_queue_init.clear();
  }
  lifecycle_lock.unlock();
  commit_lock.unlock();

  PRINT_WARNING(YELLOW "[RESET]: starting estimator segment %llu at sensor timestamp %.5f (interval %.3f s)\n" RESET,
                static_cast<unsigned long long>(new_generation), timestamp_in_imu, params.auto_reset_interval);
  return true;
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // Keep the pipeline stable until every consumer accepts this IMU sample. No
  // called feed method re-enters VioManager, so this lock cannot self-deadlock.
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx);
  if (params.auto_reset_interval > 0.0) {
    buffer_imu_for_reset_locked(message);
  }
  const bool initialized_now = is_initialized_vio.load();

  // The camera thread publishes a conservative retention watermark. Reading
  // clone containers here would race asynchronous camera state updates.
  double oldest_time = imu_retention_timestamp.load();
  if (!initialized_now) {
    oldest_time = message.timestamp - params.init_options.init_window_time + params.calib_camimu_dt - 0.10;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!initialized_now) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (initialized_now && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt.load())) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();
  std::lock_guard<std::mutex> update_lock(state_update_mtx);

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  if (message_const.sensor_ids.empty() || message_const.sensor_ids.size() != message_const.images.size() ||
      message_const.images.size() != message_const.masks.size()) {
    throw std::invalid_argument("camera measurement must contain one image and mask per sensor ID");
  }
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }
  ov_core::CameraData message = message_const;

  if (periodic_reset_due(message.timestamp)) {
    reset_system(message.timestamp);
  }
  // The VGGT tracker/callback use the processed grid; the main tracker keeps raw images.
  if (params.use_vggt_init()) {
    ov_core::CameraData message_vggt;
    message_vggt.timestamp = message.timestamp;
    message_vggt.sensor_ids = message.sensor_ids;

    for (size_t i = 0; i < message.sensor_ids.size(); i++) {
      const size_t cam_id = message.sensor_ids.at(i);
      const auto plan_it = params.init_options.vggt_preprocess_plans.find(cam_id);
      if (plan_it == params.init_options.vggt_preprocess_plans.end()) {
        throw std::runtime_error("missing VGGT preprocessing plan for camera " + std::to_string(cam_id));
      }
      ov_core::VGGTPreprocessOutput processed =
          ov_core::applyVGGTPreprocessPlan(plan_it->second, message.images.at(i), message.masks.at(i));
      message_vggt.images.push_back(processed.image);
      message_vggt.masks.push_back(processed.mask);
    }

    if (trackFEATS_vggt != nullptr) {
      trackFEATS_vggt->feed_new_camera(message_vggt);
    }

    if (processed_camera_callback) {
      processed_camera_callback(message_vggt);
    }

    // message remains in the raw camera domain for the main tracker.
  }

  // Standard downsampling (only when NOT using vggt_init)
  else if (!params.use_vggt_init() && params.downsample_cameras) {
    for (size_t i = 0; i < message.sensor_ids.size(); i++) {
      cv::Mat img = message.images.at(i);
      cv::Mat mask = message.masks.at(i);
      cv::Mat img_temp, mask_temp;
      cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
      message.images.at(i) = img_temp;
      cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
      message.masks.at(i) = mask_temp;
    }
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    std::lock_guard<std::mutex> update_lock(state_update_mtx);
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
  }

  // Call on our propagate and update function
  std::lock_guard<std::mutex> update_lock(state_update_mtx);
  do_feature_propagate_update(message);
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    propagator->propagate_and_clone(state, message.timestamp);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      if ((int)cams.second.size() > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  updaterMSCKF->update(state, featsup_MSCKF);
  propagator->invalidate_cache();
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
    propagator->invalidate_cache();
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  updaterSLAM->delayed_init(state, feats_slam_DELAYED);
  rT6 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  // First do anchor change if we are about to lose an anchor pose
  updaterSLAM->change_anchors(state);

  // Cleanup any features older than the marginalization time
  if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
    trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
    }
  }

  // Finally marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  imu_retention_timestamp.store(state->margtimestep());
  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }

  // Debug for imu intrinsics
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_GYROtoIMU->value()(0), state->_calib_imu_GYROtoIMU->value()(1),
               state->_calib_imu_GYROtoIMU->value()(2), state->_calib_imu_GYROtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_ACCtoIMU->value()(0), state->_calib_imu_ACCtoIMU->value()(1),
               state->_calib_imu_ACCtoIMU->value()(2), state->_calib_imu_ACCtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.do_calib_imu_g_sensitivity) {
    PRINT_INFO("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_tg->value()(0),
               state->_calib_imu_tg->value()(1), state->_calib_imu_tg->value()(2), state->_calib_imu_tg->value()(3),
               state->_calib_imu_tg->value()(4), state->_calib_imu_tg->value()(5), state->_calib_imu_tg->value()(6),
               state->_calib_imu_tg->value()(7), state->_calib_imu_tg->value()(8));
  }
}
