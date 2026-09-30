/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Modifications Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "types/IMU.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#ifndef OV_MSCKF_TEST_CONFIG_PATH
#error "OV_MSCKF_TEST_CONFIG_PATH must name a complete estimator configuration"
#endif

namespace {

class TestVioManager : public ov_msckf::VioManager {
public:
  explicit TestVioManager(ov_msckf::VioManagerOptions &options) : VioManager(options) {}

  bool reset_due(double timestamp) const { return periodic_reset_due(timestamp); }
  bool reset(double timestamp) { return reset_system(timestamp); }
  void buffer_imu(const ov_core::ImuData &sample) {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    buffer_imu_for_reset_locked(sample);
  }
  std::vector<double> replay_timestamps(double timestamp) const {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    std::vector<double> timestamps;
    for (const ov_core::ImuData &sample : select_reset_imu_replay_locked(timestamp)) {
      timestamps.push_back(sample.timestamp);
    }
    return timestamps;
  }
  std::shared_ptr<std::atomic<bool>> start_completed_worker_for_test() {
    const auto finished = std::make_shared<std::atomic<bool>>(false);
    std::lock_guard<std::mutex> lock(init_workers_mtx);
    init_workers.emplace_back();
    init_workers.back().finished = finished;
    init_workers.back().thread = std::thread([finished] { finished->store(true, std::memory_order_release); });
    return finished;
  }
  void reap_workers_for_test() { reap_finished_initialization_workers(); }
  size_t worker_count_for_test() {
    std::lock_guard<std::mutex> lock(init_workers_mtx);
    return init_workers.size();
  }
  void clear_reset_imu_for_test() {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    reset_imu_history.clear();
  }
  std::shared_ptr<ov_type::IMU> active_imu_for_test() {
    std::lock_guard<std::mutex> lock(lifecycle_mtx);
    return state->_imu;
  }
  static std::shared_ptr<ov_type::IMU> clone_imu_for_test(const std::shared_ptr<ov_type::IMU> &active_imu) {
    return clone_imu_for_initialization(active_imu);
  }
  static bool commit_imu_for_test(const std::shared_ptr<ov_type::IMU> &temporary_imu,
                                  const std::shared_ptr<ov_type::IMU> &active_imu,
                                  const Eigen::MatrixXd &covariance,
                                  std::vector<std::shared_ptr<ov_type::Type>> &order) {
    return commit_initialized_imu(temporary_imu, active_imu, covariance, order);
  }
};

} // namespace

int main() {
  ov_core::Printer::setPrintLevel("SILENT");
  auto parser = std::make_shared<ov_core::YamlParser>(OV_MSCKF_TEST_CONFIG_PATH);
  ov_msckf::VioManagerOptions options;
  options.print_and_load(parser);
  if (!parser->successful()) {
    std::cerr << "Failed to load reset regression configuration\n";
    return 1;
  }

  ov_msckf::VioManagerOptions invalid_options = options;
  invalid_options.auto_reset_interval = -0.1;
  bool rejected_negative = false;
  try {
    invalid_options.print_and_load_estimator();
  } catch (const std::invalid_argument &) {
    rejected_negative = true;
  }
  invalid_options.auto_reset_interval = std::numeric_limits<double>::infinity();
  bool rejected_infinite = false;
  try {
    invalid_options.print_and_load_estimator();
  } catch (const std::invalid_argument &) {
    rejected_infinite = true;
  }
  if (!rejected_negative || !rejected_infinite) {
    std::cerr << "Invalid reset interval was accepted\n";
    return 1;
  }

  options.auto_reset_interval = 0.5;
  options.use_multi_threading_subs = false;

  TestVioManager manager(options);

  const auto active_imu = manager.active_imu_for_test();
  const Eigen::MatrixXd active_value_before_worker = active_imu->value();
  const Eigen::MatrixXd active_fej_before_worker = active_imu->fej();
  const auto temporary_imu = TestVioManager::clone_imu_for_test(active_imu);
  Eigen::MatrixXd initialized_value = temporary_imu->value();
  Eigen::MatrixXd initialized_fej = temporary_imu->fej();
  initialized_value(4) = 12.5;
  initialized_fej(4) = -3.25;
  std::thread simulated_initialization_worker([temporary_imu, initialized_value, initialized_fej] {
    temporary_imu->set_value(initialized_value);
    temporary_imu->set_fej(initialized_fej);
  });
  simulated_initialization_worker.join();
  if (!active_imu->value().isApprox(active_value_before_worker) || !active_imu->fej().isApprox(active_fej_before_worker)) {
    std::cerr << "Initializer worker temporary state leaked into the active IMU before commit\n";
    return 1;
  }
  std::vector<std::shared_ptr<ov_type::Type>> invalid_initialization_order;
  const Eigen::MatrixXd initialization_covariance = Eigen::MatrixXd::Identity(temporary_imu->size(), temporary_imu->size());
  if (TestVioManager::commit_imu_for_test(temporary_imu, active_imu, initialization_covariance, invalid_initialization_order) ||
      !active_imu->value().isApprox(active_value_before_worker) || !active_imu->fej().isApprox(active_fej_before_worker)) {
    std::cerr << "Rejected initializer result modified the active IMU\n";
    return 1;
  }
  const auto nonfinite_imu = TestVioManager::clone_imu_for_test(temporary_imu);
  Eigen::MatrixXd nonfinite_value = nonfinite_imu->value();
  nonfinite_value(0) = std::numeric_limits<double>::quiet_NaN();
  nonfinite_imu->set_value(nonfinite_value);
  std::vector<std::shared_ptr<ov_type::Type>> nonfinite_order = {nonfinite_imu};
  if (TestVioManager::commit_imu_for_test(nonfinite_imu, active_imu, initialization_covariance, nonfinite_order) ||
      !active_imu->value().isApprox(active_value_before_worker) || !active_imu->fej().isApprox(active_fej_before_worker)) {
    std::cerr << "Non-finite initializer result modified the active IMU\n";
    return 1;
  }
  std::vector<std::shared_ptr<ov_type::Type>> initialization_order = {temporary_imu};
  if (!TestVioManager::commit_imu_for_test(temporary_imu, active_imu, initialization_covariance, initialization_order) ||
      initialization_order.front().get() != active_imu.get() || !active_imu->value().isApprox(initialized_value) ||
      !active_imu->fej().isApprox(initialized_fej)) {
    std::cerr << "Accepted initializer IMU was not atomically copied and remapped at commit\n";
    return 1;
  }

  for (int iteration = 0; iteration < 64; ++iteration) {
    const auto finished = manager.start_completed_worker_for_test();
    while (!finished->load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    manager.reap_workers_for_test();
    if (manager.worker_count_for_test() != 0) {
      std::cerr << "Completed initialization workers accumulated in the registry\n";
      return 1;
    }
  }

  ov_core::ImuData imu_sample;
  imu_sample.wm.setZero();
  imu_sample.am.setZero();
  for (const double timestamp : {10.1, 9.9, 10.0, 10.1}) {
    imu_sample.timestamp = timestamp;
    manager.buffer_imu(imu_sample);
  }
  const std::vector<double> replay = manager.replay_timestamps(10.0);
  if (replay.size() != 3 || replay.at(0) != 9.9 || replay.at(1) != 10.0 || replay.at(2) != 10.1) {
    std::cerr << "Reset replay did not retain an ordered boundary predecessor without duplicates\n";
    return 1;
  }

  if (manager.reset_due(10.5)) {
    std::cerr << "Reset was allowed before the segment initialized\n";
    return 1;
  }
  for (const double timestamp : {10.4, 10.6, 20.0}) {
    imu_sample.timestamp = timestamp;
    manager.buffer_imu(imu_sample);
  }
  const std::vector<double> delayed_replay = manager.replay_timestamps(10.5);
  if (delayed_replay.size() != 2 || delayed_replay.at(0) != 10.6 || delayed_replay.at(1) != 20.0) {
    std::cerr << "Bounded reset replay retained an unexpected stale camera boundary\n";
    return 1;
  }

  Eigen::Matrix<double, 17, 1> ground_truth = Eigen::Matrix<double, 17, 1>::Zero();
  ground_truth(0) = 10.25;
  ground_truth(4) = 1.0;
  manager.initialize_with_gt(ground_truth);
  const auto initial_pipeline = manager.get_pipeline_snapshot();
  const auto initial_timings = manager.get_initialization_timings();
  if (initial_pipeline.segment_id != 0 || initial_timings.size() != 1 || initial_timings.front().segment_id != 0 ||
      initial_timings.front().reason != "startup" || initial_timings.front().diagnostics.complete()) {
    std::cerr << "Ground-truth segment record or unavailable diagnostics were not retained\n";
    return 1;
  }
  if (manager.reset_due(10.749) || !manager.reset_due(10.75)) {
    std::cerr << "Positive reset interval did not start at successful initialization\n";
    return 1;
  }

  const auto before_replay_race = manager.get_pipeline_snapshot();
  manager.clear_reset_imu_for_test();
  if (manager.reset(10.75) || manager.get_pipeline_snapshot().segment_id != before_replay_race.segment_id) {
    std::cerr << "Reset ignored loss of its previously validated IMU bracket\n";
    return 1;
  }
  for (const double timestamp : {10.7, 10.8}) {
    imu_sample.timestamp = timestamp;
    manager.buffer_imu(imu_sample);
  }

  if (!manager.reset(10.75)) {
    std::cerr << "Valid bracketed reset was rejected\n";
    return 1;
  }
  const auto second_pipeline = manager.get_pipeline_snapshot();
  if (second_pipeline.segment_id != 1 || second_pipeline.state == initial_pipeline.state ||
      second_pipeline.propagator == initial_pipeline.propagator || second_pipeline.initialized) {
    std::cerr << "Reset did not replace the estimator pipeline\n";
    return 1;
  }

  if (manager.reset_due(11.25)) {
    std::cerr << "Uninitialized segment was reset repeatedly\n";
    return 1;
  }
  if (manager.reset(11.25)) {
    std::cerr << "Unbracketed reset replaced the estimator pipeline\n";
    return 1;
  }
  const auto third_pipeline = manager.get_pipeline_snapshot();
  const auto retained_timings = manager.get_initialization_timings();
  if (third_pipeline.segment_id != 1 || third_pipeline.state != second_pipeline.state || retained_timings.size() != 1 ||
      retained_timings.front().segment_id != 0) {
    std::cerr << "Rejected reset changed the pipeline or lost an earlier timing event\n";
    return 1;
  }

  ground_truth(0) = 10.9;
  manager.initialize_with_gt(ground_truth);
  const auto committed_timings = manager.get_initialization_timings();
  if (committed_timings.size() != 2 || committed_timings.front().reason != "startup" ||
      committed_timings.back().segment_id != 1 || committed_timings.back().reason != "periodic_reset" ||
      committed_timings.front().diagnostics.complete() || committed_timings.back().diagnostics.complete()) {
    std::cerr << "Ground-truth reset segment did not retain its reason and unavailable diagnostics\n";
    return 1;
  }

  return 0;
}
