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

#include "ros/ROSVisualizerHelper.h"
#include "state/State.h"
#include "state/StateOptions.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

int main() {
  ov_msckf::StateOptions options;
  auto state = std::make_shared<ov_msckf::State>(options);
  state->_timestamp = 10.0;

  Eigen::Matrix<double, 1, 1> time_offset;
  time_offset << 0.25;
  state->_calib_dt_CAMtoIMU->set_value(time_offset);

  Eigen::Matrix<double, 16, 1> imu_state = Eigen::Matrix<double, 16, 1>::Zero();
  imu_state.block<4, 1>(0, 0) << 0.182574, -0.365148, 0.547723, 0.730297;
  imu_state.block<3, 1>(4, 0) << 1.25, -2.5, 3.75;
  state->_imu->set_value(imu_state);

  std::ostringstream output;
  ov_msckf::ROSVisualizerHelper::save_tum_pose_to_file(state, output);

  std::istringstream row(output.str());
  double values[8] = {};
  for (double &value : values) {
    if (!(row >> value) || !std::isfinite(value)) {
      std::cerr << "TUM writer did not produce eight finite values\n";
      return 1;
    }
  }
  std::string extra;
  if (row >> extra) {
    std::cerr << "TUM writer produced an extra field: " << extra << "\n";
    return 1;
  }

  constexpr double tolerance = 1e-9;
  const double expected[8] = {10.25, 1.25, -2.5, 3.75, 0.182574, -0.365148, 0.547723, 0.730297};
  for (int i = 0; i < 8; ++i) {
    if (std::abs(values[i] - expected[i]) > tolerance) {
      std::cerr << "Unexpected TUM field " << i << ": " << values[i] << "\n";
      return 1;
    }
  }

  std::ostringstream segmented_output;
  ov_msckf::ROSVisualizerHelper::save_tum_header_to_file(10.0, segmented_output);
  ov_init::InitializationDiagnostics diagnostics;
  diagnostics.has_timing = true;
  diagnostics.has_linear_state = true;
  diagnostics.has_nonlinear_state = true;
  diagnostics.first_attempt_oldest_time = 1403636580.0;
  diagnostics.success_oldest_time = 1403636581.5;
  diagnostics.init_window_time = 1.0;
  diagnostics.linear_state = ov_init::make_initialization_state(
      1403636582.0, Eigen::Vector4d(0.1, 0.2, 0.3, 0.9), Eigen::Vector3d(1.0, 2.0, 3.0),
      Eigen::Vector3d(4.0, 5.0, 6.0));
  diagnostics.nonlinear_state = ov_init::make_initialization_state(
      1403636580.5, Eigen::Vector4d(0.2, 0.3, 0.4, 0.8), Eigen::Vector3d(-1.0, -2.0, -3.0),
      Eigen::Vector3d(-4.0, -5.0, -6.0));
  if (!diagnostics.complete()) {
    std::cerr << "Complete public initialization diagnostics were rejected\n";
    return 1;
  }
  ov_msckf::ROSVisualizerHelper::save_segment_header_to_file(7, "startup", diagnostics, segmented_output);
  ov_msckf::ROSVisualizerHelper::save_tum_pose_to_file(state, segmented_output);
  ov_msckf::ROSVisualizerHelper::save_segment_header_to_file(8, "periodic_reset", diagnostics, segmented_output);
  ov_msckf::ROSVisualizerHelper::save_tum_pose_to_file(state, segmented_output);

  std::istringstream segmented_rows(segmented_output.str());
  std::string line;
  int marker_count = 0;
  int timing_count = 0;
  int linear_count = 0;
  int nonlinear_count = 0;
  int pose_count = 0;
  while (std::getline(segmented_rows, line)) {
    if (!line.empty() && line.front() == '#') {
      if (line.find("# segment ") == 0) {
        if (line.find(" reason startup") == std::string::npos && line.find(" reason periodic_reset") == std::string::npos) {
          std::cerr << "Segment marker has no recognized reason\n";
          return 1;
        }
        ++marker_count;
      } else if (line.find("# init_timing ") == 0) {
        if (line.find(" first_attempt_oldest_time ") == std::string::npos ||
            line.find(" success_oldest_time ") == std::string::npos || line.find(" init_window_time ") == std::string::npos) {
          std::cerr << "Initialization timing row is not self-describing\n";
          return 1;
        }
        ++timing_count;
      } else if (line.find("# init_linear_state ") == 0 || line.find("# init_nonlinear_state ") == 0) {
        std::istringstream state_row(line);
        std::string hash;
        std::string label;
        state_row >> hash >> label;
        int fields = 0;
        double field = 0.0;
        while (state_row >> field) {
          if (!std::isfinite(field)) {
            std::cerr << "Initialization state contains a non-finite field\n";
            return 1;
          }
          ++fields;
        }
        if (fields != 11) {
          std::cerr << "Initialization state does not contain eleven numeric fields\n";
          return 1;
        }
        if (label == "init_linear_state") {
          ++linear_count;
        } else {
          ++nonlinear_count;
        }
      }
      continue;
    }
    std::istringstream pose_row(line);
    int fields = 0;
    double field = 0.0;
    while (pose_row >> field) {
      if (!std::isfinite(field)) {
        std::cerr << "Segmented trajectory contains a non-finite pose field\n";
        return 1;
      }
      ++fields;
    }
    if (fields != 8) {
      std::cerr << "Segmented trajectory pose does not contain eight fields\n";
      return 1;
    }
    ++pose_count;
  }
  if (marker_count != 2 || timing_count != 2 || linear_count != 2 || nonlinear_count != 2 || pose_count != 2) {
    std::cerr << "Single-file trajectory did not retain complete metadata for both reset segments\n";
    return 1;
  }
  const std::string serialized = segmented_output.str();
  if (serialized.find("# openvins_result_version 2\n") == std::string::npos ||
      serialized.find("# auto_reset_interval_seconds 10.000000000\n") == std::string::npos ||
      serialized.find("initialization_time_seconds") != std::string::npos) {
    std::cerr << "Versioned trajectory header or raw-only timing schema is invalid\n";
    return 1;
  }

  std::ostringstream unavailable_output;
  ov_msckf::ROSVisualizerHelper::save_segment_header_to_file(0, "startup", ov_init::InitializationDiagnostics(),
                                                              unavailable_output);
  if (unavailable_output.str() != "# segment 0 reason startup\n") {
    std::cerr << "Unavailable static or ground-truth diagnostics were not omitted cleanly\n";
    return 1;
  }

  return 0;
}
