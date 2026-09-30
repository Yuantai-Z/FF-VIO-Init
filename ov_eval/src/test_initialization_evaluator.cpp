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

#include "InitializationEvaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

bool require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
  }
  return condition;
}

bool near(double actual, double expected, double tolerance = 1e-8) {
  return std::abs(actual - expected) <= tolerance;
}

bool state_is_zero(const ov_eval::InitializationStateEvaluation &evaluation) {
  return evaluation.rotation_degrees.valid && evaluation.gravity_degrees.valid &&
         evaluation.velocity_meters_per_second.valid && near(evaluation.rotation_degrees.value, 0.0, 1e-6) &&
         near(evaluation.gravity_degrees.value, 0.0, 1e-6) && near(evaluation.velocity_meters_per_second.value, 0.0, 1e-8);
}

bool diagnostics_contain(const ov_eval::InitializationSegmentEvaluation &evaluation, const std::string &needle) {
  for (const std::string &diagnostic : evaluation.diagnostics) {
    if (diagnostic.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  const std::string data_directory = OV_EVAL_TEST_DATA_DIR;
  std::string error;
  ov_eval::InitializationResultFile result;
  if (!require(ov_eval::parse_initialization_result(data_directory + "/initialization_result_v2.tum", result, error), error) ||
      !require(result.has_version && result.version == 2, "result version was not parsed") ||
      !require(result.has_auto_reset_interval && near(result.auto_reset_interval_seconds, 3.0), "reset interval was not parsed") ||
      !require(result.segments.size() == 2, "expected two reset-separated segments") ||
      !require(result.segments[0].id == 0 && result.segments[1].id == 1, "segment IDs were not parsed")) {
    return 1;
  }

  ov_eval::GroundTruthTrajectory ground_truth;
  if (!require(ov_eval::load_initialization_ground_truth(data_directory + "/groundtruth.tum", ground_truth, error), error)) {
    return 1;
  }
  const ov_eval::InitializationSegmentEvaluation first =
      ov_eval::evaluate_initialization_segment(result.segments[0], ground_truth, ov_eval::InitializationAlignment::POSYAW);
  const ov_eval::InitializationSegmentEvaluation second =
      ov_eval::evaluate_initialization_segment(result.segments[1], ground_truth, ov_eval::InitializationAlignment::POSYAW);
  if (!require(state_is_zero(first.linear) && state_is_zero(first.nonlinear), "segment 0 initialization errors should be zero") ||
      !require(state_is_zero(second.linear) && state_is_zero(second.nonlinear),
               "segment 1 must be aligned independently and have zero errors") ||
      !require(first.time_seconds.valid && near(first.time_seconds.value, 2.0), "segment 0 old Time(s) formula is wrong") ||
      !require(second.time_seconds.valid && near(second.time_seconds.value, 1.5), "segment 1 old Time(s) formula is wrong")) {
    return 1;
  }

  const std::string report = ov_eval::format_initialization_report({first, second});
  if (!require(std::count(report.begin(), report.end(), '\n') == 5, "compact report must contain only header, sequences, and mean") ||
      !require(report.find("| Sequence | Lin Rot (deg) |") == 0, "compact report header is missing") ||
      !require(report.find("| 0 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.0000 | 2.000 |") != std::string::npos,
               "segment 0 report row is wrong") ||
      !require(report.find("| 1 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.0000 | 1.500 |") != std::string::npos,
               "segment 1 report row is wrong") ||
      !require(report.find("| Mean | 0.000 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.0000 | 1.750 |") != std::string::npos,
               "mean report row is wrong") ||
      !require(report.find("Diagnostics") == std::string::npos && report.find("startup") == std::string::npos,
               "compact report must not contain prose or diagnostics")) {
    return 1;
  }

  ov_eval::GroundTruthTrajectory csv_ground_truth;
  if (!require(ov_eval::load_initialization_ground_truth(data_directory + "/groundtruth.csv", csv_ground_truth, error), error) ||
      !require(csv_ground_truth.poses.size() == 3 && near(csv_ground_truth.poses.front().timestamp, 20.0),
               "EuRoC CSV timestamps were not converted from nanoseconds") ||
      !require(near(csv_ground_truth.poses.back().pose(0), 2.0), "EuRoC CSV positions were not loaded")) {
    return 1;
  }

  ov_eval::InitializationSegment incomplete = result.segments[0];
  incomplete.timing.present = false;
  incomplete.linear_state.present = false;
  const ov_eval::InitializationSegmentEvaluation incomplete_evaluation =
      ov_eval::evaluate_initialization_segment(incomplete, ground_truth, ov_eval::InitializationAlignment::POSYAW);
  if (!require(!incomplete_evaluation.time_seconds.valid, "missing timing must be N/A") ||
      !require(!incomplete_evaluation.linear.rotation_degrees.valid, "missing linear state must be N/A") ||
      !require(incomplete_evaluation.nonlinear.rotation_degrees.valid, "available nonlinear state should still be evaluated")) {
    return 1;
  }

  ov_eval::GroundTruthTrajectory velocity_ground_truth;
  const double velocity_timestamps[4] = {0.0, 0.015625, 0.03125, 0.046875};
  const double velocity_positions[4][2] = {{0.0, 0.0}, {0.0, 0.015625}, {0.03125, 0.0}, {0.09375, 0.015625}};
  for (size_t index = 0; index < 4; index++) {
    ov_eval::TimedPose pose;
    pose.timestamp = velocity_timestamps[index];
    pose.pose << velocity_positions[index][0], velocity_positions[index][1], 0.0, 0.0, 0.0, 0.0, 1.0;
    velocity_ground_truth.poses.push_back(pose);
  }
  ov_eval::InitializationSegment velocity_segment;
  velocity_segment.id = 7;
  velocity_segment.reason = "startup";
  velocity_segment.poses = velocity_ground_truth.poses;
  velocity_segment.linear_state.present = true;
  velocity_segment.linear_state.timestamp = 0.0234375;
  velocity_segment.linear_state.pose << 0.015625, 0.0078125, 0.0, 0.0, 0.0, 0.0, 1.0;
  velocity_segment.linear_state.velocity.setZero();
  velocity_segment.nonlinear_state = velocity_segment.linear_state;
  velocity_segment.nonlinear_state.timestamp = 0.0546875;
  velocity_segment.nonlinear_state.pose << 0.09375, 0.015625, 0.0, 0.0, 0.0, 0.0, 1.0;
  velocity_segment.nonlinear_state.velocity << 1.0, 2.0, 2.0;
  const ov_eval::InitializationSegmentEvaluation velocity_evaluation =
      ov_eval::evaluate_initialization_segment(velocity_segment, velocity_ground_truth, ov_eval::InitializationAlignment::POSYAW);
  if (!require(velocity_evaluation.linear.velocity_meters_per_second.valid &&
                   near(velocity_evaluation.linear.velocity_meters_per_second.value, 3.0),
               "an equidistant velocity timestamp must keep the upper ground-truth sample") ||
      !require(velocity_evaluation.nonlinear.velocity_meters_per_second.valid &&
                   near(velocity_evaluation.nonlinear.velocity_meters_per_second.value, 3.0),
               "a timestamp after the last ground-truth pose must use the legacy zero-velocity fallback") ||
      !require(diagnostics_contain(velocity_evaluation, "legacy zero-velocity fallback"),
               "the zero-velocity fallback must be reported as a diagnostic")) {
    return 1;
  }

  ov_eval::InitializationSegment too_short = result.segments[0];
  too_short.poses.resize(2);
  const ov_eval::InitializationSegmentEvaluation too_short_evaluation =
      ov_eval::evaluate_initialization_segment(too_short, ground_truth, ov_eval::InitializationAlignment::POSYAW);
  if (!require(!too_short_evaluation.linear.rotation_degrees.valid && too_short_evaluation.time_seconds.valid,
               "insufficient poses must be N/A without discarding valid Time(s)")) {
    return 1;
  }

  const std::string malformed_path = "/tmp/ov_eval_duplicate_segment_regression.tum";
  {
    std::ofstream malformed(malformed_path);
    malformed << "# openvins_result_version 2\n"
              << "# auto_reset_interval_seconds 1.0\n"
              << "# segment 0 reason startup\n"
              << "# segment 0 reason periodic_reset\n";
  }
  ov_eval::InitializationResultFile malformed_result;
  const bool malformed_accepted = ov_eval::parse_initialization_result(malformed_path, malformed_result, error);
  std::remove(malformed_path.c_str());
  if (!require(!malformed_accepted && error.find("strictly increasing") != std::string::npos,
               "duplicate segment IDs must be rejected explicitly")) {
    return 1;
  }

  const std::string unversioned_path = "/tmp/ov_eval_unversioned_result_regression.tum";
  {
    std::ofstream unversioned(unversioned_path);
    unversioned << "# auto_reset_interval_seconds 0.0\n"
                << "# segment 0 reason startup\n";
  }
  ov_eval::InitializationResultFile unversioned_result;
  const bool unversioned_accepted = ov_eval::parse_initialization_result(unversioned_path, unversioned_result, error);
  std::remove(unversioned_path.c_str());
  if (!require(!unversioned_accepted && error.find("openvins_result_version") != std::string::npos,
               "unversioned result files must be rejected explicitly")) {
    return 1;
  }

  const std::string unavailable_path = "/tmp/ov_eval_unavailable_metadata_regression.tum";
  {
    std::ofstream unavailable(unavailable_path);
    unavailable << "# openvins_result_version 2\n"
                << "# auto_reset_interval_seconds 0.0\n"
                << "# segment 0 reason startup\n"
                << "# init_timing N/A\n"
                << "# init_linear_state N/A\n"
                << "# init_nonlinear_state N/A\n";
  }
  ov_eval::InitializationResultFile unavailable_result;
  const bool unavailable_accepted =
      ov_eval::parse_initialization_result(unavailable_path, unavailable_result, error);
  std::remove(unavailable_path.c_str());
  if (!require(unavailable_accepted && unavailable_result.segments.size() == 1 &&
                   !unavailable_result.segments.front().timing.present &&
                   !unavailable_result.segments.front().linear_state.present &&
                   !unavailable_result.segments.front().nonlinear_state.present,
               "explicit N/A metadata must remain unavailable without failing the parser")) {
    return 1;
  }

  std::cout << "Initialization evaluator regression passed\n";
  return 0;
}
