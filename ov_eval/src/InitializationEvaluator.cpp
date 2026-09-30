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

#include "alignment/AlignTrajectory.h"
#include "alignment/AlignUtils.h"
#include "utils/Loader.h"
#include "utils/quat_ops.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace ov_eval {
namespace {

constexpr double kAssociationToleranceSeconds = 0.02;

std::string trim(const std::string &text) {
  const size_t begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const size_t end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

std::vector<std::string> split_whitespace(const std::string &text) {
  std::istringstream stream(text);
  std::vector<std::string> fields;
  std::string field;
  while (stream >> field) {
    fields.push_back(field);
  }
  return fields;
}

bool parse_double(const std::string &text, double &value) {
  try {
    size_t consumed = 0;
    value = std::stod(text, &consumed);
    return consumed == text.size() && std::isfinite(value);
  } catch (const std::exception &) {
    return false;
  }
}

bool parse_integer(const std::string &text, int &value) {
  try {
    size_t consumed = 0;
    long parsed = std::stol(text, &consumed);
    if (consumed != text.size() || parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
      return false;
    }
    value = static_cast<int>(parsed);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool parse_segment_id(const std::string &text, uint64_t &value) {
  if (text.empty() || !std::all_of(text.begin(), text.end(), [](char character) { return character >= '0' && character <= '9'; })) {
    return false;
  }
  try {
    size_t consumed = 0;
    value = std::stoull(text, &consumed);
    return consumed == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

bool normalize_pose_quaternion(TimedPose &timed_pose) {
  const double norm = timed_pose.pose.tail<4>().norm();
  if (!std::isfinite(norm) || norm < 1e-12) {
    return false;
  }
  timed_pose.pose.tail<4>() /= norm;
  return timed_pose.pose.allFinite();
}

bool parse_pose_fields(const std::vector<std::string> &fields, size_t offset, TimedPose &timed_pose) {
  if (fields.size() != offset + 8) {
    return false;
  }
  double values[8];
  for (size_t index = 0; index < 8; index++) {
    if (!parse_double(fields[offset + index], values[index])) {
      return false;
    }
  }
  timed_pose.timestamp = values[0];
  timed_pose.pose << values[1], values[2], values[3], values[4], values[5], values[6], values[7];
  return normalize_pose_quaternion(timed_pose);
}

bool parse_initialization_state(const std::vector<std::string> &fields, InitializationState &state) {
  if (fields.size() != 12) {
    return false;
  }
  double values[11];
  for (size_t index = 0; index < 11; index++) {
    if (!parse_double(fields[index + 1], values[index])) {
      return false;
    }
  }
  TimedPose timed_pose;
  timed_pose.timestamp = values[0];
  timed_pose.pose << values[1], values[2], values[3], values[4], values[5], values[6], values[7];
  if (!normalize_pose_quaternion(timed_pose)) {
    return false;
  }
  state.present = true;
  state.timestamp = timed_pose.timestamp;
  state.pose = timed_pose.pose;
  state.velocity << values[8], values[9], values[10];
  return state.velocity.allFinite();
}

std::string line_error(size_t line_number, const std::string &message) {
  return "line " + std::to_string(line_number) + ": " + message;
}

EvaluationValue unavailable(const std::string &reason) {
  EvaluationValue result;
  result.unavailable_reason = reason;
  return result;
}

EvaluationValue available(double value) {
  EvaluationValue result;
  result.valid = std::isfinite(value);
  result.value = value;
  if (!result.valid) {
    result.unavailable_reason = "calculation produced a non-finite value";
  }
  return result;
}

std::string display_value(const EvaluationValue &value, int precision) {
  if (!value.valid) {
    return "N/A";
  }
  std::ostringstream output;
  output << std::fixed << std::setprecision(precision) << value.value;
  return output.str();
}

struct MeanAccumulator {
  double sum = 0.0;
  size_t count = 0;

  void append(const EvaluationValue &value) {
    if (value.valid) {
      sum += value.value;
      count++;
    }
  }

  EvaluationValue mean() const {
    if (count == 0) {
      return unavailable("no valid sequence values");
    }
    return available(sum / static_cast<double>(count));
  }
};

void append_report_row(std::ostream &output, const std::string &sequence, const InitializationStateEvaluation &linear,
                       const InitializationStateEvaluation &nonlinear, const EvaluationValue &time) {
  output << "| " << sequence << " | " << display_value(linear.rotation_degrees, 3) << " | "
         << display_value(linear.gravity_degrees, 3) << " | " << display_value(linear.velocity_meters_per_second, 4) << " | "
         << display_value(nonlinear.rotation_degrees, 3) << " | " << display_value(nonlinear.gravity_degrees, 3) << " | "
         << display_value(nonlinear.velocity_meters_per_second, 4) << " | " << display_value(time, 3) << " |\n";
}

void add_diagnostic(InitializationSegmentEvaluation &evaluation, const std::string &diagnostic) {
  if (std::find(evaluation.diagnostics.begin(), evaluation.diagnostics.end(), diagnostic) == evaluation.diagnostics.end()) {
    evaluation.diagnostics.push_back(diagnostic);
  }
}

bool associate_poses(const std::vector<TimedPose> &estimate, const GroundTruthTrajectory &ground_truth,
                     std::vector<Eigen::Matrix<double, 7, 1>> &estimate_poses,
                     std::vector<Eigen::Matrix<double, 7, 1>> &ground_truth_poses) {
  std::vector<double> estimate_timestamps;
  std::vector<double> ground_truth_timestamps;
  for (const TimedPose &estimate_pose : estimate) {
    estimate_timestamps.push_back(estimate_pose.timestamp);
    estimate_poses.push_back(estimate_pose.pose);
  }
  for (const TimedPose &ground_truth_pose : ground_truth.poses) {
    ground_truth_timestamps.push_back(ground_truth_pose.timestamp);
    ground_truth_poses.push_back(ground_truth_pose.pose);
  }
  AlignUtils::perform_association(0.0, kAssociationToleranceSeconds, estimate_timestamps, ground_truth_timestamps,
                                  estimate_poses, ground_truth_poses);
  return estimate_poses.size() >= 3;
}

bool interpolate_ground_truth(double timestamp, const GroundTruthTrajectory &ground_truth,
                              Eigen::Matrix<double, 7, 1> &pose) {
  if (ground_truth.poses.empty()) {
    return false;
  }
  const auto lower = std::lower_bound(ground_truth.poses.begin(), ground_truth.poses.end(), timestamp,
                                      [](const TimedPose &entry, double query) { return entry.timestamp < query; });
  if (lower == ground_truth.poses.end()) {
    pose = ground_truth.poses.back().pose;
    return true;
  }
  if (lower == ground_truth.poses.begin()) {
    pose = lower->pose;
    return true;
  }

  const TimedPose &right = *lower;
  const TimedPose &left = *(lower - 1);
  const double alpha = (timestamp - left.timestamp) / (right.timestamp - left.timestamp);
  pose.head<3>() = (1.0 - alpha) * left.pose.head<3>() + alpha * right.pose.head<3>();
  Eigen::Vector4d left_quaternion = left.pose.tail<4>();
  Eigen::Vector4d right_quaternion = right.pose.tail<4>();
  if (left_quaternion.dot(right_quaternion) < 0.0) {
    right_quaternion = -right_quaternion;
  }
  pose.tail<4>() = (1.0 - alpha) * left_quaternion + alpha * right_quaternion;
  const double norm = pose.tail<4>().norm();
  if (!std::isfinite(norm) || norm < 1e-12) {
    return false;
  }
  pose.tail<4>() /= norm;
  return pose.allFinite();
}

bool ground_truth_velocity(double timestamp, const GroundTruthTrajectory &ground_truth, Eigen::Vector3d &velocity) {
  if (ground_truth.poses.size() < 2) {
    return false;
  }
  const auto lower = std::lower_bound(ground_truth.poses.begin(), ground_truth.poses.end(), timestamp,
                                      [](const TimedPose &entry, double query) { return entry.timestamp < query; });
  if (lower == ground_truth.poses.end()) {
    return false;
  }
  size_t index = static_cast<size_t>(std::distance(ground_truth.poses.begin(), lower));
  if (index > 0 && std::abs(ground_truth.poses[index - 1].timestamp - timestamp) <
                       std::abs(ground_truth.poses[index].timestamp - timestamp)) {
    index--;
  }
  if (std::abs(ground_truth.poses[index].timestamp - timestamp) > kAssociationToleranceSeconds) {
    return false;
  }

  size_t left = index;
  size_t right = index;
  if (index > 0 && index + 1 < ground_truth.poses.size()) {
    left = index - 1;
    right = index + 1;
  } else if (index + 1 < ground_truth.poses.size()) {
    right = index + 1;
  } else if (index > 0) {
    left = index - 1;
  }
  const double duration = ground_truth.poses[right].timestamp - ground_truth.poses[left].timestamp;
  if (duration <= 1e-6) {
    return false;
  }
  velocity = (ground_truth.poses[right].pose.head<3>() - ground_truth.poses[left].pose.head<3>()) / duration;
  return velocity.allFinite();
}

Eigen::Matrix<double, 7, 1> transform_pose(const Eigen::Matrix<double, 7, 1> &pose, const Eigen::Matrix3d &rotation,
                                           const Eigen::Vector3d &translation, double scale) {
  Eigen::Matrix<double, 7, 1> transformed;
  transformed.head<3>() = scale * rotation * pose.head<3>() + translation;
  const Eigen::Vector4d alignment_quaternion = ov_core::rot_2_quat(rotation);
  transformed.tail<4>() = ov_core::quat_multiply(pose.tail<4>(), ov_core::Inv(alignment_quaternion));
  return transformed;
}

InitializationStateEvaluation evaluate_state(const InitializationState &state, const GroundTruthTrajectory &ground_truth,
                                               const Eigen::Matrix3d &alignment_rotation,
                                               const Eigen::Vector3d &alignment_translation, double alignment_scale,
                                               const std::string &state_name, InitializationSegmentEvaluation &segment_evaluation) {
  InitializationStateEvaluation evaluation;
  if (!state.present) {
    const std::string reason = "missing # init_" + state_name + "_state";
    evaluation.rotation_degrees = unavailable(reason);
    evaluation.gravity_degrees = unavailable(reason);
    evaluation.velocity_meters_per_second = unavailable(reason);
    add_diagnostic(segment_evaluation, reason);
    return evaluation;
  }

  Eigen::Matrix<double, 7, 1> ground_truth_pose;
  if (!interpolate_ground_truth(state.timestamp, ground_truth, ground_truth_pose)) {
    const std::string reason = "ground truth pose is unavailable at the " + state_name + " state timestamp";
    evaluation.rotation_degrees = unavailable(reason);
    evaluation.gravity_degrees = unavailable(reason);
    evaluation.velocity_meters_per_second = unavailable(reason);
    add_diagnostic(segment_evaluation, reason);
    return evaluation;
  }

  const Eigen::Matrix<double, 7, 1> aligned =
      transform_pose(state.pose, alignment_rotation, alignment_translation, alignment_scale);
  const Eigen::Matrix3d rotation_error =
      ov_core::quat_2_Rot(aligned.tail<4>()).transpose() * ov_core::quat_2_Rot(ground_truth_pose.tail<4>());
  evaluation.rotation_degrees = available(180.0 / M_PI * ov_core::log_so3(rotation_error).norm());
  Eigen::Vector3d gravity_direction = rotation_error.col(2);
  gravity_direction.normalize();
  const double vertical_component = std::max(-1.0, std::min(1.0, gravity_direction(2)));
  evaluation.gravity_degrees = available(180.0 / M_PI * std::acos(vertical_component));

  Eigen::Vector3d reference_velocity;
  if (ground_truth_velocity(state.timestamp, ground_truth, reference_velocity)) {
    evaluation.velocity_meters_per_second = available((alignment_rotation * state.velocity - reference_velocity).norm());
  } else {
    evaluation.velocity_meters_per_second = available((alignment_rotation * state.velocity).norm());
    add_diagnostic(segment_evaluation, "ground truth velocity is unavailable within 0.020 s of the " + state_name +
                                           " state timestamp; using the legacy zero-velocity fallback");
  }
  return evaluation;
}

} // namespace

bool parse_initialization_result(const std::string &path, InitializationResultFile &result, std::string &error) {
  result = InitializationResultFile();
  std::ifstream input(path);
  if (!input.is_open()) {
    error = "unable to open result file: " + path;
    return false;
  }

  InitializationSegment *current_segment = nullptr;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    line_number++;
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    if (line.front() == '#') {
      const std::vector<std::string> fields = split_whitespace(trim(line.substr(1)));
      if (fields.empty()) {
        continue;
      }
      if (fields[0] == "openvins_result_version") {
        if (current_segment != nullptr || result.has_version || fields.size() != 2 || !parse_integer(fields[1], result.version)) {
          error = line_error(line_number, "invalid or repeated openvins_result_version");
          return false;
        }
        if (result.version != 2) {
          error = line_error(line_number, "unsupported openvins_result_version " + std::to_string(result.version));
          return false;
        }
        result.has_version = true;
      } else if (fields[0] == "auto_reset_interval_seconds") {
        if (current_segment != nullptr || result.has_auto_reset_interval || fields.size() != 2 ||
            !parse_double(fields[1], result.auto_reset_interval_seconds) || result.auto_reset_interval_seconds < 0.0) {
          error = line_error(line_number, "invalid or repeated auto_reset_interval_seconds");
          return false;
        }
        result.has_auto_reset_interval = true;
      } else if (fields[0] == "segment") {
        uint64_t segment_id = 0;
        if (fields.size() != 4 || fields[2] != "reason" || !parse_segment_id(fields[1], segment_id) ||
            (fields[3] != "startup" && fields[3] != "periodic_reset")) {
          error = line_error(line_number, "expected '# segment <id> reason <startup|periodic_reset>'");
          return false;
        }
        if (!result.segments.empty() && segment_id <= result.segments.back().id) {
          error = line_error(line_number, "segment IDs must be strictly increasing and unique");
          return false;
        }
        result.segments.emplace_back();
        current_segment = &result.segments.back();
        current_segment->id = segment_id;
        current_segment->reason = fields[3];
      } else if (fields[0] == "init_timing") {
        if (current_segment == nullptr) {
          error = line_error(line_number, "init_timing appears before a segment header");
          return false;
        }
        RawInitializationTiming &timing = current_segment->timing;
        if (fields.size() == 2 && fields[1] == "N/A") {
          if (timing.present) {
            error = line_error(line_number, "init_timing N/A follows recorded timing metadata");
            return false;
          }
          continue;
        }
        if (timing.present || fields.size() != 7 || fields[1] != "first_attempt_oldest_time" ||
            fields[3] != "success_oldest_time" || fields[5] != "init_window_time" ||
            !parse_double(fields[2], timing.first_attempt_oldest_time) || !parse_double(fields[4], timing.success_oldest_time) ||
            !parse_double(fields[6], timing.init_window_time) || timing.init_window_time < 0.0) {
          error = line_error(line_number, "invalid or repeated init_timing metadata");
          return false;
        }
        timing.present = true;
      } else if (fields[0] == "init_linear_state" || fields[0] == "init_nonlinear_state") {
        if (current_segment == nullptr) {
          error = line_error(line_number, fields[0] + " appears before a segment header");
          return false;
        }
        InitializationState &state = fields[0] == "init_linear_state" ? current_segment->linear_state : current_segment->nonlinear_state;
        if (fields.size() == 2 && fields[1] == "N/A") {
          if (state.present) {
            error = line_error(line_number, fields[0] + " N/A follows a recorded state");
            return false;
          }
          continue;
        }
        if (state.present || !parse_initialization_state(fields, state)) {
          error = line_error(line_number, "invalid or repeated " + fields[0]);
          return false;
        }
      }
      continue;
    }

    if (current_segment == nullptr) {
      error = line_error(line_number, "pose row appears before a segment header");
      return false;
    }
    const std::vector<std::string> fields = split_whitespace(line);
    TimedPose pose;
    if (!parse_pose_fields(fields, 0, pose)) {
      error = line_error(line_number, "expected an 8-column finite TUM pose row");
      return false;
    }
    if (!current_segment->poses.empty() && pose.timestamp <= current_segment->poses.back().timestamp) {
      error = line_error(line_number, "pose timestamps must be strictly increasing within each segment");
      return false;
    }
    current_segment->poses.push_back(pose);
  }

  if (result.segments.empty()) {
    error = "result file contains no '# segment' records";
    return false;
  }
  if (!result.has_version) {
    error = "result file is missing '# openvins_result_version 2'";
    return false;
  }
  if (!result.has_auto_reset_interval) {
    error = "result file is missing '# auto_reset_interval_seconds <value>'";
    return false;
  }
  return true;
}

bool load_initialization_ground_truth(const std::string &path, GroundTruthTrajectory &ground_truth, std::string &error) {
  ground_truth = GroundTruthTrajectory();
  std::ifstream probe(path);
  if (!probe.is_open()) {
    error = "unable to open ground-truth file: " + path;
    return false;
  }
  probe.close();

  std::vector<double> timestamps;
  std::vector<Eigen::Matrix<double, 7, 1>> poses;
  std::vector<Eigen::Matrix3d> orientation_covariances;
  std::vector<Eigen::Matrix3d> position_covariances;
  const bool is_csv = path.size() >= 4 && path.substr(path.size() - 4) == ".csv";
  if (is_csv) {
    Loader::load_data_csv(path, timestamps, poses, orientation_covariances, position_covariances);
  } else {
    Loader::load_data(path, timestamps, poses, orientation_covariances, position_covariances);
  }

  if (timestamps.size() != poses.size() || timestamps.empty()) {
    error = "ground-truth loader returned inconsistent or empty data";
    return false;
  }
  for (size_t index = 0; index < timestamps.size(); index++) {
    TimedPose pose;
    pose.timestamp = timestamps[index];
    pose.pose = poses[index];
    if (!std::isfinite(pose.timestamp) || !normalize_pose_quaternion(pose)) {
      error = "ground truth contains a non-finite value or invalid quaternion at row " + std::to_string(index + 1);
      return false;
    }
    if (!ground_truth.poses.empty() && pose.timestamp <= ground_truth.poses.back().timestamp) {
      error = "ground-truth timestamps must be strictly increasing (row " + std::to_string(index + 1) + ")";
      return false;
    }
    ground_truth.poses.push_back(pose);
  }
  return true;
}

bool parse_initialization_alignment(const std::string &text, InitializationAlignment &alignment) {
  if (text == "posyaw") {
    alignment = InitializationAlignment::POSYAW;
  } else if (text == "se3") {
    alignment = InitializationAlignment::SE3;
  } else if (text == "sim3") {
    alignment = InitializationAlignment::SIM3;
  } else {
    return false;
  }
  return true;
}

const char *initialization_alignment_name(InitializationAlignment alignment) {
  switch (alignment) {
    case InitializationAlignment::POSYAW:
      return "posyaw";
    case InitializationAlignment::SE3:
      return "se3";
    case InitializationAlignment::SIM3:
      return "sim3";
  }
  return "unknown";
}

InitializationSegmentEvaluation evaluate_initialization_segment(const InitializationSegment &segment,
                                                                 const GroundTruthTrajectory &ground_truth,
                                                                 InitializationAlignment alignment) {
  InitializationSegmentEvaluation evaluation;
  evaluation.id = segment.id;
  evaluation.reason = segment.reason;

  if (!segment.linear_state.present) {
    add_diagnostic(evaluation, "missing # init_linear_state");
  }
  if (!segment.nonlinear_state.present) {
    add_diagnostic(evaluation, "missing # init_nonlinear_state");
  }
  if (!segment.timing.present) {
    evaluation.time_seconds = unavailable("missing # init_timing");
    add_diagnostic(evaluation, "missing # init_timing");
  } else {
    const double elapsed = segment.timing.success_oldest_time - segment.timing.first_attempt_oldest_time +
                           segment.timing.init_window_time;
    if (!std::isfinite(elapsed) || elapsed < 0.0) {
      evaluation.time_seconds = unavailable("init_timing produces a negative or non-finite Time(s)");
      add_diagnostic(evaluation, evaluation.time_seconds.unavailable_reason);
    } else {
      evaluation.time_seconds = available(elapsed);
    }
  }

  std::vector<Eigen::Matrix<double, 7, 1>> estimate_poses;
  std::vector<Eigen::Matrix<double, 7, 1>> ground_truth_poses;
  if (segment.poses.size() < 3) {
    const std::string reason = "fewer than 3 segment poses are available for alignment";
    evaluation.linear.rotation_degrees = unavailable(reason);
    evaluation.linear.gravity_degrees = unavailable(reason);
    evaluation.linear.velocity_meters_per_second = unavailable(reason);
    evaluation.nonlinear = evaluation.linear;
    add_diagnostic(evaluation, reason);
    return evaluation;
  }
  if (!associate_poses(segment.poses, ground_truth, estimate_poses, ground_truth_poses)) {
    const std::string reason = "fewer than 3 ground-truth associations are available within 0.020 s";
    evaluation.linear.rotation_degrees = unavailable(reason);
    evaluation.linear.gravity_degrees = unavailable(reason);
    evaluation.linear.velocity_meters_per_second = unavailable(reason);
    evaluation.nonlinear = evaluation.linear;
    add_diagnostic(evaluation, reason);
    return evaluation;
  }

  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
  double scale = 1.0;
  const std::string alignment_name = initialization_alignment_name(alignment);
  AlignTrajectory::align_trajectory(estimate_poses, ground_truth_poses, rotation, translation, scale, alignment_name, -1);
  if (!rotation.allFinite() || !translation.allFinite() || !std::isfinite(scale) || scale <= 0.0) {
    const std::string reason = "trajectory alignment is degenerate";
    evaluation.linear.rotation_degrees = unavailable(reason);
    evaluation.linear.gravity_degrees = unavailable(reason);
    evaluation.linear.velocity_meters_per_second = unavailable(reason);
    evaluation.nonlinear = evaluation.linear;
    add_diagnostic(evaluation, reason);
    return evaluation;
  }

  evaluation.linear = evaluate_state(segment.linear_state, ground_truth, rotation, translation, scale, "linear", evaluation);
  evaluation.nonlinear = evaluate_state(segment.nonlinear_state, ground_truth, rotation, translation, scale, "nonlinear", evaluation);
  return evaluation;
}

std::string format_initialization_report(const std::vector<InitializationSegmentEvaluation> &evaluations) {
  std::ostringstream output;
  output << "| Sequence | Lin Rot (deg) | Lin Grav (deg) | Lin Vel (m/s) | NL Rot (deg) | NL Grav (deg) | NL Vel (m/s) | Time(s) |\n";
  output << "|---:|---:|---:|---:|---:|---:|---:|---:|\n";

  MeanAccumulator linear_rotation;
  MeanAccumulator linear_gravity;
  MeanAccumulator linear_velocity;
  MeanAccumulator nonlinear_rotation;
  MeanAccumulator nonlinear_gravity;
  MeanAccumulator nonlinear_velocity;
  MeanAccumulator initialization_time;
  for (const InitializationSegmentEvaluation &evaluation : evaluations) {
    append_report_row(output, std::to_string(evaluation.id), evaluation.linear, evaluation.nonlinear, evaluation.time_seconds);
    linear_rotation.append(evaluation.linear.rotation_degrees);
    linear_gravity.append(evaluation.linear.gravity_degrees);
    linear_velocity.append(evaluation.linear.velocity_meters_per_second);
    nonlinear_rotation.append(evaluation.nonlinear.rotation_degrees);
    nonlinear_gravity.append(evaluation.nonlinear.gravity_degrees);
    nonlinear_velocity.append(evaluation.nonlinear.velocity_meters_per_second);
    initialization_time.append(evaluation.time_seconds);
  }

  InitializationStateEvaluation linear_mean;
  linear_mean.rotation_degrees = linear_rotation.mean();
  linear_mean.gravity_degrees = linear_gravity.mean();
  linear_mean.velocity_meters_per_second = linear_velocity.mean();
  InitializationStateEvaluation nonlinear_mean;
  nonlinear_mean.rotation_degrees = nonlinear_rotation.mean();
  nonlinear_mean.gravity_degrees = nonlinear_gravity.mean();
  nonlinear_mean.velocity_meters_per_second = nonlinear_velocity.mean();
  append_report_row(output, "Mean", linear_mean, nonlinear_mean, initialization_time.mean());
  return output.str();
}

} // namespace ov_eval
