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

#ifndef OV_EVAL_INITIALIZATION_EVALUATOR_H
#define OV_EVAL_INITIALIZATION_EVALUATOR_H

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <vector>

namespace ov_eval {

enum class InitializationAlignment { POSYAW, SE3, SIM3 };

struct TimedPose {
  double timestamp = -1.0;
  Eigen::Matrix<double, 7, 1> pose = Eigen::Matrix<double, 7, 1>::Zero();
};

struct InitializationState {
  bool present = false;
  double timestamp = -1.0;
  Eigen::Matrix<double, 7, 1> pose = Eigen::Matrix<double, 7, 1>::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
};

struct RawInitializationTiming {
  bool present = false;
  double first_attempt_oldest_time = -1.0;
  double success_oldest_time = -1.0;
  double init_window_time = -1.0;
};

struct InitializationSegment {
  uint64_t id = 0;
  std::string reason;
  RawInitializationTiming timing;
  InitializationState linear_state;
  InitializationState nonlinear_state;
  std::vector<TimedPose> poses;
};

struct InitializationResultFile {
  bool has_version = false;
  int version = 0;
  bool has_auto_reset_interval = false;
  double auto_reset_interval_seconds = 0.0;
  std::vector<InitializationSegment> segments;
};

struct GroundTruthTrajectory {
  std::vector<TimedPose> poses;
};

struct EvaluationValue {
  bool valid = false;
  double value = 0.0;
  std::string unavailable_reason;
};

struct InitializationStateEvaluation {
  EvaluationValue rotation_degrees;
  EvaluationValue gravity_degrees;
  EvaluationValue velocity_meters_per_second;
};

struct InitializationSegmentEvaluation {
  uint64_t id = 0;
  std::string reason;
  InitializationStateEvaluation linear;
  InitializationStateEvaluation nonlinear;
  EvaluationValue time_seconds;
  std::vector<std::string> diagnostics;
};

bool parse_initialization_result(const std::string &path, InitializationResultFile &result, std::string &error);

/// Load native OpenVINS TUM ground truth, or EuRoC CSV when path ends in .csv.
bool load_initialization_ground_truth(const std::string &path, GroundTruthTrajectory &ground_truth, std::string &error);

bool parse_initialization_alignment(const std::string &text, InitializationAlignment &alignment);

const char *initialization_alignment_name(InitializationAlignment alignment);

InitializationSegmentEvaluation evaluate_initialization_segment(const InitializationSegment &segment,
                                                                 const GroundTruthTrajectory &ground_truth,
                                                                 InitializationAlignment alignment);

/// Format one compact table containing every sequence and the per-column mean.
std::string format_initialization_report(const std::vector<InitializationSegmentEvaluation> &evaluations);

} // namespace ov_eval

#endif // OV_EVAL_INITIALIZATION_EVALUATOR_H
