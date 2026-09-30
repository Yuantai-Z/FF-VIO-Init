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

#ifndef OV_INIT_INITIALIZATIONDIAGNOSTICS_H
#define OV_INIT_INITIALIZATIONDIAGNOSTICS_H

#include <Eigen/Eigen>
#include <cmath>
#include <limits>

namespace ov_init {

/// One initialization state in the legacy evaluator ordering: timestamp, position, JPL quaternion, velocity.
struct InitializationState {
  InitializationState() {
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    timestamp = invalid;
    position.setConstant(invalid);
    quaternion.setConstant(invalid);
    velocity.setConstant(invalid);
  }

  bool valid() const {
    return std::isfinite(timestamp) && position.allFinite() && quaternion.allFinite() && velocity.allFinite();
  }

  double timestamp;
  Eigen::Vector3d position;
  Eigen::Vector4d quaternion;
  Eigen::Vector3d velocity;
};

/// Minimal successful-initialization data retained until its estimator segment is committed.
struct InitializationDiagnostics {
  bool has_timing = false;
  bool has_linear_state = false;
  bool has_nonlinear_state = false;
  double first_attempt_oldest_time = std::numeric_limits<double>::quiet_NaN();
  double success_oldest_time = std::numeric_limits<double>::quiet_NaN();
  double init_window_time = std::numeric_limits<double>::quiet_NaN();
  InitializationState linear_state;
  InitializationState nonlinear_state;

  bool timing_valid() const {
    return has_timing && std::isfinite(first_attempt_oldest_time) && std::isfinite(success_oldest_time) &&
           std::isfinite(init_window_time);
  }

  bool complete() const {
    return timing_valid() && has_linear_state && linear_state.valid() && has_nonlinear_state && nonlinear_state.valid();
  }
};

inline InitializationState make_initialization_state(double timestamp, const Eigen::Vector4d &quaternion,
                                                       const Eigen::Vector3d &position, const Eigen::Vector3d &velocity) {
  InitializationState state;
  state.timestamp = timestamp;
  state.position = position;
  state.quaternion = quaternion;
  state.velocity = velocity;
  return state;
}

} // namespace ov_init

#endif // OV_INIT_INITIALIZATIONDIAGNOSTICS_H
