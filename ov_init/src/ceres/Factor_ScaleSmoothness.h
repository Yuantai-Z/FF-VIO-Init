/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 * Copyright (C) 2025-2026 Yuantai-Z
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

#ifndef OV_INIT_CERES_FACTOR_SCALESMOOTHNESS_H
#define OV_INIT_CERES_FACTOR_SCALESMOOTHNESS_H

#include <ceres/ceres.h>
#include <cmath>

namespace ov_init {

/**
 * @brief Factor for a smoothness constraint between adjacent scale regions
 *
 * This factor constrains adjacent scale parameters to be equal:
 * residual = sqrt_weight * (scale_i - scale_{i-1})
 *
 * Supports both direct and softplus parameterization.
 */
class Factor_ScaleSmoothness : public ceres::SizedCostFunction<1, 1, 1> {
public:
  /**
   * @brief Constructor
   * @param weight Weight for the smoothness constraint (information = weight)
   * @param use_softplus Whether scale uses softplus parameterization
   */
  Factor_ScaleSmoothness(double weight, bool use_softplus)
      : sqrt_weight_(std::sqrt(weight)), use_softplus_(use_softplus) {}

  /**
   * @brief Evaluate the residual and jacobians
   * @param parameters [0]: scale_prev, [1]: scale_curr
   */
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    double s_prev = parameters[0][0];
    double s_curr = parameters[1][0];

    double actual_scale_prev, actual_scale_curr;
    double d_prev, d_curr; // derivatives

    if (use_softplus_) {
      // softplus(x) = log(1 + exp(x)), numerically stable version
      double t_prev = std::max(s_prev, 0.0);
      actual_scale_prev = std::log1p(std::exp(-std::abs(s_prev))) + t_prev + 1e-5;
      double sigmoid_prev = 1.0 / (1.0 + std::exp(-s_prev));
      d_prev = sigmoid_prev;

      double t_curr = std::max(s_curr, 0.0);
      actual_scale_curr = std::log1p(std::exp(-std::abs(s_curr))) + t_curr + 1e-5;
      double sigmoid_curr = 1.0 / (1.0 + std::exp(-s_curr));
      d_curr = sigmoid_curr;
    } else {
      actual_scale_prev = s_prev;
      actual_scale_curr = s_curr;
      d_prev = 1.0;
      d_curr = 1.0;
    }

    // Residual: sqrt_weight * (scale_curr - scale_prev)
    residuals[0] = sqrt_weight_ * (actual_scale_curr - actual_scale_prev);

    // Jacobians
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        // d_residual / d_scale_prev = -sqrt_weight * d_prev
        jacobians[0][0] = -sqrt_weight_ * d_prev;
      }
      if (jacobians[1] != nullptr) {
        // d_residual / d_scale_curr = sqrt_weight * d_curr
        jacobians[1][0] = sqrt_weight_ * d_curr;
      }
    }

    return true;
  }

private:
  double sqrt_weight_;
  bool use_softplus_;
};

} // namespace ov_init

#endif // OV_INIT_CERES_FACTOR_SCALESMOOTHNESS_H
