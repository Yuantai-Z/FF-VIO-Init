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

#ifndef OV_INIT_CERES_FACTOR_SCALEPRIOR_H
#define OV_INIT_CERES_FACTOR_SCALEPRIOR_H

#include <ceres/ceres.h>
#include <cmath>

namespace ov_init {

/**
 * @brief Factor for scale prior constraint in nonlinear optimization
 *
 * This factor adds a soft constraint on the scale parameter to push it towards
 * a prior value based on empirical observation. Supports both direct and
 * softplus parameterization.
 *
 * Cost: weight * (actual_scale - prior_scale)^2
 *
 * Where actual_scale = softplus(param) + 1e-5 if use_softplus, else param
 */
class Factor_ScalePrior : public ceres::SizedCostFunction<1, 1> {
public:
  /**
   * @brief Constructor
   * @param prior_scale Prior value for scale
   * @param weight Weight for the prior constraint (information = weight)
   * @param use_softplus Whether scale uses softplus parameterization
   */
  Factor_ScalePrior(double prior_scale, double weight, bool use_softplus)
      : prior_scale_(prior_scale), sqrt_weight_(std::sqrt(weight)), use_softplus_(use_softplus) {}

  /**
   * @brief Evaluate the residual and jacobian
   */
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    double s_param = parameters[0][0];

    // Compute actual scale based on parameterization
    double actual_scale;
    double d_scale_d_param; // derivative of actual_scale w.r.t. s_param

    if (use_softplus_) {
      // softplus(x) = log(1 + exp(x)), numerically stable version
      double t = std::max(s_param, 0.0);
      actual_scale = std::log1p(std::exp(-std::abs(s_param))) + t + 1e-5;

      // sigmoid(x) = 1 / (1 + exp(-x)) = d(softplus)/dx
      double sigmoid = 1.0 / (1.0 + std::exp(-s_param));
      d_scale_d_param = sigmoid;
    } else {
      actual_scale = s_param;
      d_scale_d_param = 1.0;
    }

    // Residual: sqrt_weight * (actual_scale - prior_scale)
    residuals[0] = sqrt_weight_ * (actual_scale - prior_scale_);

    // Jacobian
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = sqrt_weight_ * d_scale_d_param;
    }

    return true;
  }

private:
  double prior_scale_;
  double sqrt_weight_;
  bool use_softplus_;
};

} // namespace ov_init

#endif // OV_INIT_CERES_FACTOR_SCALEPRIOR_H
