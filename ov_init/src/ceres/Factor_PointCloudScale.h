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

#ifndef OV_INIT_CERES_FACTOR_POINTCLOUDSCALE_H
#define OV_INIT_CERES_FACTOR_POINTCLOUDSCALE_H

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <cmath>

namespace ov_init {

/**
 * @brief Point cloud scale constraint factor
 *
 * This factor constrains the estimated feature position to match
 * the VGGT point cloud position transformed with rotation, translation, and scale.
 *
 * Residual: sqrt_weight * (p_FinG - (scale * R_C0toG * p_FinC0 + t_C0toG))
 *
 * where:
 * - R_C0toG = R_I0toG * R_CtoI (rotation from camera C0 to global G)
 * - t_C0toG = -R_I0toG * R_CtoI * p_IinC (translation offset from extrinsics)
 *
 * Parameters (optimized):
 * - [0] p_FinG: Estimated feature position in global frame (3 DOF)
 * - [1] scale: Scale parameter (1 DOF, can be softplus parameterized)
 *
 * Measurements (fixed):
 * - p_FinC0: VGGT point position in camera frame C0
 * - R_C0toG: Rotation from C0 to G frame
 * - t_C0toG: Translation offset
 */
class Factor_PointCloudScale : public ceres::SizedCostFunction<3, 3, 1> {
public:
  /**
   * @brief Constructor
   * @param p_FinC0 VGGT point position in camera frame C0
   * @param R_C0toG Rotation matrix from C0 to G frame
   * @param t_C0toG Translation offset (extrinsic contribution)
   * @param weight Weight for the constraint (information = weight)
   * @param use_softplus Whether scale uses softplus parameterization
   */
  Factor_PointCloudScale(const Eigen::Vector3d &p_FinC0,
                         const Eigen::Matrix3d &R_C0toG,
                         const Eigen::Vector3d &t_C0toG,
                         double weight, bool use_softplus)
      : p_FinC0_(p_FinC0), R_C0toG_(R_C0toG), t_C0toG_(t_C0toG),
        sqrt_weight_(std::sqrt(weight)), use_softplus_(use_softplus) {}

  /**
   * @brief Evaluate the residual and jacobians
   * @param parameters [0]: p_FinG (3), [1]: scale (1)
   * @param residuals Output residual (3x1)
   * @param jacobians Output jacobians (optional)
   */
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override {
    Eigen::Vector3d p_FinG = Eigen::Map<const Eigen::Vector3d>(parameters[0]);
    double s_param = parameters[1][0];

    // Compute actual scale (with optional softplus)
    double scale, d_scale_d_param;
    if (use_softplus_) {
      // softplus(x) = log(1 + exp(x)), numerically stable version
      double t = std::max(s_param, 0.0);
      scale = std::log1p(std::exp(-std::abs(s_param))) + t + 1e-5;
      // Evaluate sigmoid without overflowing exp() for large negative inputs.
      if (s_param >= 0.0) {
        const double exp_neg = std::exp(-s_param);
        d_scale_d_param = 1.0 / (1.0 + exp_neg);
      } else {
        const double exp_pos = std::exp(s_param);
        d_scale_d_param = exp_pos / (1.0 + exp_pos);
      }
    } else {
      scale = s_param;
      d_scale_d_param = 1.0;
    }

    // Predicted point in G frame: scale * R * p_FinC0 + t
    Eigen::Vector3d p_FinG_pred = scale * R_C0toG_ * p_FinC0_ + t_C0toG_;

    // Residual: r = sqrt_weight * (p_FinG - p_FinG_pred)
    Eigen::Vector3d res = sqrt_weight_ * (p_FinG - p_FinG_pred);
    residuals[0] = res(0);
    residuals[1] = res(1);
    residuals[2] = res(2);

    if (jacobians != nullptr) {
      // J w.r.t. p_FinG: sqrt_weight * I_3x3
      if (jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J0(jacobians[0]);
        J0 = sqrt_weight_ * Eigen::Matrix3d::Identity();
      }
      // J w.r.t. scale: -sqrt_weight * d_scale_d_param * R * p_FinC0
      if (jacobians[1] != nullptr) {
        Eigen::Map<Eigen::Vector3d> J1(jacobians[1]);
        J1 = -sqrt_weight_ * d_scale_d_param * R_C0toG_ * p_FinC0_;
      }
    }
    return true;
  }

private:
  Eigen::Vector3d p_FinC0_;   ///< VGGT point in C0 frame
  Eigen::Matrix3d R_C0toG_;   ///< Rotation from C0 to G
  Eigen::Vector3d t_C0toG_;   ///< Translation offset
  double sqrt_weight_;
  bool use_softplus_;
};

} // namespace ov_init

#endif // OV_INIT_CERES_FACTOR_POINTCLOUDSCALE_H
