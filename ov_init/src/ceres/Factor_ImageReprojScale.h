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

#ifndef OV_INIT_CERES_FACTOR_IMAGEREPROJSCALE_H
#define OV_INIT_CERES_FACTOR_IMAGEREPROJSCALE_H

#include <Eigen/Eigen>
#include <ceres/ceres.h>

#include "cam/CamBase.h"
#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"

namespace ov_init {

/**
 * @brief Factor for depth scale-based image reprojection
 *
 * This factor constrains the depth scale parameter such that VGGT point cloud
 * (scaled by lambda) reprojects correctly to the measured pixel coordinates.
 *
 * Residual equation:
 * Given:
 * - uv_meas: measured pixel coordinates
 * - p_c0: VGGT point cloud in camera C0 frame (first frame)
 * - lambda: pointcloud scale factor (regional or global)
 *
 * Compute:
 * 1. p_FinI0 = R_ItoC^T * (lambda * p_c0 - p_IinC)
 * 2. p_FinG = R_GtoI0^T * p_FinI0, with I0 at the global origin
 * 3. p_FinIk = R_GtoIk * (p_FinG - p_IkinG)
 * 4. p_FinCk = R_ItoC * p_FinIk + p_IinC
 * 5. uv_proj = distort(p_FinCk)
 * 6. residual = sqrt_info * (uv_proj - uv_meas)
 *
 * Parameters:
 * [0] q_GtoIk (4dof): orientation of IMU frame Ik in global frame
 * [1] p_IkinG (3dof): position of IMU frame Ik in global frame
 * [2] lambda  (1dof): depth scale factor
 * [3] q_ItoC  (4dof): camera-IMU extrinsic rotation
 * [4] p_IinC  (3dof): camera-IMU extrinsic translation
 */
class Factor_ImageReprojScale : public ceres::CostFunction {

public:
  /**
   * @brief Constructor
   * @param uv_meas_ Measured pixel coordinates
   * @param point3d_c0_ VGGT point cloud in C0 frame (unscaled)
   * @param pix_sigma_ Pixel measurement noise standard deviation
   * @param confidence_ Confidence weight for this observation (default 1.0)
   * @param camera_vals_ Camera intrinsic parameters [fx, fy, cx, cy, k1, k2, k3, k4]
   * @param R_GtoI0_ Rotation from gravity-aligned global frame to first IMU frame
   * @param is_scale_softplus_ Use softplus parameterization for scale
   * @param is_fisheye_ Must be false; preprocessing exposes a pinhole/radtan camera
   */
  Factor_ImageReprojScale(const Eigen::Vector2d &uv_meas_, const Eigen::Vector3d &point3d_c0_, double pix_sigma_, double confidence_,
                          const Eigen::Matrix<double, 8, 1> &camera_vals_,
                          const Eigen::Matrix3d &R_GtoI0_, bool is_scale_softplus_, bool is_fisheye_);

  /**
   * @brief Evaluate the residual and jacobians
   * @param parameters Array of parameter blocks
   * @param residuals Output residual vector (2x1)
   * @param jacobians Output jacobian matrices (optional)
   * @return True if successful
   */
  bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

  /// Gate for chi-squared test (default 1.0)
  double gate = 1.0;

private:
  /// Measured pixel coordinates
  Eigen::Vector2d uv_meas;

  /// VGGT point in C0 frame (unscaled)
  Eigen::Vector3d point3d_c0;

  /// Square root information matrix (2x2)
  Eigen::Matrix<double, 2, 2> sqrtQ;

  /// Camera intrinsic parameters
  Eigen::Matrix<double, 8, 1> camera_vals;

  /// Rotation from gravity-aligned global frame to first IMU frame (I0)
  Eigen::Matrix3d R_GtoI0;

  bool is_scale_softplus = false;

  /// Camera model type
  bool is_fisheye;
};

} // namespace ov_init

#endif // OV_INIT_CERES_FACTOR_IMAGEREPROJSCALE_H
