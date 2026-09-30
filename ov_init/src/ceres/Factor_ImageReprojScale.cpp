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

#include "Factor_ImageReprojScale.h"

#include "utils/quat_ops.h"

using namespace ov_init;

Factor_ImageReprojScale::Factor_ImageReprojScale(const Eigen::Vector2d &uv_meas_, const Eigen::Vector3d &point3d_c0_, double pix_sigma_, double confidence_,
                                                 const Eigen::Matrix<double, 8, 1> &camera_vals_,
                                                 const Eigen::Matrix3d &R_GtoI0_, bool is_scale_softplus_, bool is_fisheye_)
    : uv_meas(uv_meas_), point3d_c0(point3d_c0_), camera_vals(camera_vals_), R_GtoI0(R_GtoI0_), is_scale_softplus(is_scale_softplus_), is_fisheye(is_fisheye_) {

  // Square root information matrix: confidence / sigma_pix (actual weight = confidence²/sigma_pix²)
  sqrtQ = Eigen::Matrix<double, 2, 2>::Identity();
  sqrtQ(0, 0) *= confidence_ / pix_sigma_;
  sqrtQ(1, 1) *= confidence_ / pix_sigma_;

  // Parameters we are a function of
  set_num_residuals(2);
  mutable_parameter_block_sizes()->push_back(4); // q_GtoIk
  mutable_parameter_block_sizes()->push_back(3); // p_IiinG
  mutable_parameter_block_sizes()->push_back(1); // scalei (scale)
  mutable_parameter_block_sizes()->push_back(4); // q_ItoC
  mutable_parameter_block_sizes()->push_back(3); // p_IinC
}

bool Factor_ImageReprojScale::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

  // Recover the current state from our parameters
  Eigen::Vector4d q_GtoIi = Eigen::Map<const Eigen::Vector4d>(parameters[0]);
  Eigen::Matrix3d R_GtoIi = ov_core::quat_2_Rot(q_GtoIi);
  Eigen::Vector3d p_IiinG = Eigen::Map<const Eigen::Vector3d>(parameters[1]);
  double s_param = parameters[2][0]; // pointcloud scale
  Eigen::Vector4d q_ItoC = Eigen::Map<const Eigen::Vector4d>(parameters[3]);
  Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(q_ItoC);
  Eigen::Vector3d p_IinC = Eigen::Map<const Eigen::Vector3d>(parameters[4]);

  // Numerically stable softplus and sigmoid
  auto softplus = [](double x) {
    const double t = std::max(x, 0.0);
    return std::log1p(std::exp(-std::abs(x))) + t; // ~softplus, no overflow
  };

  double scalei = 1;
  if (is_scale_softplus)
    scalei = softplus(s_param) + 1e-5; // softplus parameterization to ensure scale > 0
  else
    scalei = s_param;
  // p_FinI0 = R_ItoC^T * (scalei * p_c0 - p_IinC)
  Eigen::Vector3d p_FinI0 = R_ItoC.transpose() * (scalei * point3d_c0 - p_IinC);
  Eigen::Vector3d p_FinG = R_GtoI0.transpose() * p_FinI0;
  Eigen::Vector3d p_FinIi = R_GtoIi * (p_FinG - p_IiinG);
  Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;

  // Normalized projected feature bearing
  Eigen::Vector2d uv_norm;
  uv_norm << p_FinCi(0) / p_FinCi(2), p_FinCi(1) / p_FinCi(2);

  // Square-root information and gate
  Eigen::Matrix<double, 2, 2> sqrtQ_gate = gate * sqrtQ;

  // Get the distorted raw image coordinate using the camera model
  // Also if jacobians are requested, then compute derivatives
  Eigen::Vector2d uv_dist;
  Eigen::MatrixXd H_dz_dzn, H_dz_dzeta;
  if (is_fisheye) {
    assert(!is_fisheye);
    ov_core::CamEqui cam(0, 0);
    cam.set_value(camera_vals);
    uv_dist = cam.distort_d(uv_norm);
    if (jacobians) {
      cam.compute_distort_jacobian(uv_norm, H_dz_dzn, H_dz_dzeta);
      H_dz_dzn = sqrtQ_gate * H_dz_dzn;
      H_dz_dzeta = sqrtQ_gate * H_dz_dzeta;
    }
  } else {
    ov_core::CamRadtan cam(0, 0);
    cam.set_value(camera_vals);
    uv_dist = cam.distort_d(uv_norm);
    if (jacobians) {
      cam.compute_distort_jacobian(uv_norm, H_dz_dzn, H_dz_dzeta);
      H_dz_dzn = sqrtQ_gate * H_dz_dzn;
      H_dz_dzeta = sqrtQ_gate * H_dz_dzeta;
    }
  }

  // Compute residual
  Eigen::Vector2d res = uv_dist - uv_meas;
  res = sqrtQ_gate * res;
  residuals[0] = res(0);
  residuals[1] = res(1);

  // Compute jacobians if requested by ceres
  if (jacobians) {

    // Normalized coordinates in respect to projection function
    Eigen::MatrixXd H_dzn_dpfc = Eigen::MatrixXd::Zero(2, 3);
    H_dzn_dpfc << 1.0 / p_FinCi(2), 0, -p_FinCi(0) / std::pow(p_FinCi(2), 2), 0, 1.0 / p_FinCi(2), -p_FinCi(1) / std::pow(p_FinCi(2), 2);
    Eigen::MatrixXd H_dz_dpfc = H_dz_dzn * H_dzn_dpfc;

    // Jacobian wrt q_GtoIi
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian(jacobians[0]);
      jacobian.block(0, 0, 2, 3) = H_dz_dpfc * R_ItoC * ov_core::skew_x(p_FinIi);
      jacobian.block(0, 3, 2, 1).setZero();
    }

    // Jacobian wrt p_IiinG
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian(jacobians[1]);
      jacobian.block(0, 0, 2, 3) = - H_dz_dpfc * R_ItoC * R_GtoIi;
    }

    // Jacobian wrt scalei (scale)
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 1>> jacobian(jacobians[2]);
      Eigen::MatrixXd H_dpfc_dpfg = R_ItoC * R_GtoIi;
      Eigen::MatrixXd H_dpfg_dscale = R_GtoI0.transpose() * R_ItoC.transpose() * point3d_c0;
      if(is_scale_softplus) {
        // Numerically stable sigmoid for softplus derivative
        auto sigmoid = [](double x) {
          if (x >= 0) {
            double e = std::exp(-x);
            return 1.0 / (1.0 + e);
          } else {
            double e = std::exp(x);
            return e / (1.0 + e);
          }
        };
        H_dpfg_dscale = H_dpfg_dscale * sigmoid(s_param);
      }
      jacobian = H_dz_dpfc * H_dpfc_dpfg * H_dpfg_dscale;
    }

    // This factor contributes zero extrinsic Jacobians.
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian(jacobians[3]);
      jacobian.setZero();
    }

    if (jacobians[4]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian(jacobians[4]);
      jacobian.setZero();
    }
  }
  return true;
}
