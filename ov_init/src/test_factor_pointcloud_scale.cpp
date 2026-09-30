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

#include "ceres/Factor_PointCloudScale.h"
#include "vggt/ScaleLayout.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool evaluate(const ov_init::Factor_PointCloudScale &factor, double scale, Eigen::Vector3d &residual,
              Eigen::Vector3d &scale_jacobian) {
  double feature[3] = {0.2, -0.4, 0.8};
  double scale_parameter[1] = {scale};
  const double *parameters[2] = {feature, scale_parameter};
  double residual_data[3] = {};
  double feature_jacobian[9] = {};
  double scale_jacobian_data[3] = {};
  double *jacobians[2] = {feature_jacobian, scale_jacobian_data};

  if (!factor.Evaluate(parameters, residual_data, jacobians)) {
    return false;
  }
  residual = Eigen::Map<Eigen::Vector3d>(residual_data);
  scale_jacobian = Eigen::Map<Eigen::Vector3d>(scale_jacobian_data);
  return residual.allFinite() && scale_jacobian.allFinite();
}

bool check_finite_difference(const ov_init::Factor_PointCloudScale &factor, double scale) {
  constexpr double step = 1e-6;
  Eigen::Vector3d residual;
  Eigen::Vector3d jacobian;
  Eigen::Vector3d residual_plus;
  Eigen::Vector3d unused_plus;
  Eigen::Vector3d residual_minus;
  Eigen::Vector3d unused_minus;
  if (!evaluate(factor, scale, residual, jacobian) || !evaluate(factor, scale + step, residual_plus, unused_plus) ||
      !evaluate(factor, scale - step, residual_minus, unused_minus)) {
    return false;
  }

  const Eigen::Vector3d numerical = (residual_plus - residual_minus) / (2.0 * step);
  const double tolerance = 1e-7 * std::max({1.0, numerical.norm(), jacobian.norm()});
  return (numerical - jacobian).norm() <= tolerance;
}

bool check_scale_layouts() {
  const ov_init::VggtScaleLayout global = ov_init::make_vggt_scale_layout(0);
  if (global.is_region() || global.scale_count != 1 || !global.smoothness_edges.empty()) {
    std::cerr << "global scale layout is invalid\n";
    return false;
  }

  const ov_init::VggtScaleLayout region1 = ov_init::make_vggt_scale_layout(1);
  if (!region1.is_region() || region1.scale_count != 1 || !region1.smoothness_edges.empty()) {
    std::cerr << "1x1 regional scale layout is invalid\n";
    return false;
  }

  const ov_init::VggtScaleLayout region2 = ov_init::make_vggt_scale_layout(2);
  const std::vector<std::pair<std::size_t, std::size_t>> expected2 = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
  if (!region2.is_region() || region2.scale_count != 4 || region2.smoothness_edges != expected2) {
    std::cerr << "2x2 regional scale layout is invalid\n";
    return false;
  }

  const ov_init::VggtScaleLayout region3 = ov_init::make_vggt_scale_layout(3);
  std::set<std::pair<std::size_t, std::size_t>> unique_edges;
  for (const auto &edge : region3.smoothness_edges) {
    if (edge.first == edge.second || edge.first >= region3.scale_count || edge.second >= region3.scale_count ||
        !unique_edges.insert(edge).second) {
      std::cerr << "3x3 regional scale layout contains an invalid edge\n";
      return false;
    }
  }
  if (!region3.is_region() || region3.scale_count != 9 || region3.smoothness_edges.size() != 12) {
    std::cerr << "3x3 regional scale layout is invalid\n";
    return false;
  }

  try {
    (void)ov_init::make_vggt_scale_layout(-1);
  } catch (const std::invalid_argument &) {
    try {
      (void)ov_init::make_vggt_scale_layout(ov_init::kMaxVggtScaleRegionN + 1);
    } catch (const std::out_of_range &) {
      return true;
    }
    std::cerr << "oversized regional scale count was accepted\n";
    return false;
  }
  std::cerr << "negative regional scale count was accepted\n";
  return false;
}

} // namespace

int main() {
  if (!check_scale_layouts()) {
    return 1;
  }

  const Eigen::Vector3d point_in_c0(1.0, -2.0, 0.5);
  const Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d translation(0.1, 0.2, -0.3);
  const ov_init::Factor_PointCloudScale softplus_factor(point_in_c0, rotation, translation, 4.0, true);

  for (double scale : std::vector<double>{-1000.0, 0.0, 1000.0}) {
    Eigen::Vector3d residual;
    Eigen::Vector3d jacobian;
    if (!evaluate(softplus_factor, scale, residual, jacobian)) {
      std::cerr << "non-finite softplus evaluation at " << scale << '\n';
      return 1;
    }
  }

  for (double scale : std::vector<double>{-10.0, 0.0, 10.0}) {
    if (!check_finite_difference(softplus_factor, scale)) {
      std::cerr << "softplus Jacobian finite-difference mismatch at " << scale << '\n';
      return 1;
    }
  }

  const ov_init::Factor_PointCloudScale linear_factor(point_in_c0, rotation, translation, 4.0, false);
  if (!check_finite_difference(linear_factor, 2.0)) {
    std::cerr << "linear-scale Jacobian finite-difference mismatch\n";
    return 1;
  }

  return 0;
}
