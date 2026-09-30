/*
 * VGGT point-cloud nonlinear method selection
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_VGGTNONLINEARMODE_H
#define OV_INIT_VGGTNONLINEARMODE_H

#include <cmath>
#include <stdexcept>

namespace ov_init {

/**
 * @brief Nonlinear graph used by the point-cloud initializer.
 *
 * FEATURE_ONLY optimizes landmarks without a point-cloud scale constraint.
 * FF optimizes poses and point-cloud scale(s), with no landmark parameter blocks.
 * SC optimizes landmarks and adds point-cloud scale constraints.
 */
enum class VggtNonlinearMethod { FEATURE_ONLY = 0, FF = 1, SC = 2 };

inline bool is_valid_vggt_nonlinear_flags(bool use_scale_optimization, double feature_pc_constraint_weight) {
  return std::isfinite(feature_pc_constraint_weight) && feature_pc_constraint_weight >= 0.0 &&
         !(use_scale_optimization && feature_pc_constraint_weight > 0.0);
}

inline VggtNonlinearMethod vggt_nonlinear_method_from_flags(bool use_scale_optimization,
                                                            double feature_pc_constraint_weight) {
  if (!is_valid_vggt_nonlinear_flags(use_scale_optimization, feature_pc_constraint_weight)) {
    throw std::invalid_argument("invalid VGGT nonlinear method flags");
  }
  if (use_scale_optimization) {
    return VggtNonlinearMethod::FF;
  }
  return feature_pc_constraint_weight > 0.0 ? VggtNonlinearMethod::SC : VggtNonlinearMethod::FEATURE_ONLY;
}

inline const char *vggt_nonlinear_method_name(VggtNonlinearMethod method) {
  switch (method) {
    case VggtNonlinearMethod::FF:
      return "FF";
    case VggtNonlinearMethod::SC:
      return "SC";
    case VggtNonlinearMethod::FEATURE_ONLY:
      return "FEATURE_ONLY";
  }
  return "FEATURE_ONLY";
}

inline const char *vggt_nonlinear_method_token(VggtNonlinearMethod method) {
  switch (method) {
    case VggtNonlinearMethod::FF:
      return "ff";
    case VggtNonlinearMethod::SC:
      return "sc";
    case VggtNonlinearMethod::FEATURE_ONLY:
      return "feature_only";
  }
  return "feature_only";
}

} // namespace ov_init

#endif // OV_INIT_VGGTNONLINEARMODE_H
