/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "vggt/NonlinearMode.h"
#include "init/InertialInitializerOptions.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

int main() {
  using ov_init::VggtNonlinearMethod;

  const ov_init::InertialInitializerOptions defaults;
  if (!defaults.init_vggt_use_scale_optimization || defaults.init_vggt_feature_pc_constraint_weight != 0.0) {
    std::cerr << "default point-cloud nonlinear method is not FF\n";
    return 1;
  }

  const auto ff = ov_init::vggt_nonlinear_method_from_flags(true, 0.0);
  const auto sc = ov_init::vggt_nonlinear_method_from_flags(false, 0.5);
  const auto feature_only = ov_init::vggt_nonlinear_method_from_flags(false, 0.0);
  if (ff != VggtNonlinearMethod::FF || sc != VggtNonlinearMethod::SC ||
      feature_only != VggtNonlinearMethod::FEATURE_ONLY) {
    std::cerr << "VGGT nonlinear truth table is invalid\n";
    return 1;
  }
  if (std::string(ov_init::vggt_nonlinear_method_token(ff)) != "ff" ||
      std::string(ov_init::vggt_nonlinear_method_token(sc)) != "sc" ||
      std::string(ov_init::vggt_nonlinear_method_token(feature_only)) != "feature_only") {
    std::cerr << "VGGT nonlinear method metadata is invalid\n";
    return 1;
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  for (const auto &invalid : {std::make_pair(true, 0.5), std::make_pair(false, -0.1), std::make_pair(false, nan),
                              std::make_pair(false, infinity)}) {
    try {
      (void)ov_init::vggt_nonlinear_method_from_flags(invalid.first, invalid.second);
      std::cerr << "invalid VGGT nonlinear flags were accepted\n";
      return 1;
    } catch (const std::invalid_argument &) {
    }
  }

  return 0;
}
