/*
 * VGGT point-cloud scale layout
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_INIT_SCALELAYOUT_H
#define OV_INIT_SCALELAYOUT_H

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ov_init {

constexpr int kMaxVggtScaleRegionN = 32;

enum class VggtScaleMode { GLOBAL, REGION };

struct VggtScaleLayout {
  VggtScaleMode mode = VggtScaleMode::GLOBAL;
  std::size_t scale_count = 1;
  std::vector<std::pair<std::size_t, std::size_t>> smoothness_edges;

  bool is_region() const { return mode == VggtScaleMode::REGION; }
};

/**
 * @brief Construct the supported VGGT point-cloud scale topology.
 * @param region_n Zero selects one global scale; positive values select an n-by-n regional grid.
 */
inline VggtScaleLayout make_vggt_scale_layout(int region_n) {
  if (region_n < 0) {
    throw std::invalid_argument("VGGT scale region count must be non-negative");
  }
  if (region_n > kMaxVggtScaleRegionN) {
    throw std::out_of_range("VGGT scale region count exceeds the supported limit");
  }

  VggtScaleLayout layout;
  if (region_n == 0) {
    return layout;
  }

  layout.mode = VggtScaleMode::REGION;
  const std::size_t n = static_cast<std::size_t>(region_n);
  layout.scale_count = n * n;
  layout.smoothness_edges.reserve(2 * n * (n - 1));
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      const std::size_t current = col + row * n;
      if (col + 1 < n) {
        layout.smoothness_edges.emplace_back(current, current + 1);
      }
      if (row + 1 < n) {
        layout.smoothness_edges.emplace_back(current, current + n);
      }
    }
  }
  return layout;
}

} // namespace ov_init

#endif // OV_INIT_SCALELAYOUT_H
