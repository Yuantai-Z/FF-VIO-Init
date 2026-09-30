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

#ifndef OV_CORE_VGGT_PREPROCESSING_H
#define OV_CORE_VGGT_PREPROCESSING_H

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <string>

namespace ov_core {

/**
 * @brief Immutable-by-convention description of the complete VGGT image preprocessing path.
 *
 * The undistortion maps convert the distorted input into a rectified pinhole image at input_size. The two following resizes and
 * output_roi describe every remaining image-domain operation. output_calib is the pinhole calibration of output_roi with zero distortion.
 */
struct VGGTPreprocessPlan {
  cv::Size input_size;
  cv::Size square_size;
  cv::Size resized_size;
  cv::Rect output_roi;
  cv::Mat map_x;
  cv::Mat map_y;
  Eigen::VectorXd output_calib;

  cv::Size output_size() const { return output_roi.size(); }
};

/**
 * @brief Image and optional exclusion mask produced by one preprocessing operation.
 */
struct VGGTPreprocessOutput {
  cv::Mat image;
  cv::Mat mask;
};

/**
 * @brief Build the camera calibration, maps, sizes, and crop used by VGGT preprocessing.
 *
 * Intrinsics are scaled with the ratios of the actual integer resize dimensions. The existing integer-coordinate calibration convention is
 * retained; no half-pixel translation is introduced.
 */
VGGTPreprocessPlan makeVGGTPreprocessPlan(int cam_id, const std::string &dist_model, const Eigen::VectorXd &cam_calib_original,
                                         const cv::Size &input_size, int target_width, double fov_scale = 0.15,
                                         bool crop_to_4_3 = true);

/**
 * @brief Apply every operation encoded by a preprocessing plan to an image and optional exclusion mask.
 *
 * Images use the legacy interpolation sequence (linear remap, linear square resize, area target resize). Masks use nearest-neighbor
 * interpolation and are returned as binary CV_8UC1 images. Inputs are never modified.
 */
VGGTPreprocessOutput applyVGGTPreprocessPlan(const VGGTPreprocessPlan &plan, const cv::Mat &image,
                                             const cv::Mat &exclusion_mask = cv::Mat());

} // namespace ov_core

#endif // OV_CORE_VGGT_PREPROCESSING_H
