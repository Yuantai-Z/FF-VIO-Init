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

#include "utils/vggt_preprocessing.h"

#include "utils/print.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>

namespace ov_core {

namespace {

void validatePlanInputs(const std::string &dist_model, const Eigen::VectorXd &calibration, const cv::Size &input_size,
                        int target_width, double fov_scale) {
  if (dist_model != "radtan" && dist_model != "equidistant") {
    throw std::invalid_argument("VGGT distortion model must be radtan or equidistant");
  }
  if (calibration.size() != 8 || !calibration.allFinite() || calibration(0) <= 0.0 || calibration(1) <= 0.0) {
    throw std::invalid_argument("VGGT camera calibration must contain eight finite values and positive focal lengths");
  }
  if (input_size.width <= 0 || input_size.height <= 0) {
    throw std::invalid_argument("VGGT input image size must be positive");
  }
  if (target_width <= 0 || target_width % 14 != 0) {
    throw std::invalid_argument("VGGT target width must be a positive multiple of 14");
  }
  if (!std::isfinite(fov_scale) || fov_scale <= 0.0) {
    throw std::invalid_argument("VGGT fisheye FOV scale must be finite and positive");
  }
}

cv::Mat applyImagePlan(const VGGTPreprocessPlan &plan, const cv::Mat &input) {
  cv::Mat rectified;
  cv::remap(input, rectified, plan.map_x, plan.map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

  cv::Mat square;
  cv::resize(rectified, square, plan.square_size, 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat resized;
  cv::resize(square, resized, plan.resized_size, 0.0, 0.0, cv::INTER_AREA);
  return resized(plan.output_roi).clone();
}

cv::Mat applyMaskPlan(const VGGTPreprocessPlan &plan, const cv::Mat &input) {
  cv::Mat rectified;
  cv::remap(input, rectified, plan.map_x, plan.map_y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(255));

  cv::Mat square;
  cv::resize(rectified, square, plan.square_size, 0.0, 0.0, cv::INTER_NEAREST);

  cv::Mat resized;
  cv::resize(square, resized, plan.resized_size, 0.0, 0.0, cv::INTER_NEAREST);

  cv::Mat binary;
  cv::threshold(resized(plan.output_roi), binary, 127.0, 255.0, cv::THRESH_BINARY);
  return binary;
}

} // namespace

VGGTPreprocessPlan makeVGGTPreprocessPlan(int cam_id, const std::string &dist_model, const Eigen::VectorXd &cam_calib_original,
                                         const cv::Size &input_size, int target_width, double fov_scale, bool crop_to_4_3) {
  validatePlanInputs(dist_model, cam_calib_original, input_size, target_width, fov_scale);

  VGGTPreprocessPlan plan;
  plan.input_size = input_size;
  Eigen::VectorXd rectified_calib = cam_calib_original;

  const cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << rectified_calib(0), 0.0, rectified_calib(2), 0.0, rectified_calib(1), rectified_calib(3), 0.0,
       0.0, 1.0);
  const cv::Mat dist_coeffs =
      (cv::Mat_<double>(4, 1) << rectified_calib(4), rectified_calib(5), rectified_calib(6), rectified_calib(7));
  cv::Mat new_camera_matrix;
  constexpr double center_zoom = 1.05;

  if (dist_model == "equidistant") {
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camera_matrix, dist_coeffs, input_size, cv::Mat(), new_camera_matrix, 1.0,
                                                            input_size, fov_scale);
    new_camera_matrix.at<double>(0, 2) = input_size.width / 2.0;
    new_camera_matrix.at<double>(1, 2) = input_size.height / 2.0;
    new_camera_matrix.at<double>(0, 0) *= center_zoom;
    new_camera_matrix.at<double>(1, 1) *= center_zoom;
    cv::fisheye::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix, input_size, CV_32FC1, plan.map_x,
                                         plan.map_y);
    PRINT_DEBUG("[CAM%d] Using fisheye undistortion model (fov_scale=%.2f), principal point centered\n", cam_id, fov_scale);
  } else {
    new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, input_size, 0.0, input_size);
    new_camera_matrix.at<double>(0, 2) = input_size.width / 2.0;
    new_camera_matrix.at<double>(1, 2) = input_size.height / 2.0;
    new_camera_matrix.at<double>(0, 0) *= center_zoom;
    new_camera_matrix.at<double>(1, 1) *= center_zoom;
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix, input_size, CV_32FC1, plan.map_x, plan.map_y);
    PRINT_DEBUG("[CAM%d] Using radtan undistortion model, principal point centered\n", cam_id);
  }

  if (!cv::checkRange(new_camera_matrix) || new_camera_matrix.at<double>(0, 0) <= 0.0 ||
      new_camera_matrix.at<double>(1, 1) <= 0.0 || plan.map_x.empty() || plan.map_y.empty() || plan.map_x.size() != input_size ||
      plan.map_y.size() != input_size || plan.map_x.type() != CV_32FC1 || plan.map_y.type() != CV_32FC1) {
    throw std::runtime_error("VGGT undistortion produced an invalid rectified camera domain");
  }

  rectified_calib(0) = new_camera_matrix.at<double>(0, 0);
  rectified_calib(1) = new_camera_matrix.at<double>(1, 1);
  rectified_calib(2) = new_camera_matrix.at<double>(0, 2);
  rectified_calib(3) = new_camera_matrix.at<double>(1, 2);
  rectified_calib.tail<4>().setZero();
  PRINT_DEBUG("[CAM%d] After undistortion: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f (centered)\n", cam_id,
              rectified_calib(0), rectified_calib(1), rectified_calib(2), rectified_calib(3));

  const double unified_focal = std::sqrt(rectified_calib(0) * rectified_calib(1));
  const double ideal_scale_x = unified_focal / rectified_calib(0);
  const double ideal_scale_y = unified_focal / rectified_calib(1);
  plan.square_size.width = static_cast<int>(std::round(input_size.width * ideal_scale_x));
  plan.square_size.height = static_cast<int>(std::round(input_size.height * ideal_scale_y));
  if (plan.square_size.width <= 0 || plan.square_size.height <= 0) {
    throw std::runtime_error("VGGT square-pixel resize produced an invalid size");
  }

  const double square_scale_x = static_cast<double>(plan.square_size.width) / input_size.width;
  const double square_scale_y = static_cast<double>(plan.square_size.height) / input_size.height;
  Eigen::VectorXd output_calib = rectified_calib;
  output_calib(0) *= square_scale_x;
  output_calib(1) *= square_scale_y;
  output_calib(2) *= square_scale_x;
  output_calib(3) *= square_scale_y;
  PRINT_DEBUG("[CAM%d] Making pixels square: ideal f=%.2f, actual scale=(%.4f, %.4f), resolution %dx%d -> %dx%d\n", cam_id,
              unified_focal, square_scale_x, square_scale_y, input_size.width, input_size.height, plan.square_size.width,
              plan.square_size.height);

  plan.resized_size.width = target_width;
  plan.resized_size.height = static_cast<int>(
      std::round(plan.square_size.height * (static_cast<double>(target_width) / plan.square_size.width) / 14.0) * 14.0);
  if (plan.resized_size.height <= 0) {
    throw std::runtime_error("VGGT target resize produced an invalid height");
  }

  const double target_scale_x = static_cast<double>(plan.resized_size.width) / plan.square_size.width;
  const double target_scale_y = static_cast<double>(plan.resized_size.height) / plan.square_size.height;
  output_calib(0) *= target_scale_x;
  output_calib(1) *= target_scale_y;
  output_calib(2) *= target_scale_x;
  output_calib(3) *= target_scale_y;
  plan.output_roi = cv::Rect(0, 0, plan.resized_size.width, plan.resized_size.height);

  if (dist_model == "equidistant" && crop_to_4_3) {
    const int target_height = static_cast<int>(std::round(target_width * 0.75 / 14.0) * 14.0);
    if (plan.resized_size.height > target_height) {
      const int crop_offset = (plan.resized_size.height - target_height) / 2;
      plan.output_roi = cv::Rect(0, crop_offset, target_width, target_height);
      output_calib(3) -= crop_offset;
      PRINT_DEBUG("[CAM%d] Fisheye center crop: %d -> %d (offset=%d), cy=%.2f\n", cam_id, plan.resized_size.height,
                  target_height, crop_offset, output_calib(3));
    }
  }

  if (!output_calib.allFinite() || output_calib(0) <= 0.0 || output_calib(1) <= 0.0 || plan.output_roi.width <= 0 ||
      plan.output_roi.height <= 0 || plan.output_roi.x < 0 || plan.output_roi.y < 0 ||
      plan.output_roi.x + plan.output_roi.width > plan.resized_size.width ||
      plan.output_roi.y + plan.output_roi.height > plan.resized_size.height) {
    throw std::runtime_error("VGGT preprocessing produced an invalid output crop");
  }

  plan.output_calib = output_calib;
  PRINT_DEBUG("[CAM%d] VGGT resize: %dx%d -> %dx%d; final %dx%d\n", cam_id, plan.square_size.width, plan.square_size.height,
              plan.resized_size.width, plan.resized_size.height, plan.output_size().width, plan.output_size().height);
  PRINT_DEBUG("[CAM%d] Final intrinsics: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f\n", cam_id, output_calib(0), output_calib(1),
              output_calib(2), output_calib(3));
  return plan;
}

VGGTPreprocessOutput applyVGGTPreprocessPlan(const VGGTPreprocessPlan &plan, const cv::Mat &image, const cv::Mat &exclusion_mask) {
  const bool valid_plan = plan.input_size.width > 0 && plan.input_size.height > 0 && plan.square_size.width > 0 &&
                          plan.square_size.height > 0 && plan.resized_size.width > 0 && plan.resized_size.height > 0 &&
                          !plan.map_x.empty() && !plan.map_y.empty() && plan.map_x.size() == plan.input_size &&
                          plan.map_y.size() == plan.input_size && plan.map_x.type() == CV_32FC1 && plan.map_y.type() == CV_32FC1 &&
                          plan.output_calib.size() == 8 && plan.output_calib.allFinite() && plan.output_calib(0) > 0.0 &&
                          plan.output_calib(1) > 0.0 && plan.output_roi.width > 0 && plan.output_roi.height > 0 &&
                          plan.output_roi.x >= 0 && plan.output_roi.y >= 0 &&
                          plan.output_roi.x + plan.output_roi.width <= plan.resized_size.width &&
                          plan.output_roi.y + plan.output_roi.height <= plan.resized_size.height;
  if (!valid_plan) {
    throw std::invalid_argument("cannot apply an invalid VGGT preprocessing plan");
  }
  if (image.empty() || image.size() != plan.input_size) {
    throw std::invalid_argument("VGGT input image does not match the preprocessing plan");
  }
  if (!exclusion_mask.empty() && (exclusion_mask.size() != plan.input_size || exclusion_mask.type() != CV_8UC1)) {
    throw std::invalid_argument("VGGT exclusion mask must be CV_8UC1 and match the input image size");
  }

  VGGTPreprocessOutput output;
  output.image = applyImagePlan(plan, image);
  if (!exclusion_mask.empty()) {
    output.mask = applyMaskPlan(plan, exclusion_mask);
  }
  if (output.image.size() != plan.output_size() || (!output.mask.empty() && output.mask.size() != plan.output_size())) {
    throw std::runtime_error("VGGT preprocessing output does not match its plan");
  }
  return output;
}

} // namespace ov_core
