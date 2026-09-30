/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2025-2026 Yuantai-Z
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "utils/vggt_preprocessing.h"

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void requireNear(double actual, double expected, double tolerance, const std::string &message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
  }
}

void requireThrowsInvalidArgument(const std::function<void()> &function, const std::string &message) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(message);
}

Eigen::VectorXd eurocCalibration() {
  Eigen::VectorXd calibration(8);
  calibration << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05;
  return calibration;
}

Eigen::VectorXd tumCalibration() {
  Eigen::VectorXd calibration(8);
  calibration << 380.81042871360756, 380.81194179427075, 510.29465304840727, 514.3304630538506,
      0.010171079892421483, -0.010816440029919381, 0.005942781769412756, -0.001662284667857643;
  return calibration;
}

Eigen::VectorXd d455Calibration() {
  Eigen::VectorXd calibration(8);
  calibration << 384.231778, 384.29036633, 319.07342043, 242.13295995, 0.06407878, 0.04445103, -0.00149072,
      -0.00294164;
  return calibration;
}

void testDatasetGeometryOracles() {
  struct GeometryCase {
    std::string name;
    std::string model;
    Eigen::VectorXd calibration;
    cv::Size input_size;
    int target_width;
    double fov_scale;
    bool crop_to_4_3;
    cv::Size square_size;
    cv::Size resized_size;
    cv::Rect output_roi;
    Eigen::Vector4d output_k;
  };

  const std::vector<GeometryCase> cases{
      {"EuRoC-504", "radtan", eurocCalibration(), cv::Size(752, 480), 504, 0.15, false, cv::Size(814, 443),
       cv::Size(504, 280), cv::Rect(0, 0, 504, 280), Eigen::Vector4d(250.26928107079041, 255.51158638000493, 252.0, 140.0)},
      {"EuRoC-518", "radtan", eurocCalibration(), cv::Size(752, 480), 518, 0.15, false, cv::Size(814, 443),
       cv::Size(518, 280), cv::Rect(0, 0, 518, 280), Eigen::Vector4d(257.221205544979, 255.51158638000493, 259.0, 140.0)},
      {"TUM-VI-1024-504", "equidistant", tumCalibration(), cv::Size(1024, 1024), 504, 0.2, true, cv::Size(1024, 1024),
       cv::Size(504, 504), cv::Rect(0, 63, 504, 378), Eigen::Vector4d(301.44280576086391, 301.44400348877804, 252.0, 189.0)},
      {"TUM-VI-1024-518", "equidistant", tumCalibration(), cv::Size(1024, 1024), 518, 0.2, true, cv::Size(1024, 1024),
       cv::Size(518, 518), cv::Rect(0, 63, 518, 392), Eigen::Vector4d(309.81621703199903, 309.81744803013299, 259.0, 196.0)},
      {"D455-504", "radtan", d455Calibration(), cv::Size(640, 480), 504, 0.2, true, cv::Size(640, 480), cv::Size(504, 378),
       cv::Rect(0, 0, 504, 378), Eigen::Vector4d(347.75680606842042, 347.94374118804933, 252.0, 189.0)},
      {"D455-518", "radtan", d455Calibration(), cv::Size(640, 480), 518, 0.2, true, cv::Size(640, 480), cv::Size(518, 392),
       cv::Rect(0, 0, 518, 392), Eigen::Vector4d(357.41671734809876, 360.83054641723635, 259.0, 196.0)},
  };

  for (const GeometryCase &test_case : cases) {
    const ov_core::VGGTPreprocessPlan plan =
        ov_core::makeVGGTPreprocessPlan(0, test_case.model, test_case.calibration, test_case.input_size,
                                        test_case.target_width, test_case.fov_scale, test_case.crop_to_4_3);
    require(plan.square_size == test_case.square_size, test_case.name + " square size changed");
    require(plan.resized_size == test_case.resized_size, test_case.name + " resized size changed");
    require(plan.output_roi == test_case.output_roi, test_case.name + " output ROI changed");
    require(plan.output_size() == test_case.output_roi.size(), test_case.name + " output_size disagrees with ROI");
    require(plan.map_x.size() == test_case.input_size && plan.map_y.size() == test_case.input_size &&
                plan.map_x.type() == CV_32FC1 && plan.map_y.type() == CV_32FC1,
            test_case.name + " rectification maps changed domain or type");
    require(plan.output_calib.size() == 8 && plan.output_calib.tail<4>().isZero(0.0),
            test_case.name + " output is not a zero-distortion pinhole calibration");
    for (int index = 0; index < 4; ++index) {
      requireNear(plan.output_calib(index), test_case.output_k(index), 1e-6,
                  test_case.name + " K[" + std::to_string(index) + "] changed");
    }
  }
}

void testIntegerResizeRatiosAndApply() {
  const ov_core::VGGTPreprocessPlan plan =
      ov_core::makeVGGTPreprocessPlan(0, "radtan", eurocCalibration(), cv::Size(752, 480), 518, 0.15, false);

  require(plan.input_size == cv::Size(752, 480), "plan input size changed");
  require(plan.square_size == cv::Size(814, 443), "unexpected rounded square-pixel size");
  require(plan.resized_size == cv::Size(518, 280), "unexpected model resize");
  require(plan.output_roi == cv::Rect(0, 0, 518, 280), "unexpected radtan output ROI");
  require(plan.output_calib.size() == 8 && plan.output_calib.tail<4>().isZero(0.0), "output must be an undistorted pinhole calibration");

  // The rectified principal point is exactly input_size/2. Updating K with the actual integer resize ratios therefore leaves it at
  // output_size/2. The former theoretical-ratio implementation produced approximately (259.144, 140.111) here.
  requireNear(plan.output_calib(2), 259.0, 1e-12, "cx does not use the actual rounded resize ratio");
  requireNear(plan.output_calib(3), 140.0, 1e-12, "cy does not use the actual rounded resize ratio");

  cv::Mat image(plan.input_size, CV_8UC1);
  for (int row = 0; row < image.rows; ++row) {
    for (int col = 0; col < image.cols; ++col) {
      image.at<uint8_t>(row, col) = static_cast<uint8_t>((3 * row + 5 * col) % 256);
    }
  }
  cv::Mat mask = cv::Mat::zeros(plan.input_size, CV_8UC1);
  cv::rectangle(mask, cv::Rect(250, 150, 80, 60), cv::Scalar(255), cv::FILLED);
  const cv::Mat image_before = image.clone();
  const cv::Mat mask_before = mask.clone();

  const ov_core::VGGTPreprocessOutput output = ov_core::applyVGGTPreprocessPlan(plan, image, mask);
  require(output.image.size() == plan.output_roi.size(), "processed image size disagrees with plan");
  require(output.mask.size() == plan.output_roi.size(), "processed mask size disagrees with plan");
  require(output.image.type() == CV_8UC1 && output.mask.type() == CV_8UC1, "processed types changed");
  cv::Mat non_binary = (output.mask != 0) & (output.mask != 255);
  require(cv::countNonZero(non_binary) == 0, "processed mask is not binary");
  require(cv::countNonZero(output.mask) > 0, "processed mask lost its invalid region");
  require(cv::norm(image, image_before, cv::NORM_INF) == 0.0 && cv::norm(mask, mask_before, cv::NORM_INF) == 0.0,
          "apply mutated its input image or mask");

  const ov_core::VGGTPreprocessOutput without_mask = ov_core::applyVGGTPreprocessPlan(plan, image);
  require(without_mask.image.size() == plan.output_roi.size() && without_mask.mask.empty(), "empty-mask apply path is invalid");

  requireThrowsInvalidArgument(
      [&]() { ov_core::applyVGGTPreprocessPlan(plan, cv::Mat::zeros(480, 751, CV_8UC1), mask); },
      "wrong input image size was accepted");
  requireThrowsInvalidArgument(
      [&]() { ov_core::applyVGGTPreprocessPlan(plan, image, cv::Mat::zeros(plan.input_size, CV_8UC3)); },
      "non-monochrome mask was accepted");
}

void testFisheyeCropAndInvalidBorderMask() {
  const ov_core::VGGTPreprocessPlan cropped =
      ov_core::makeVGGTPreprocessPlan(0, "equidistant", tumCalibration(), cv::Size(1024, 1024), 518, 0.2, true);
  require(cropped.square_size == cv::Size(1024, 1024), "unexpected TUM-VI square-pixel size");
  require(cropped.resized_size == cv::Size(518, 518), "unexpected TUM-VI resized size");
  require(cropped.output_roi == cv::Rect(0, 63, 518, 392), "unexpected TUM-VI crop ROI");
  requireNear(cropped.output_calib(2), 259.0, 1e-12, "fisheye cx is not centered");
  requireNear(cropped.output_calib(3), 196.0, 1e-12, "fisheye cy did not account for the crop");

  // A wide-FOV plan deliberately samples outside the raw image. Those pixels must become invalid (255), not valid (0), in the mask.
  const ov_core::VGGTPreprocessPlan wide =
      ov_core::makeVGGTPreprocessPlan(0, "equidistant", tumCalibration(), cv::Size(1024, 1024), 518, 2.0, false);
  const cv::Mat image = cv::Mat::zeros(wide.input_size, CV_8UC1);
  const cv::Mat mask = cv::Mat::zeros(wide.input_size, CV_8UC1);
  const ov_core::VGGTPreprocessOutput output = ov_core::applyVGGTPreprocessPlan(wide, image, mask);
  require(cv::countNonZero(output.mask) > 0, "out-of-image fisheye samples were not marked invalid");
  cv::Mat non_binary = (output.mask != 0) & (output.mask != 255);
  require(cv::countNonZero(non_binary) == 0, "wide-FOV mask is not binary");
}

void testBuilderValidation() {
  const Eigen::VectorXd calibration = eurocCalibration();
  requireThrowsInvalidArgument(
      [&]() { ov_core::makeVGGTPreprocessPlan(0, "unknown", calibration, cv::Size(752, 480), 518); },
      "unknown distortion model was accepted");
  requireThrowsInvalidArgument(
      [&]() { ov_core::makeVGGTPreprocessPlan(0, "radtan", calibration, cv::Size(752, 480), 500); },
      "target width not divisible by 14 was accepted");
  requireThrowsInvalidArgument(
      [&]() { ov_core::makeVGGTPreprocessPlan(0, "radtan", calibration, cv::Size(0, 480), 518); },
      "invalid input size was accepted");
  requireThrowsInvalidArgument(
      [&]() { ov_core::makeVGGTPreprocessPlan(0, "radtan", calibration, cv::Size(752, 480), 518, 0.0); },
      "non-positive FOV scale was accepted");

  Eigen::VectorXd invalid_calibration = calibration;
  invalid_calibration(0) = 0.0;
  requireThrowsInvalidArgument(
      [&]() { ov_core::makeVGGTPreprocessPlan(0, "radtan", invalid_calibration, cv::Size(752, 480), 518); },
      "non-positive focal length was accepted");

  ov_core::VGGTPreprocessPlan invalid_plan =
      ov_core::makeVGGTPreprocessPlan(0, "radtan", calibration, cv::Size(752, 480), 518);
  invalid_plan.output_roi.x = invalid_plan.resized_size.width;
  requireThrowsInvalidArgument(
      [&]() { ov_core::applyVGGTPreprocessPlan(invalid_plan, cv::Mat::zeros(invalid_plan.input_size, CV_8UC1)); },
      "out-of-bounds output ROI was accepted");
}

} // namespace

int main() {
  try {
    testDatasetGeometryOracles();
    testIntegerResizeRatiosAndApply();
    testFisheyeCropAndInvalidBorderMask();
    testBuilderValidation();
  } catch (const std::exception &exception) {
    std::cerr << "test_vggt_preprocessing failed: " << exception.what() << std::endl;
    return 1;
  }
  std::cout << "test_vggt_preprocessing passed" << std::endl;
  return 0;
}
