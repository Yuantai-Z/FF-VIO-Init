/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 * Modifications Copyright (C) 2025-2026 Yuantai-Z
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

#ifndef OV_CORE_OPENCV_LAMBDA_BODY_H
#define OV_CORE_OPENCV_LAMBDA_BODY_H

#include <functional>
#include <opencv2/opencv.hpp>
#include <utility>

namespace ov_core {

/*
 * @brief Helper class to do OpenCV parallelization
 *
 * Small local adapter used by the inherited tracking code on OpenCV versions
 * whose parallel_for_ overload requires a cv::ParallelLoopBody object.
 */
class LambdaBody final : public cv::ParallelLoopBody {
public:
  explicit LambdaBody(std::function<void(const cv::Range &)> callback) : callback_(std::move(callback)) {}

  void operator()(const cv::Range &range) const override { callback_(range); }

private:
  std::function<void(const cv::Range &)> callback_;
};

} /* namespace ov_core */

#endif /* OV_CORE_OPENCV_LAMBDA_BODY_H */
