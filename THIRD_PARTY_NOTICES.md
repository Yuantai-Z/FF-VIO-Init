# Third-Party Notices

This file records the origin and licensing notices for third-party material
retained in this repository. The repository as a combined work is distributed
under the terms in [`LICENSE`](LICENSE); the notices and license terms below
continue to apply to their respective components.

## OpenVINS origin and modification notice

This repository is a modified derivative of
[OpenVINS](https://github.com/rpng/open_vins). The upstream baseline is commit
[`d0075dd86f2d96618f14829e6c33425f9278610a`](https://github.com/rpng/open_vins/commit/d0075dd86f2d96618f14829e6c33425f9278610a).
The derivative changes were made during 2025-2026.

The retained OpenVINS file headers identify the following copyright notices:

```text
Copyright (C) 2018-2023 Patrick Geneva
Copyright (C) 2018-2023 Guoquan Huang
Copyright (C) 2018-2023 OpenVINS Contributors
Copyright (C) 2018-2019 Kevin Eckenhoff
```

The retained OpenVINS code and this derivative are provided under the GNU
General Public License; see the root `LICENSE` and the applicable source-file
headers for the precise terms. This repository is not an official OpenVINS
release. The upstream OpenVINS authors and contributors are not responsible
for these modifications and do not provide support or warranty for them.

The regression fixtures in `ov_eval/example/` are retained from the same
OpenVINS baseline. Their upstream `readme.txt` records the trajectory sources;
they are included only to exercise the inherited evaluator and are not a
project dataset.

`config/tum_vi/mask_tumvi0_1024.png` and
`config/tum_vi/mask_tumvi1_1024.png` are project-produced, two-times
nearest-neighbor grayscale adaptations of the corresponding 512-pixel
OpenVINS mask assets. They remain covered by this derivative's GPL terms.

## Continuous preintegration (MIT)

The continuous preintegration implementation under `ov_core/src/cpi/` retains
the following notice:

```text
MIT License
Copyright (c) 2018 Kevin Eckenhoff
Copyright (c) 2018 Patrick Geneva
Copyright (c) 2018 Guoquan Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## matplotlib-cpp (MIT)

`ov_core/src/plot/matplotlibcpp.h` is distributed with
`ov_core/src/plot/LICENSE`, which contains the following notice:

```text
The MIT License (MIT)

Copyright (c) 2014 Benno Evers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GTSAM-derived rotation logic (BSD 3-Clause)

The `log_so3` and `log_se3` logic in `ov_core/src/utils/quat_ops.h` is based on
[GTSAM's rotation implementation](https://github.com/borglab/gtsam/blob/26356db/gtsam/geometry/SO3.cpp).
The applicable BSD notice is:

```text
Copyright (c) 2010, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## LSD-SLAM depth-color fragment (GPL-3.0-or-later)

The sparse-depth color visualization in
`ov_msckf/src/ros/ROS1Visualizer.cpp` and
`ov_msckf/src/ros/ROS2Visualizer.cpp` contains a small fragment adapted from
[`lsd_slam_core/src/util/globalFuncs.cpp`](https://github.com/tum-vision/lsd_slam/blob/d1e6f0e1a027889985d2e6b4c0fe7a90b0c75067/lsd_slam_core/src/util/globalFuncs.cpp).
The source file carries this notice:

```text
This file is part of LSD-SLAM.

Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
For more information see <http://vision.in.tum.de/lsdslam>

LSD-SLAM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LSD-SLAM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
```

The complete GNU GPL version 3 text is included in the root `LICENSE`.

## External VGGT and Pi3X model materials

The GPL-covered `vggt_ros` package in this repository is an integration
wrapper. It contains no VGGT or Pi3/Pi3X model source, trained model weights,
checkpoint, model cache, or upstream Python environment. Those materials must
be obtained separately and are not relicensed by this repository.

### VGGT

Meta distributes the official
[VGGT source repository](https://github.com/facebookresearch/vggt) under its
custom
[VGGT License and incorporated Acceptable Use Policy](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt).
That agreement defines its covered research materials to include model code,
trained weights, and inference/training-enabling code. It is not this
repository's GPL-3.0 license and should not be described as an open-source
license merely because the source is publicly accessible.

Checkpoint conditions can differ. Users must review the model card and access
conditions for the exact checkpoint and revision selected by `model_name` and
`model_revision`, including, as applicable,
[`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) or the separately
offered
[`facebook/VGGT-1B-Commercial`](https://huggingface.co/facebook/VGGT-1B-Commercial).
This notice does not interpret those terms or represent that a checkpoint is
suitable for a particular use.

### Pi3 source and Pi3X weights

The official [Pi3 source repository](https://github.com/yyfz/Pi3) currently
identifies its code license as
[BSD 3-Clause](https://github.com/yyfz/Pi3/blob/main/LICENSE). The separately
published [`yyfz233/Pi3X` model card](https://huggingface.co/yyfz233/Pi3X)
currently identifies the Pi3X weights as CC BY-NC 4.0 and non-commercial.
These are distinct grants: the source-code license does not grant rights to the
weights, and the weight terms do not replace the source-code license.

Upstream repositories and model cards can change. Users and distributors must
verify the terms for the exact source revision and weight snapshot they obtain.
No conclusion about those external terms is incorporated into this notice.

## External build and runtime dependencies

ROS, catkin, Eigen, OpenCV, Boost, Ceres Solver, Python, and other build or
runtime packages are external dependencies and are not relicensed by this
repository. They remain subject to the licenses supplied by their respective
authors and distributors.
