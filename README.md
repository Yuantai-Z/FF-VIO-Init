<p align="center">
  <h1 align="center"><strong>Efficient Feature-Free Initialization for Monocular Visual-Inertial Systems Using a Feed-Forward 3D Model</strong></h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Lh2CthAAAAAJ&hl">Yuantai Zhang</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=f7ox7CIAAAAJ">Jiaqi Yang</a><sup>1</sup>,
    <a href="https://huajian-zeng.github.io/">Huajian Zeng</a><sup>1</sup>,
    <a href="https://changhao-chen.github.io/">Changhao Chen</a><sup>2</sup>,
    <a href="https://sites.google.com/view/haoangli/homepage">Haoang Li</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=6JscxDkAAAAJ&hl=zh-CN">Liang Li</a><sup>3</sup>,
    <a href="https://mbzuai.ac.ae/study/faculty/dezhen-song/">Dezhen Song</a><sup>1</sup>,
    <a href="https://xingxingzuo.github.io/">Xingxing Zuo</a><sup>1,*</sup>
    <br>
    <sup>1</sup>MBZUAI, <sup>2</sup>HKUST (GZ), <sup>3</sup>ZJU
    <br>
    <sup>*</sup>Corresponding author
  </p>
  <p align="center">🎉 <strong>Accepted to RSS 2026</strong></p>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.17327-red)](https://arxiv.org/abs/2605.17327)
[![Video](https://img.shields.io/badge/Video-%E2%96%B6-red?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=uxTh21D8log)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Built on OpenVINS](https://img.shields.io/badge/Built%20on-OpenVINS-orange.svg)](https://github.com/rpng/open_vins)

</div>

## Abstract

Fast and reliable initialization is critical for monocular visual-inertial
navigation systems (VINS), as it establishes the starting conditions for
subsequent state estimation. Despite steady progress, most existing methods
heavily rely on visual feature correspondences and require 3--4 seconds of
sensory data for successful initialization, which limits their applicability
and efficiency. With the advent of feed-forward 3D models that can directly
predict point clouds from images, we revisit the visual-inertial initialization
problem from a concise perspective. We propose a feature-free initialization
framework that leverages up-to-scale point clouds predicted by a feed-forward
3D model, thereby removing visual feature tracking and landmark estimation
from initialization. Experiments on public and self-collected datasets show
high success rates, short initialization times, and robust performance in
visually degraded environments.

## Supplementary Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=uxTh21D8log">
    <img src="https://img.youtube.com/vi/uxTh21D8log/maxresdefault.jpg" alt="Supplementary video" width="80%">
  </a>
</p>

## Installation

Build the ROS packages in a catkin workspace:

```bash
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/Yuantai-Z/FF-VIO-Init.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src --rosdistro noetic -r -y
catkin build ov_core ov_eval ov_init ov_msckf vggt_ros --no-status
source devel/setup.bash
```

Install a CUDA-compatible PyTorch build and VGGT in a separate Python
environment:

```bash
python3.10 -m venv /absolute/path/to/vggt-venv
source /absolute/path/to/vggt-venv/bin/activate
git clone https://github.com/facebookresearch/vggt.git /absolute/path/to/vggt
python -m pip install -r /absolute/path/to/vggt/requirements.txt
python -m pip install -e /absolute/path/to/vggt
```

## Configuration

Parameters are documented directly in the configuration files:

- `config/euroc_mav/estimator_config.yaml`
- `config/tum_vi/estimator_config.yaml`
- `config/Self-RealSenseD455/estimator_config.yaml`
- `vggt_ros/config/vggt.yaml`
- `vggt_ros/config/pi3x.yaml`

The supplied estimator configurations use FF by default. The inference node
receives the processed OpenVINS cam0 stream, not the original bag image.
VGGT and Pi3X use their original upstream pretrained weights; this project
does not provide or use fine-tuned weights.

## EuRoC Example

Start the inference node first and wait for its `ready` message.

Terminal 1:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export VGGT_ROS_PYTHON=/absolute/path/to/vggt-venv/bin/python3
export VGGT_SOURCE_PATH=/absolute/path/to/vggt
roslaunch vggt_ros inference.launch model:=vggt
```

Terminal 2:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch ov_msckf serial.launch \
  config:=euroc_mav \
  bag:=/absolute/path/to/V1_01_easy.bag \
  bag_start:=4 \
  max_cameras:=1 \
  use_stereo:=false
```

With the default `ROS_HOME`, the result is written to:

```text
~/.ros/output/Euroc/V1_01/trajectory.tum
```

## Evaluation

```bash
rosrun ov_eval eval_initialization \
  ~/.ros/output/Euroc/V1_01/trajectory.tum \
  /absolute/path/to/mav0/state_groundtruth_estimate0/data.csv \
  posyaw
```

The evaluator writes `trajectory_init_eval.md` beside the trajectory. The
report contains one row per sequence and a final `Mean` row. When periodic
reset is enabled, every sequence is evaluated in its own local frame.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@INPROCEEDINGS{ZhangY-RSS-26,
    AUTHOR    = {Yuantai Zhang AND Jiaqi Yang AND Huajian Zeng AND Changhao Chen AND Haoang Li AND Liang Li AND Dezhen Song AND Xingxing Zuo},
    TITLE     = {{Efficient Feature-Free Initialization for Monocular Visual-Inertial Systems Using A Feed-Forward 3D Model}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2026},
    ADDRESS   = {Sydney, Australia},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2026.XXII.047}
}
```

## Acknowledgements

This work is built on top of
[OpenVINS](https://github.com/rpng/open_vins) and uses the
[VGGT](https://github.com/facebookresearch/vggt) and
[Pi3X](https://github.com/yyfz/Pi3) feed-forward 3D models. We thank all teams
for their contributions and open-source releases.

## License

This project is released under GPL-3.0. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
