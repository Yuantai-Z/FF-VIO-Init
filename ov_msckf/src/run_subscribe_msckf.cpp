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

#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/dataset_reader.h"

#if ROS_AVAILABLE == 1
#include "ros/ROS1Visualizer.h"
#include <ros/ros.h>
#elif ROS_AVAILABLE == 2
#include "ros/ROS2Visualizer.h"
#include <rclcpp/rclcpp.hpp>
#endif

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;
#if ROS_AVAILABLE == 1
std::shared_ptr<ROS1Visualizer> viz;
#elif ROS_AVAILABLE == 2
std::shared_ptr<ROS2Visualizer> viz;
#endif

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }
#if ROS_AVAILABLE == 1
  // Launch our ros node
  ros::init(argc, argv, "run_subscribe_msckf");
  auto nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);
#elif ROS_AVAILABLE == 2
  // Launch our ros node
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("run_subscribe_msckf", options);
  node->get_parameter<std::string>("config_path", config_path);
#endif

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
#if ROS_AVAILABLE == 1
  parser->set_node_handler(nh);

  // Mirror output settings from the YAML onto ROS parameters for the visualizer.
  bool save_total_state = false;
  bool save_total_state_tum = false;
  parser->parse_config("save_total_state", save_total_state);
  parser->parse_config("save_total_state_tum", save_total_state_tum, false);
  nh->setParam("save_total_state", save_total_state);
  nh->setParam("save_total_state_tum", save_total_state_tum);

  if (save_total_state) {
    std::string filepath_est = "state_estimate.txt";
    parser->parse_config("filepath_est", filepath_est);
    nh->setParam("filepath_est", filepath_est);

    if (!save_total_state_tum) {
      std::string filepath_std = "state_deviation.txt";
      std::string filepath_gt = "state_groundtruth.txt";
      parser->parse_config("filepath_std", filepath_std);
      parser->parse_config("filepath_gt", filepath_gt);
      nh->setParam("filepath_std", filepath_std);
      nh->setParam("filepath_gt", filepath_gt);
    }
  }
#elif ROS_AVAILABLE == 2
  parser->set_node(node);

  // Parse YAML parameters and set them to ROS2 node parameters
  bool save_total_state = false;
  bool save_total_state_tum = false;
  std::string filepath_est = "state_estimate.txt";
  std::string filepath_std = "state_deviation.txt";
  std::string filepath_gt = "state_groundtruth.txt";

  parser->parse_config("save_total_state", save_total_state);
  parser->parse_config("save_total_state_tum", save_total_state_tum, false);
  parser->parse_config("filepath_est", filepath_est);
  parser->parse_config("filepath_std", filepath_std);
  parser->parse_config("filepath_gt", filepath_gt);

  // Set these parameters to the ROS2 node so Visualizer can access them
  node->declare_parameter("save_total_state", save_total_state);
  node->declare_parameter("save_total_state_tum", save_total_state_tum);
  node->declare_parameter("filepath_est", filepath_est);
  node->declare_parameter("filepath_std", filepath_std);
  node->declare_parameter("filepath_gt", filepath_gt);

#endif

  // Verbosity
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = true;
  sys = std::make_shared<VioManager>(params);
#if ROS_AVAILABLE == 1
  viz = std::make_shared<ROS1Visualizer>(nh, sys);
  viz->setup_subscribers(parser);
#elif ROS_AVAILABLE == 2
  viz = std::make_shared<ROS2Visualizer>(node, sys);
  viz->setup_subscribers(parser);


#endif

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Spin off to ROS
  PRINT_DEBUG("done...spinning to ros\n");
#if ROS_AVAILABLE == 1
  // ros::spin();
  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();
  spinner.stop();
  viz->wait_for_update_worker();
  viz->wait_for_image_publish_worker();
  sys->wait_for_initialization_worker();
#elif ROS_AVAILABLE == 2
  // rclcpp::spin(node);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
#endif

  // Final visualization
  viz->visualize_final();
#if ROS_AVAILABLE == 1
  ros::shutdown();
#elif ROS_AVAILABLE == 2
  rclcpp::shutdown();
#endif

  // Done!
  return EXIT_SUCCESS;
}
