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

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <memory>
#include <thread>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "ros/ROS1Visualizer.h"
#include "utils/dataset_reader.h"

using namespace ov_msckf;

std::shared_ptr<VioManager> sys;
std::shared_ptr<ROS1Visualizer> viz;

// Helper: extract a sensor_msgs::Image::ConstPtr from a bag message that may be
// either Image or CompressedImage. CompressedImage is decoded via cv_bridge.
static sensor_msgs::Image::ConstPtr extract_image_msg(const rosbag::MessageInstance &msg) {
  // Fast path: already raw Image
  sensor_msgs::Image::ConstPtr img = msg.instantiate<sensor_msgs::Image>();
  if (img != nullptr) {
    return img;
  }
  // Compressed path: decode and wrap in a fresh Image message
  sensor_msgs::CompressedImage::ConstPtr cimg = msg.instantiate<sensor_msgs::CompressedImage>();
  if (cimg == nullptr) {
    return nullptr;
  }
  try {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(cimg);
    sensor_msgs::ImagePtr out(new sensor_msgs::Image());
    out->header = cimg->header;
    cv_ptr->encoding = cv_ptr->encoding.empty() ? "bgr8" : cv_ptr->encoding;
    cv_ptr->toImageMsg(*out);
    return out;
  } catch (cv_bridge::Exception &e) {
    PRINT_ERROR(RED "[SERIAL]: cv_bridge decode failed: %s\n" RESET, e.what());
    return nullptr;
  }
}

// Main function
int main(int argc, char **argv) {
  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  // Launch our ros node
  ros::init(argc, argv, "ros1_serial_msckf");
  auto nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  parser->set_node_handler(nh);

  // Verbosity
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);
  // Load save_total_state from config and set ROS parameter
  bool save_total_state = false;
  parser->parse_config("save_total_state", save_total_state);
  nh->setParam("save_total_state", save_total_state);

  // Optional TUM-format logging flag
  bool save_total_state_tum = false;
  parser->parse_config("save_total_state_tum", save_total_state_tum, false);
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

  // Create our VIO system
  VioManagerOptions params;
  params.print_and_load(parser);
  // params.num_opencv_threads = 0; // uncomment if you want repeatability
  // params.use_multi_threading_pubs = 0; // uncomment if you want repeatability
  params.use_multi_threading_subs = false;
  sys = std::make_shared<VioManager>(params);
  viz = std::make_shared<ROS1Visualizer>(nh, sys);

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "[SERIAL]: unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Our imu topic
  std::string topic_imu;
  nh->param<std::string>("topic_imu", topic_imu, "/imu0");
  parser->parse_external("relative_config_imu", "imu0", "rostopic", topic_imu);
  PRINT_DEBUG("[SERIAL]: imu: %s\n", topic_imu.c_str());

  // Our camera topics
  std::vector<std::string> topic_cameras;
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    std::string cam_topic;
    nh->param<std::string>("topic_camera" + std::to_string(i), cam_topic, "/cam" + std::to_string(i) + "/image_raw");
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
    topic_cameras.emplace_back(cam_topic);
    PRINT_DEBUG("[SERIAL]: cam: %s\n", cam_topic.c_str());
  }

  // Location of the ROS bag we want to read in
  std::string path_to_bag;
  nh->param<std::string>("path_bag", path_to_bag, "");
  PRINT_DEBUG("[SERIAL]: ros bag path is: %s\n", path_to_bag.c_str());

  // Load groundtruth if we have it
  // NOTE: needs to be a csv ASL format file
  std::map<double, Eigen::Matrix<double, 17, 1>> gt_states;
  if (nh->hasParam("path_gt")) {
    std::string path_to_gt;
    nh->param<std::string>("path_gt", path_to_gt, "");
    if (!path_to_gt.empty()) {
      ov_core::DatasetReader::load_gt_file(path_to_gt, gt_states);
      PRINT_DEBUG("[SERIAL]: gt file path is: %s\n", path_to_gt.c_str());
    }
  }

  // Get our start location and how much of the bag we want to play
  // Make the bag duration < 0 to just process to the end of the bag
  double bag_start, bag_durr;
  nh->param<double>("bag_start", bag_start, 0);
  nh->param<double>("bag_durr", bag_durr, -1);
  PRINT_DEBUG("[SERIAL]: bag start: %.1f\n", bag_start);
  PRINT_DEBUG("[SERIAL]: bag duration: %.1f\n", bag_durr);

  // Real-time playback factor (0 = disabled, 1.0 = real-time, 2.0 = 2x speed)
  double bag_realtime_factor = 0.0;
  parser->parse_config("bag_realtime_factor", bag_realtime_factor, false);
  nh->param<double>("bag_realtime_factor", bag_realtime_factor, bag_realtime_factor);
  PRINT_DEBUG("[SERIAL]: realtime factor: %.1f\n", bag_realtime_factor);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Load rosbag here, and find messages we can play
  rosbag::Bag bag;
  bag.open(path_to_bag, rosbag::bagmode::Read);

  // We should load the bag as a view
  // Here we go from beginning of the bag to the end of the bag
  rosbag::View view_full;
  rosbag::View view;

  // Start a few seconds in from the full view time
  // If we have a negative duration then use the full bag length
  view_full.addQuery(bag);
  ros::Time time_init = view_full.getBeginTime();
  time_init += ros::Duration(bag_start);
  ros::Time time_finish = (bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(bag_durr);
  PRINT_DEBUG("time start = %.6f\n", time_init.toSec());
  PRINT_DEBUG("time end   = %.6f\n", time_finish.toSec());
  view.addQuery(bag, time_init, time_finish);

  // Check to make sure we have data to play
  if (view.size() == 0) {
    PRINT_ERROR(RED "[SERIAL]: No messages to play on specified topics.  Exiting.\n" RESET);
    ros::shutdown();
    return EXIT_FAILURE;
  }

  // We going to loop through and collect a list of all messages
  // This is done so we can access arbitrary points in the bag
  // NOTE: if we instantiate messages here, this requires the whole bag to be read
  // NOTE: thus we just check the topic which allows us to quickly loop through the index
  // NOTE: see this PR https://github.com/ros/ros_comm/issues/117
  double max_camera_time = -1;
  std::vector<rosbag::MessageInstance> msgs;
  for (const rosbag::MessageInstance &msg : view) {
    if (!ros::ok())
      break;
    if (msg.getTopic() == topic_imu) {
      // if (msg.instantiate<sensor_msgs::Imu>() == nullptr) {
      //   PRINT_ERROR(RED "[SERIAL]: IMU topic has unmatched message types!!\n" RESET);
      //   PRINT_ERROR(RED "[SERIAL]: Supports: sensor_msgs::Imu\n" RESET);
      //   return EXIT_FAILURE;
      // }
      msgs.push_back(msg);
    }
    for (int i = 0; i < params.state_options.num_cameras; i++) {
      if (msg.getTopic() == topic_cameras.at(i)) {
        // sensor_msgs::CompressedImage::ConstPtr img_c = msg.instantiate<sensor_msgs::CompressedImage>();
        // sensor_msgs::Image::ConstPtr img_i = msg.instantiate<sensor_msgs::Image>();
        // if (img_c == nullptr && img_i == nullptr) {
        //   PRINT_ERROR(RED "[SERIAL]: Image topic has unmatched message types!!\n" RESET);
        //   PRINT_ERROR(RED "[SERIAL]: Supports: sensor_msgs::Image and sensor_msgs::CompressedImage\n" RESET);
        //   return EXIT_FAILURE;
        // }
        msgs.push_back(msg);
        max_camera_time = std::max(max_camera_time, msg.getTime().toSec());
      }
    }
  }
  PRINT_DEBUG("[SERIAL]: total of %zu messages!\n", msgs.size());

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Loop through our message array, and lets process them
  std::set<int> used_index;

  // For real-time playback - track cumulative time from start
  auto wall_start = std::chrono::steady_clock::now();
  ros::Time bag_start_time = msgs.at(0).getTime();

  for (int m = 0; m < (int)msgs.size(); m++) {

    // End once we reach the last time, or skip if before beginning time (shouldn't happen)
    if (!ros::ok() || msgs.at(m).getTime() > time_finish || msgs.at(m).getTime().toSec() > max_camera_time)
      break;
    if (msgs.at(m).getTime() < time_init)
      continue;

    // Skip messages that we have already used
    if (used_index.find(m) != used_index.end()) {
      used_index.erase(m);
      continue;
    }

    // Real-time playback with catch-up mode
    if (bag_realtime_factor > 0) {
      ros::Time current_msg_time = msgs.at(m).getTime();
      double bag_elapsed = (current_msg_time - bag_start_time).toSec();
      auto actual_wall = std::chrono::steady_clock::now() - wall_start;

      // Normal target time (1x speed)
      auto target_wall_normal = std::chrono::duration<double>(bag_elapsed / bag_realtime_factor);
      // Catch-up target time (2x speed)
      auto target_wall_catchup = std::chrono::duration<double>(bag_elapsed / (bag_realtime_factor * 2.0));

      if (actual_wall < target_wall_normal) {
        // Ahead of normal progress, wait to maintain set speed
        std::this_thread::sleep_for(target_wall_normal - actual_wall);
      } else if (actual_wall < target_wall_catchup) {
        // Behind but within catch-up range, catch up at 2x set speed
        std::this_thread::sleep_for(target_wall_catchup - actual_wall);
      }
      // else: Too far behind (beyond 2x speed range), don't wait
    }

    // IMU processing
    if (msgs.at(m).getTopic() == topic_imu) {
      // PRINT_DEBUG("processing imu = %.3f sec\n", msgs.at(m).getTime().toSec() - time_init.toSec());
      viz->callback_inertial(msgs.at(m).instantiate<sensor_msgs::Imu>());
    }

    // Camera processing
    for (int cam_id = 0; cam_id < params.state_options.num_cameras; cam_id++) {

      // Skip if this message is not a camera topic
      if (msgs.at(m).getTopic() != topic_cameras.at(cam_id))
        continue;

      // We have a matching camera topic here, now find the other cameras for this time
      // For each camera, we will find the nearest timestamp (within 0.02sec) that is greater than the current
      // If we are unable, then this message should just be skipped since it isn't a sync'ed pair!
      std::map<int, int> camid_to_msg_index;
      double meas_time = msgs.at(m).getTime().toSec();
      for (int cam_idt = 0; cam_idt < params.state_options.num_cameras; cam_idt++) {
        if (cam_idt == cam_id) {
          camid_to_msg_index.insert({cam_id, m});
          continue;
        }
        int cam_idt_idx = -1;
        for (int mt = m; mt < (int)msgs.size(); mt++) {
          if (msgs.at(mt).getTopic() != topic_cameras.at(cam_idt))
            continue;
          if (std::abs(msgs.at(mt).getTime().toSec() - meas_time) < 0.02)
            cam_idt_idx = mt;
          break;
        }
        if (cam_idt_idx != -1) {
          camid_to_msg_index.insert({cam_idt, cam_idt_idx});
        }
      }

      // Skip processing if we were unable to find any messages
      if ((int)camid_to_msg_index.size() != params.state_options.num_cameras) {
        PRINT_DEBUG(YELLOW "[SERIAL]: Unable to find stereo pair for message %d at %.2f into bag (will skip!)\n" RESET, m,
                    meas_time - time_init.toSec());
        continue;
      }

      // Check if we should initialize using the groundtruth
      Eigen::Matrix<double, 17, 1> imustate;
      if (!gt_states.empty() && !sys->initialized() && ov_core::DatasetReader::get_gt_state(meas_time, imustate, gt_states)) {
        // biases are pretty bad normally, so zero them
        // imustate.block(11,0,6,1).setZero();
        sys->initialize_with_gt(imustate);
      }

      // Pass our data into our visualizer callbacks!
      // PRINT_DEBUG("processing cam = %.3f sec\n", msgs.at(m).getTime().toSec() - time_init.toSec());
      if (params.state_options.num_cameras == 1) {
        viz->callback_monocular(extract_image_msg(msgs.at(camid_to_msg_index.at(0))), 0);
      } else if (params.state_options.num_cameras == 2) {
        auto msg0 = msgs.at(camid_to_msg_index.at(0));
        auto msg1 = msgs.at(camid_to_msg_index.at(1));
        used_index.insert(camid_to_msg_index.at(0)); // skip this message
        used_index.insert(camid_to_msg_index.at(1)); // skip this message
        viz->callback_stereo(extract_image_msg(msg0), extract_image_msg(msg1), 0, 1);
      } else {
        PRINT_ERROR(RED "[SERIAL]: We currently only support 1 or 2 camera serial input....\n" RESET);
        return EXIT_FAILURE;
      }

      break;
    }
  }

  // Final visualization
  viz->wait_for_image_publish_worker();
  sys->wait_for_initialization_worker();
  viz->visualize_final();
  fflush(stdout);
  fflush(stderr);
  _exit(0);
  // Done!
  return EXIT_SUCCESS;
}
