# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.0.2)
project(vggt_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs
)

catkin_install_python(PROGRAMS
  scripts/inference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()
