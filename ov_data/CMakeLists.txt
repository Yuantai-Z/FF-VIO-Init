cmake_minimum_required(VERSION 3.3)
project(ov_data)

# Dataset payloads are not distributed; keep only the ROS 1 placeholder package.
find_package(catkin REQUIRED)
catkin_package()

install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
