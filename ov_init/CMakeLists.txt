cmake_minimum_required(VERSION 3.3)
project(ov_init)

# Ignore anaconda paths to avoid library conflicts during linking
# This prevents CMake from finding incompatible libraries in anaconda installation
if(EXISTS "$ENV{HOME}/anaconda3")
    list(APPEND CMAKE_IGNORE_PATH "$ENV{HOME}/anaconda3")
endif()

# Include libraries (if we don't have OpenCV 4, fall back to OpenCV 3)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Ceres REQUIRED)
message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION} " | CERES: " ${Ceres_VERSION})

# Require C++14 for the initializer implementation.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized")

# The point-cloud transport uses ROS 1 message types and roscpp directly.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
