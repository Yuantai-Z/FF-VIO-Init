cmake_minimum_required(VERSION 3.3)
project(ov_eval)

# Ignore Anaconda paths so ROS Noetic libraries are not mixed with an
# incompatible Conda Boost/Python stack during linking.
if(EXISTS "$ENV{HOME}/anaconda3")
    list(APPEND CMAKE_IGNORE_PATH "$ENV{HOME}/anaconda3")
endif()

# Include libraries
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)

# By default we build with ROS, but you can disable this and just build as a library
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)

# check if we have our python libs files (will search for python3 then python2 installs)
# sudo apt-get install python-matplotlib python-numpy python-dev
# https://cmake.org/cmake/help/v3.10/module/FindPythonLibs.html
find_package(PythonLibs QUIET)
option(DISABLE_MATPLOTLIB "Disable or enable matplotlib plot scripts in ov_eval" OFF)
if (PYTHONLIBS_FOUND AND NOT DISABLE_MATPLOTLIB)
    add_definitions(-DHAVE_PYTHONLIBS=1)
    message(STATUS "PYTHON VERSION: " ${PYTHONLIBS_VERSION_STRING})
    message(STATUS "PYTHON INCLUDE: " ${PYTHON_INCLUDE_DIRS})
    message(STATUS "PYTHON LIBRARIES: " ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})
    list(APPEND thirdparty_libraries ${PYTHON_LIBRARIES})
endif ()

# We need c++14 for ROS2, thus just require it for everybody
# NOTE: To future self, hope this isn't an issue...
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized")

# This release supports the ROS 1 build path. ROS1.cmake also contains the
# inherited non-ROS fallback used when ENABLE_ROS is disabled.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
