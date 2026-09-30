cmake_minimum_required(VERSION 3.3)
project(ov_core)

# Ignore anaconda paths to avoid library conflicts during linking
# This prevents CMake from finding incompatible libraries in anaconda installation
if(EXISTS "$ENV{HOME}/anaconda3")
    list(APPEND CMAKE_IGNORE_PATH "$ENV{HOME}/anaconda3")
endif()

# Include libraries (if we don't have opencv 4, then fallback to opencv 3)
# The OpenCV version needs to match the one used by cv_bridge otherwise you will get a segmentation fault!
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    # set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
    find_package(OpenCV 4 REQUIRED)
endif ()
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
message(STATUS "OPENCV: " ${OpenCV_VERSION} " | BOOST: " ${Boost_VERSION})

# By default we build with ROS, but you can disable this and just build as a library
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)

# If we will compile with aruco support
option(ENABLE_ARUCO_TAGS "Enable or disable aruco tag (disable if no contrib modules)" ON)
if (NOT ENABLE_ARUCO_TAGS)
    add_definitions(-DENABLE_ARUCO_TAGS=0)
    message(WARNING "DISABLING ARUCOTAG TRACKING!")
else ()
    add_definitions(-DENABLE_ARUCO_TAGS=1)
endif ()

# OpenVINS plotting helpers optionally embed matplotlib through Python.
find_package(PythonLibs QUIET)
option(DISABLE_MATPLOTLIB "Disable or enable matplotlib plot scripts" OFF)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")

# This release supports the ROS 1 build path. ROS1.cmake also contains the
# inherited non-ROS fallback used when ENABLE_ROS is disabled.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS1.cmake)
