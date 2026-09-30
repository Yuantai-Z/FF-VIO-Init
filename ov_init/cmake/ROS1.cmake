cmake_minimum_required(VERSION 3.3)

# Find ROS build system and every message package used by the library or tools.
find_package(catkin REQUIRED COMPONENTS
        ov_core
        roscpp
        sensor_msgs
        std_msgs
)

# Describe ROS project
add_definitions(-DROS_AVAILABLE=1)
catkin_package(
        CATKIN_DEPENDS ov_core roscpp sensor_msgs std_msgs
        INCLUDE_DIRS src/
        LIBRARIES ov_init_lib
)

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CERES_LIBRARIES}
        ${catkin_LIBRARIES}
)

##################################################
# Make the shared library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/ceres/Factor_GenericPrior.cpp
        src/ceres/Factor_ImageReprojCalib.cpp
        src/ceres/Factor_ImageReprojScale.cpp
        src/ceres/Factor_ImuCPIv1.cpp
        src/ceres/State_JPLQuatLocal.cpp
        src/init/InertialInitializer.cpp
        src/dynamic/DynamicInitializer.cpp
        src/static/StaticInitializer.cpp
        src/vggt/VGGTInitializer.cpp
)
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_init_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(ov_init_lib ${thirdparty_libraries})
target_include_directories(ov_init_lib PUBLIC src/)
install(TARGETS ov_init_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)


##################################################
# Make binary files!
##################################################

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../config/
        DESTINATION ${CATKIN_GLOBAL_SHARE_DESTINATION}/config
)

add_executable(test_vggt_nonlinear_mode src/test_vggt_nonlinear_mode.cpp)
target_link_libraries(test_vggt_nonlinear_mode ov_init_lib ${thirdparty_libraries})
install(TARGETS test_vggt_nonlinear_mode
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(test_factor_pointcloud_scale src/test_factor_pointcloud_scale.cpp)
target_link_libraries(test_factor_pointcloud_scale ov_init_lib ${thirdparty_libraries})
install(TARGETS test_factor_pointcloud_scale
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(test_vggt_pointcloud_protocol src/test_vggt_pointcloud_protocol.cpp)
target_link_libraries(test_vggt_pointcloud_protocol ov_init_lib ${thirdparty_libraries})
install(TARGETS test_vggt_pointcloud_protocol
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
