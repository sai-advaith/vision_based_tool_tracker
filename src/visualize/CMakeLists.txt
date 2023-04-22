cmake_minimum_required(VERSION 2.8.3)
project(visualize)

## Compile as C++11, supported in ROS Kinetic and newer

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  tf
  roslint
  nav_msgs
  apriltag_ros
)
###################################
## catkin specific configuration ##
###################################


add_message_files(
  FILES
  FilterErr.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  roscpp
  rospy
  nav_msgs
  geometry_msgs
  sensor_msgs
  apriltag_ros
  std_msgs
)
## Executables
##add_executable(visualize
##                src/motionModel.cpp)


###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${apriltag_INCLUDE_DIRS}
  ${apriltag_ros_INCLUDE_DIRS}
  ${usb_cam_INCLDUE_DIRS}
)
# Install headers
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY nodes/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}/nodes
  FILES_MATCHING PATTERN "*.py"
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

roslint_python()
roslint_cpp()
