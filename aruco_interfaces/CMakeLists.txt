cmake_minimum_required(VERSION 3.0.2)
project(aruco_interfaces)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
  std_msgs
  message_generation
)

add_message_files(
  FILES
  ArucoMarkers.msg
)
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS message_runtime
)
