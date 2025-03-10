cmake_minimum_required(VERSION 3.0.2)
project(aruco_pose_estimation)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  geometry_msgs
  sensor_msgs
  std_msgs
  aruco_interfaces
)

catkin_python_setup()

catkin_package()


# Install Python executables
catkin_install_python(PROGRAMS
    scripts/aruco_node.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
