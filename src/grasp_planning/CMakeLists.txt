cmake_minimum_required(VERSION 3.16)
project(grasp_planning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(grasp_interfaces REQUIRED)
find_package(relaxed_ik_lite REQUIRED)

add_executable(pick_place_node src/pick_place_node.cpp)
target_link_libraries(pick_place_node relaxed_ik_lite::relaxed_ik_lite)
ament_target_dependencies(pick_place_node
  rclcpp rclcpp_action moveit_ros_planning_interface moveit_core moveit_msgs control_msgs
  shape_msgs std_srvs tf2_eigen grasp_interfaces)

install(TARGETS pick_place_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/analyze_benchmark.py scripts/wrist_camera_tf.py scripts/benchmark_protocol.py
  scripts/protocol_report.py DESTINATION lib/${PROJECT_NAME})

ament_package()
