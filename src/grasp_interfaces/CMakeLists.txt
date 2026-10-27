cmake_minimum_required(VERSION 3.16)
project(grasp_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Grasp.msg"
  "srv/DetectGrasps.srv"
  "srv/RunTrial.srv"
  "srv/SetScene.srv"
  DEPENDENCIES geometry_msgs)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
