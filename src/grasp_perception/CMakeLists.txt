cmake_minimum_required(VERSION 3.16)
project(grasp_perception LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Eigen3 REQUIRED)
# PCL's config pulls in VTK, whose Ubuntu 24.04 package expects MPI::MPI_C to exist.
find_package(MPI QUIET COMPONENTS C)
if(NOT TARGET MPI::MPI_C)
  add_library(MPI::MPI_C INTERFACE IMPORTED)
endif()
find_package(PCL REQUIRED COMPONENTS common io filters features search segmentation sample_consensus)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
# Builds as a ROS 2 package under colcon, or as a plain CMake library (core algorithms
# and tests only) when ament is not available.
find_package(ament_cmake QUIET)

add_library(grasp_perception_core
  src/cloud_processing.cpp
  src/contour.cpp
  src/efd.cpp
  src/grasp_synthesis.cpp)
target_include_directories(grasp_perception_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})
target_link_libraries(grasp_perception_core PUBLIC Eigen3::Eigen ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_compile_definitions(grasp_perception_core PUBLIC ${PCL_DEFINITIONS})
set_target_properties(grasp_perception_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(detect_grasps_pcd tools/detect_grasps_pcd.cpp)
target_link_libraries(detect_grasps_pcd grasp_perception_core)

if(ament_cmake_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(tf2_eigen REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(grasp_interfaces REQUIRED)

  add_executable(perception_node src/perception_node.cpp)
  target_link_libraries(perception_node grasp_perception_core)
  ament_target_dependencies(perception_node
    rclcpp sensor_msgs geometry_msgs visualization_msgs tf2_ros tf2_eigen pcl_conversions grasp_interfaces)

  install(TARGETS grasp_perception_core EXPORT export_grasp_perception
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
  install(TARGETS perception_node detect_grasps_pcd DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY include/ DESTINATION include)
  install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
  ament_export_targets(export_grasp_perception HAS_LIBRARY_TARGET)
  ament_export_dependencies(Eigen3 PCL OpenCV)

  if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_grasp_perception test/test_segmentation.cpp test/test_efd.cpp test/test_grasp_synthesis.cpp)
    target_link_libraries(test_grasp_perception grasp_perception_core)
  endif()
  ament_package()
else()
  enable_testing()
  find_package(GTest REQUIRED)
  include(GoogleTest)
  add_executable(test_grasp_perception test/test_segmentation.cpp test/test_efd.cpp test/test_grasp_synthesis.cpp)
  target_link_libraries(test_grasp_perception grasp_perception_core GTest::gtest_main)
  gtest_discover_tests(test_grasp_perception)
endif()
