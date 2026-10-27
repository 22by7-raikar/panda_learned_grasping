cmake_minimum_required(VERSION 3.16)
project(relaxed_ik_lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Eigen3 REQUIRED)
# Builds as a ROS 2 package under colcon, or as a plain CMake library (for CI and
# quick iteration) when ament is not available.
find_package(ament_cmake QUIET)

add_library(${PROJECT_NAME} src/panda_kinematics.cpp src/relaxed_ik.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(ik_benchmark test/ik_benchmark.cpp)
target_link_libraries(ik_benchmark ${PROJECT_NAME})

if(ament_cmake_FOUND)
  install(DIRECTORY include/ DESTINATION include)
  install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
  install(TARGETS ik_benchmark DESTINATION lib/${PROJECT_NAME})
  ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
  ament_export_dependencies(Eigen3)

  if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_relaxed_ik test/test_relaxed_ik.cpp)
    target_link_libraries(test_relaxed_ik ${PROJECT_NAME})
  endif()
  ament_package()
else()
  enable_testing()
  find_package(GTest REQUIRED)
  include(GoogleTest)
  add_executable(test_relaxed_ik test/test_relaxed_ik.cpp)
  target_link_libraries(test_relaxed_ik ${PROJECT_NAME} GTest::gtest_main)
  gtest_discover_tests(test_relaxed_ik)
endif()
