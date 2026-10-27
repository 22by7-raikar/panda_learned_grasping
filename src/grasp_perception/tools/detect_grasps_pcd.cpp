// Offline grasp detection on a saved cloud (PCD, robot base frame):
//
//   detect_grasps_pcd scene.pcd [viewpoint_x viewpoint_y viewpoint_z]
//
// Prints segmentation results and the ranked grasps. Handy for tuning parameters on
// recorded data (ros2 run pcl_ros pointcloud_to_pcd) without running the robot.

#include <pcl/io/pcd_io.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "grasp_perception/cloud_processing.hpp"
#include "grasp_perception/grasp_synthesis.hpp"

using namespace grasp_perception;

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s cloud.pcd [vx vy vz]\n", argv[0]);
    return 2;
  }
  auto cloud = std::make_shared<Cloud>();
  if (pcl::io::loadPCDFile(argv[1], *cloud) != 0) {
    std::fprintf(stderr, "could not read %s\n", argv[1]);
    return 1;
  }
  Eigen::Vector3f viewpoint(0.5f, 0.0f, 0.6f);
  if (argc >= 5) {
    viewpoint = Eigen::Vector3f(std::atof(argv[2]), std::atof(argv[3]), std::atof(argv[4]));
  }

  const auto t0 = std::chrono::steady_clock::now();
  const auto scene = segmentTabletop(cloud, SegmentationParams{}, viewpoint);
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<ObjectGrasps> per_object;
  const auto grasps = synthesizeGrasps(scene, GraspSynthesisParams{}, &per_object);
  const auto t2 = std::chrono::steady_clock::now();

  const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  std::printf("%zu points -> %zu filtered, plane %s (%.3f %.3f %.3f %.3f)\n", cloud->size(),
    scene.filtered->size(), scene.plane_found ? "found" : "NOT found", scene.plane[0], scene.plane[1],
    scene.plane[2], scene.plane[3]);
  for (size_t i = 0; i < scene.objects.size(); ++i) {
    const auto & o = scene.objects[i];
    const size_t n_grasps = i < per_object.size() ? per_object[i].grasps.size() : 0;
    std::printf("object %zu: %zu pts, centroid (%.3f %.3f %.3f), height %.3f, %zu feasible grasps\n", i,
      o.points->size(), o.centroid.x(), o.centroid.y(), o.centroid.z(), o.max_pt.z(), n_grasps);
  }
  std::printf("segmentation %.1f ms, grasp synthesis %.1f ms\n", ms(t0, t1), ms(t1, t2));
  std::printf("rank obj  score  width   x      y      z     yaw(deg) alpha_a alpha_b\n");
  for (size_t i = 0; i < grasps.size(); ++i) {
    const auto & g = grasps[i];
    const Eigen::Vector3d y = g.pose.linear().col(1);
    std::printf("%4zu %3d  %.3f  %.3f  %.3f  %.3f  %.3f  %7.1f  %6.1f  %6.1f\n", i, g.object_id, g.score, g.width,
      g.pose.translation().x(), g.pose.translation().y(), g.pose.translation().z(),
      std::atan2(y.y(), y.x()) * 180.0 / M_PI, g.alpha_a * 180.0 / M_PI, g.alpha_b * 180.0 / M_PI);
  }
  return grasps.empty() ? 3 : 0;
}
