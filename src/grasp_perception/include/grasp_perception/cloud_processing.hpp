#pragma once

#include <pcl/ModelCoefficients.h>

#include <Eigen/Core>
#include <vector>

#include "grasp_perception/types.hpp"

namespace grasp_perception
{

struct SegmentationParams
{
  // Workspace crop in the robot base frame [m].
  Eigen::Vector3f crop_min{0.15f, -0.4f, -0.05f};
  Eigen::Vector3f crop_max{0.85f, 0.4f, 0.4f};
  float voxel_leaf = 0.003f;
  int outlier_mean_k = 30;
  double outlier_stddev = 2.0;  // lower values also erode object edges
  // Table plane: RANSAC with the normal constrained to the base z axis.
  double plane_distance = 0.008;
  double plane_max_angle = 0.15;  // [rad] between plane normal and +z
  int plane_max_iterations = 500;
  double min_height_above_plane = 0.008;
  // Euclidean clustering of what is left.
  double cluster_tolerance = 0.015;
  int min_cluster_size = 80;
  int max_cluster_size = 50000;
  double normal_radius = 0.012;
};

struct SceneObject
{
  NormalCloud::Ptr points;   // with normals, oriented toward the viewpoint
  Eigen::Vector3d centroid;
  Eigen::Vector3d min_pt;
  Eigen::Vector3d max_pt;
};

struct Segmentation
{
  Cloud::Ptr filtered;           // cropped, downsampled, outlier-free scene
  Cloud::Ptr above_plane;        // everything on top of the table (collision checks)
  Eigen::Vector4d plane{0, 0, 1, 0};  // ax + by + cz + d = 0, normal pointing up
  bool plane_found = false;
  std::vector<SceneObject> objects;
};

/// Tabletop segmentation: crop -> voxel grid -> statistical outlier removal ->
/// RANSAC table plane -> keep points above the plane -> Euclidean clusters -> normals.
///
/// The input must already be in the robot base frame (z up). viewpoint is the camera
/// origin in that frame and is used to orient the normals outward.
Segmentation segmentTabletop(
  const Cloud::ConstPtr & cloud, const SegmentationParams & params,
  const Eigen::Vector3f & viewpoint);

/// Signed distance from p to the plane (positive above).
inline double planeDistance(const Eigen::Vector4d & plane, const Eigen::Vector3d & p)
{
  return plane.head<3>().dot(p) + plane[3];
}

}  // namespace grasp_perception
