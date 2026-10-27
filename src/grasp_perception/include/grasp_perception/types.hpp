#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Geometry>

namespace grasp_perception
{

using PointT = pcl::PointXYZRGB;
using Cloud = pcl::PointCloud<PointT>;
using NormalCloud = pcl::PointCloud<pcl::PointNormal>;

/// Parallel-jaw gripper geometry in the TCP frame (Franka Hand convention):
/// z = approach direction, y = closing direction, origin between the fingertips.
/// The fingers occupy z in [-finger_length, 0]; the palm sits behind them.
struct GripperGeometry
{
  double max_width = 0.08;
  double finger_length = 0.045;    // along z
  double finger_width = 0.02;      // along x
  double finger_thickness = 0.012; // along y, per finger
  double palm_depth = 0.06;        // along z, behind the fingers
  double palm_width = 0.20;        // along y
  double palm_thickness = 0.06;    // along x
  double open_clearance = 0.01;    // extra opening per side during the approach
};

struct ScoreWeights
{
  double antipodal = 1.0;    // contact normals vs. the contact line (friction cone)
  double support = 0.5;      // object points between the fingers (contact area)
  double centroid = 0.75;    // grasp line close to the object centre (less torque)
  double flatness = 0.5;     // low outline curvature at the contacts
  double width_margin = 0.25;
  double depth = 0.5;        // fingers reach deep below the top (shallow pinches slip)
};

struct GraspCandidate
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();  // TCP pose in the cloud frame
  double width = 0.0;        // object width between the contacts
  double score = 0.0;
  int object_id = -1;
  Eigen::Vector3d contact_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d contact_b = Eigen::Vector3d::Zero();
  double alpha_a = 0.0;      // [rad] angle between inward normal and contact line
  double alpha_b = 0.0;
  // Sub-scores in [0, 1], kept for logging and benchmarking.
  double antipodal_quality = 0.0;
  double support = 0.0;
  double centroid = 0.0;
  double flatness = 0.0;
  double width_margin = 0.0;
  double depth = 0.0;
};

}  // namespace grasp_perception
