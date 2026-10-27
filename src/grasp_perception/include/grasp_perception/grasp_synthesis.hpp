#pragma once

#include <vector>

#include "grasp_perception/cloud_processing.hpp"
#include "grasp_perception/contour.hpp"
#include "grasp_perception/efd.hpp"
#include "grasp_perception/types.hpp"

namespace grasp_perception
{

struct GraspSynthesisParams
{
  GripperGeometry gripper;
  ScoreWeights weights;
  ContourParams contour;

  int efd_harmonics = 20;         // more = follows corners closer, but keeps more noise
  int contour_samples = 120;      // candidate contact points along the EFD curve
  double friction_coefficient = 0.4;  // friction cone half-angle = atan(mu)
  double min_width = 0.005;
  double flatness_length = 0.01;  // [m] curvature scale for the flatness score
  int max_pairs_per_object = 200; // best pairs (by friction angle) passed to 3D checks

  double max_grasp_depth = 0.04;  // fingertips at most this far below the object top
  double min_contact_depth = 0.006;
  double depth_step = 0.005;      // back-off step when the deepest grasp collides
  double pregrasp_distance = 0.10;  // approach length that must be collision free
  double table_clearance = 0.004;
  int min_support_points = 8;
  double support_saturation = 40.0;   // support count that scores 1.0
  int max_intruding_points = 3;       // non-object points tolerated between fingers

  double nms_distance = 0.01;
  double nms_angle = 0.3;
  int max_candidates = 20;
  /// Grasps are flipped 180 deg about the approach axis so their x axis agrees with
  /// this hint. The Franka Hand is symmetric, and this keeps joint 7 away from its limits.
  Eigen::Vector3d x_axis_hint{1.0, 0.0, 0.0};  // TCP x axis in the Panda ready pose
};

struct ObjectGrasps
{
  ObjectContour contour;
  std::vector<Eigen::Vector2d> efd_curve;  // dense samples of the fitted curve, for display
  std::vector<GraspCandidate> grasps;
};

/// Grasps for all segmented objects, best first. If per_object is given it receives the
/// contour and fitted curve of each object (for visualisation).
std::vector<GraspCandidate> synthesizeGrasps(
  const Segmentation & scene, const GraspSynthesisParams & params,
  std::vector<ObjectGrasps> * per_object = nullptr);

/// Contact pairs on the EFD curve that satisfy the friction-cone (antipodal) condition
/// and whose connecting line stays inside the outline. Returned as top-down grasps,
/// not yet collision checked.
std::vector<GraspCandidate> contourGrasps(
  int object_id, const ObjectContour & contour, const EllipticFourier & efd, const GraspSynthesisParams & params);

/// Check the gripper and its approach against the scene and table, then fill in the
/// remaining scores. Returns false if the grasp is infeasible.
bool evaluateGrasp(
  GraspCandidate & grasp, const SceneObject & object, const Segmentation & scene,
  const GraspSynthesisParams & params);

/// Right-handed TCP frame from a centre, closing axis (y) and approach axis (z).
Eigen::Isometry3d makeGraspFrame(const Eigen::Vector3d & center, const Eigen::Vector3d & closing, const Eigen::Vector3d & approach);

/// Non-maximum suppression; the gripper's 180 deg symmetry about z is taken into account.
std::vector<GraspCandidate> suppressDuplicates(std::vector<GraspCandidate> grasps, double distance, double angle);

}  // namespace grasp_perception
