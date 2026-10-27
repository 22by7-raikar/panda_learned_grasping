#pragma once

#include <Eigen/Geometry>
#include <optional>
#include <vector>

#include "grasp_perception/cloud_processing.hpp"

namespace grasp_perception
{

struct ContourParams
{
  double resolution = 0.002;  // [m] raster cell size for the top-view silhouette
  int closing_radius = 2;     // [cells] fills gaps between voxel-grid points
  double min_area = 2e-4;     // [m^2] ignore silhouettes smaller than this
  /// Only points within this distance of the object's top form the outline (the "top
  /// surface"); 0 uses the whole object. Lower parts, e.g. a mug's handle base or the
  /// sides seen in perspective, would otherwise widen the outline the fingers close on.
  double top_layer_thickness = 0.02;
};

/// Outer outline of an object seen from above, in a 2D frame lying on the table plane.
struct ObjectContour
{
  std::vector<Eigen::Vector2d> points;  // ordered, closed, counter-clockwise
  Eigen::Vector3d origin;               // point on the table plane
  Eigen::Vector3d u;                    // in-plane axes; (u, v, up) is right-handed
  Eigen::Vector3d v;
  Eigen::Vector3d up;
  double top_height = 0.0;              // highest object point above the table

  Eigen::Vector2d to2d(const Eigen::Vector3d & p) const
  {
    const Eigen::Vector3d d = p - origin;
    return {d.dot(u), d.dot(v)};
  }
  Eigen::Vector3d to3d(const Eigen::Vector2d & p, double height) const
  {
    return origin + p.x() * u + p.y() * v + height * up;
  }
  Eigen::Vector3d direction3d(const Eigen::Vector2d & d) const { return d.x() * u + d.y() * v; }
};

/// Top-surface outline: keep the top layer of the object, project it onto the table,
/// rasterise, close small gaps (morphological closing acts like the concave hull's alpha:
/// gaps narrower than ~2 * closing_radius cells are bridged, concavities wider than that
/// are kept) and trace the outer boundary. Holes are ignored: a parallel gripper can only
/// touch the outside from above.
std::optional<ObjectContour> extractContour(
  const SceneObject & object, const Eigen::Vector4d & plane, const ContourParams & params);

}  // namespace grasp_perception
