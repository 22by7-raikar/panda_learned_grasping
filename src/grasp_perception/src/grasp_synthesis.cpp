#include "grasp_perception/grasp_synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace grasp_perception
{
namespace
{

constexpr double kWidthSafety = 0.004;  // never plan a grasp at exactly max opening
constexpr int kInsideChecks = 7;        // samples along the contact line

double clamp01(double v)
{
  return std::min(1.0, std::max(0.0, v));
}

double rotationAngle(const Eigen::Matrix3d & a, const Eigen::Matrix3d & b)
{
  return Eigen::AngleAxisd(a.transpose() * b).angle();
}

/// Lowest point of the opened fingers and palm, as signed height above the table.
double lowestGripperPoint(const Eigen::Isometry3d & pose, double half_open, const GripperGeometry & g, const Eigen::Vector4d & plane)
{
  double lowest = std::numeric_limits<double>::max();
  const double y_out = half_open + g.finger_thickness;
  for (double sx : {-0.5 * g.finger_width, 0.5 * g.finger_width}) {
    for (double sy : {-y_out, -half_open, half_open, y_out}) {
      for (double sz : {0.0, -g.finger_length}) {
        lowest = std::min(lowest, planeDistance(plane, pose * Eigen::Vector3d(sx, sy, sz)));
      }
    }
  }
  for (double sx : {-0.5 * g.palm_thickness, 0.5 * g.palm_thickness}) {
    for (double sy : {-0.5 * g.palm_width, 0.5 * g.palm_width}) {
      lowest = std::min(lowest, planeDistance(plane, pose * Eigen::Vector3d(sx, sy, -g.finger_length)));
    }
  }
  return lowest;
}

}  // namespace

Eigen::Isometry3d makeGraspFrame(const Eigen::Vector3d & center, const Eigen::Vector3d & closing, const Eigen::Vector3d & approach)
{
  const Eigen::Vector3d z = approach.normalized();
  const Eigen::Vector3d y = (closing - closing.dot(z) * z).normalized();
  const Eigen::Vector3d x = y.cross(z);
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.linear().col(0) = x;
  t.linear().col(1) = y;
  t.linear().col(2) = z;
  t.translation() = center;
  return t;
}

std::vector<GraspCandidate> contourGrasps(
  int object_id, const ObjectContour & contour, const EllipticFourier & efd, const GraspSynthesisParams & params)
{
  const int n = params.contour_samples;
  const double cone = std::atan(params.friction_coefficient);
  const double max_width = params.gripper.max_width - kWidthSafety;

  std::vector<Eigen::Vector2d> pts(n);
  std::vector<Eigen::Vector2d> normals(n);
  std::vector<double> curvature(n);
  const auto ts = efd.sampleParameters(n);
  for (int i = 0; i < n; ++i) {
    pts[i] = efd.point(ts[i]);
    normals[i] = efd.outwardNormal(ts[i]);
    curvature[i] = efd.curvature(ts[i]);
  }
  // Denser polygon of the smoothed outline for the inside test.
  std::vector<Eigen::Vector2d> outline;
  for (double t : efd.sampleParameters(4 * n)) {
    outline.push_back(efd.point(t));
  }

  // Fingertips as deep as the fingers allow, but above the table.
  const double tcp_height = std::max(params.table_clearance, contour.top_height - params.max_grasp_depth);
  const double contact_height = 0.5 * (tcp_height + contour.top_height);
  if (contour.top_height - tcp_height < params.min_contact_depth) {
    return {};  // too flat to pinch from above
  }

  std::vector<GraspCandidate> out;
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const Eigen::Vector2d w = pts[j] - pts[i];
      const double d = w.norm();
      if (d < params.min_width || d > max_width) {
        continue;
      }
      const Eigen::Vector2d wh = w / d;
      // Finger at i pushes along +w, finger at j along -w. Each push must lie inside
      // the friction cone around that contact's inward normal.
      const double alpha_i = std::acos(std::clamp(-normals[i].dot(wh), -1.0, 1.0));
      const double alpha_j = std::acos(std::clamp(normals[j].dot(wh), -1.0, 1.0));
      if (alpha_i > cone || alpha_j > cone) {
        continue;
      }
      // Reject pairs that pinch across a gap, e.g. the two arms of a clamp.
      bool inside = true;
      for (int k = 1; k <= kInsideChecks && inside; ++k) {
        inside = insidePolygon(outline, pts[i] + (static_cast<double>(k) / (kInsideChecks + 1)) * w);
      }
      if (!inside) {
        continue;
      }

      GraspCandidate g;
      const Eigen::Vector2d mid = 0.5 * (pts[i] + pts[j]);
      g.pose = makeGraspFrame(contour.to3d(mid, tcp_height), contour.direction3d(wh), -contour.up);
      g.width = d;
      g.object_id = object_id;
      g.contact_a = contour.to3d(pts[i], contact_height);
      g.contact_b = contour.to3d(pts[j], contact_height);
      g.alpha_a = alpha_i;
      g.alpha_b = alpha_j;
      g.antipodal_quality = 1.0 - std::max(alpha_i, alpha_j) / cone;
      const double k_max = std::max(std::abs(curvature[i]), std::abs(curvature[j]));
      g.flatness = 1.0 / (1.0 + params.flatness_length * k_max);
      out.push_back(g);
    }
  }

  std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) {
      return a.antipodal_quality > b.antipodal_quality;
    });
  if (static_cast<int>(out.size()) > params.max_pairs_per_object) {
    out.resize(params.max_pairs_per_object);
  }
  return out;
}

bool evaluateGrasp(
  GraspCandidate & grasp, const SceneObject & object, const Segmentation & scene, const GraspSynthesisParams & params)
{
  const auto & g = params.gripper;
  if (grasp.width < params.min_width || grasp.width > g.max_width - kWidthSafety) {
    return false;
  }
  const double half_open = 0.5 * std::min(g.max_width, grasp.width + 2.0 * g.open_clearance);
  const Eigen::Vector3d z = grasp.pose.linear().col(2);

  // Lift along the approach axis until the fingers clear the table.
  if (scene.plane_found) {
    const double lowest = lowestGripperPoint(grasp.pose, half_open, g, scene.plane);
    const double need = params.table_clearance - lowest;
    if (need > 0.0) {
      const double descent = -z.dot(scene.plane.head<3>());
      if (descent < 0.2) {
        return false;
      }
      grasp.pose.translation() -= (need / descent) * z;
    }
  }

  // Collision and support checks in the gripper frame. Fingers and palm are swept back
  // along the approach by pregrasp_distance, so the straight-line approach is covered.
  const double fx = 0.5 * g.finger_width;
  const double sweep = params.pregrasp_distance;
  auto between_fingers = [&](const Eigen::Vector3d & q) {
      return std::abs(q.x()) <= fx && std::abs(q.y()) < half_open && q.z() <= 0.0 && q.z() >= -g.finger_length;
    };

  // How deep the fingertips reach below the object's top, along the approach.
  double top_along_z = std::numeric_limits<double>::max();
  for (const auto & pt : *object.points) {
    top_along_z = std::min(top_along_z, z.dot(pt.getVector3fMap().cast<double>()));
  }

  // Try the deepest grasp first; if the hand hits something, back off in steps
  // (a shallower pinch can clear a taller neighbour under the palm).
  Eigen::Isometry3d inv;
  int support = 0;
  bool feasible = false;
  const Eigen::Isometry3d deepest = grasp.pose;
  for (double back = 0.0; !feasible; back += params.depth_step) {
    grasp.pose = deepest;
    grasp.pose.translation() -= back * z;
    if (z.dot(grasp.pose.translation()) - top_along_z < params.min_contact_depth) {
      return false;
    }
    inv = grasp.pose.inverse();
    bool collision = false;
    int region_scene = 0;
    for (const auto & pt : *scene.above_plane) {
      const Eigen::Vector3d q = inv * pt.getVector3fMap().cast<double>();
      const double ay = std::abs(q.y());
      const bool in_finger = std::abs(q.x()) <= fx && ay >= half_open && ay <= half_open + g.finger_thickness &&
        q.z() <= 0.0 && q.z() >= -g.finger_length - sweep;
      const bool in_palm = std::abs(q.x()) <= 0.5 * g.palm_thickness && ay <= 0.5 * g.palm_width &&
        q.z() < -g.finger_length && q.z() >= -g.finger_length - g.palm_depth - sweep;
      if (in_finger || in_palm) {
        collision = true;
        break;
      }
      region_scene += between_fingers(q) ? 1 : 0;
    }
    if (collision) {
      continue;
    }
    support = 0;
    for (const auto & pt : *object.points) {
      support += between_fingers(inv * pt.getVector3fMap().cast<double>()) ? 1 : 0;
    }
    if (support < params.min_support_points || region_scene - support > params.max_intruding_points) {
      return false;
    }
    feasible = true;
  }

  const Eigen::Vector3d c_local = inv * object.centroid;
  const double offset = std::hypot(c_local.x(), c_local.y());

  grasp.support = clamp01(support / params.support_saturation);
  grasp.centroid = std::exp(-(offset * offset) / (2.0 * 0.02 * 0.02));
  grasp.width_margin = clamp01((g.max_width - grasp.width) / g.max_width);
  grasp.depth = clamp01((z.dot(grasp.pose.translation()) - top_along_z) / params.max_grasp_depth);

  const auto & w = params.weights;
  const double total = w.antipodal + w.support + w.centroid + w.flatness + w.width_margin + w.depth;
  grasp.score = (w.antipodal * grasp.antipodal_quality + w.support * grasp.support +
    w.centroid * grasp.centroid + w.flatness * grasp.flatness + w.width_margin * grasp.width_margin +
    w.depth * grasp.depth) / total;
  return true;
}

std::vector<GraspCandidate> suppressDuplicates(std::vector<GraspCandidate> grasps, double distance, double angle)
{
  std::sort(grasps.begin(), grasps.end(), [](const auto & a, const auto & b) { return a.score > b.score; });
  const Eigen::Matrix3d flip = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  std::vector<GraspCandidate> kept;
  for (const auto & g : grasps) {
    bool duplicate = false;
    for (const auto & k : kept) {
      if (k.object_id != g.object_id || (k.pose.translation() - g.pose.translation()).norm() > distance) {
        continue;
      }
      const double rot = std::min(
        rotationAngle(k.pose.linear(), g.pose.linear()), rotationAngle(k.pose.linear(), g.pose.linear() * flip));
      if (rot < angle) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      kept.push_back(g);
    }
  }
  return kept;
}

std::vector<GraspCandidate> synthesizeGrasps(
  const Segmentation & scene, const GraspSynthesisParams & params, std::vector<ObjectGrasps> * per_object)
{
  std::vector<GraspCandidate> feasible;
  const Eigen::Vector4d plane = scene.plane_found ? scene.plane : Eigen::Vector4d(0, 0, 1, 0);
  for (size_t i = 0; i < scene.objects.size(); ++i) {
    const auto & obj = scene.objects[i];
    const auto contour = extractContour(obj, plane, params.contour);
    if (!contour) {
      continue;
    }
    const EllipticFourier efd(contour->points, params.efd_harmonics);

    ObjectGrasps record;
    for (auto & c : contourGrasps(static_cast<int>(i), *contour, efd, params)) {
      if (!evaluateGrasp(c, obj, scene, params)) {
        continue;
      }
      if (c.pose.linear().col(0).dot(params.x_axis_hint) < 0.0) {
        c.pose.linear().col(0) *= -1.0;
        c.pose.linear().col(1) *= -1.0;
        std::swap(c.contact_a, c.contact_b);
        std::swap(c.alpha_a, c.alpha_b);
      }
      feasible.push_back(c);
      record.grasps.push_back(c);
    }
    if (per_object) {
      record.contour = *contour;
      for (double t : efd.sampleParameters(4 * params.contour_samples)) {
        record.efd_curve.push_back(efd.point(t));
      }
      per_object->push_back(std::move(record));
    }
  }
  auto kept = suppressDuplicates(std::move(feasible), params.nms_distance, params.nms_angle);
  if (static_cast<int>(kept.size()) > params.max_candidates) {
    kept.resize(params.max_candidates);
  }
  return kept;
}

}  // namespace grasp_perception
