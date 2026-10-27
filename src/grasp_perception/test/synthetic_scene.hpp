#pragma once

#include <cmath>
#include <random>

#include "grasp_perception/types.hpp"

// Synthetic tabletop clouds for tests: surfaces sampled on a regular grid with a little
// Gaussian noise, all in the robot base frame with the table at z = 0.
namespace synthetic
{

using grasp_perception::Cloud;
using grasp_perception::PointT;

inline PointT makePoint(double x, double y, double z, std::mt19937 & rng, double noise)
{
  std::normal_distribution<double> n(0.0, noise);
  PointT p;
  p.x = static_cast<float>(x + n(rng));
  p.y = static_cast<float>(y + n(rng));
  p.z = static_cast<float>(z + n(rng));
  p.r = p.g = p.b = 200;
  return p;
}

inline void addTable(Cloud & cloud, std::mt19937 & rng, double step = 0.004, double noise = 0.001)
{
  for (double x = 0.25; x <= 0.75; x += step) {
    for (double y = -0.25; y <= 0.25; y += step) {
      cloud.push_back(makePoint(x, y, 0.0, rng, noise));
    }
  }
}

/// Axis-aligned box resting on the table (top and four sides; the camera never sees the
/// bottom), rotated by yaw about its centre.
inline void addBox(
  Cloud & cloud, std::mt19937 & rng, double cx, double cy, double sx, double sy, double sz,
  double yaw = 0.0, double step = 0.003, double noise = 0.0007)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  auto put = [&](double lx, double ly, double z) {
      cloud.push_back(makePoint(cx + c * lx - s * ly, cy + s * lx + c * ly, z, rng, noise));
    };
  for (double lx = -sx / 2; lx <= sx / 2 + 1e-9; lx += step) {
    for (double ly = -sy / 2; ly <= sy / 2 + 1e-9; ly += step) {
      put(lx, ly, sz);
    }
  }
  for (double z = step; z < sz; z += step) {
    for (double lx = -sx / 2; lx <= sx / 2 + 1e-9; lx += step) {
      put(lx, -sy / 2, z);
      put(lx, sy / 2, z);
    }
    for (double ly = -sy / 2; ly <= sy / 2 + 1e-9; ly += step) {
      put(-sx / 2, ly, z);
      put(sx / 2, ly, z);
    }
  }
}

inline void addCylinder(
  Cloud & cloud, std::mt19937 & rng, double cx, double cy, double radius, double height,
  double step = 0.003, double noise = 0.0007)
{
  for (double r = 0.0; r <= radius; r += step) {
    const int n = std::max(1, static_cast<int>(2 * M_PI * r / step));
    for (int k = 0; k < n; ++k) {
      const double a = 2 * M_PI * k / n;
      cloud.push_back(makePoint(cx + r * std::cos(a), cy + r * std::sin(a), height, rng, noise));
    }
  }
  const int n = static_cast<int>(2 * M_PI * radius / step);
  for (double z = step; z < height; z += step) {
    for (int k = 0; k < n; ++k) {
      const double a = 2 * M_PI * k / n;
      cloud.push_back(makePoint(cx + radius * std::cos(a), cy + radius * std::sin(a), z, rng, noise));
    }
  }
}

/// A "V" like an open spring clamp seen from above: two 2.5 cm wide arms joined at the
/// hinge, spreading apart. Grasps must not pinch across the gap between the arms.
inline void addVShape(Cloud & cloud, std::mt19937 & rng, double cx, double cy, double height = 0.025)
{
  const double arm_length = 0.14;
  const double arm_width = 0.025;
  for (double side : {-1.0, 1.0}) {
    const double yaw = side * 0.35;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    // Arm starts at the hinge (cx, cy) and extends along +x rotated by yaw.
    addBox(cloud, rng, cx + c * arm_length / 2, cy + s * arm_length / 2, arm_length, arm_width, height, yaw);
  }
}

}  // namespace synthetic
