#include <gtest/gtest.h>

#include <cmath>

#include "grasp_perception/efd.hpp"

using grasp_perception::EllipticFourier;
using grasp_perception::insidePolygon;
using grasp_perception::signedArea;

namespace
{

std::vector<Eigen::Vector2d> circle(double r, int n, bool ccw = true)
{
  std::vector<Eigen::Vector2d> pts;
  for (int i = 0; i < n; ++i) {
    const double a = (ccw ? 1.0 : -1.0) * 2.0 * M_PI * i / n;
    pts.emplace_back(0.1 + r * std::cos(a), -0.2 + r * std::sin(a));
  }
  return pts;
}

std::vector<Eigen::Vector2d> rectangle(double w, double h, double step)
{
  std::vector<Eigen::Vector2d> pts;
  for (double x = -w / 2; x < w / 2; x += step) {
    pts.emplace_back(x, -h / 2);
  }
  for (double y = -h / 2; y < h / 2; y += step) {
    pts.emplace_back(w / 2, y);
  }
  for (double x = w / 2; x > -w / 2; x -= step) {
    pts.emplace_back(x, h / 2);
  }
  for (double y = h / 2; y > -h / 2; y -= step) {
    pts.emplace_back(-w / 2, y);
  }
  return pts;
}

}  // namespace

TEST(Efd, CircleHasConstantRadiusAndCurvature)
{
  for (bool ccw : {true, false}) {
    const EllipticFourier efd(circle(0.03, 200, ccw), 10);
    EXPECT_EQ(efd.counterClockwise(), ccw);
    for (double t : efd.sampleParameters(50)) {
      const Eigen::Vector2d p = efd.point(t);
      const Eigen::Vector2d radial = (p - Eigen::Vector2d(0.1, -0.2));
      EXPECT_NEAR(radial.norm(), 0.03, 1e-4);
      EXPECT_NEAR(efd.curvature(t), 1.0 / 0.03, 0.5);
      EXPECT_GT(efd.outwardNormal(t).dot(radial.normalized()), 0.999);
    }
  }
}

TEST(Efd, RectangleIsReconstructedAndCornersAreSmoothed)
{
  const auto rect = rectangle(0.06, 0.04, 0.001);
  const EllipticFourier efd(rect, 30);
  double max_err = 0.0;
  for (double t : efd.sampleParameters(400)) {
    const Eigen::Vector2d p = efd.point(t);
    // Distance to the rectangle boundary.
    const double dx = std::abs(p.x()) - 0.03;
    const double dy = std::abs(p.y()) - 0.02;
    const double d = (dx > 0 || dy > 0) ? std::hypot(std::max(dx, 0.0), std::max(dy, 0.0)) : std::max(dx, dy);
    max_err = std::max(max_err, std::abs(d));
  }
  EXPECT_LT(max_err, 0.002);

  // Mid-side normals point straight out of the long faces.
  bool found = false;
  for (double t : efd.sampleParameters(400)) {
    const Eigen::Vector2d p = efd.point(t);
    if (std::abs(p.x()) < 0.002 && p.y() > 0) {
      EXPECT_GT(efd.outwardNormal(t).y(), 0.99);
      EXPECT_LT(std::abs(efd.curvature(t)), 5.0);
      found = true;
    }
  }
  EXPECT_TRUE(found);
}

TEST(Efd, PolygonHelpers)
{
  const auto c = circle(0.05, 100);
  EXPECT_NEAR(signedArea(c), M_PI * 0.05 * 0.05, 1e-4);
  EXPECT_TRUE(insidePolygon(c, Eigen::Vector2d(0.1, -0.2)));
  EXPECT_FALSE(insidePolygon(c, Eigen::Vector2d(0.2, -0.2)));
}
