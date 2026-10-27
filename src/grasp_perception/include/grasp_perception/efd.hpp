#pragma once

#include <Eigen/Core>
#include <vector>

namespace grasp_perception
{

/// Elliptical Fourier Descriptors of a closed 2D contour (Kuhl & Giardina, 1982).
///
///   x(t) = A0 + sum_n a_n cos(2 pi n t / T) + b_n sin(2 pi n t / T)
///   y(t) = C0 + sum_n c_n cos(2 pi n t / T) + d_n sin(2 pi n t / T)
///
/// t is arc length along the input polygon and T its perimeter. Truncating to a few
/// harmonics low-pass filters the noisy, pixelated outline from the depth camera, and the
/// series gives tangents, normals and curvature analytically instead of by differencing.
class EllipticFourier
{
public:
  /// Fit to an ordered, closed contour (first point is not repeated at the end).
  EllipticFourier(const std::vector<Eigen::Vector2d> & contour, int harmonics);

  Eigen::Vector2d point(double t) const;
  Eigen::Vector2d derivative(double t) const;
  Eigen::Vector2d secondDerivative(double t) const;

  /// Unit normal pointing out of the shape.
  Eigen::Vector2d outwardNormal(double t) const;

  /// Signed curvature; positive where the outline is convex.
  double curvature(double t) const;

  /// n points equally spaced in the curve parameter.
  std::vector<double> sampleParameters(int n) const;

  double period() const { return period_; }
  int harmonics() const { return static_cast<int>(a_.size()); }
  bool counterClockwise() const { return ccw_; }

private:
  double period_ = 1.0;
  double a0_ = 0.0;
  double c0_ = 0.0;
  bool ccw_ = true;
  std::vector<double> a_, b_, c_, d_;
};

/// Signed area of a polygon (positive if counter-clockwise).
double signedArea(const std::vector<Eigen::Vector2d> & polygon);

/// Even-odd point-in-polygon test.
bool insidePolygon(const std::vector<Eigen::Vector2d> & polygon, const Eigen::Vector2d & p);

}  // namespace grasp_perception
