#include "grasp_perception/efd.hpp"

#include <cmath>
#include <stdexcept>

namespace grasp_perception
{

double signedArea(const std::vector<Eigen::Vector2d> & polygon)
{
  double area = 0.0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const auto & p = polygon[i];
    const auto & q = polygon[(i + 1) % polygon.size()];
    area += p.x() * q.y() - q.x() * p.y();
  }
  return 0.5 * area;
}

bool insidePolygon(const std::vector<Eigen::Vector2d> & polygon, const Eigen::Vector2d & p)
{
  bool inside = false;
  for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
    const auto & a = polygon[i];
    const auto & b = polygon[j];
    if ((a.y() > p.y()) != (b.y() > p.y()) &&
      p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x())
    {
      inside = !inside;
    }
  }
  return inside;
}

EllipticFourier::EllipticFourier(const std::vector<Eigen::Vector2d> & contour, int harmonics)
{
  if (contour.size() < 3 || harmonics < 1) {
    throw std::invalid_argument("EllipticFourier needs >= 3 points and >= 1 harmonic");
  }
  ccw_ = signedArea(contour) > 0.0;

  // Segments of the closed polygon, skipping repeated points.
  std::vector<Eigen::Vector2d> delta;
  std::vector<double> dt;
  std::vector<Eigen::Vector2d> start;
  for (size_t i = 0; i < contour.size(); ++i) {
    const Eigen::Vector2d d = contour[(i + 1) % contour.size()] - contour[i];
    const double len = d.norm();
    if (len > 1e-12) {
      delta.push_back(d);
      dt.push_back(len);
      start.push_back(contour[i]);
    }
  }
  double total = 0.0;
  std::vector<double> t(dt.size() + 1, 0.0);
  for (size_t p = 0; p < dt.size(); ++p) {
    total += dt[p];
    t[p + 1] = total;
  }
  period_ = total;

  // DC terms: mean of the piecewise-linear curve over arc length (exact for a polygon).
  for (size_t p = 0; p < dt.size(); ++p) {
    const Eigen::Vector2d mid = start[p] + 0.5 * delta[p];
    a0_ += mid.x() * dt[p];
    c0_ += mid.y() * dt[p];
  }
  a0_ /= period_;
  c0_ /= period_;

  a_.assign(harmonics, 0.0);
  b_.assign(harmonics, 0.0);
  c_.assign(harmonics, 0.0);
  d_.assign(harmonics, 0.0);
  for (int n = 1; n <= harmonics; ++n) {
    const double w = 2.0 * M_PI * n / period_;
    const double k = period_ / (2.0 * n * n * M_PI * M_PI);
    for (size_t p = 0; p < dt.size(); ++p) {
      const double dcos = std::cos(w * t[p + 1]) - std::cos(w * t[p]);
      const double dsin = std::sin(w * t[p + 1]) - std::sin(w * t[p]);
      const double rx = delta[p].x() / dt[p];
      const double ry = delta[p].y() / dt[p];
      a_[n - 1] += k * rx * dcos;
      b_[n - 1] += k * rx * dsin;
      c_[n - 1] += k * ry * dcos;
      d_[n - 1] += k * ry * dsin;
    }
  }
}

Eigen::Vector2d EllipticFourier::point(double t) const
{
  Eigen::Vector2d p(a0_, c0_);
  for (size_t i = 0; i < a_.size(); ++i) {
    const double w = 2.0 * M_PI * (i + 1) / period_;
    const double c = std::cos(w * t);
    const double s = std::sin(w * t);
    p.x() += a_[i] * c + b_[i] * s;
    p.y() += c_[i] * c + d_[i] * s;
  }
  return p;
}

Eigen::Vector2d EllipticFourier::derivative(double t) const
{
  Eigen::Vector2d d = Eigen::Vector2d::Zero();
  for (size_t i = 0; i < a_.size(); ++i) {
    const double w = 2.0 * M_PI * (i + 1) / period_;
    const double c = std::cos(w * t);
    const double s = std::sin(w * t);
    d.x() += w * (-a_[i] * s + b_[i] * c);
    d.y() += w * (-c_[i] * s + d_[i] * c);
  }
  return d;
}

Eigen::Vector2d EllipticFourier::secondDerivative(double t) const
{
  Eigen::Vector2d d = Eigen::Vector2d::Zero();
  for (size_t i = 0; i < a_.size(); ++i) {
    const double w = 2.0 * M_PI * (i + 1) / period_;
    const double c = std::cos(w * t);
    const double s = std::sin(w * t);
    d.x() -= w * w * (a_[i] * c + b_[i] * s);
    d.y() -= w * w * (c_[i] * c + d_[i] * s);
  }
  return d;
}

Eigen::Vector2d EllipticFourier::outwardNormal(double t) const
{
  const Eigen::Vector2d d = derivative(t).normalized();
  // For a counter-clockwise curve the outside is to the right of the tangent.
  return ccw_ ? Eigen::Vector2d(d.y(), -d.x()) : Eigen::Vector2d(-d.y(), d.x());
}

double EllipticFourier::curvature(double t) const
{
  const Eigen::Vector2d d1 = derivative(t);
  const Eigen::Vector2d d2 = secondDerivative(t);
  const double k = (d1.x() * d2.y() - d1.y() * d2.x()) / std::pow(d1.squaredNorm(), 1.5);
  return ccw_ ? k : -k;
}

std::vector<double> EllipticFourier::sampleParameters(int n) const
{
  std::vector<double> ts(n);
  for (int i = 0; i < n; ++i) {
    ts[i] = period_ * i / n;
  }
  return ts;
}

}  // namespace grasp_perception
