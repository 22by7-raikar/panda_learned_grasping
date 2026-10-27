#include "grasp_perception/contour.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include "grasp_perception/efd.hpp"

namespace grasp_perception
{

std::optional<ObjectContour> extractContour(
  const SceneObject & object, const Eigen::Vector4d & plane, const ContourParams & params)
{
  if (!object.points || object.points->size() < 3) {
    return std::nullopt;
  }
  ObjectContour out;
  out.up = plane.head<3>().normalized();
  out.origin = -plane[3] * out.up;
  out.u = out.up.unitOrthogonal();
  out.v = out.up.cross(out.u);

  out.top_height = -std::numeric_limits<double>::max();
  for (const auto & pt : *object.points) {
    out.top_height = std::max(out.top_height, planeDistance(plane, pt.getVector3fMap().cast<double>()));
  }
  const double min_height = params.top_layer_thickness > 0.0 ?
    out.top_height - params.top_layer_thickness : -std::numeric_limits<double>::max();

  std::vector<Eigen::Vector2d> uv;
  uv.reserve(object.points->size());
  Eigen::Vector2d lo = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d hi = -lo;
  for (const auto & pt : *object.points) {
    const Eigen::Vector3d p = pt.getVector3fMap().cast<double>();
    if (planeDistance(plane, p) < min_height) {
      continue;
    }
    uv.push_back(out.to2d(p));
    lo = lo.cwiseMin(uv.back());
    hi = hi.cwiseMax(uv.back());
  }
  if (uv.size() < 3) {
    return std::nullopt;
  }

  const double res = params.resolution;
  const int pad = params.closing_radius + 2;
  const int cols = static_cast<int>(std::ceil((hi.x() - lo.x()) / res)) + 1 + 2 * pad;
  const int rows = static_cast<int>(std::ceil((hi.y() - lo.y()) / res)) + 1 + 2 * pad;
  cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);
  for (const auto & p : uv) {
    const int c = static_cast<int>(std::lround((p.x() - lo.x()) / res)) + pad;
    const int r = static_cast<int>(std::lround((p.y() - lo.y()) / res)) + pad;
    mask.at<uint8_t>(r, c) = 255;
  }
  if (params.closing_radius > 0) {
    const int k = 2 * params.closing_radius + 1;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k}));
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  if (contours.empty()) {
    return std::nullopt;
  }
  const auto largest = std::max_element(contours.begin(), contours.end(), [](const auto & a, const auto & b) {
      return cv::contourArea(a) < cv::contourArea(b);
    });
  if (largest->size() < 3) {
    return std::nullopt;
  }

  for (const auto & px : *largest) {
    out.points.emplace_back(lo.x() + (px.x - pad) * res, lo.y() + (px.y - pad) * res);
  }
  const double area = signedArea(out.points);
  if (std::abs(area) < params.min_area) {
    return std::nullopt;
  }
  if (area < 0.0) {
    std::reverse(out.points.begin(), out.points.end());
  }
  return out;
}

}  // namespace grasp_perception
