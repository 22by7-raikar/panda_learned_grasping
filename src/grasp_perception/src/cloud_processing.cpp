#include "grasp_perception/cloud_processing.hpp"

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <cmath>

namespace grasp_perception
{
namespace
{

Cloud::Ptr cropAndDownsample(const Cloud::ConstPtr & in, const SegmentationParams & p)
{
  auto cropped = std::make_shared<Cloud>();
  pcl::CropBox<PointT> crop;
  crop.setInputCloud(in);
  crop.setMin(Eigen::Vector4f(p.crop_min.x(), p.crop_min.y(), p.crop_min.z(), 1.0f));
  crop.setMax(Eigen::Vector4f(p.crop_max.x(), p.crop_max.y(), p.crop_max.z(), 1.0f));
  crop.filter(*cropped);

  auto down = std::make_shared<Cloud>();
  pcl::VoxelGrid<PointT> voxel;
  voxel.setInputCloud(cropped);
  voxel.setLeafSize(p.voxel_leaf, p.voxel_leaf, p.voxel_leaf);
  voxel.filter(*down);

  if (static_cast<int>(down->size()) <= p.outlier_mean_k) {
    return down;
  }
  auto clean = std::make_shared<Cloud>();
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(down);
  sor.setMeanK(p.outlier_mean_k);
  sor.setStddevMulThresh(p.outlier_stddev);
  sor.filter(*clean);
  return clean;
}

bool fitTablePlane(const Cloud::ConstPtr & cloud, const SegmentationParams & p, Eigen::Vector4d & plane)
{
  if (cloud->size() < 3) {
    return false;
  }
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients(true);
  // Constraining the normal to +z stops RANSAC from picking a large box face.
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setAxis(Eigen::Vector3f::UnitZ());
  seg.setEpsAngle(p.plane_max_angle);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(p.plane_distance);
  seg.setMaxIterations(p.plane_max_iterations);
  seg.setInputCloud(cloud);

  pcl::PointIndices inliers;
  pcl::ModelCoefficients coeffs;
  seg.segment(inliers, coeffs);
  if (inliers.indices.empty() || coeffs.values.size() != 4) {
    return false;
  }
  plane = Eigen::Vector4d(coeffs.values[0], coeffs.values[1], coeffs.values[2], coeffs.values[3]);
  plane /= plane.head<3>().norm();
  if (plane[2] < 0.0) {
    plane = -plane;
  }
  return true;
}

NormalCloud::Ptr estimateNormals(const Cloud::ConstPtr & points, double radius, const Eigen::Vector3f & viewpoint)
{
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  ne.setInputCloud(points);
  ne.setSearchMethod(std::make_shared<pcl::search::KdTree<PointT>>());
  ne.setRadiusSearch(radius);
  ne.setViewPoint(viewpoint.x(), viewpoint.y(), viewpoint.z());
  pcl::PointCloud<pcl::Normal> normals;
  ne.compute(normals);

  auto out = std::make_shared<NormalCloud>();
  out->reserve(points->size());
  for (size_t i = 0; i < points->size(); ++i) {
    const auto & n = normals[i];
    if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z)) {
      continue;
    }
    pcl::PointNormal pn;
    pn.x = (*points)[i].x;
    pn.y = (*points)[i].y;
    pn.z = (*points)[i].z;
    pn.normal_x = n.normal_x;
    pn.normal_y = n.normal_y;
    pn.normal_z = n.normal_z;
    pn.curvature = n.curvature;
    out->push_back(pn);
  }
  return out;
}

}  // namespace

Segmentation segmentTabletop(
  const Cloud::ConstPtr & cloud, const SegmentationParams & params, const Eigen::Vector3f & viewpoint)
{
  Segmentation result;
  result.filtered = cropAndDownsample(cloud, params);
  result.above_plane = std::make_shared<Cloud>();
  result.plane_found = fitTablePlane(result.filtered, params, result.plane);

  for (const auto & pt : *result.filtered) {
    const Eigen::Vector3d p(pt.x, pt.y, pt.z);
    if (!result.plane_found || planeDistance(result.plane, p) > params.min_height_above_plane) {
      result.above_plane->push_back(pt);
    }
  }
  if (result.above_plane->empty()) {
    return result;
  }

  auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
  tree->setInputCloud(result.above_plane);
  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(params.cluster_tolerance);
  ec.setMinClusterSize(params.min_cluster_size);
  ec.setMaxClusterSize(params.max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(result.above_plane);
  ec.extract(clusters);

  for (const auto & indices : clusters) {
    auto pts = std::make_shared<Cloud>();
    pcl::copyPointCloud(*result.above_plane, indices, *pts);

    SceneObject obj;
    obj.points = estimateNormals(pts, params.normal_radius, viewpoint);
    if (obj.points->empty()) {
      continue;
    }
    Eigen::Vector4f c;
    pcl::compute3DCentroid(*pts, c);
    PointT min_pt;
    PointT max_pt;
    pcl::getMinMax3D(*pts, min_pt, max_pt);
    obj.centroid = c.head<3>().cast<double>();
    obj.min_pt = min_pt.getVector3fMap().cast<double>();
    obj.max_pt = max_pt.getVector3fMap().cast<double>();
    result.objects.push_back(obj);
  }
  return result;
}

}  // namespace grasp_perception
