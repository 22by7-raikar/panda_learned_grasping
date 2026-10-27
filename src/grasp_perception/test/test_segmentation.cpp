#include <gtest/gtest.h>

#include "grasp_perception/cloud_processing.hpp"
#include "synthetic_scene.hpp"

using grasp_perception::Cloud;
using grasp_perception::segmentTabletop;
using grasp_perception::SegmentationParams;

namespace
{
const Eigen::Vector3f kViewpoint(0.5f, 0.0f, 0.6f);
}

TEST(Segmentation, FindsTableAndSeparatesObjects)
{
  std::mt19937 rng(0);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  synthetic::addBox(*cloud, rng, 0.45, -0.1, 0.04, 0.06, 0.10);
  synthetic::addCylinder(*cloud, rng, 0.55, 0.1, 0.025, 0.08);

  const auto seg = segmentTabletop(cloud, SegmentationParams{}, kViewpoint);
  ASSERT_TRUE(seg.plane_found);
  EXPECT_NEAR(seg.plane[2], 1.0, 1e-3);
  EXPECT_NEAR(seg.plane[3], 0.0, 2e-3);
  ASSERT_EQ(seg.objects.size(), 2u);

  bool saw_box = false;
  bool saw_cylinder = false;
  for (const auto & obj : seg.objects) {
    if ((obj.centroid.head<2>() - Eigen::Vector2d(0.45, -0.1)).norm() < 0.01) {
      saw_box = true;
      EXPECT_NEAR(obj.max_pt.z(), 0.10, 0.005);
    }
    if ((obj.centroid.head<2>() - Eigen::Vector2d(0.55, 0.1)).norm() < 0.01) {
      saw_cylinder = true;
    }
    // Normals are oriented toward the camera, so top-face normals point up (edges
    // excluded: their normals blend the top and the side).
    int up = 0;
    int top = 0;
    for (const auto & p : *obj.points) {
      const bool interior = p.x > obj.min_pt.x() + 0.01 && p.x < obj.max_pt.x() - 0.01 &&
        p.y > obj.min_pt.y() + 0.01 && p.y < obj.max_pt.y() - 0.01;
      if (interior && p.z > obj.max_pt.z() - 0.002) {
        ++top;
        up += p.normal_z > 0.9f ? 1 : 0;
      }
    }
    EXPECT_GT(top, 10);
    EXPECT_GT(up, 0.95 * top);
  }
  EXPECT_TRUE(saw_box);
  EXPECT_TRUE(saw_cylinder);
}

TEST(Segmentation, CropRemovesOutOfWorkspacePoints)
{
  std::mt19937 rng(1);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  synthetic::addBox(*cloud, rng, 0.95, 0.0, 0.04, 0.04, 0.05);  // beyond crop_max.x
  const auto seg = segmentTabletop(cloud, SegmentationParams{}, kViewpoint);
  EXPECT_TRUE(seg.objects.empty());
}

TEST(Segmentation, EmptyCloudIsHandled)
{
  auto cloud = std::make_shared<Cloud>();
  const auto seg = segmentTabletop(cloud, SegmentationParams{}, kViewpoint);
  EXPECT_FALSE(seg.plane_found);
  EXPECT_TRUE(seg.objects.empty());
}
