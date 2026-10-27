#include <gtest/gtest.h>

#include "grasp_perception/grasp_synthesis.hpp"
#include "synthetic_scene.hpp"

using grasp_perception::Cloud;
using grasp_perception::GraspCandidate;
using grasp_perception::GraspSynthesisParams;
using grasp_perception::ObjectGrasps;
using grasp_perception::segmentTabletop;
using grasp_perception::SegmentationParams;
using grasp_perception::synthesizeGrasps;

namespace
{

std::vector<GraspCandidate> run(const Cloud::Ptr & cloud, std::vector<ObjectGrasps> * per_object = nullptr)
{
  const auto seg = segmentTabletop(cloud, SegmentationParams{}, Eigen::Vector3f(0.5f, 0.0f, 0.6f));
  return synthesizeGrasps(seg, GraspSynthesisParams{}, per_object);
}

void expectWellFormed(const GraspCandidate & g)
{
  const Eigen::Matrix3d r = g.pose.linear();
  EXPECT_TRUE((r.transpose() * r).isIdentity(1e-9));
  EXPECT_NEAR(r.determinant(), 1.0, 1e-9);
  EXPECT_NEAR(r.col(2).z(), -1.0, 1e-3);  // top-down approach
  EXPECT_GE(g.score, 0.0);
  EXPECT_LE(g.score, 1.0);
}

}  // namespace

TEST(GraspSynthesis, BoxIsGraspedAcrossParallelFaces)
{
  std::mt19937 rng(2);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  const double yaw = 0.4;
  synthetic::addBox(*cloud, rng, 0.5, 0.0, 0.04, 0.10, 0.08, yaw);

  const auto grasps = run(cloud);
  ASSERT_FALSE(grasps.empty());
  const auto & best = grasps.front();
  expectWellFormed(best);
  // 10 cm side does not fit in the 8 cm hand, so it must close across the 4 cm side.
  EXPECT_NEAR(best.width, 0.04, 0.006);
  const Eigen::Vector3d box_x(std::cos(yaw), std::sin(yaw), 0.0);
  EXPECT_GT(std::abs(best.pose.linear().col(1).dot(box_x)), 0.97);
  EXPECT_LT((best.pose.translation().head<2>() - Eigen::Vector2d(0.5, 0.0)).norm(), 0.015);
  // Fingertips below the top, above the table.
  EXPECT_LT(best.pose.translation().z(), 0.08 - 0.006);
  EXPECT_GT(best.pose.translation().z(), 0.0);
  EXPECT_GT(best.antipodal_quality, 0.8);
}

TEST(GraspSynthesis, CylinderWidthMatchesDiameter)
{
  std::mt19937 rng(3);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  synthetic::addCylinder(*cloud, rng, 0.5, 0.05, 0.03, 0.10);
  const auto grasps = run(cloud);
  ASSERT_FALSE(grasps.empty());
  expectWellFormed(grasps.front());
  EXPECT_NEAR(grasps.front().width, 0.06, 0.006);
  EXPECT_LT((grasps.front().pose.translation().head<2>() - Eigen::Vector2d(0.5, 0.05)).norm(), 0.01);
}

TEST(GraspSynthesis, ObjectWiderThanGripperHasNoGrasp)
{
  std::mt19937 rng(4);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  synthetic::addBox(*cloud, rng, 0.5, 0.0, 0.10, 0.12, 0.05);
  EXPECT_TRUE(run(cloud).empty());
}

TEST(GraspSynthesis, VShapeIsNotPinchedAcrossTheGap)
{
  std::mt19937 rng(5);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  synthetic::addVShape(*cloud, rng, 0.42, 0.0);

  std::vector<ObjectGrasps> per_object;
  const auto grasps = run(cloud, &per_object);
  ASSERT_EQ(per_object.size(), 1u);
  ASSERT_FALSE(grasps.empty());
  for (const auto & g : grasps) {
    expectWellFormed(g);
    // Every grasp's centre must lie on the object, not in the gap between the arms.
    const Eigen::Vector2d c = per_object[0].contour.to2d(g.pose.translation());
    EXPECT_TRUE(grasp_perception::insidePolygon(per_object[0].efd_curve, c));
  }
}

TEST(GraspSynthesis, NeighbourBlocksFingerOnThatSide)
{
  std::mt19937 rng(6);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  // 4 x 4 cm box with a neighbour 1.5 cm away along +y: no room for a finger there.
  // The neighbour is only 3 mm lower, too little for a pinch above it, while the
  // fingers are long enough that the palm clears it when closing along x.
  synthetic::addBox(*cloud, rng, 0.5, 0.0, 0.04, 0.04, 0.06);
  synthetic::addBox(*cloud, rng, 0.5, 0.055, 0.04, 0.04, 0.057);

  const auto grasps = run(cloud);
  ASSERT_FALSE(grasps.empty());
  bool box_grasped = false;
  for (const auto & g : grasps) {
    if ((g.pose.translation().head<2>() - Eigen::Vector2d(0.5, 0.0)).norm() > 0.015) {
      continue;
    }
    // Closing along y would put a finger into the neighbour.
    EXPECT_LT(std::abs(g.pose.linear().col(1).y()), 0.9);
    if (!box_grasped) {
      // Best grasp on this box closes along x.
      EXPECT_GT(std::abs(g.pose.linear().col(1).x()), 0.95);
      box_grasped = true;
    }
  }
  EXPECT_TRUE(box_grasped);
}

TEST(GraspSynthesis, BacksOffToPinchAboveLowerNeighbour)
{
  std::mt19937 rng(7);
  auto cloud = std::make_shared<Cloud>();
  synthetic::addTable(*cloud, rng);
  // Neighbour 1.5 cm away and 3 cm lower: closing along y is possible, but only as a
  // shallow pinch above the neighbour.
  synthetic::addBox(*cloud, rng, 0.5, 0.0, 0.04, 0.04, 0.08);
  synthetic::addBox(*cloud, rng, 0.5, 0.055, 0.04, 0.04, 0.05);
  const auto grasps = run(cloud);
  bool found = false;
  for (const auto & g : grasps) {
    if ((g.pose.translation().head<2>() - Eigen::Vector2d(0.5, 0.0)).norm() < 0.015 &&
      std::abs(g.pose.linear().col(1).y()) > 0.95)
    {
      found = true;
      EXPECT_GT(g.pose.translation().z(), 0.05);  // fingertips above the neighbour
      EXPECT_LT(g.depth, 0.8);                    // 3 cm of the 4 cm max depth at most
    }
  }
  EXPECT_TRUE(found);
}
