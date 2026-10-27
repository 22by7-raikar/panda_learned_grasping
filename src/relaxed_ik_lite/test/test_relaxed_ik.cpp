#include <gtest/gtest.h>

#include <random>

#include "relaxed_ik_lite/relaxed_ik.hpp"

using relaxed_ik_lite::grooveLoss;
using relaxed_ik_lite::GrooveParams;
using relaxed_ik_lite::JointVector;
using relaxed_ik_lite::kPandaDof;
using relaxed_ik_lite::PandaKinematics;
using relaxed_ik_lite::RelaxedIk;
using relaxed_ik_lite::rotationDistance;

namespace
{

JointVector randomConfig(std::mt19937 & rng, double margin = 0.2)
{
  JointVector q;
  for (int i = 0; i < kPandaDof; ++i) {
    const double lo = PandaKinematics::lowerLimits()[i];
    const double hi = PandaKinematics::upperLimits()[i];
    const double m = std::min(margin, 0.25 * (hi - lo));
    q[i] = std::uniform_real_distribution<double>(lo + m, hi - m)(rng);
  }
  return q;
}

}  // namespace

TEST(PandaKinematics, ReadyPosePointsGripperDown)
{
  PandaKinematics kin;
  const Eigen::Isometry3d flange = kin.flangePose(PandaKinematics::readyPose());
  // Published value for the Panda "ready" pose flange.
  EXPECT_NEAR(flange.translation().x(), 0.3069, 1e-3);
  EXPECT_NEAR(flange.translation().y(), 0.0, 1e-6);
  EXPECT_NEAR(flange.translation().z(), 0.5903, 1e-3);

  const Eigen::Isometry3d tcp = kin.tcpPose(PandaKinematics::readyPose());
  EXPECT_NEAR(tcp.linear().col(2).dot(-Eigen::Vector3d::UnitZ()), 1.0, 1e-6);
  EXPECT_NEAR(tcp.translation().z(), 0.5903 - PandaKinematics::kHandTcpOffset, 1e-3);
  // Fingers close along the world y axis in the ready pose.
  EXPECT_NEAR(std::abs(tcp.linear().col(1).dot(Eigen::Vector3d::UnitY())), 1.0, 1e-6);
}

TEST(PandaKinematics, JacobianMatchesFiniteDifferences)
{
  PandaKinematics kin;
  std::mt19937 rng(1);
  for (int trial = 0; trial < 10; ++trial) {
    const JointVector q = randomConfig(rng);
    const auto j = kin.jacobian(q);
    const double h = 1e-7;
    for (int i = 0; i < kPandaDof; ++i) {
      JointVector qp = q;
      qp[i] += h;
      const Eigen::Vector3d dp = (kin.tcpPose(qp).translation() - kin.tcpPose(q).translation()) / h;
      EXPECT_LT((dp - j.block<3, 1>(0, i)).norm(), 1e-5);
    }
  }
}

TEST(GrooveLoss, MinimumAtTarget)
{
  const GrooveParams p{0.3, 0.2, 5.0};
  EXPECT_DOUBLE_EQ(grooveLoss(0.3, p), -1.0);
  EXPECT_GT(grooveLoss(0.25, p), -1.0);
  EXPECT_GT(grooveLoss(0.35, p), -1.0);
  EXPECT_GT(grooveLoss(2.0, p), grooveLoss(1.0, p));
}

TEST(RelaxedIk, SolveGoalReachesRandomReachablePoses)
{
  RelaxedIk ik;
  PandaKinematics kin;
  std::mt19937 rng(7);
  int solved = 0;
  const int trials = 50;
  for (int t = 0; t < trials; ++t) {
    const Eigen::Isometry3d goal = kin.tcpPose(randomConfig(rng));
    ik.reset(PandaKinematics::readyPose());
    const auto r = ik.solveGoal(goal);
    if (!r) {
      continue;
    }
    ++solved;
    EXPECT_LT(r->position_error, 1e-3);
    EXPECT_LT(r->orientation_error, 1e-2);
    EXPECT_TRUE((r->q.array() >= PandaKinematics::lowerLimits().array()).all());
    EXPECT_TRUE((r->q.array() <= PandaKinematics::upperLimits().array()).all());
  }
  EXPECT_GE(solved, static_cast<int>(0.9 * trials));
}

TEST(RelaxedIk, SolveGoalRejectsUnreachablePose)
{
  RelaxedIk ik;
  Eigen::Isometry3d goal = Eigen::Isometry3d::Identity();
  goal.translation() = Eigen::Vector3d(1.5, 0.0, 0.5);
  EXPECT_FALSE(ik.solveGoal(goal).has_value());
}

TEST(RelaxedIk, TrackLineGivesStraightSmoothApproach)
{
  RelaxedIk ik;
  ik.reset(PandaKinematics::readyPose());
  const Eigen::Isometry3d start = ik.kinematics().tcpPose(ik.current());
  Eigen::Isometry3d goal = start;
  // 12 cm top-down approach, like pre-grasp -> grasp.
  goal.translation() += Eigen::Vector3d(0.05, 0.03, -0.12);

  const auto path = ik.trackLine(goal, 0.005);
  ASSERT_TRUE(path.has_value());
  ASSERT_GE(path->size(), 24u);

  const Eigen::Vector3d dir = (goal.translation() - start.translation()).normalized();
  JointVector prev = PandaKinematics::readyPose();
  for (const auto & q : *path) {
    const Eigen::Isometry3d tcp = ik.kinematics().tcpPose(q);
    const Eigen::Vector3d v = tcp.translation() - start.translation();
    const double off_line = (v - v.dot(dir) * dir).norm();
    EXPECT_LT(off_line, 1.5e-3);
    EXPECT_LT(rotationDistance(tcp.linear(), start.linear()), 1e-2);
    EXPECT_LT((q - prev).cwiseAbs().maxCoeff(), 0.05);
    prev = q;
  }
  const Eigen::Isometry3d end = ik.kinematics().tcpPose(path->back());
  EXPECT_LT((end.translation() - goal.translation()).norm(), 1e-3);
  EXPECT_TRUE(ik.current().isApprox(path->back()));
}

TEST(RelaxedIk, FailedTrackLineRestoresHistory)
{
  RelaxedIk ik;
  ik.reset(PandaKinematics::readyPose());
  Eigen::Isometry3d goal = ik.kinematics().tcpPose(ik.current());
  goal.translation() = Eigen::Vector3d(1.5, 0.0, 0.3);
  EXPECT_FALSE(ik.trackLine(goal).has_value());
  EXPECT_TRUE(ik.current().isApprox(PandaKinematics::readyPose()));
}

TEST(RelaxedIk, AnalyticGradientMatchesFiniteDifferences)
{
  relaxed_ik_lite::RelaxedIkConfig config;
  config.weights.joint_limits = 0.0;  // the penalty is only C1 at the 90% boundary
  RelaxedIk ik(config);
  std::mt19937 rng(3);
  ik.reset(randomConfig(rng));
  for (int trial = 0; trial < 10; ++trial) {
    const JointVector q = randomConfig(rng);
    const Eigen::Isometry3d goal = ik.kinematics().tcpPose(randomConfig(rng));
    for (bool motion : {false, true}) {
      const JointVector g = ik.gradient(q, goal, motion);
      for (int i = 0; i < kPandaDof; ++i) {
        JointVector qp = q;
        JointVector qm = q;
        qp[i] += 1e-6;
        qm[i] -= 1e-6;
        const double fd = (ik.cost(qp, goal, motion) - ik.cost(qm, goal, motion)) / 2e-6;
        EXPECT_NEAR(g[i], fd, 1e-4 * std::max(1.0, std::abs(fd)));
      }
    }
  }
}
