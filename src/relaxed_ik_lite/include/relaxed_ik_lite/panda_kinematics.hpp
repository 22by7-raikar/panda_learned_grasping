#pragma once

#include <Eigen/Geometry>
#include <array>

namespace relaxed_ik_lite
{

constexpr int kPandaDof = 7;
using JointVector = Eigen::Matrix<double, kPandaDof, 1>;

/// Franka Emika Panda forward kinematics (modified DH, as published by Franka).
///
/// Frames: panda_link0 (base) -> panda_link8 (flange) -> panda_hand -> panda_hand_tcp.
/// The TCP sits between the fingertips; its z axis is the approach direction and the
/// fingers close along its y axis.
class PandaKinematics
{
public:
  /// Distance from panda_hand to the fingertip centre (Franka Hand default).
  static constexpr double kHandTcpOffset = 0.1034;

  explicit PandaKinematics(double tcp_offset = kHandTcpOffset);

  /// Pose of panda_hand_tcp in panda_link0.
  Eigen::Isometry3d tcpPose(const JointVector & q) const;

  /// Pose of the flange (panda_link8) in panda_link0.
  Eigen::Isometry3d flangePose(const JointVector & q) const;

  /// Poses of link1..link7 and the flange; used for the Jacobian and debugging.
  std::array<Eigen::Isometry3d, kPandaDof + 1> linkPoses(const JointVector & q) const;

  /// 6x7 geometric Jacobian of the TCP (linear rows first) in the base frame.
  Eigen::Matrix<double, 6, kPandaDof> jacobian(const JointVector & q) const;

  /// Yoshikawa manipulability sqrt(det(J J^T)).
  double manipulability(const JointVector & q) const;

  static const JointVector & lowerLimits();
  static const JointVector & upperLimits();
  /// The "ready" configuration used by franka_ros / MoveIt demos.
  static const JointVector & readyPose();

private:
  Eigen::Isometry3d hand_from_flange_;
  Eigen::Isometry3d tcp_from_hand_;
};

}  // namespace relaxed_ik_lite
