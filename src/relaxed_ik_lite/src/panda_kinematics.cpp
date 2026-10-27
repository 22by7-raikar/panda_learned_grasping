#include "relaxed_ik_lite/panda_kinematics.hpp"

#include <cmath>

namespace relaxed_ik_lite
{
namespace
{

struct DhRow
{
  double a;
  double d;
  double alpha;
};

// Modified (Craig) DH parameters for joints 1..7 followed by the fixed flange.
constexpr std::array<DhRow, kPandaDof + 1> kDh = {{
  {0.0, 0.333, 0.0},
  {0.0, 0.0, -M_PI_2},
  {0.0, 0.316, M_PI_2},
  {0.0825, 0.0, M_PI_2},
  {-0.0825, 0.384, -M_PI_2},
  {0.0, 0.0, M_PI_2},
  {0.088, 0.0, M_PI_2},
  {0.0, 0.107, 0.0},
}};

Eigen::Isometry3d dhTransform(const DhRow & row, double theta)
{
  // T = RotX(alpha) * TransX(a) * RotZ(theta) * TransZ(d)
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.rotate(Eigen::AngleAxisd(row.alpha, Eigen::Vector3d::UnitX()));
  t.translate(Eigen::Vector3d(row.a, 0.0, 0.0));
  t.rotate(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));
  t.translate(Eigen::Vector3d(0.0, 0.0, row.d));
  return t;
}

}  // namespace

PandaKinematics::PandaKinematics(double tcp_offset)
{
  hand_from_flange_ = Eigen::Isometry3d::Identity();
  hand_from_flange_.rotate(Eigen::AngleAxisd(-M_PI_4, Eigen::Vector3d::UnitZ()));
  tcp_from_hand_ = Eigen::Isometry3d::Identity();
  tcp_from_hand_.translate(Eigen::Vector3d(0.0, 0.0, tcp_offset));
}

std::array<Eigen::Isometry3d, kPandaDof + 1> PandaKinematics::linkPoses(const JointVector & q) const
{
  std::array<Eigen::Isometry3d, kPandaDof + 1> poses;
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  for (int i = 0; i < kPandaDof; ++i) {
    t = t * dhTransform(kDh[i], q[i]);
    poses[i] = t;
  }
  poses[kPandaDof] = t * dhTransform(kDh[kPandaDof], 0.0);
  return poses;
}

Eigen::Isometry3d PandaKinematics::flangePose(const JointVector & q) const
{
  return linkPoses(q)[kPandaDof];
}

Eigen::Isometry3d PandaKinematics::tcpPose(const JointVector & q) const
{
  return flangePose(q) * hand_from_flange_ * tcp_from_hand_;
}

Eigen::Matrix<double, 6, kPandaDof> PandaKinematics::jacobian(const JointVector & q) const
{
  const auto links = linkPoses(q);
  const Eigen::Vector3d p_tcp = (links[kPandaDof] * hand_from_flange_ * tcp_from_hand_).translation();
  Eigen::Matrix<double, 6, kPandaDof> j;
  for (int i = 0; i < kPandaDof; ++i) {
    // With modified DH, joint i rotates about the z axis of frame i.
    const Eigen::Vector3d z = links[i].linear().col(2);
    const Eigen::Vector3d p = links[i].translation();
    j.block<3, 1>(0, i) = z.cross(p_tcp - p);
    j.block<3, 1>(3, i) = z;
  }
  return j;
}

double PandaKinematics::manipulability(const JointVector & q) const
{
  const auto j = jacobian(q);
  const double det = (j * j.transpose()).determinant();
  return det > 0.0 ? std::sqrt(det) : 0.0;
}

const JointVector & PandaKinematics::lowerLimits()
{
  static const JointVector v =
    (JointVector() << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973).finished();
  return v;
}

const JointVector & PandaKinematics::upperLimits()
{
  static const JointVector v =
    (JointVector() << 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973).finished();
  return v;
}

const JointVector & PandaKinematics::readyPose()
{
  static const JointVector v =
    (JointVector() << 0.0, -M_PI_4, 0.0, -3.0 * M_PI_4, 0.0, M_PI_2, M_PI_4).finished();
  return v;
}

}  // namespace relaxed_ik_lite
