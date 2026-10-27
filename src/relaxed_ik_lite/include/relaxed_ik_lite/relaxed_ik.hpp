#pragma once

#include <deque>
#include <optional>
#include <vector>

#include "relaxed_ik_lite/panda_kinematics.hpp"

namespace relaxed_ik_lite
{

/// Groove loss from Rakita et al., "RelaxedIK" (RSS 2018):
///   f(x) = -exp(-(x - s)^2 / (2 c^2)) + r (x - s)^4
/// The Gaussian gives a steep, well-conditioned pull near the target and the quartic
/// keeps a gradient far away from it, so terms with very different units can be summed.
struct GrooveParams
{
  double s = 0.0;
  double c = 0.2;
  double r = 5.0;
};

double grooveLoss(double x, const GrooveParams & p);
double grooveDerivative(double x, const GrooveParams & p);

struct ObjectiveWeights
{
  double position = 50.0;
  double orientation = 40.0;
  double joint_velocity = 5.0;
  double joint_acceleration = 4.0;
  double joint_jerk = 3.0;
  double joint_limits = 2.0;
  double manipulability = 0.0;  // off by default, costs a Jacobian per evaluation
};

struct RelaxedIkConfig
{
  ObjectiveWeights weights;
  GrooveParams position_groove{0.0, 0.2, 5.0};
  GrooveParams orientation_groove{0.0, 0.2, 5.0};
  GrooveParams motion_groove{0.0, 0.2, 5.0};
  int max_iterations = 300;
  double gradient_tolerance = 1e-7;
  /// Random restarts used by solveGoal() when the seeded solve misses the goal.
  int restarts = 16;
  double position_tolerance = 1e-3;     // [m]
  double orientation_tolerance = 1e-2;  // [rad]
  unsigned int seed = 42;
};

struct IkResult
{
  JointVector q;
  double position_error = 0.0;
  double orientation_error = 0.0;
  double cost = 0.0;
  int iterations = 0;
  bool converged = false;
};

/// Optimisation-based IK in the style of RelaxedIK.
///
/// Instead of solving for an exact pose, every call minimises a weighted sum of
/// groove losses: end-effector pose match, joint velocity/acceleration/jerk relative to
/// the previous solutions, and a soft joint-limit term. Because the motion terms pull
/// toward the last solution, a stream of nearby goals gives a smooth joint trajectory
/// with no IK branch flips, which is what we want for the Cartesian approach segment.
///
/// Not included (unlike the full RelaxedIK/CollisionIK): self- and environment-collision
/// terms. Callers validate the result against the MoveIt planning scene.
class RelaxedIk
{
public:
  explicit RelaxedIk(RelaxedIkConfig config = {}, PandaKinematics kinematics = PandaKinematics());

  /// Clear the motion history and treat q as the current (stationary) state.
  void reset(const JointVector & q);

  /// One streaming step: solve for goal seeded from (and smoothed toward) the history,
  /// then push the answer onto the history. This is the RelaxedIK "tracking" mode.
  IkResult track(const Eigen::Isometry3d & goal);

  /// One-shot goal: like track(), but without motion terms and with random restarts if
  /// the seeded solve fails. Used to turn a grasp pose into a joint-space goal for OMPL.
  /// Among acceptable solutions it returns the one closest to the current state.
  std::optional<IkResult> solveGoal(const Eigen::Isometry3d & goal);

  /// Follow a straight line from the current TCP pose to goal, using track() at most
  /// max_step metres (and max_step_rot radians) apart. Returns std::nullopt if any
  /// waypoint misses its target or the joints jump more than max_joint_step.
  std::optional<std::vector<JointVector>> trackLine(
    const Eigen::Isometry3d & goal, double max_step = 0.005, double max_step_rot = 0.05,
    double max_joint_step = 0.1);

  double cost(const JointVector & q, const Eigen::Isometry3d & goal, bool motion_terms) const;
  JointVector gradient(const JointVector & q, const Eigen::Isometry3d & goal, bool motion_terms) const;

  const JointVector & current() const { return history_.front(); }
  const PandaKinematics & kinematics() const { return kin_; }
  const RelaxedIkConfig & config() const { return config_; }

private:
  IkResult optimise(const JointVector & seed, const Eigen::Isometry3d & goal, bool motion_terms) const;
  void push(const JointVector & q);

  RelaxedIkConfig config_;
  PandaKinematics kin_;
  std::deque<JointVector> history_;  // front = most recent
};

/// Rotation angle between two orientations, in [0, pi].
double rotationDistance(const Eigen::Matrix3d & a, const Eigen::Matrix3d & b);

}  // namespace relaxed_ik_lite
