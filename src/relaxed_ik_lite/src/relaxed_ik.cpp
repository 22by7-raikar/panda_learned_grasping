#include "relaxed_ik_lite/relaxed_ik.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <cmath>
#include <random>

namespace relaxed_ik_lite
{
namespace
{

constexpr double kFiniteDiffStep = 1e-6;
constexpr int kMaxLineSearchSteps = 30;
constexpr int kMaxTrackRepeats = 5;

JointVector clampToLimits(const JointVector & q)
{
  return q.cwiseMax(PandaKinematics::lowerLimits()).cwiseMin(PandaKinematics::upperLimits());
}

/// Zero the gradient components that would push a joint further past its active bound,
/// so convergence is judged on the feasible directions only.
JointVector projectGradient(const JointVector & q, const JointVector & g)
{
  JointVector out = g;
  const auto & lo = PandaKinematics::lowerLimits();
  const auto & hi = PandaKinematics::upperLimits();
  for (int i = 0; i < kPandaDof; ++i) {
    if ((q[i] <= lo[i] && g[i] > 0.0) || (q[i] >= hi[i] && g[i] < 0.0)) {
      out[i] = 0.0;
    }
  }
  return out;
}

double jointLimitPenalty(const JointVector & q)
{
  // Quadratic penalty once a joint is in the outer 10% of its range. The optimiser
  // also clamps to the hard limits; this keeps solutions away from them.
  const auto & lo = PandaKinematics::lowerLimits();
  const auto & hi = PandaKinematics::upperLimits();
  double penalty = 0.0;
  for (int i = 0; i < kPandaDof; ++i) {
    const double mid = 0.5 * (lo[i] + hi[i]);
    const double half = 0.5 * (hi[i] - lo[i]);
    const double x = std::abs(q[i] - mid) / half;
    if (x > 0.9) {
      penalty += 100.0 * (x - 0.9) * (x - 0.9);
    }
  }
  return penalty;
}

}  // namespace

double grooveLoss(double x, const GrooveParams & p)
{
  const double d = x - p.s;
  return -std::exp(-(d * d) / (2.0 * p.c * p.c)) + p.r * d * d * d * d;
}

double grooveDerivative(double x, const GrooveParams & p)
{
  const double d = x - p.s;
  return std::exp(-(d * d) / (2.0 * p.c * p.c)) * d / (p.c * p.c) + 4.0 * p.r * d * d * d;
}

double rotationDistance(const Eigen::Matrix3d & a, const Eigen::Matrix3d & b)
{
  // atan2 form stays accurate near zero, unlike acos((trace - 1) / 2).
  const Eigen::Matrix3d r = a.transpose() * b;
  const Eigen::Vector3d axis(r(2, 1) - r(1, 2), r(0, 2) - r(2, 0), r(1, 0) - r(0, 1));
  return std::atan2(0.5 * axis.norm(), 0.5 * (r.trace() - 1.0));
}

RelaxedIk::RelaxedIk(RelaxedIkConfig config, PandaKinematics kinematics)
: config_(config), kin_(kinematics)
{
  reset(PandaKinematics::readyPose());
}

void RelaxedIk::reset(const JointVector & q)
{
  history_.assign(3, clampToLimits(q));
}

void RelaxedIk::push(const JointVector & q)
{
  history_.push_front(q);
  history_.pop_back();
}

double RelaxedIk::cost(const JointVector & q, const Eigen::Isometry3d & goal, bool motion_terms) const
{
  const auto & w = config_.weights;
  const Eigen::Isometry3d tcp = kin_.tcpPose(q);
  const double pos_err = (tcp.translation() - goal.translation()).norm();
  const double ori_err = rotationDistance(tcp.linear(), goal.linear());

  double c = w.position * grooveLoss(pos_err, config_.position_groove) +
    w.orientation * grooveLoss(ori_err, config_.orientation_groove);

  if (motion_terms) {
    const JointVector & q1 = history_[0];
    const JointVector & q2 = history_[1];
    const JointVector & q3 = history_[2];
    const double vel = (q - q1).norm();
    const double acc = (q - 2.0 * q1 + q2).norm();
    const double jerk = (q - 3.0 * q1 + 3.0 * q2 - q3).norm();
    c += w.joint_velocity * grooveLoss(vel, config_.motion_groove) +
      w.joint_acceleration * grooveLoss(acc, config_.motion_groove) +
      w.joint_jerk * grooveLoss(jerk, config_.motion_groove);
  }

  c += w.joint_limits * jointLimitPenalty(q);
  if (w.manipulability > 0.0) {
    c -= w.manipulability * kin_.manipulability(q);
  }
  return c;
}

JointVector RelaxedIk::gradient(const JointVector & q, const Eigen::Isometry3d & goal, bool motion_terms) const
{
  // Analytic gradient: one FK + Jacobian per evaluation instead of 14 FK calls.
  const auto & w = config_.weights;
  const Eigen::Isometry3d tcp = kin_.tcpPose(q);
  const auto jac = kin_.jacobian(q);
  JointVector g = JointVector::Zero();

  const Eigen::Vector3d dp = tcp.translation() - goal.translation();
  const double pos_err = dp.norm();
  if (pos_err > 1e-12) {
    g += w.position * grooveDerivative(pos_err, config_.position_groove) *
      (jac.topRows<3>().transpose() * (dp / pos_err));
  }

  // d(angle)/dq = u^T J_w, with u the world-frame axis of R * R_goal^T.
  const Eigen::AngleAxisd err(tcp.linear() * goal.linear().transpose());
  if (err.angle() > 1e-12) {
    g += w.orientation * grooveDerivative(err.angle(), config_.orientation_groove) *
      (jac.bottomRows<3>().transpose() * err.axis());
  }

  if (motion_terms) {
    const JointVector & q1 = history_[0];
    const JointVector & q2 = history_[1];
    const JointVector & q3 = history_[2];
    const std::array<std::pair<double, JointVector>, 3> terms = {{
      {w.joint_velocity, q - q1},
      {w.joint_acceleration, q - 2.0 * q1 + q2},
      {w.joint_jerk, q - 3.0 * q1 + 3.0 * q2 - q3},
    }};
    for (const auto & [weight, v] : terms) {
      const double n = v.norm();
      if (n > 1e-12) {
        g += weight * grooveDerivative(n, config_.motion_groove) * (v / n);
      }
    }
  }

  const auto & lo = PandaKinematics::lowerLimits();
  const auto & hi = PandaKinematics::upperLimits();
  for (int i = 0; i < kPandaDof; ++i) {
    const double mid = 0.5 * (lo[i] + hi[i]);
    const double half = 0.5 * (hi[i] - lo[i]);
    const double x = std::abs(q[i] - mid) / half;
    if (x > 0.9) {
      g[i] += w.joint_limits * 200.0 * (x - 0.9) * (q[i] > mid ? 1.0 : -1.0) / half;
    }
  }

  if (w.manipulability > 0.0) {
    for (int i = 0; i < kPandaDof; ++i) {
      JointVector qp = q;
      JointVector qm = q;
      qp[i] += kFiniteDiffStep;
      qm[i] -= kFiniteDiffStep;
      g[i] -= w.manipulability * (kin_.manipulability(qp) - kin_.manipulability(qm)) / (2.0 * kFiniteDiffStep);
    }
  }
  return g;
}

IkResult RelaxedIk::optimise(const JointVector & seed, const Eigen::Isometry3d & goal, bool motion_terms) const
{
  // Projected BFGS with an Armijo backtracking line search. The original RelaxedIK uses
  // SLSQP; with only box constraints, BFGS plus clamping behaves the same and has no
  // dependency beyond Eigen.
  using Matrix7d = Eigen::Matrix<double, kPandaDof, kPandaDof>;
  JointVector x = clampToLimits(seed);
  double f = cost(x, goal, motion_terms);
  JointVector g = gradient(x, goal, motion_terms);
  Matrix7d h_inv = Matrix7d::Identity();

  IkResult result;
  int iter = 0;
  for (; iter < config_.max_iterations; ++iter) {
    if (projectGradient(x, g).norm() < config_.gradient_tolerance) {
      break;
    }
    // Joints sitting on a bound with the gradient pushing outward are held fixed for
    // this step; otherwise clamping would silently eat the step and stall BFGS.
    const JointVector g_free = projectGradient(x, g);
    JointVector d = -h_inv * g_free;
    for (int i = 0; i < kPandaDof; ++i) {
      if (g_free[i] == 0.0 && g[i] != 0.0) {
        d[i] = 0.0;
      }
    }
    if (g_free.dot(d) >= 0.0) {
      h_inv.setIdentity();
      d = -g_free;
    }

    JointVector x_new = x;
    double f_new = f;
    bool accepted = false;
    for (int attempt = 0; attempt < 2 && !accepted; ++attempt) {
      double step = 1.0;
      for (int ls = 0; ls < kMaxLineSearchSteps; ++ls) {
        x_new = clampToLimits(x + step * d);
        f_new = cost(x_new, goal, motion_terms);
        if (f_new <= f + 1e-4 * g.dot(x_new - x)) {
          accepted = true;
          break;
        }
        step *= 0.5;
      }
      if (!accepted) {
        // The quasi-Newton direction was poor; fall back to steepest descent once.
        h_inv.setIdentity();
        d = -g_free;
      }
    }
    if (!accepted) {
      break;
    }

    const JointVector g_new = gradient(x_new, goal, motion_terms);
    const JointVector s = x_new - x;
    const JointVector y = g_new - g;
    const double sy = s.dot(y);
    if (sy > 1e-12) {
      const double rho = 1.0 / sy;
      const Matrix7d i_rsy = Matrix7d::Identity() - rho * s * y.transpose();
      h_inv = i_rsy * h_inv * i_rsy.transpose() + rho * s * s.transpose();
    }
    x = x_new;
    f = f_new;
    g = g_new;
    if (s.norm() < 1e-10) {
      break;
    }
  }

  const Eigen::Isometry3d tcp = kin_.tcpPose(x);
  result.q = x;
  result.cost = f;
  result.iterations = iter;
  result.position_error = (tcp.translation() - goal.translation()).norm();
  result.orientation_error = rotationDistance(tcp.linear(), goal.linear());
  result.converged = result.position_error < config_.position_tolerance &&
    result.orientation_error < config_.orientation_tolerance;
  return result;
}

IkResult RelaxedIk::track(const Eigen::Isometry3d & goal)
{
  IkResult r = optimise(current(), goal, true);
  push(r.q);
  return r;
}

std::optional<IkResult> RelaxedIk::solveGoal(const Eigen::Isometry3d & goal)
{
  const JointVector q0 = current();
  IkResult seeded = optimise(q0, goal, false);
  if (seeded.converged) {
    return seeded;
  }

  std::mt19937 rng(config_.seed);
  std::optional<IkResult> best;
  const auto & lo = PandaKinematics::lowerLimits();
  const auto & hi = PandaKinematics::upperLimits();
  for (int k = 0; k < config_.restarts; ++k) {
    JointVector seed;
    for (int i = 0; i < kPandaDof; ++i) {
      seed[i] = std::uniform_real_distribution<double>(lo[i], hi[i])(rng);
    }
    IkResult r = optimise(seed, goal, false);
    if (!r.converged) {
      continue;
    }
    if (!best || (r.q - q0).norm() < (best->q - q0).norm()) {
      best = r;
    }
  }
  return best;
}

std::optional<std::vector<JointVector>> RelaxedIk::trackLine(
  const Eigen::Isometry3d & goal, double max_step, double max_step_rot, double max_joint_step)
{
  const auto saved_history = history_;
  const Eigen::Isometry3d start = kin_.tcpPose(current());
  const double dist = (goal.translation() - start.translation()).norm();
  const double angle = rotationDistance(start.linear(), goal.linear());
  const int n = std::max(
    {1, static_cast<int>(std::ceil(dist / max_step)), static_cast<int>(std::ceil(angle / max_step_rot))});

  const Eigen::Quaterniond q_start(start.linear());
  const Eigen::Quaterniond q_goal(goal.linear());

  std::vector<JointVector> waypoints;
  waypoints.reserve(n);
  JointVector prev = current();
  for (int i = 1; i <= n; ++i) {
    const double t = static_cast<double>(i) / n;
    Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
    target.translation() = (1.0 - t) * start.translation() + t * goal.translation();
    target.linear() = q_start.slerp(t, q_goal).toRotationMatrix();

    // RelaxedIK trades a little accuracy for smoothness on each tick; re-solving the
    // same target lets it settle, like the real solver does when the goal is held.
    IkResult r = track(target);
    for (int k = 1; k < kMaxTrackRepeats && !r.converged; ++k) {
      r = track(target);
    }
    if (!r.converged) {
      // Near singular poses the motion terms keep RelaxedIK lagging behind the target.
      // Finish with a pose-only solve seeded from the current solution: it stays on the
      // same IK branch, and the joint-step check below still guards continuity.
      const IkResult polished = optimise(current(), target, false);
      if (polished.converged) {
        history_.front() = polished.q;
        r = polished;
      }
    }
    if (!r.converged || (r.q - prev).cwiseAbs().maxCoeff() > max_joint_step) {
      history_ = saved_history;
      return std::nullopt;
    }
    waypoints.push_back(r.q);
    prev = r.q;
  }
  return waypoints;
}

}  // namespace relaxed_ik_lite
