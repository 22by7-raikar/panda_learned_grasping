// Offline benchmark for the IK solver: success rate, accuracy and solve time on random
// reachable goals, plus smoothness of tracked straight-line approaches.
//
//   ik_benchmark [num_goals]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "relaxed_ik_lite/relaxed_ik.hpp"

using relaxed_ik_lite::JointVector;
using relaxed_ik_lite::kPandaDof;
using relaxed_ik_lite::PandaKinematics;
using relaxed_ik_lite::RelaxedIk;

namespace
{

double percentile(std::vector<double> v, double p)
{
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const auto idx = static_cast<size_t>(p * (v.size() - 1));
  return v[idx];
}

}  // namespace

int main(int argc, char ** argv)
{
  const int n = argc > 1 ? std::atoi(argv[1]) : 500;
  PandaKinematics kin;
  RelaxedIk ik;
  std::mt19937 rng(0);

  std::vector<double> goal_ms;
  std::vector<double> pos_err;
  int goal_ok = 0;
  for (int t = 0; t < n; ++t) {
    JointVector q;
    for (int i = 0; i < kPandaDof; ++i) {
      q[i] = std::uniform_real_distribution<double>(
        PandaKinematics::lowerLimits()[i] + 0.1, PandaKinematics::upperLimits()[i] - 0.1)(rng);
    }
    ik.reset(PandaKinematics::readyPose());
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = ik.solveGoal(kin.tcpPose(q));
    goal_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    if (r) {
      ++goal_ok;
      pos_err.push_back(r->position_error * 1e3);
    }
  }

  std::vector<double> tick_ms;
  std::vector<double> max_step;
  int line_ok = 0;
  const int lines = std::max(1, n / 5);
  for (int t = 0; t < lines; ++t) {
    ik.reset(PandaKinematics::readyPose());
    Eigen::Isometry3d goal = kin.tcpPose(ik.current());
    goal.translation() += Eigen::Vector3d(
      std::uniform_real_distribution<double>(-0.15, 0.15)(rng),
      std::uniform_real_distribution<double>(-0.15, 0.15)(rng),
      std::uniform_real_distribution<double>(-0.25, 0.0)(rng));
    const auto t0 = std::chrono::steady_clock::now();
    const auto path = ik.trackLine(goal);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!path) {
      continue;
    }
    ++line_ok;
    tick_ms.push_back(ms / path->size());
    JointVector prev = PandaKinematics::readyPose();
    double m = 0.0;
    for (const auto & q : *path) {
      m = std::max(m, (q - prev).cwiseAbs().maxCoeff());
      prev = q;
    }
    max_step.push_back(m);
  }

  std::printf("solveGoal: %d/%d solved (%.1f%%)\n", goal_ok, n, 100.0 * goal_ok / n);
  std::printf("  time   [ms] p50 %.2f  p95 %.2f\n", percentile(goal_ms, 0.5), percentile(goal_ms, 0.95));
  std::printf("  pos err[mm] p50 %.3f p95 %.3f\n", percentile(pos_err, 0.5), percentile(pos_err, 0.95));
  std::printf("trackLine: %d/%d lines tracked\n", line_ok, lines);
  std::printf("  per-waypoint [ms] p50 %.3f  p95 %.3f\n", percentile(tick_ms, 0.5), percentile(tick_ms, 0.95));
  std::printf("  max joint step [rad] p50 %.4f  p95 %.4f\n", percentile(max_step, 0.5), percentile(max_step, 0.95));
  return 0;
}
