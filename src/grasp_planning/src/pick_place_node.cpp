// Pick-and-place state machine for the Franka Panda with MoveIt 2.
//
// observe -> detect grasps -> plan -> approach -> close -> lift -> place -> release.
//
// Two ways to combine OMPL and the RelaxedIK solver (parameter `mode`):
//
//  ik_goal_ompl            RelaxedIK turns the pre-grasp and grasp poses into joint goals;
//                          OMPL plans both the transit and the final approach in joint space.
//  ompl_transit_ik_approach  OMPL plans the transit to the pre-grasp; RelaxedIK tracks a
//                          straight line from pre-grasp to grasp (and back up for the lift),
//                          each waypoint checked against the planning scene.
//
// Every attempt is appended to a CSV so the two modes and the two OMPL planners
// (RRTConnect, PRM) can be compared on the same scenes.
//
// With protocol.enabled the grasp is scored like the lab's benchmarking studies
// (Bekiroglu et al. RA-L 2020 protocol): after closing, the object is lifted, rotated
// about the wrist (yaw) and shaken (pitch), one point per stage it survives. With
// serve_trials the node waits for ~/run_trial calls from the benchmark driver instead
// of running cycles on its own.

#include <Eigen/Geometry>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <control_msgs/action/gripper_command.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit_msgs/srv/get_state_validity.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "grasp_interfaces/srv/detect_grasps.hpp"
#include "grasp_interfaces/srv/run_trial.hpp"
#include <std_srvs/srv/trigger.hpp>
#include "relaxed_ik_lite/relaxed_ik.hpp"

namespace grasp_planning
{

using grasp_interfaces::msg::Grasp;
using grasp_interfaces::srv::DetectGrasps;
using grasp_interfaces::srv::RunTrial;
using moveit::planning_interface::MoveGroupInterface;
using relaxed_ik_lite::JointVector;
using relaxed_ik_lite::RelaxedIk;
using Clock = std::chrono::steady_clock;

enum class Mode { IkGoalOmpl, OmplTransitIkApproach };

/// One row of the benchmark CSV.
struct AttemptLog
{
  std::string mode;
  std::string planner;
  std::string grasp_source;
  int grasp_rank = -1;
  double grasp_score = 0.0;
  double detect_time = 0.0;
  double transit_plan_time = 0.0;
  double approach_plan_time = 0.0;
  double transit_joint_length = 0.0;
  double approach_joint_length = 0.0;
  double approach_max_line_deviation = 0.0;  // TCP distance from the straight approach line
  bool planned = false;
  bool executed = false;
  bool lifted = false;  // gripper did not close fully, i.e. something is in the hand
  // Benchmark protocol (only filled when protocol.enabled).
  std::string object;
  std::string pose_id;
  bool lift_passed = false;
  bool yaw_passed = false;
  bool shake_passed = false;
  int points = 0;
  std::string failure;
};

struct ProtocolConfig
{
  bool enabled = false;
  double lift_height = 0.40;     // TCP height above the table after lifting [m]
  double yaw_angle = M_PI / 4;   // wrist rotation each way [rad]
  int shake_joint = 4;           // 0-based joint index
  double shake_angle = M_PI / 8; // amplitude each way [rad]
  int shake_cycles = 2;
  double shake_velocity_scaling = 0.6;
  std::string held_check = "gripper";  // "gripper" (finger width) or "operator"
};

double seconds(Clock::time_point since)
{
  return std::chrono::duration<double>(Clock::now() - since).count();
}

JointVector toJointVector(const std::vector<double> & v)
{
  JointVector q;
  for (int i = 0; i < relaxed_ik_lite::kPandaDof; ++i) {
    q[i] = v.at(i);
  }
  return q;
}

std::vector<double> toStd(const JointVector & q)
{
  return {q.data(), q.data() + q.size()};
}

class PickPlace
{
public:
  explicit PickPlace(const rclcpp::Node::SharedPtr & node)
  : node_(node),
    arm_(node, node->get_parameter("arm_group").as_string()),
    hand_(node, node->get_parameter("hand_group").as_string())
  {
    const auto p = [&](const std::string & name) { return node_->get_parameter(name); };
    mode_ = p("mode").as_string() == "ik_goal_ompl" ? Mode::IkGoalOmpl : Mode::OmplTransitIkApproach;
    planner_id_ = p("planner_id").as_string();
    observe_joints_ = p("observe_joints").as_double_array();
    place_joints_ = p("place_joints").as_double_array();
    pregrasp_distance_ = p("pregrasp_distance").as_double();
    lift_distance_ = p("lift_distance").as_double();
    gripper_mode_ = p("gripper_mode").as_string();
    max_grasp_attempts_ = static_cast<int>(p("max_grasp_attempts").as_int());
    velocity_scaling_ = p("velocity_scaling").as_double();
    acceleration_scaling_ = p("acceleration_scaling").as_double();
    csv_path_ = p("benchmark_csv").as_string();
    grasp_source_ = p("grasp_service").as_string();
    table_height_ = p("table_height").as_double();
    protocol_.enabled = p("protocol.enabled").as_bool();
    protocol_.lift_height = p("protocol.lift_height").as_double();
    protocol_.yaw_angle = p("protocol.yaw_angle").as_double();
    protocol_.shake_joint = static_cast<int>(p("protocol.shake_joint").as_int());
    protocol_.shake_angle = p("protocol.shake_angle").as_double();
    protocol_.shake_cycles = static_cast<int>(p("protocol.shake_cycles").as_int());
    protocol_.shake_velocity_scaling = p("protocol.shake_velocity_scaling").as_double();
    protocol_.held_check = p("protocol.held_check").as_string();

    arm_.setPlannerId(planner_id_);
    arm_.setPlanningTime(p("planning_time").as_double());
    arm_.setNumPlanningAttempts(static_cast<int>(p("planning_attempts").as_int()));
    arm_.setMaxVelocityScalingFactor(velocity_scaling_);
    arm_.setMaxAccelerationScalingFactor(acceleration_scaling_);

    detect_client_ = node_->create_client<DetectGrasps>(grasp_source_);
    validity_client_ = node_->create_client<moveit_msgs::srv::GetStateValidity>("/check_state_validity");
    operator_client_ = node_->create_client<std_srvs::srv::Trigger>("/grasp_benchmark/is_object_held");
    if (gripper_mode_ == "action") {
      gripper_client_ = rclcpp_action::create_client<control_msgs::action::GripperCommand>(
        node_, p("gripper_action").as_string());
    }

    addTable(p("table_height").as_double());
    RCLCPP_INFO(
      node_->get_logger(), "mode=%s planner=%s grasps from %s",
      mode_ == Mode::IkGoalOmpl ? "ik_goal_ompl" : "ompl_transit_ik_approach", planner_id_.c_str(),
      grasp_source_.c_str());
  }

  /// One full pick-and-place cycle. Returns false if nothing could be picked.
  bool runCycle(const std::string & object = "", const std::string & pose_id = "", AttemptLog * result = nullptr)
  {
    AttemptLog log;
    log.object = object;
    log.pose_id = pose_id;
    log.mode = mode_ == Mode::IkGoalOmpl ? "ik_goal_ompl" : "ompl_transit_ik_approach";
    log.planner = planner_id_;
    log.grasp_source = grasp_source_;

    if (result) {
      *result = log;
      result->failure = "no_grasp";
    }
    if (!moveToJoints(observe_joints_)) {
      RCLCPP_ERROR(node_->get_logger(), "Could not reach the observation pose");
      if (result) {
        result->failure = "observe";
      }
      return false;
    }
    openGripper(0.08);

    const auto t_detect = Clock::now();
    const auto grasps = detect();
    log.detect_time = seconds(t_detect);
    if (grasps.empty()) {
      RCLCPP_WARN(node_->get_logger(), "No grasps detected");
      return false;
    }

    const int n = std::min<int>(max_grasp_attempts_, grasps.size());
    for (int rank = 0; rank < n; ++rank) {
      AttemptLog attempt = log;
      attempt.grasp_rank = rank;
      attempt.grasp_score = grasps[rank].score;
      const bool picked = tryGrasp(grasps[rank], attempt);
      writeLog(attempt);
      if (result) {
        *result = attempt;
      }
      if (picked) {
        return true;
      }
      if (attempt.executed) {
        break;  // the robot moved; the scene may have changed, so re-observe
      }
    }
    return false;
  }

private:
  bool tryGrasp(const Grasp & grasp, AttemptLog & log)
  {
    Eigen::Isometry3d grasp_pose;
    tf2::fromMsg(grasp.pose.pose, grasp_pose);
    // Pre-grasp: back off along the approach axis (TCP z).
    const Eigen::Isometry3d pregrasp = grasp_pose * Eigen::Translation3d(0.0, 0.0, -pregrasp_distance_);

    RelaxedIk ik;
    const JointVector q_now = toJointVector(arm_.getCurrentJointValues());
    ik.reset(q_now);
    const auto q_pre = ik.solveGoal(pregrasp);
    if (!q_pre) {
      log.failure = "ik_pregrasp";
      return false;
    }

    // Transit: OMPL, current state -> pre-grasp joint goal (same in both modes).
    MoveGroupInterface::Plan transit;
    arm_.setStartStateToCurrentState();
    arm_.setJointValueTarget(toStd(q_pre->q));
    auto t = Clock::now();
    const bool transit_ok = static_cast<bool>(arm_.plan(transit));
    log.transit_plan_time = seconds(t);
    if (!transit_ok) {
      log.failure = "plan_transit";
      return false;
    }
    log.transit_joint_length = jointPathLength(transit.trajectory_);

    // Approach: pre-grasp -> grasp.
    moveit_msgs::msg::RobotTrajectory approach;
    t = Clock::now();
    const bool approach_ok = planApproach(q_pre->q, grasp_pose, approach);
    log.approach_plan_time = seconds(t);
    if (!approach_ok) {
      log.failure = "plan_approach";
      return false;
    }
    log.approach_joint_length = jointPathLength(approach);
    log.approach_max_line_deviation = lineDeviation(approach, pregrasp.translation(), grasp_pose.translation());
    log.planned = true;

    openGripper(std::min(0.08, grasp.width + 0.02));
    if (!static_cast<bool>(arm_.execute(transit)) || !static_cast<bool>(arm_.execute(approach))) {
      log.executed = true;
      log.failure = "execute_approach";
      return false;
    }
    log.executed = true;

    const bool holding = closeGripper(grasp.width);
    attachObject(grasp);

    Eigen::Isometry3d lifted = grasp_pose;
    lifted.translation().z() = protocol_.enabled ? table_height_ + protocol_.lift_height
                                                 : lifted.translation().z() + lift_distance_;
    if (!moveLinear(lifted)) {
      log.failure = "lift";
      openGripper(0.08);
      releaseObject();
      return false;
    }
    log.lifted = holding;
    if (protocol_.enabled) {
      runProtocolStages(grasp, holding, log);
    }

    if (!moveToJoints(place_joints_)) {
      log.failure = "place";
    }
    openGripper(0.08);
    releaseObject();
    return holding;
  }

  /// Lift, yaw and shake checks; one point per stage the object survives. A stage is
  /// only attempted if the previous one passed.
  void runProtocolStages(const Grasp & grasp, bool closed_on_object, AttemptLog & log)
  {
    log.lift_passed = closed_on_object && objectHeld(grasp, "lift");
    if (log.lift_passed) {
      std::vector<double> q = arm_.getCurrentJointValues();
      const double q7 = q.at(6);
      const double lo = relaxed_ik_lite::PandaKinematics::lowerLimits()[6];
      const double hi = relaxed_ik_lite::PandaKinematics::upperLimits()[6];
      bool ok = true;
      for (double target : {q7 + protocol_.yaw_angle, q7 - protocol_.yaw_angle, q7}) {
        q[6] = std::clamp(target, lo, hi);
        ok = ok && moveToJoints(q);
      }
      log.yaw_passed = ok && objectHeld(grasp, "yaw");
    }
    if (log.yaw_passed) {
      std::vector<double> q = arm_.getCurrentJointValues();
      const int j = protocol_.shake_joint;
      const double q0 = q.at(j);
      const double lo = relaxed_ik_lite::PandaKinematics::lowerLimits()[j];
      const double hi = relaxed_ik_lite::PandaKinematics::upperLimits()[j];
      arm_.setMaxVelocityScalingFactor(protocol_.shake_velocity_scaling);
      arm_.setMaxAccelerationScalingFactor(protocol_.shake_velocity_scaling);
      bool ok = true;
      for (int c = 0; c < protocol_.shake_cycles && ok; ++c) {
        for (double target : {q0 + protocol_.shake_angle, q0 - protocol_.shake_angle}) {
          q[j] = std::clamp(target, lo, hi);
          ok = ok && moveToJoints(q);
        }
      }
      q[j] = q0;
      ok = ok && moveToJoints(q);
      arm_.setMaxVelocityScalingFactor(velocity_scaling_);
      arm_.setMaxAccelerationScalingFactor(acceleration_scaling_);
      log.shake_passed = ok && objectHeld(grasp, "shake");
    }
    log.points = static_cast<int>(log.lift_passed) + static_cast<int>(log.yaw_passed) +
      static_cast<int>(log.shake_passed);
    RCLCPP_INFO(
      node_->get_logger(), "Protocol %s @ %s: lift %d yaw %d shake %d -> %d/3", log.object.c_str(),
      log.pose_id.c_str(), log.lift_passed, log.yaw_passed, log.shake_passed, log.points);
  }

  /// Is the object still in the hand? Either from the finger opening (the fingers close
  /// past the object width when it slips out) or by asking the operator.
  bool objectHeld(const Grasp & grasp, const std::string & stage)
  {
    if (protocol_.held_check == "operator") {
      if (!operator_client_->wait_for_service(std::chrono::seconds(5))) {
        RCLCPP_ERROR(node_->get_logger(), "No operator (/grasp_benchmark/is_object_held); counting as dropped");
        return false;
      }
      RCLCPP_INFO(node_->get_logger(), "Asking operator after stage '%s'", stage.c_str());
      auto future = operator_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
      future.wait();
      return future.get()->success;
    }
    const auto fingers = hand_.getCurrentJointValues();
    const double opening = fingers.empty() ? 0.0 : 2.0 * fingers.front();
    return opening > 0.5 * grasp.width;
  }

  bool planApproach(const JointVector & q_pre, const Eigen::Isometry3d & grasp_pose, moveit_msgs::msg::RobotTrajectory & out)
  {
    RelaxedIk ik;
    ik.reset(q_pre);
    if (mode_ == Mode::IkGoalOmpl) {
      const auto q_grasp = ik.solveGoal(grasp_pose);
      if (!q_grasp) {
        return false;
      }
      moveit::core::RobotState start(*arm_.getCurrentState());
      start.setJointGroupPositions(arm_.getName(), toStd(q_pre));
      arm_.setStartState(start);
      arm_.setJointValueTarget(toStd(q_grasp->q));
      MoveGroupInterface::Plan plan;
      const bool ok = static_cast<bool>(arm_.plan(plan));
      arm_.setStartStateToCurrentState();
      out = plan.trajectory_;
      return ok;
    }
    const auto path = ik.trackLine(grasp_pose);
    return path && buildTrajectory(q_pre, *path, out);
  }

  /// Straight TCP line between two poses: RelaxedIK in ompl_transit_ik_approach mode,
  /// OMPL to an IK goal otherwise.
  bool moveLinear(const Eigen::Isometry3d & to)
  {
    const JointVector q_now = toJointVector(arm_.getCurrentJointValues());
    RelaxedIk ik;
    ik.reset(q_now);
    moveit_msgs::msg::RobotTrajectory traj;
    if (mode_ == Mode::OmplTransitIkApproach) {
      const auto path = ik.trackLine(to);
      if (!path || !buildTrajectory(q_now, *path, traj)) {
        return false;
      }
      return static_cast<bool>(arm_.execute(traj));
    }
    const auto q = ik.solveGoal(to);
    return q && moveToJoints(toStd(q->q));
  }

  /// Time-parameterise RelaxedIK waypoints and check each against the planning scene.
  /// RelaxedIK itself has no environment collision term, so this check is required.
  bool buildTrajectory(const JointVector & q_start, const std::vector<JointVector> & path, moveit_msgs::msg::RobotTrajectory & out)
  {
    const auto model = arm_.getRobotModel();
    robot_trajectory::RobotTrajectory traj(model, arm_.getName());
    moveit::core::RobotState state(*arm_.getCurrentState());
    state.setJointGroupPositions(arm_.getName(), toStd(q_start));
    traj.addSuffixWayPoint(state, 0.0);
    for (const auto & q : path) {
      state.setJointGroupPositions(arm_.getName(), toStd(q));
      state.update();
      if (!isStateValid(state)) {
        RCLCPP_WARN(node_->get_logger(), "RelaxedIK waypoint in collision");
        return false;
      }
      traj.addSuffixWayPoint(state, 0.0);
    }
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    if (!totg.computeTimeStamps(traj, velocity_scaling_, acceleration_scaling_)) {
      return false;
    }
    traj.getRobotTrajectoryMsg(out);
    return true;
  }

  bool isStateValid(const moveit::core::RobotState & state)
  {
    if (!validity_client_->wait_for_service(std::chrono::seconds(2))) {
      RCLCPP_ERROR(node_->get_logger(), "/check_state_validity unavailable");
      return false;
    }
    auto req = std::make_shared<moveit_msgs::srv::GetStateValidity::Request>();
    moveit::core::robotStateToRobotStateMsg(state, req->robot_state, false);
    // As a diff, the check keeps move_group's attached objects (the grasped box).
    req->robot_state.is_diff = true;
    req->group_name = arm_.getName();
    auto future = validity_client_->async_send_request(req);
    if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
      return false;
    }
    return future.get()->valid;
  }

  std::vector<Grasp> detect()
  {
    if (!detect_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(node_->get_logger(), "%s unavailable", grasp_source_.c_str());
      return {};
    }
    auto req = std::make_shared<DetectGrasps::Request>();
    req->timeout = 3.0;
    auto future = detect_client_->async_send_request(req);
    if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      return {};
    }
    const auto res = future.get();
    RCLCPP_INFO(node_->get_logger(), "Detection: %s (%.1f ms)", res->message.c_str(), 1e3 * res->compute_time);
    return res->grasps;
  }

  bool moveToJoints(const std::vector<double> & joints)
  {
    arm_.setStartStateToCurrentState();
    arm_.setJointValueTarget(joints);
    MoveGroupInterface::Plan plan;
    return static_cast<bool>(arm_.plan(plan)) && static_cast<bool>(arm_.execute(plan));
  }

  void openGripper(double width)
  {
    setGripper(width, 0.0);
  }

  /// Returns true if the fingers stopped before closing fully, i.e. something is held.
  bool closeGripper(double object_width)
  {
    setGripper(std::max(0.0, object_width - 0.01), 40.0);
    const auto fingers = hand_.getCurrentJointValues();
    const double opening = fingers.empty() ? 0.0 : 2.0 * fingers.front();
    return opening > 0.5 * object_width;
  }

  void setGripper(double width, double effort)
  {
    if (gripper_mode_ == "action") {
      if (!gripper_client_->wait_for_action_server(std::chrono::seconds(2))) {
        RCLCPP_ERROR(node_->get_logger(), "Gripper action server unavailable");
        return;
      }
      control_msgs::action::GripperCommand::Goal goal;
      goal.command.position = 0.5 * width;  // per-finger position
      goal.command.max_effort = effort;
      auto handle = gripper_client_->async_send_goal(goal);
      if (handle.wait_for(std::chrono::seconds(2)) == std::future_status::ready && handle.get()) {
        auto result = gripper_client_->async_get_result(handle.get());
        result.wait_for(std::chrono::seconds(5));
      }
      return;
    }
    // MoveIt "hand" group (simulation / mock hardware).
    // panda_finger_joint2 mimics joint1, so the group has a single active variable.
    hand_.setJointValueTarget("panda_finger_joint1", 0.5 * width);
    MoveGroupInterface::Plan plan;
    if (static_cast<bool>(hand_.plan(plan))) {
      hand_.execute(plan);
    }
  }

  void addTable(double height)
  {
    moveit_msgs::msg::CollisionObject table;
    table.header.frame_id = arm_.getPlanningFrame();
    table.id = "table";
    shape_msgs::msg::SolidPrimitive box;
    box.type = shape_msgs::msg::SolidPrimitive::BOX;
    box.dimensions = {1.2, 1.4, 0.05};
    geometry_msgs::msg::Pose pose;
    pose.position.x = 0.5;
    pose.position.z = height - 0.025;
    pose.orientation.w = 1.0;
    table.primitives.push_back(box);
    table.primitive_poses.push_back(pose);
    table.operation = moveit_msgs::msg::CollisionObject::ADD;
    scene_.applyCollisionObject(table);
  }

  /// Approximate the grasped object by its bounding box and attach it to the hand,
  /// so the lift and place motions account for it.
  void attachObject(const Grasp & grasp)
  {
    moveit_msgs::msg::AttachedCollisionObject attached;
    attached.link_name = "panda_hand";
    attached.object.header.frame_id = arm_.getPlanningFrame();
    attached.object.id = "grasped_object";
    shape_msgs::msg::SolidPrimitive box;
    box.type = shape_msgs::msg::SolidPrimitive::BOX;
    // Shrunk slightly and lifted off the table, so the attached box does not start in
    // collision with the table it is resting on.
    box.dimensions = {
      std::max(0.01, grasp.object_dimensions.x - 0.005), std::max(0.01, grasp.object_dimensions.y - 0.005),
      std::max(0.01, grasp.object_dimensions.z - 0.01)};
    geometry_msgs::msg::Pose pose;
    pose.position = grasp.object_center;
    pose.position.z += 0.005;
    pose.orientation.w = 1.0;
    attached.object.primitives.push_back(box);
    attached.object.primitive_poses.push_back(pose);
    attached.object.operation = moveit_msgs::msg::CollisionObject::ADD;
    attached.touch_links = {"panda_hand", "panda_leftfinger", "panda_rightfinger"};
    scene_.applyAttachedCollisionObject(attached);
  }

  void releaseObject()
  {
    moveit_msgs::msg::AttachedCollisionObject detach;
    detach.object.id = "grasped_object";
    detach.link_name = "panda_hand";
    detach.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
    scene_.applyAttachedCollisionObject(detach);
    scene_.removeCollisionObjects({"grasped_object"});
  }

  static double jointPathLength(const moveit_msgs::msg::RobotTrajectory & traj)
  {
    const auto & pts = traj.joint_trajectory.points;
    double length = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) {
      double d2 = 0.0;
      for (size_t j = 0; j < pts[i].positions.size(); ++j) {
        const double d = pts[i].positions[j] - pts[i - 1].positions[j];
        d2 += d * d;
      }
      length += std::sqrt(d2);
    }
    return length;
  }

  /// Largest TCP distance from the straight pre-grasp -> grasp line. Near zero for the
  /// RelaxedIK approach; OMPL in joint space gives no such guarantee.
  static double lineDeviation(const moveit_msgs::msg::RobotTrajectory & traj, const Eigen::Vector3d & a, const Eigen::Vector3d & b)
  {
    relaxed_ik_lite::PandaKinematics kin;
    const Eigen::Vector3d dir = (b - a).normalized();
    double worst = 0.0;
    for (const auto & pt : traj.joint_trajectory.points) {
      if (pt.positions.size() < static_cast<size_t>(relaxed_ik_lite::kPandaDof)) {
        continue;
      }
      const Eigen::Vector3d v = kin.tcpPose(toJointVector(pt.positions)).translation() - a;
      worst = std::max(worst, (v - v.dot(dir) * dir).norm());
    }
    return worst;
  }

  void writeLog(const AttemptLog & l)
  {
    if (csv_path_.empty()) {
      return;
    }
    const bool fresh = !std::ifstream(csv_path_).good();
    std::ofstream f(csv_path_, std::ios::app);
    if (fresh) {
      f << "mode,planner,grasp_source,object,pose_id,grasp_rank,grasp_score,detect_time,transit_plan_time,"
        "approach_plan_time,transit_joint_length,approach_joint_length,approach_max_line_deviation,"
        "planned,executed,lifted,lift_passed,yaw_passed,shake_passed,points,failure\n";
    }
    f << l.mode << ',' << l.planner << ',' << l.grasp_source << ',' << l.object << ',' << l.pose_id << ','
      << l.grasp_rank << ',' << l.grasp_score << ','
      << l.detect_time << ',' << l.transit_plan_time << ',' << l.approach_plan_time << ','
      << l.transit_joint_length << ',' << l.approach_joint_length << ',' << l.approach_max_line_deviation << ','
      << l.planned << ',' << l.executed << ',' << l.lifted << ',' << l.lift_passed << ',' << l.yaw_passed << ','
      << l.shake_passed << ',' << l.points << ',' << l.failure << '\n';
  }

  rclcpp::Node::SharedPtr node_;
  MoveGroupInterface arm_;
  MoveGroupInterface hand_;
  moveit::planning_interface::PlanningSceneInterface scene_;
  rclcpp::Client<DetectGrasps>::SharedPtr detect_client_;
  rclcpp::Client<moveit_msgs::srv::GetStateValidity>::SharedPtr validity_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr operator_client_;
  rclcpp_action::Client<control_msgs::action::GripperCommand>::SharedPtr gripper_client_;

  Mode mode_;
  std::string planner_id_;
  std::string grasp_source_;
  std::string gripper_mode_;
  std::string csv_path_;
  std::vector<double> observe_joints_;
  std::vector<double> place_joints_;
  ProtocolConfig protocol_;
  double table_height_;
  double pregrasp_distance_;
  double lift_distance_;
  double velocity_scaling_;
  double acceleration_scaling_;
  int max_grasp_attempts_;
};

}  // namespace grasp_planning

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("pick_place", options);

  // Defaults for anything not given in the YAML.
  const auto declare = [&](const std::string & name, const rclcpp::ParameterValue & value) {
      if (!node->has_parameter(name)) {
        node->declare_parameter(name, value);
      }
    };
  declare("arm_group", rclcpp::ParameterValue(std::string("panda_arm")));
  declare("hand_group", rclcpp::ParameterValue(std::string("hand")));
  declare("mode", rclcpp::ParameterValue(std::string("ompl_transit_ik_approach")));
  declare("planner_id", rclcpp::ParameterValue(std::string("RRTConnectkConfigDefault")));
  declare("planning_time", rclcpp::ParameterValue(2.0));
  declare("planning_attempts", rclcpp::ParameterValue(1));
  declare("velocity_scaling", rclcpp::ParameterValue(0.3));
  declare("acceleration_scaling", rclcpp::ParameterValue(0.3));
  declare("observe_joints", rclcpp::ParameterValue(std::vector<double>{0.0, 0.125, 0.0, -0.745, 0.0, 0.871, 0.785}));
  declare("place_joints", rclcpp::ParameterValue(std::vector<double>{-0.9, 0.125, 0.0, -0.745, 0.0, 0.871, -0.115}));
  declare("pregrasp_distance", rclcpp::ParameterValue(0.10));
  declare("lift_distance", rclcpp::ParameterValue(0.10));
  declare("table_height", rclcpp::ParameterValue(0.0));
  declare("gripper_mode", rclcpp::ParameterValue(std::string("moveit")));
  declare("gripper_action", rclcpp::ParameterValue(std::string("/panda_gripper/gripper_action")));
  declare("grasp_service", rclcpp::ParameterValue(std::string("/grasp_perception/detect_grasps")));
  declare("max_grasp_attempts", rclcpp::ParameterValue(5));
  declare("cycles", rclcpp::ParameterValue(1));
  declare("benchmark_csv", rclcpp::ParameterValue(std::string("")));
  declare("serve_trials", rclcpp::ParameterValue(false));
  declare("protocol.enabled", rclcpp::ParameterValue(false));
  declare("protocol.lift_height", rclcpp::ParameterValue(0.40));
  declare("protocol.yaw_angle", rclcpp::ParameterValue(M_PI / 4));
  declare("protocol.shake_joint", rclcpp::ParameterValue(4));
  declare("protocol.shake_angle", rclcpp::ParameterValue(M_PI / 8));
  declare("protocol.shake_cycles", rclcpp::ParameterValue(2));
  declare("protocol.shake_velocity_scaling", rclcpp::ParameterValue(0.6));
  declare("protocol.held_check", rclcpp::ParameterValue(std::string("gripper")));

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  std::thread spinner([&exec]() { exec.spin(); });

  if (node->get_parameter("serve_trials").as_bool()) {
    // Benchmark mode: one trial per ~/run_trial call (see scripts/benchmark_protocol.py).
    grasp_planning::PickPlace app(node);
    std::mutex trial_mutex;
    auto group = node->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto srv = node->create_service<grasp_planning::RunTrial>(
      "~/run_trial",
      [&](const grasp_planning::RunTrial::Request::SharedPtr req, grasp_planning::RunTrial::Response::SharedPtr res) {
        std::lock_guard<std::mutex> lock(trial_mutex);
        grasp_planning::AttemptLog log;
        app.runCycle(req->object_name, req->pose_id, &log);
        res->grasp_success = log.lift_passed;
        res->lift_passed = log.lift_passed;
        res->yaw_passed = log.yaw_passed;
        res->shake_passed = log.shake_passed;
        res->points = log.points;
        res->max_points = 3;
        res->failure = log.failure;
        res->detect_time = log.detect_time;
        res->planning_time = log.transit_plan_time + log.approach_plan_time;
      },
      rmw_qos_profile_services_default, group);
    RCLCPP_INFO(node->get_logger(), "Waiting for ~/run_trial requests");
    spinner.join();
    return 0;
  }

  {
    grasp_planning::PickPlace app(node);
    const int cycles = static_cast<int>(node->get_parameter("cycles").as_int());
    for (int i = 0; (cycles <= 0 || i < cycles) && rclcpp::ok(); ++i) {
      RCLCPP_INFO(node->get_logger(), "Cycle %d", i + 1);
      if (!app.runCycle() && cycles <= 0) {
        break;  // continuous mode stops when the table is cleared
      }
    }
  }

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
