// ROS 2 wrapper around the classical grasp pipeline.
//
// Subscribes to the wrist camera point cloud and serves ~/detect_grasps: on request it
// waits for a frame captured after the call (the arm has stopped at the observation
// pose by then), transforms it into the robot base frame, segments the tabletop, fits
// EFD outlines and returns ranked grasps. Markers show outlines, contacts and grippers.

#include <pcl/common/transforms.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "grasp_interfaces/msg/grasp.hpp"
#include "grasp_interfaces/srv/detect_grasps.hpp"
#include "grasp_perception/cloud_processing.hpp"
#include "grasp_perception/grasp_synthesis.hpp"

namespace grasp_perception
{

using sensor_msgs::msg::PointCloud2;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using DetectGrasps = grasp_interfaces::srv::DetectGrasps;

class PerceptionNode : public rclcpp::Node
{
public:
  explicit PerceptionNode(const rclcpp::NodeOptions & options)
  : Node("grasp_perception", options),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    base_frame_ = declare_parameter("base_frame", std::string("panda_link0"));
    frame_timeout_ = declare_parameter("frame_timeout", 2.0);
    num_markers_ = declare_parameter("num_grasp_markers", 5);
    loadParams();

    cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = cb_group_;
    cloud_sub_ = create_subscription<PointCloud2>(
      declare_parameter("cloud_topic", std::string("/camera/camera/depth/color/points")), rclcpp::SensorDataQoS(),
      [this](PointCloud2::ConstSharedPtr msg) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          latest_ = msg;
        }
        cv_.notify_all();
      },
      sub_opts);

    markers_pub_ = create_publisher<MarkerArray>("~/markers", 1);
    objects_pub_ = create_publisher<PointCloud2>("~/objects", 1);
    service_ = create_service<DetectGrasps>(
      "~/detect_grasps",
      [this](const DetectGrasps::Request::SharedPtr req, DetectGrasps::Response::SharedPtr res) {
        handle(*req, *res);
      },
      rmw_qos_profile_services_default, cb_group_);

    RCLCPP_INFO(get_logger(), "Ready: base frame '%s'", base_frame_.c_str());
  }

private:
  void loadParams()
  {
    auto & s = seg_params_;
    const auto crop_min = declare_parameter("segmentation.crop_min", std::vector<double>{0.15, -0.4, -0.05});
    const auto crop_max = declare_parameter("segmentation.crop_max", std::vector<double>{0.85, 0.4, 0.4});
    s.crop_min = Eigen::Vector3d(crop_min[0], crop_min[1], crop_min[2]).cast<float>();
    s.crop_max = Eigen::Vector3d(crop_max[0], crop_max[1], crop_max[2]).cast<float>();
    s.voxel_leaf = static_cast<float>(declare_parameter("segmentation.voxel_leaf", static_cast<double>(s.voxel_leaf)));
    s.outlier_mean_k = declare_parameter("segmentation.outlier_mean_k", s.outlier_mean_k);
    s.outlier_stddev = declare_parameter("segmentation.outlier_stddev", s.outlier_stddev);
    s.plane_distance = declare_parameter("segmentation.plane_distance", s.plane_distance);
    s.plane_max_angle = declare_parameter("segmentation.plane_max_angle", s.plane_max_angle);
    s.plane_max_iterations = declare_parameter("segmentation.plane_max_iterations", s.plane_max_iterations);
    s.min_height_above_plane = declare_parameter("segmentation.min_height_above_plane", s.min_height_above_plane);
    s.cluster_tolerance = declare_parameter("segmentation.cluster_tolerance", s.cluster_tolerance);
    s.min_cluster_size = declare_parameter("segmentation.min_cluster_size", s.min_cluster_size);
    s.max_cluster_size = declare_parameter("segmentation.max_cluster_size", s.max_cluster_size);
    s.normal_radius = declare_parameter("segmentation.normal_radius", s.normal_radius);

    auto & g = grasp_params_;
    g.gripper.max_width = declare_parameter("gripper.max_width", g.gripper.max_width);
    g.gripper.finger_length = declare_parameter("gripper.finger_length", g.gripper.finger_length);
    g.gripper.finger_width = declare_parameter("gripper.finger_width", g.gripper.finger_width);
    g.gripper.finger_thickness = declare_parameter("gripper.finger_thickness", g.gripper.finger_thickness);
    g.gripper.palm_depth = declare_parameter("gripper.palm_depth", g.gripper.palm_depth);
    g.gripper.palm_width = declare_parameter("gripper.palm_width", g.gripper.palm_width);
    g.gripper.palm_thickness = declare_parameter("gripper.palm_thickness", g.gripper.palm_thickness);
    g.gripper.open_clearance = declare_parameter("gripper.open_clearance", g.gripper.open_clearance);

    g.weights.antipodal = declare_parameter("weights.antipodal", g.weights.antipodal);
    g.weights.support = declare_parameter("weights.support", g.weights.support);
    g.weights.centroid = declare_parameter("weights.centroid", g.weights.centroid);
    g.weights.flatness = declare_parameter("weights.flatness", g.weights.flatness);
    g.weights.width_margin = declare_parameter("weights.width_margin", g.weights.width_margin);
    g.weights.depth = declare_parameter("weights.depth", g.weights.depth);

    g.contour.resolution = declare_parameter("contour.resolution", g.contour.resolution);
    g.contour.closing_radius = declare_parameter("contour.closing_radius", g.contour.closing_radius);
    g.contour.min_area = declare_parameter("contour.min_area", g.contour.min_area);
    g.contour.top_layer_thickness = declare_parameter("contour.top_layer_thickness", g.contour.top_layer_thickness);

    g.efd_harmonics = declare_parameter("grasp.efd_harmonics", g.efd_harmonics);
    g.contour_samples = declare_parameter("grasp.contour_samples", g.contour_samples);
    g.friction_coefficient = declare_parameter("grasp.friction_coefficient", g.friction_coefficient);
    g.min_width = declare_parameter("grasp.min_width", g.min_width);
    g.flatness_length = declare_parameter("grasp.flatness_length", g.flatness_length);
    g.max_pairs_per_object = declare_parameter("grasp.max_pairs_per_object", g.max_pairs_per_object);
    g.max_grasp_depth = declare_parameter("grasp.max_grasp_depth", g.max_grasp_depth);
    g.min_contact_depth = declare_parameter("grasp.min_contact_depth", g.min_contact_depth);
    g.depth_step = declare_parameter("grasp.depth_step", g.depth_step);
    g.pregrasp_distance = declare_parameter("grasp.pregrasp_distance", g.pregrasp_distance);
    g.table_clearance = declare_parameter("grasp.table_clearance", g.table_clearance);
    g.min_support_points = declare_parameter("grasp.min_support_points", g.min_support_points);
    g.support_saturation = declare_parameter("grasp.support_saturation", g.support_saturation);
    g.max_intruding_points = declare_parameter("grasp.max_intruding_points", g.max_intruding_points);
    g.nms_distance = declare_parameter("grasp.nms_distance", g.nms_distance);
    g.nms_angle = declare_parameter("grasp.nms_angle", g.nms_angle);
    g.max_candidates = declare_parameter("grasp.max_candidates", g.max_candidates);
    const auto hint = declare_parameter("grasp.x_axis_hint", std::vector<double>{1.0, 0.0, 0.0});
    g.x_axis_hint = Eigen::Vector3d(hint[0], hint[1], hint[2]);
  }

  PointCloud2::ConstSharedPtr waitForFreshCloud(const rclcpp::Time & after, double timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout);
    while (!latest_ || rclcpp::Time(latest_->header.stamp, get_clock()->get_clock_type()) < after) {
      if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
        return nullptr;
      }
    }
    return latest_;
  }

  void handle(const DetectGrasps::Request & req, DetectGrasps::Response & res)
  {
    const double timeout = req.timeout > 0.0 ? req.timeout : frame_timeout_;
    const auto msg = waitForFreshCloud(now(), timeout);
    if (!msg) {
      res.success = false;
      res.message = "no point cloud received after the request";
      return;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(
        base_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.5));
    } catch (const tf2::TransformException & e) {
      res.success = false;
      res.message = std::string("TF lookup failed: ") + e.what();
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    Cloud raw;
    pcl::fromROSMsg(*msg, raw);
    auto cloud = std::make_shared<Cloud>();
    const Eigen::Isometry3d base_from_camera = tf2::transformToEigen(tf);
    pcl::transformPointCloud(raw, *cloud, Eigen::Affine3f(base_from_camera.matrix().cast<float>()));

    const Eigen::Vector3f viewpoint = base_from_camera.translation().cast<float>();
    const Segmentation scene = segmentTabletop(cloud, seg_params_, viewpoint);
    std::vector<ObjectGrasps> per_object;
    const auto grasps = synthesizeGrasps(scene, grasp_params_, &per_object);
    res.compute_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std_msgs::msg::Header header;
    header.frame_id = base_frame_;
    header.stamp = msg->header.stamp;
    for (const auto & g : grasps) {
      const auto & obj = scene.objects[g.object_id];
      grasp_interfaces::msg::Grasp out;
      out.pose.header = header;
      out.pose.pose = tf2::toMsg(g.pose);
      out.width = g.width;
      out.score = g.score;
      out.object_id = g.object_id;
      out.source = "efd";
      out.contact_a = tf2::toMsg(g.contact_a);
      out.contact_b = tf2::toMsg(g.contact_b);
      out.object_center = tf2::toMsg(Eigen::Vector3d(0.5 * (obj.min_pt + obj.max_pt)));
      const Eigen::Vector3d dims = obj.max_pt - obj.min_pt;
      out.object_dimensions.x = dims.x();
      out.object_dimensions.y = dims.y();
      out.object_dimensions.z = dims.z();
      res.grasps.push_back(out);
    }
    res.success = !grasps.empty();
    res.message = std::to_string(scene.objects.size()) + " objects, " + std::to_string(grasps.size()) + " grasps";
    RCLCPP_INFO(get_logger(), "%s in %.1f ms", res.message.c_str(), 1e3 * res.compute_time);

    publishObjects(scene, header);
    publishMarkers(grasps, per_object, header);
  }

  void publishObjects(const Segmentation & scene, const std_msgs::msg::Header & header)
  {
    Cloud colored;
    for (size_t i = 0; i < scene.objects.size(); ++i) {
      const uint8_t r = static_cast<uint8_t>(80 + 60 * (i % 3));
      const uint8_t g = static_cast<uint8_t>(200 - 50 * (i % 4));
      for (const auto & p : *scene.objects[i].points) {
        PointT q;
        q.x = p.x;
        q.y = p.y;
        q.z = p.z;
        q.r = r;
        q.g = g;
        q.b = 220;
        colored.push_back(q);
      }
    }
    PointCloud2 out;
    pcl::toROSMsg(colored, out);
    out.header = header;
    objects_pub_->publish(out);
  }

  void publishMarkers(
    const std::vector<GraspCandidate> & grasps, const std::vector<ObjectGrasps> & per_object,
    const std_msgs::msg::Header & header)
  {
    MarkerArray arr;
    Marker clear;
    clear.action = Marker::DELETEALL;
    arr.markers.push_back(clear);
    int id = 0;

    // EFD outline of each object, drawn at its top height.
    for (const auto & obj : per_object) {
      Marker m;
      m.header = header;
      m.ns = "efd_outline";
      m.id = id++;
      m.type = Marker::LINE_STRIP;
      m.scale.x = 0.002;
      m.color.g = 0.8f;
      m.color.b = 1.0f;
      m.color.a = 1.0f;
      m.pose.orientation.w = 1.0;
      for (size_t k = 0; k <= obj.efd_curve.size(); ++k) {
        const auto & p2 = obj.efd_curve[k % obj.efd_curve.size()];
        m.points.push_back(tf2::toMsg(Eigen::Vector3d(obj.contour.to3d(p2, obj.contour.top_height))));
      }
      arr.markers.push_back(m);
    }

    const auto & gr = grasp_params_.gripper;
    const int n = std::min<int>(num_markers_, grasps.size());
    for (int i = 0; i < n; ++i) {
      const auto & g = grasps[i];
      const float alpha = i == 0 ? 1.0f : 0.35f;

      // Fingers and palm as boxes in the TCP frame.
      const double half_open = 0.5 * std::min(gr.max_width, g.width + 2.0 * gr.open_clearance);
      const std::array<std::pair<Eigen::Vector3d, Eigen::Vector3d>, 3> boxes = {{
        {{0.0, half_open + 0.5 * gr.finger_thickness, -0.5 * gr.finger_length},
          {gr.finger_width, gr.finger_thickness, gr.finger_length}},
        {{0.0, -half_open - 0.5 * gr.finger_thickness, -0.5 * gr.finger_length},
          {gr.finger_width, gr.finger_thickness, gr.finger_length}},
        {{0.0, 0.0, -gr.finger_length - 0.01},
          {gr.finger_width, 2.0 * (half_open + gr.finger_thickness), 0.02}},
      }};
      for (const auto & [offset, size] : boxes) {
        Marker m;
        m.header = header;
        m.ns = "grasps";
        m.id = id++;
        m.type = Marker::CUBE;
        Eigen::Isometry3d pose = g.pose;
        pose.translate(offset);
        m.pose = tf2::toMsg(pose);
        m.scale.x = size.x();
        m.scale.y = size.y();
        m.scale.z = size.z();
        m.color.r = i == 0 ? 0.1f : 0.9f;
        m.color.g = i == 0 ? 0.9f : 0.5f;
        m.color.b = 0.1f;
        m.color.a = alpha;
        arr.markers.push_back(m);
      }

      Marker contacts;
      contacts.header = header;
      contacts.ns = "contacts";
      contacts.id = id++;
      contacts.type = Marker::SPHERE_LIST;
      contacts.scale.x = contacts.scale.y = contacts.scale.z = 0.008;
      contacts.color.b = 1.0f;
      contacts.color.a = alpha;
      contacts.pose.orientation.w = 1.0;
      contacts.points.push_back(tf2::toMsg(g.contact_a));
      contacts.points.push_back(tf2::toMsg(g.contact_b));
      arr.markers.push_back(contacts);
    }
    markers_pub_->publish(arr);
  }

  std::string base_frame_;
  double frame_timeout_;
  int num_markers_;
  SegmentationParams seg_params_;
  GraspSynthesisParams grasp_params_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Subscription<PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr objects_pub_;
  rclcpp::Service<DetectGrasps>::SharedPtr service_;

  std::mutex mutex_;
  std::condition_variable cv_;
  PointCloud2::ConstSharedPtr latest_;
};

}  // namespace grasp_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<grasp_perception::PerceptionNode>(rclcpp::NodeOptions());
  // Multi-threaded so the cloud subscription keeps running while a service call waits.
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
