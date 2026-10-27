"""ROS 2 node for the learned pipeline: Faster R-CNN grasp rectangles -> 3D grasps.

Serves ~/detect_grasps (grasp_interfaces/DetectGrasps), the same interface as the
classical perception node, so the planner switches detectors by service name only.
"""

import threading
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
from grasp_interfaces.msg import Grasp
from grasp_interfaces.srv import DetectGrasps
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from grasp_learning.evaluate import load_model
from grasp_learning.model import make_rgd, predict
from grasp_learning.projection import ProjectionParams, rect_to_grasp, rotation_to_quaternion


def image_to_numpy(msg: Image) -> np.ndarray:
    if msg.encoding in ("rgb8", "bgr8"):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        return img[..., ::-1] if msg.encoding == "bgr8" else img
    if msg.encoding in ("16UC1", "mono16"):
        d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return d.astype(np.float32) / 1000.0
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width].copy()
    raise ValueError(f"unsupported encoding {msg.encoding}")


def matrix_to_pose(m: np.ndarray) -> Pose:
    x, y, z, w = rotation_to_quaternion(m[:3, :3])
    return Pose(position=Point(x=float(m[0, 3]), y=float(m[1, 3]), z=float(m[2, 3])),
                orientation=Quaternion(x=x, y=y, z=z, w=w))


def tf_to_matrix(tf) -> np.ndarray:
    q, t = tf.transform.rotation, tf.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = [t.x, t.y, t.z]
    return m


class FrcnnGraspNode(Node):
    def __init__(self):
        super().__init__("grasp_learning")
        self.base_frame = self.declare_parameter("base_frame", "panda_link0").value
        model_path = self.declare_parameter("model_path", "").value
        self.top_k = self.declare_parameter("top_k", 10).value
        self.min_score = self.declare_parameter("min_score", 0.3).value
        self.frame_timeout = self.declare_parameter("frame_timeout", 2.0).value
        prefix = self.declare_parameter("topic_prefix", "/camera/camera").value
        self.params = ProjectionParams(
            max_grasp_depth=self.declare_parameter("max_grasp_depth", 0.04).value,
            min_contact_depth=self.declare_parameter("min_contact_depth", 0.006).value,
            table_clearance=self.declare_parameter("table_clearance", 0.004).value,
        )
        device = self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu").value
        if not model_path:
            raise RuntimeError("model_path is required (train with `ros2 run grasp_learning train`)")
        self.device = torch.device(device)
        self.model = load_model(model_path, self.device)
        self.image_size = self.model.transform.min_size[0]

        self.lock = threading.Condition()
        self.latest = None
        group = ReentrantCallbackGroup()
        subs = [
            Subscriber(self, Image, f"{prefix}/color/image_raw", qos_profile=qos_profile_sensor_data,
                       callback_group=group),
            Subscriber(self, Image, f"{prefix}/aligned_depth_to_color/image_raw",
                       qos_profile=qos_profile_sensor_data, callback_group=group),
            Subscriber(self, CameraInfo, f"{prefix}/color/camera_info", qos_profile=qos_profile_sensor_data,
                       callback_group=group),
        ]
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=5, slop=0.05)
        self.sync.registerCallback(self.on_frame)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.markers_pub = self.create_publisher(MarkerArray, "~/markers", 1)
        self.create_service(DetectGrasps, "~/detect_grasps", self.handle, callback_group=group)
        self.get_logger().info(f"Loaded {model_path} on {self.device}")

    def on_frame(self, color, depth, info):
        with self.lock:
            self.latest = (color, depth, info)
            self.lock.notify_all()

    def wait_for_frame(self, after: Time, timeout: float):
        deadline = time.monotonic() + timeout
        with self.lock:
            while self.latest is None or Time.from_msg(self.latest[0].header.stamp) < after:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.lock.wait(remaining)
            return self.latest

    def handle(self, req, res):
        frame = self.wait_for_frame(self.get_clock().now(), req.timeout or self.frame_timeout)
        if frame is None:
            res.success, res.message = False, "no RGB-D frame received after the request"
            return res
        color_msg, depth_msg, info = frame
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, color_msg.header.frame_id,
                                                 Time.from_msg(color_msg.header.stamp), Duration(seconds=0.5))
        except TransformException as e:
            res.success, res.message = False, f"TF lookup failed: {e}"
            return res

        t0 = time.perf_counter()
        rgb, depth = image_to_numpy(color_msg), image_to_numpy(depth_msg)
        k = np.array(info.k).reshape(3, 3)
        # Square centre crop -> network input size, as in training.
        h, w = depth.shape
        side = min(h, w)
        x0, y0 = (w - side) // 2, (h - side) // 2
        rgd = make_rgd(rgb[y0:y0 + side, x0:x0 + side], depth[y0:y0 + side, x0:x0 + side],
                       depth_channel=self.model.depth_channel)
        rgd = torch.nn.functional.interpolate(rgd[None], size=(self.image_size, self.image_size),
                                              mode="bilinear", align_corners=False)[0]
        rects = predict(self.model, rgd.to(self.device), top_k=self.top_k, min_score=self.min_score)
        scale = side / self.image_size
        rects = [r.scaled(scale).shifted(x0, y0) for r in rects]

        base_t_cam = tf_to_matrix(tf)
        header = color_msg.header
        header.frame_id = self.base_frame
        for i, r in enumerate(rects):
            g = rect_to_grasp(r, depth, k, base_t_cam, self.params)
            if g is None:
                continue
            msg = Grasp(source="frcnn", width=float(g.width), score=float(r.score), object_id=i)
            msg.pose = PoseStamped(header=header, pose=matrix_to_pose(g.pose))
            msg.contact_a = Point(x=float(g.contact_a[0]), y=float(g.contact_a[1]), z=float(g.contact_a[2]))
            msg.contact_b = Point(x=float(g.contact_b[0]), y=float(g.contact_b[1]), z=float(g.contact_b[2]))
            msg.object_center = Point(x=float(g.object_center[0]), y=float(g.object_center[1]),
                                      z=float(g.object_center[2]))
            msg.object_dimensions = Vector3(x=float(g.object_dimensions[0]), y=float(g.object_dimensions[1]),
                                            z=float(g.object_dimensions[2]))
            res.grasps.append(msg)
        res.compute_time = time.perf_counter() - t0
        res.success = bool(res.grasps)
        res.message = f"{len(rects)} rectangles, {len(res.grasps)} grasps"
        self.get_logger().info(f"{res.message} in {1e3 * res.compute_time:.1f} ms")
        self.publish_markers(res.grasps)
        return res

    def publish_markers(self, grasps):
        arr = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, g in enumerate(grasps[:5]):
            # The closing line between the two contacts.
            m = Marker(header=g.pose.header, ns="frcnn_grasps", id=i, type=Marker.LINE_LIST,
                       pose=Pose(orientation=Quaternion(w=1.0)), points=[g.contact_a, g.contact_b])
            m.scale.x = 0.004
            m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 0.9, 0.2, 1.0) if i == 0 else (0.9, 0.5, 0.1, 0.35)
            arr.markers.append(m)
        self.markers_pub.publish(arr)


def main():
    rclpy.init()
    node = FrcnnGraspNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
