"""Synthetic wrist RealSense for running the pipeline on MoveIt mock hardware.

Renders the scene from the camera's current TF pose and publishes the same topics as
realsense2_camera 4.x (camera_namespace:=camera camera_name:=camera, pointcloud.enable
and align_depth.enable on):

    /camera/camera/depth/color/points                 sensor_msgs/PointCloud2 (xyz + rgb)
    /camera/camera/color/image_raw                    sensor_msgs/Image (rgb8)
    /camera/camera/aligned_depth_to_color/image_raw   sensor_msgs/Image (16UC1, mm)
    /camera/camera/color/camera_info                  sensor_msgs/CameraInfo

It also fakes the physics of a grasp: when the fingers close on an object (TCP inside
the object, finger opening narrower than the object) the object follows the hand, and
when they open it drops onto the table where it is.
"""

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

from grasp_sim.renderer import Intrinsics, object_footprint_contains, primitives_from_scene, render
import yaml
from grasp_interfaces.srv import SetScene

from grasp_sim.scene import load_scene, parse_scene

TCP_OFFSET = 0.1034  # panda_hand -> fingertip centre


def transform_to_matrix(tf) -> np.ndarray:
    q = tf.transform.rotation
    t = tf.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def make_cloud(header, points: np.ndarray, colors: np.ndarray) -> PointCloud2:
    rgb = ((colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8)
           | colors[:, 2].astype(np.uint32))
    data = np.zeros(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["rgb"] = rgb
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(points)
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


class SyntheticCamera(Node):
    def __init__(self):
        super().__init__("synthetic_camera")
        self.base_frame = self.declare_parameter("base_frame", "panda_link0").value
        self.camera_frame = self.declare_parameter("camera_frame", "camera_color_optical_frame").value
        self.hand_frame = self.declare_parameter("hand_frame", "panda_hand").value
        scene_file = self.declare_parameter("scene_file", "").value
        rate = self.declare_parameter("rate", 5.0).value
        prefix = self.declare_parameter("topic_prefix", "/camera/camera").value

        self.scene, self.table, self.poses = load_scene(scene_file)
        self.intr = Intrinsics()
        self.rng = np.random.default_rng(0)
        self.attached = None  # (name, hand_T_object)
        self.finger_opening = 0.08

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cloud_pub = self.create_publisher(PointCloud2, f"{prefix}/depth/color/points", qos_profile_sensor_data)
        self.color_pub = self.create_publisher(Image, f"{prefix}/color/image_raw", qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(
            Image, f"{prefix}/aligned_depth_to_color/image_raw", qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, f"{prefix}/color/camera_info", qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_service(SetScene, "~/set_scene", self.on_set_scene)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"Scene '{scene_file}' with {len(self.poses)} objects")

    def on_set_scene(self, req, res):
        try:
            self.scene, self.table, self.poses = parse_scene(yaml.safe_load(req.scene_yaml))
            self.attached = None
            res.success, res.message = True, f"{len(self.poses)} objects"
        except (yaml.YAMLError, KeyError, TypeError, ValueError) as e:
            res.success, res.message = False, str(e)
        return res

    def lookup(self, target: str, source: str):
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(f"TF {target} <- {source}: {e}", throttle_duration_sec=5.0)
            return None, None
        return transform_to_matrix(tf), tf.header.stamp

    def on_joints(self, msg: JointState):
        if "panda_finger_joint1" in msg.name:
            self.finger_opening = 2.0 * msg.position[msg.name.index("panda_finger_joint1")]

    def update_grasp(self):
        hand, _ = self.lookup(self.base_frame, self.hand_frame)
        if hand is None:
            return
        tcp = hand @ np.array([0.0, 0.0, TCP_OFFSET, 1.0])
        if self.attached is None and self.finger_opening < 0.075:
            for name, obj in self.scene["objects"].items():
                if object_footprint_contains(obj, self.poses[name], tcp[:3]):
                    self.attached = (name, np.linalg.inv(hand) @ self.poses[name])
                    self.get_logger().info(f"Grasped '{name}'")
                    break
        elif self.attached is not None:
            name, hand_t_obj = self.attached
            pose = hand @ hand_t_obj
            if self.finger_opening >= 0.075:
                # Released: drop upright onto the table.
                yaw = np.arctan2(pose[1, 0], pose[0, 0])
                c, s = np.cos(yaw), np.sin(yaw)
                pose = np.eye(4)
                pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                pose[:3, 3] = [*(hand @ hand_t_obj)[:2, 3], self.table.height]
                self.attached = None
                self.get_logger().info(f"Released '{name}'")
            self.poses[name] = pose

    def tick(self):
        self.update_grasp()
        cam, _ = self.lookup(self.base_frame, self.camera_frame)
        if cam is None:
            return
        frame = render(cam, primitives_from_scene(self.scene, self.poses), self.table, self.intr, rng=self.rng)

        h = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.camera_frame)
        self.cloud_pub.publish(make_cloud(h, frame.points, frame.point_colors))

        color = Image(header=h, height=self.intr.height, width=self.intr.width, encoding="rgb8",
                      is_bigendian=0, step=3 * self.intr.width, data=frame.color.tobytes())
        depth_mm = np.nan_to_num(frame.depth * 1000.0, nan=0.0).clip(0, 65535).astype(np.uint16)
        depth = Image(header=h, height=self.intr.height, width=self.intr.width, encoding="16UC1",
                      is_bigendian=0, step=2 * self.intr.width, data=depth_mm.tobytes())
        info = CameraInfo(header=h, height=self.intr.height, width=self.intr.width, distortion_model="plumb_bob")
        info.k = self.intr.k.flatten().tolist()
        info.d = [0.0] * 5
        info.r = np.eye(3).flatten().tolist()
        info.p = np.hstack([self.intr.k, np.zeros((3, 1))]).flatten().tolist()
        self.color_pub.publish(color)
        self.depth_pub.publish(depth)
        self.info_pub.publish(info)


def main():
    rclpy.init()
    node = SyntheticCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
