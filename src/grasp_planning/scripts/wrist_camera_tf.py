#!/usr/bin/env python3
"""Publish the hand-eye calibration as a static transform, read from parameters."""

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class WristCameraTf(Node):
    def __init__(self):
        super().__init__("wrist_camera_tf")
        parent = self.declare_parameter("parent_frame", "panda_hand").value
        child = self.declare_parameter("child_frame", "camera_color_optical_frame").value
        xyz = self.declare_parameter("xyz", [0.05, 0.0, 0.04]).value
        quat = self.declare_parameter("quat_xyzw", [0.0, 0.0, 0.0, 1.0]).value

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = quat
        self._broadcaster = StaticTransformBroadcaster(self)
        self._broadcaster.sendTransform(t)
        self.get_logger().info(f"{parent} -> {child}: xyz={list(xyz)} quat={list(quat)}")


def main():
    rclpy.init()
    rclpy.spin(WristCameraTf())


if __name__ == "__main__":
    main()
