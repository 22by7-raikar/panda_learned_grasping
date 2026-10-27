from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("grasp_perception"), "config", "perception.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("cloud_topic", default_value="/camera/camera/depth/color/points"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="grasp_perception",
            executable="perception_node",
            name="grasp_perception",
            output="screen",
            parameters=[
                config,
                {
                    "cloud_topic": LaunchConfiguration("cloud_topic"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                },
            ],
        ),
    ])
