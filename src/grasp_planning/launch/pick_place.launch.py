"""Full pipeline: MoveIt 2 (Panda), wrist camera TF, grasp detector and pick-and-place.

    ros2 launch grasp_planning pick_place.launch.py                 # simulated camera, mock arm
    ros2 launch grasp_planning pick_place.launch.py detector:=learned model_path:=/models/frcnn.pt
    ros2 launch grasp_planning pick_place.launch.py mode:=ik_goal_ompl planner_id:=PRMkConfigDefault
    ros2 launch grasp_planning pick_place.launch.py sim:=false launch_moveit:=false  # real robot
    ros2 launch grasp_planning pick_place.launch.py serve_trials:=true protocol:=true  # benchmark
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

ARGS = {
    "sim": ("true", "Synthetic wrist camera instead of a RealSense"),
    "launch_moveit": ("true", "Start move_group, RViz and mock controllers (moveit_resources demo)"),
    "detector": ("classical", "classical (PCL + EFD) or learned (Faster R-CNN)"),
    "model_path": ("", "Faster R-CNN checkpoint for detector:=learned"),
    "mode": ("ompl_transit_ik_approach", "ompl_transit_ik_approach or ik_goal_ompl"),
    "planner_id": ("RRTConnectkConfigDefault", "RRTConnectkConfigDefault or PRMkConfigDefault"),
    "cycles": ("1", "pick-and-place cycles; <= 0 runs until the table is clear"),
    "benchmark_csv": ("/tmp/pick_place_benchmark.csv", "where to append per-attempt results"),
    "scene": ("default", "simulated scene file in grasp_sim/config/scenes (without .yaml)"),
    "serve_trials": ("false", "wait for benchmark_protocol.py instead of running cycles"),
    "protocol": ("false", "score each grasp with the lift / yaw / shake stages"),
    "held_check": ("gripper", "gripper (finger width) or operator (asked by benchmark_protocol.py)"),
    "gripper_mode": ("moveit", "moveit (hand group, mock/sim) or action (GripperCommand, real Franka Hand)"),
}


def launch_setup(context):
    detector = LaunchConfiguration("detector").perform(context)
    sim = LaunchConfiguration("sim").perform(context) == "true"
    grasp_service = "/grasp_learning/detect_grasps" if detector == "learned" else "/grasp_perception/detect_grasps"

    moveit_config = MoveItConfigsBuilder("moveit_resources_panda").to_moveit_configs()
    nodes = []

    # Hand-eye calibration result (easy_handeye2): panda_hand -> camera optical frame.
    camera_tf = PathJoinSubstitution([FindPackageShare("grasp_planning"), "config", "wrist_camera.yaml"])
    nodes.append(Node(
        package="grasp_planning",
        executable="wrist_camera_tf.py",
        name="wrist_camera_tf",
        parameters=[camera_tf],
    ))

    if sim:
        scene = LaunchConfiguration("scene").perform(context)
        nodes.append(Node(
            package="grasp_sim",
            executable="synthetic_camera",
            name="synthetic_camera",
            output="screen",
            parameters=[{
                "scene_file": PathJoinSubstitution(
                    [FindPackageShare("grasp_sim"), "config", "scenes", scene + ".yaml"]),
            }],
        ))

    if detector == "learned":
        nodes.append(Node(
            package="grasp_learning",
            executable="frcnn_grasp_node",
            name="grasp_learning",
            output="screen",
            parameters=[
                PathJoinSubstitution([FindPackageShare("grasp_learning"), "config", "frcnn.yaml"]),
                {"model_path": LaunchConfiguration("model_path")},
            ],
        ))
    else:
        nodes.append(Node(
            package="grasp_perception",
            executable="perception_node",
            name="grasp_perception",
            output="screen",
            parameters=[PathJoinSubstitution([FindPackageShare("grasp_perception"), "config", "perception.yaml"])],
        ))

    pick_place = Node(
        package="grasp_planning",
        executable="pick_place_node",
        name="pick_place",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            PathJoinSubstitution([FindPackageShare("grasp_planning"), "config", "pick_place.yaml"]),
            {
                "mode": LaunchConfiguration("mode"),
                "planner_id": LaunchConfiguration("planner_id"),
                "grasp_service": grasp_service,
                "cycles": int(LaunchConfiguration("cycles").perform(context)),
                "benchmark_csv": LaunchConfiguration("benchmark_csv"),
                "serve_trials": LaunchConfiguration("serve_trials").perform(context) == "true",
                "protocol.enabled": LaunchConfiguration("protocol").perform(context) == "true",
                "protocol.held_check": LaunchConfiguration("held_check"),
                "gripper_mode": LaunchConfiguration("gripper_mode"),
            },
        ],
    )
    # Give move_group and the detector time to come up.
    nodes.append(TimerAction(period=8.0, actions=[pick_place]))
    return nodes


def generate_launch_description():
    moveit_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("moveit_resources_panda_moveit_config"), "launch", "demo.launch.py"])),
        condition=IfCondition(LaunchConfiguration("launch_moveit")),
    )
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=d, description=desc) for name, (d, desc) in ARGS.items()]
        + [moveit_demo, OpaqueFunction(function=launch_setup)]
    )
