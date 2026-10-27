#!/usr/bin/env python3
"""Run the grasp benchmarking protocol: every object at every placement, scored by the
pick_place node's lift / yaw / shake stages.

Start the pipeline in trial mode first:

    ros2 launch grasp_planning pick_place.launch.py serve_trials:=true protocol:=true

Then, real robot (operator places objects and judges each stage):

    ros2 run grasp_planning benchmark_protocol.py --algorithm efd --config franka_d435 \\
        --operator --out results/efd_franka_d435.csv

Simulation (objects placed automatically through the synthetic camera):

    ros2 run grasp_planning benchmark_protocol.py --algorithm efd --config sim --sim

Trials run as the paper's: every object at every placement, repeated `trials` times
(3 in the study: 10 x 6 x 3 = 180 per configuration). Results are appended one row per
trial; rerunning resumes where it stopped.
Summarise with protocol_report.py.
"""

import argparse
import csv
import os
import threading
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from grasp_interfaces.srv import RunTrial, SetScene
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

FIELDS = ["algorithm", "config", "object", "pose_id", "trial", "x", "y", "yaw", "grasp_success", "lift_passed",
          "yaw_passed", "shake_passed", "points", "failure", "detect_time", "planning_time", "timestamp"]


class Operator(Node):
    """Hosts /grasp_benchmark/is_object_held and calls the pick_place trial service."""

    def __init__(self):
        super().__init__("grasp_benchmark")
        self.trial = self.create_client(RunTrial, "/pick_place/run_trial")
        self.scene = self.create_client(SetScene, "/synthetic_camera/set_scene")
        self.prompt_lock = threading.Lock()
        self.create_service(Trigger, "/grasp_benchmark/is_object_held", self.ask_held)

    def ask_held(self, _req, res):
        with self.prompt_lock:
            answer = input("    Is the object still in the gripper? [y/n] ").strip().lower()
        res.success = answer.startswith("y")
        res.message = answer
        return res

    def call(self, client, request, timeout=None):
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"service {client.srv_name} not available")
        future = client.call_async(request)
        start = time.monotonic()
        while not future.done():
            if timeout and time.monotonic() - start > timeout:
                raise TimeoutError(client.srv_name)
            time.sleep(0.05)
        return future.result()


def done_trials(path, algorithm, config):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["object"], r["pose_id"], int(r["trial"])) for r in csv.DictReader(f)
                if r["algorithm"] == algorithm and r["config"] == config}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default=os.path.join(
        get_package_share_directory("grasp_planning"), "config", "benchmark_protocol.yaml"))
    p.add_argument("--algorithm", required=True, help="label, e.g. efd or frcnn")
    p.add_argument("--config", required=True, help="label for the setup, e.g. franka_d435_lab_light")
    p.add_argument("--out", default="benchmark_results.csv")
    p.add_argument("--sim", action="store_true", help="place objects via the synthetic camera")
    p.add_argument("--trials", type=int, help="repetitions of the full object x pose set (default: from the protocol file)")
    p.add_argument("--operator", action="store_true",
                   help="prompt for placement (real robot); use with protocol.held_check: operator")
    args = p.parse_args(argv)

    with open(args.protocol) as f:
        protocol = yaml.safe_load(f)
    n_trials = args.trials or protocol.get("trials", 1)
    trials = [(obj, pid, pose, t) for t in range(1, n_trials + 1)
              for obj in protocol["objects"] for pid, pose in protocol["poses"].items()]
    finished = done_trials(args.out, args.algorithm, args.config)

    rclpy.init()
    node = Operator()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    new_file = not os.path.exists(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, (obj, pid, (x, y, yaw), trial) in enumerate(trials):
            if (obj, pid, trial) in finished:
                continue
            print(f"[{i + 1}/{len(trials)}] trial {trial}: {obj} at {pid} (x={x:.3f}, y={y:.3f}, yaw={yaw:.2f})")
            if args.sim:
                scene = {"objects": {obj: {"ycb": obj, "pose": [x, y, yaw]}}}
                res = node.call(node.scene, SetScene.Request(scene_yaml=yaml.safe_dump(scene)))
                if not res.success:
                    raise RuntimeError(res.message)
                time.sleep(0.5)  # let a fresh frame go out
            elif args.operator:
                with node.prompt_lock:
                    input("    Place the object on the template, clear the workspace, press Enter ")
            result = node.call(node.trial, RunTrial.Request(object_name=obj, pose_id=pid))
            row = {
                "algorithm": args.algorithm, "config": args.config, "object": obj, "pose_id": pid, "trial": trial,
                "x": x, "y": y, "yaw": yaw, "grasp_success": result.grasp_success,
                "lift_passed": result.lift_passed, "yaw_passed": result.yaw_passed,
                "shake_passed": result.shake_passed, "points": result.points, "failure": result.failure,
                "detect_time": round(result.detect_time, 4), "planning_time": round(result.planning_time, 4),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            writer.writerow(row)
            f.flush()
            print(f"    -> {result.points}/{result.max_points} {result.failure}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
