"""Scene files (YAML) and offline rendering to PCD/PNG for the C++ tools."""

import argparse
import os

import numpy as np
import yaml

from grasp_sim.renderer import Intrinsics, Table, pose_xyz_yaw, primitives_from_scene, render

# Wrist camera optical frame in panda_link0 at the default observation pose
# (observe_joints in grasp_planning/config/pick_place.yaml with wrist_camera.yaml):
# 0.8 m above the table looking straight down, as in the benchmarking study.
OBSERVE_CAMERA_POSE = np.array([
    [0.0, 1.0, 0.0, 0.500],
    [1.0, 0.0, 0.0, 0.000],
    [0.0, 0.0, -1.0, 0.800],
    [0.0, 0.0, 0.0, 1.0],
])


def load_ycb_library(path=None):
    """Primitive approximations of YCB objects (config/ycb_objects.yaml)."""
    path = path or os.path.join(os.path.dirname(__file__), "..", "config", "ycb_objects.yaml")
    if not os.path.exists(path):
        try:
            from ament_index_python.packages import get_package_share_directory
            path = os.path.join(get_package_share_directory("grasp_sim"), "config", "ycb_objects.yaml")
        except (ImportError, LookupError):
            return {}
    with open(path) as f:
        return yaml.safe_load(f)


def parse_scene(scene: dict, ycb_library: dict = None):
    """Returns (scene dict, Table, {name: 4x4 pose}). Objects may say `ycb: <id>` to use
    the primitive approximation from the YCB library instead of listing parts."""
    t = scene.get("table", {})
    table = Table(
        height=t.get("height", 0.0),
        x_range=tuple(t.get("x_range", (0.15, 0.9))),
        y_range=tuple(t.get("y_range", (-0.55, 0.55))),
    )
    poses = {}
    for name, obj in scene["objects"].items():
        if "ycb" in obj:
            library = ycb_library if ycb_library is not None else load_ycb_library()
            obj.update({k: v for k, v in library[obj["ycb"]].items() if k not in obj})
        x, y, yaw = obj["pose"]
        poses[name] = pose_xyz_yaw(x, y, table.height, yaw)
    return scene, table, poses


def load_scene(path):
    with open(path) as f:
        return parse_scene(yaml.safe_load(f))


def write_pcd(path, points, colors):
    """ASCII PCD with x y z rgb (packed float), readable by pcl::io::loadPCDFile."""
    rgb = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2].astype(np.uint32)
    rgb_f = rgb.astype(np.uint32).view(np.float32)
    with open(path, "w") as f:
        f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA ascii\n")
        for p, c in zip(points, rgb_f):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c:.9e}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a scene from the observation pose to PCD (base frame).")
    parser.add_argument("scene", help="scene YAML")
    parser.add_argument("pcd", help="output .pcd (points in panda_link0)")
    parser.add_argument("--png", help="also save the colour image")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    scene, table, poses = load_scene(args.scene)
    frame = render(OBSERVE_CAMERA_POSE, primitives_from_scene(scene, poses), table, Intrinsics(),
                   rng=np.random.default_rng(args.seed))
    world = frame.points @ OBSERVE_CAMERA_POSE[:3, :3].T + OBSERVE_CAMERA_POSE[:3, 3]
    write_pcd(args.pcd, world, frame.point_colors)
    if args.png:
        from PIL import Image
        Image.fromarray(frame.color).save(args.png)
    cam = OBSERVE_CAMERA_POSE[:3, 3]
    print(f"{len(world)} points -> {args.pcd}; viewpoint {cam[0]:.3f} {cam[1]:.3f} {cam[2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
